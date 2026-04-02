#!/usr/bin/env python3
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import numpy as np
from tqdm import tqdm
import joblib

from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import r2_score, mean_squared_error

import os, sys
from pathlib import Path

sys.path.append("..")
TEST_ROOT = Path(__file__).resolve().parents[2]
SGL_SIM_ROOT = TEST_ROOT / "simulator-sgl"
BIN_ROOT = Path(__file__).resolve().parents[3] / "ross"

for _path in (TEST_ROOT, SGL_SIM_ROOT, BIN_ROOT):
    path_str = str(_path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from tests.load_traces_lmdb import parse_log_file, extract_params_from_path
import util
from bench_config import BenchmarkConfig, build_parser, DEFAULT_RESERVE

from common.models import BaseModel, get_model
from common.plot import plot_and_save_distribution, plot_x_vs_time
from common.config import InferenceConfig
from common.features import SGLRegressionFeatures, PlatformPerf
from print_summary import print_analysis_summary

from xgb_linear import perform_xgboost_regression

FEATURE_LIST = [
    # Workload
    "batch_size",
    "total_tokens",
    "avg_len",
    "max_len",
    "sq_sum",
    "sq_avg",
    
    # Model
    "num_hidden_layers",
    "hidden_size",
    "num_attention_heads",
    "num_key_value_heads",
    "intermediate_size",
    "tensor_parallel_size",
    "pipeline_parallel_size",
    
    "vocab_size",
    
    # Calculated Features
    "head_dim",
    "gqa_ratio",
    "ffn_expansion_ratio",
    "attn_flops_log",
    "mlp_flops_log",
    "kv_cache_log",
    "attn_memory_log",
    "total_params_log",
    "attn_flops_per_rank_log",
    "mlp_flops_per_rank_log",
    "kv_cache_per_rank_log",
    "tp_comm_log",
    "pipeline_overhead_proxy",
    "layers_x_hidden_sq",
    "layers_x_hidden_x_ffn",
    "batch_x_seq_x_hidden",
    
    # Platform Perf
    "platform_gemm_flops_per_ms",
    "platform_attention_flops_per_ms",
    "platform_memory_bandwidth_gbps",
    "theoretical_fp16_flops_per_ms",
    "theoretical_memory_bandwidth_gbps",
    "platform_nccl_max_bandwidth_gbps",
    "platform_nccl_avg_latency_ms"
]

def clean_data_list(all_data, target_key='time_ms', percentile=99.5):
    cleaned_data = [
        d for d in all_data
        if target_key in d and 0.0 < d[target_key] < 50.0 
    ]
    removed_count = len(all_data) - len(cleaned_data)
    print(f"=== Data Cleaning ({target_key}) ===")
    print(f"    Original samples : {len(all_data)}")
    print(f"    Removed outliers : {removed_count} ({(removed_count / len(all_data)):.2%})")
    return cleaned_data

def balance_data_distribution(all_data, target_key='time_ms'):
    print(f"\n=== Balancing Data Distribution ===")
    df = pd.DataFrame(all_data)
    n_total_input = len(df)

    mask_high = df[target_key] >= 2.0
    df_high = df[mask_high].copy()
    df_low = df[~mask_high].copy()
    
    n_high = len(df_high)
    target_n_low = int(n_high)
    
    print(f"Original Statistics:")
    print(f"  > 2.0ms (Keep All): {n_high:7d} ")
    print(f"  < 2.0ms (To Sample): {len(df_low):7d} (Target: ~{target_n_low})")
    print("-" * 50)

    if not df_low.empty:
        bins_fine = list(np.arange(0, 1.0, 0.1))
        bins_coarse = list(np.arange(1.0, 2.01, 0.2))
        bins = sorted(list(set([round(b, 2) for b in bins_fine + bins_coarse])))
        
        num_bins = len(bins) - 1
        

        calculated_max_samples = int(target_n_low / num_bins)
        calculated_max_samples = 3000 # int(calculated_max_samples * 1.05) 
        
        print(f"Dynamic Sampling Config:")
        print(f"  Total Bins (<2ms): {num_bins}")
        print(f"  Max Samples/Bin  : {calculated_max_samples}")
        
        df_low['bin'] = pd.cut(df_low[target_key], bins=bins, include_lowest=True, labels=False)

        def sample_group(group):
            if len(group) > calculated_max_samples:
                return group.sample(n=calculated_max_samples, random_state=42)
            return group

        df_low_balanced = df_low.groupby('bin', observed=True, group_keys=False).apply(sample_group)
        
        if 'bin' in df_low_balanced.columns:
            df_low_balanced = df_low_balanced.drop(columns=['bin'])
    else:
        df_low_balanced = df_low

    balanced_df = pd.concat([df_low_balanced, df_high], ignore_index=True)    
    n_final = len(balanced_df)
    n_final_high = len(df_high)
    n_final_low = len(df_low_balanced)
    
    print(f"\nBalanced Result:")
    print(f"  Total Samples: {n_final}")
    print(f"  > 2.0ms: {n_final_high:7d} ({n_final_high/n_final:.2%})")
    print(f"  < 2.0ms: {n_final_low:7d} ({n_final_low/n_final:.2%})")
    print(f"Reduction: {n_final / n_total_input:.2%}")
    print("====================================")

    return balanced_df.to_dict('records')

def parse_raw_features(log_file: str, model_uri: str, platform_perf: PlatformPerf, target: str, stage: str = '', disaggregation_mode: bool = False) -> List[Dict]:
    """
    Parse vLLM log file to extract profiler forward timing entries.
    """
    log_data = parse_log_file(log_file)
    data = []
    config_params = extract_params_from_path(log_file, disaggregation_mode, stage)
    if not config_params:
        return []
    inference_config = InferenceConfig(
        pp_size=config_params['pp'],
        tp_size=config_params['tp'],
    )
    model = get_model(model_uri, inference_config)
    for idx, timing_data in enumerate(log_data):
        if timing_data.get('stage_type', None) != stage and target != "post-forward":
            continue
        # Extract required fields
        seq_lens = timing_data.get('seq_lens', [])
        time_ms = timing_data.get('forward_gpu_ms')
        if target == 'post-forward':
            time_ms = timing_data.get('post_gpu_ms')
        if target == 'pre-forward':
            # if idx < len(log_data) - 1:
            #     total_time = (log_data[idx + 1]['iteration_start'] - timing_data['iteration_start']) * 1000
            # else:
            #     continue
            # total_time = timing_data.get('total_time_ms')
            # time_ms = total_time - timing_data.get('post_gpu_ms') - timing_data.get('forward_gpu_ms')
            time_ms = timing_data['pre_sched_ms'] + timing_data['pre_process_ms']
            if time_ms <= 0:
                continue

        feature_vals = SGLRegressionFeatures(
            batch_size=len(seq_lens),
            seq_lens=seq_lens,
            inference_config=inference_config,
            model=model,
            platform_perf=platform_perf
        )
        feature_dict = { 'time_ms': time_ms, 'source_file': log_file }
        for col in FEATURE_LIST:
            if col.startswith("platform_") or col.startswith('theoretical_'):
                feature_dict[col] = getattr(feature_vals.platform_perf, col)
            else:
                feature_dict[col] = getattr(feature_vals, col)

        entry = {
            'time_ms': time_ms,
            'seq_lens': timing_data.get('seq_lens'),
            'source_file': log_file,
            'model': model_uri,
            'features': feature_dict
        }
        # print(json.dumps(entry))
        data.append(entry)
    
    return data

def split_train_val(df: pd.DataFrame, feature_cols: List[str], val_ratio: float, group_split: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return X_train, X_val, y_train, y_val with optional group-wise split by model/parallel (from source_file path).
    """
    X = df[feature_cols].copy().fillna(0)
    y = df['time_ms'].astype(float).values
    if val_ratio <= 0 or val_ratio >= 1:
        return X.values, None, y, None
    if group_split and 'source_file' in df.columns:
        # Extract model/parallel group from source_file (segment starting with model_pp_tp)
        def get_config_group(p):
            try:
                return Path(p).resolve().parent
            except Exception:
                raise RuntimeError(f"grouping error on path: ${p}")

        groups = df['source_file'].apply(get_config_group)
        gss = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=42)
        train_idx, val_idx = next(gss.split(X, y, groups))
        print("split result:", len(train_idx), len(val_idx))
        return X.values[train_idx], X.values[val_idx], y[train_idx], y[val_idx]
    else:
        X_tr, X_va, y_tr, y_va = train_test_split(X.values, y, test_size=val_ratio, random_state=42)
        return X_tr, X_va, y_tr, y_va

def save_trained_model(results: Dict, output_dir: str):
    """
    Save the trained model, and metadata for later inference.
    """
    print(f"\nSaving trained model to {output_dir}...")
    
    model_name = results['model_name']
    model = results['model']
    feature_importance = results['feature_importance']
    
    # Create model directory
    model_dir = os.path.join(output_dir, f'{model_name}_model')
    os.makedirs(model_dir, exist_ok=True)
    
    # --- Save model file ---
    if model_name == 'xgboost':
        model_path = os.path.join(model_dir, 'model.json')
        model.save_model(model_path)
        print(f"  XGBoost model saved to: {model_path}")
    elif model_name == 'linear':
        model_path = os.path.join(model_dir, 'model.joblib')
        joblib.dump(model, model_path)
        print(f"  Linear model saved to: {model_path}")
    else:
        return
    
    # --- Save feature importance ---
    feature_importance_path = os.path.join(model_dir, 'feature_importance.csv')
    feature_importance.to_csv(feature_importance_path, index=False)
    print(f"  Feature importance saved to: {feature_importance_path}")
    
    # --- Save model metadata ---
    metadata = {
        'model_name': model_name,
        # Save relative filenames for portability
        'model_path': os.path.basename(model_path),
        'feature_importance_path': os.path.basename(feature_importance_path),
        'training_metrics': {
            'r2': results['r2'],
            'rmse': results['rmse'],
            'mae': results['mae'],
            'mape': results['mape'],
            'median_ape': results['median_ape']
        },
        'feature_names': list(results['X_original'].columns),
        'intercept': results.get('intercept'),
        'scale_needed': results.get('scale_needed', False)
    }
    
    # Add validation metrics if available
    if 'val_metrics' in results and results['val_metrics'] is not None:
        metadata['validation_metrics'] = results['val_metrics']
    
    metadata_path = os.path.join(model_dir, 'model_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"  Model metadata saved to: {metadata_path}")
    print(f"  Model and assets saved successfully in: {model_dir}")

def process_single_rank_log(_tuple):
    rank_log_path, model_uri, hw_yaml_path, target, stage, disaggregation_mode = _tuple
    platform_perf = PlatformPerf(hw_yaml_path)
    try:
        data = parse_raw_features(rank_log_path, model_uri, platform_perf, target, stage, disaggregation_mode)
        if len(data) == 0:
            return ([], rank_log_path)
        else:
            return (data, None)
    except Exception as e:
        raise RuntimeError(f"Error processing {rank_log_path}: {e}", file=sys.stderr)

def main():
    MACHINE_STATS_PREFIX = '/volume/ycao03/SiLLM-OP/test/collector'
    parser = build_parser()

    parser.add_argument('--target', type=str, default='forward', help='Regression Target')
    parser.add_argument('--saved_model_name', type=str, default='saved_models', help='Directory Name of Saved Models')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for plots')
    
    parser.add_argument('--regressor', type=str, default='linear', choices=['linear', 'xgb', 'ftt'], help='Regression model to use')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Validation split ratio (0-1). Set 0 to disable splitting.')
    parser.add_argument('--group_split', action='store_true', help='If set, split train/val by model/parallel groups to test cross-model generalization.')

    parser.add_argument('--stage', type=str, default='prefill', choices=['prefill', 'decode'], help='trainning target: prefill / decode')

    args = parser.parse_args()
    conf = BenchmarkConfig(args)
    
    util.echo_line(util.line_width, "-", "🔥 Benchmark Configuration")
    util.echo_info(conf.summary())
    
    logs = []
    for backend_opts in conf.backend_opts:
        if "--ross-config" in conf.args['sglang'][0]:
            conf_args = conf.args['sglang'][0]["--ross-config"]
            if "mem_fraction_static" in conf_args:
                mem_fraction_static_list = conf_args["mem_fraction_static"]
        for model in conf.models:
            for parallel in conf.parallel:
                for batch in conf.batches:
                    for input in conf.inputs:
                        for mem_fraction_static in mem_fraction_static_list:
                            conf.set_curr('sglang', model, parallel, batch, input)
                            conf.input = input

                            hw_yaml = f"{MACHINE_STATS_PREFIX}/{backend_opts[0].lower()}/platform_features.yaml"
                            assert(os.path.exists(hw_yaml) == True)

                            conf.test_dst = conf.test_dst.replace(conf.gpuname, backend_opts[0])
                            conf.test_dst = conf.test_dst.replace(conf.backend_info[conf.backends[0]]['version'], backend_opts[1])      
                            conf.test_dst = conf.test_dst.replace(str(DEFAULT_RESERVE), str(mem_fraction_static))
                            if not os.path.exists(conf.test_dst):
                                continue

                            rank_logs = []
                            for root, _, files in os.walk(conf.test_dst):
                                for f in files:
                                    if f.find(f"rank") != -1 and f.endswith(".txt") and root.find('tmp') == -1:
                                        rank_logs.append(os.path.join(root, f)) 
                            if len(rank_logs) == 0:
                                continue
                            logs.append({
                                "log_dir": conf.test_dst,
                                "model": model,
                                "hw_yaml": hw_yaml,
                                "parallel": parallel,
                                "disaggregation_mode": parallel.find("@") != -1,
                                "batch_size": batch,
                                "iosl": f"{input['isl'][0]}@{input['osl'][0]}",
                                "rank_logs": rank_logs,
                            })
    assert(logs)
    tasks, all_data, failed_files = [], [], []
    for log_items in logs:
        model_uri = log_items["model"]
        hw_yaml_path = log_items['hw_yaml']
        disaggregation_mode = log_items['disaggregation_mode']
        
        for rank_log in log_items["rank_logs"]:
            tasks.append((rank_log, model_uri, hw_yaml_path, args.target, args.stage, disaggregation_mode))

    num_workers = min(32, os.cpu_count() or 8)
    print(f"Total tasks: {len(tasks)}, workers: {num_workers}")

    try:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_task = {
                executor.submit(process_single_rank_log, task): task
                for task in tasks
            }
            for future in tqdm(as_completed(future_to_task), total=len(tasks), desc="Processing logs"):
                try:
                    data, failed = future.result()
                    all_data.extend(data)
                    if failed:
                        failed_files.append(failed)
                except Exception as e:
                    task = future_to_task[future]
                    raise RuntimeError(f"Unexpected error in task {task}: {e}", file=sys.stderr)
    except KeyboardInterrupt:
        print("\nUser interrupted. Exiting...", file=sys.stderr)
        sys.exit(1)

    if not all_data:
        raise RuntimeError("NULL Input data")

    if args.target == 'pre-forward':

        if args.stage == 'decode':
            # time_dist = [data['time_ms'] for data in all_data]
            # plot_and_save_distribution(time_dist, f'time_dist_{args.stage}.png')
            all_data = clean_data_list(all_data, target_key='time_ms')
            all_data = balance_data_distribution(all_data, target_key='time_ms')
        # exit(0)

    df = pd.DataFrame([data['features'] for data in all_data])
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    model_dir = os.path.join(args.output_dir, args.saved_model_name)
    os.makedirs(model_dir, exist_ok=True)    
    
    feature_cols = []
    for col in FEATURE_LIST:
        if col in df.columns:
            unique_vals = df[col].unique()
            std_val = df[col].std() if pd.api.types.is_numeric_dtype(df[col]) else 'N/A'
            print(f"  {col}: {len(unique_vals)} unique (std: {std_val})")
            if pd.api.types.is_numeric_dtype(df[col]) and df[col].std() > 0:
                feature_cols.append(col)
        else:
            assert(f"  {col}: NOT FOUND in data")
    
    # Remove features that don't exist or have no variance
    existing_features = [col for col in feature_cols if col in df.columns and df[col].std() > 0]
    if args.val_ratio > 0:
        print(f"\nCreating validation split: val_ratio={args.val_ratio}, group_split={args.group_split}")
        
        X_tr, X_va, y_tr, y_va = split_train_val(df, existing_features, args.val_ratio, args.group_split)
        val_data_for_eval = (X_va, y_va)

        # Train/evaluate based on regressor
        if args.regressor == 'xgb':
            # Use the dedicated function for XGBoost
            results = perform_xgboost_regression(
                df=pd.DataFrame(X_tr, columns=existing_features), 
                feature_cols=existing_features, 
                val_data=val_data_for_eval,
                y_values=y_tr,
                target=args.target
            )            
            # Add validation metrics if available
            if X_va is not None:
                y_va_pred = results['model'].predict(X_va)
                va_r2 = r2_score(y_va, y_va_pred)
                va_rmse = np.sqrt(mean_squared_error(y_va, y_va_pred))
                va_mae = np.mean(np.abs(y_va - y_va_pred))
                va_mape = np.mean(np.abs((y_va - y_va_pred) / y_va)) * 100
                va_median_ape = np.median(np.abs((y_va - y_va_pred) / y_va)) * 100
                results['val_metrics'] = {
                    'r2': va_r2,
                    'rmse': va_rmse,
                    'mae': va_mae,
                    'mape': va_mape,
                    'median_ape': va_median_ape
                }
        elif args.regressor == 'linear':
            print("\n=== Training Linear Regression Model ===")

            from sklearn.linear_model import LinearRegression
            model = LinearRegression() 
            model.fit(X_tr, y_tr)
            y_tr_pred = model.predict(X_tr)
            
            r2 = r2_score(y_tr, y_tr_pred)
            rmse = np.sqrt(mean_squared_error(y_tr, y_tr_pred))
            mae = np.mean(np.abs(y_tr - y_tr_pred))
            mape = np.mean(np.abs((y_tr - y_tr_pred) / y_tr)) * 100
            median_ape = np.median(np.abs((y_tr - y_tr_pred) / y_tr)) * 100
            print(f"Linear Model Train Metrics: R2={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")
            
            coef_df = pd.DataFrame({
                'Feature': existing_features,
                'Importance': model.coef_
            }).sort_values(by='Importance', key=abs, ascending=False)
            
            results = {
                'model': model,
                'model_name': 'linear',
                'feature_importance': coef_df,
                'X_original': pd.DataFrame(X_tr, columns=existing_features),
                'r2': r2, 'rmse': rmse, 'mae': mae, 
                'mape': mape, 'median_ape': median_ape,
                'intercept': model.intercept_,
                'scale_needed': False
            }
            
            if X_va is not None:
                y_va_pred = model.predict(X_va)
                va_r2 = r2_score(y_va, y_va_pred)
                va_rmse = np.sqrt(mean_squared_error(y_va, y_va_pred))
                va_mae = np.mean(np.abs(y_va - y_va_pred))
                va_mape = np.mean(np.abs((y_va - y_va_pred) / y_va)) * 100
                va_median_ape = np.median(np.abs((y_va - y_va_pred) / y_va)) * 100
                
                results['val_metrics'] = {
                    'r2': va_r2,
                    'rmse': va_rmse,
                    'mae': va_mae,
                    'mape': va_mape,
                    'median_ape': va_median_ape
                }
                print(f"Linear Model Val Metrics:   R2={va_r2:.4f}, RMSE={va_rmse:.4f}, MAE={va_mae:.4f}")
    else:
        raise RuntimeError("Should Specify VAL_RATIO")

    # Prepare data source info
    raw_data_for_summary = all_data if all_data else None
    print_analysis_summary(df, results, raw_data=raw_data_for_summary)

    # Save trained model in a subdirectory
    save_trained_model(results, model_dir)
    print(f"\nAnalysis complete! Check {args.output_dir}/ for output files.")
    print(f"Trained model saved in: {model_dir}")

if __name__ == "__main__":
    main() 