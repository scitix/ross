#!/usr/bin/env python3
import json
import argparse
import pandas as pd
import numpy as np

from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import r2_score, mean_squared_error

import os, sys
from pathlib import Path

TEST_ROOT = Path(__file__).resolve().parents[2]
# SGL_SIM_ROOT = TEST_ROOT / "simulator-sgl"
VLLM_SIM_ROOT = TEST_ROOT / "simulator-vllm"
BIN_ROOT = Path(__file__).resolve().parents[3] / "ross"

for _path in (TEST_ROOT, VLLM_SIM_ROOT, BIN_ROOT):
    path_str = str(_path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from tests.load_traces_lmdb import parse_log_file, extract_params_from_path
import util
from bench_config import BenchmarkConfig, build_parser

from common.models import BaseModel, get_model
from common.config import InferenceConfig
from common.features import RegressionFeatures, PlatformPerf
from print_summary import print_analysis_summary

from xgb_linear import perform_regression_analysis, perform_xgboost_regression


FEATURE_LIST = [
    # Workload
    "batch_size",
    "prefill_reqs",
    "decode_reqs",
    "prefill_total_tokens",
    "decode_total_tokens",
    "prefill_avg_len",
    "prefill_max_len",
    "decode_avg_len",
    "decode_max_len",
    
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
        # Extract required fields
        time_ms = timing_data.get('gpu_forward_time_ms', 0.0) # forward time
        if target == 'post-forward':
            time_ms = timing_data.get('total_sampling_time_ms', 0.0)
        if target == 'pre-forward':
            time_ms = timing_data.get('total_time_ms', 0.0) - timing_data['gpu_forward_time_ms'] - timing_data['total_sampling_time_ms']
            # if idx == len(log_data) - 1:
            #     continue
            # next_timing_data = log_data[idx + 1]
            # iteration_time = next_timing_data['sched_start_time'] - timing_data['sched_start_time']
            # time_ms = iteration_time * 1000 - timing_data['gpu_forward_time_ms'] - timing_data['total_sampling_time_ms']
            if time_ms < 0:
                continue

        feature_vals = RegressionFeatures(
            batch_size=timing_data.get('batch_size', 0),
            prefill_seq_lens = timing_data.get('prefill_seq_lens', []),
            decode_seq_lens = timing_data.get('decode_seq_lens', []),
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
            'prefill_seq_lens': timing_data.get('prefill_seq_lens', []),
            'decode_seq_lens': timing_data.get('decode_seq_lens', []),
            'source_file': log_file,
            'model': model_uri,
            'features': feature_dict
        }
        # print(json.dumps(entry))
        data.append(entry)
    
    return data

def split_train_val(df: pd.DataFrame, feature_cols: List[str], val_ratio: float, group_split: bool, stage: str = '', disaggregation_mode: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    Save the trained model, scaler, and metadata for later inference.
    (Refactored to use joblib and save the scaler)
    """
    print(f"\nSaving trained model to {output_dir}...")
    
    model_name = results['model_name']
    model = results['model']
    scaler = results['scaler']
    feature_importance = results['feature_importance']
    
    # Create model directory
    model_dir = os.path.join(output_dir, f'{model_name}_model')
    os.makedirs(model_dir, exist_ok=True)
    
    # --- Save model file ---
    if model_name == 'xgboost':
        model_path = os.path.join(model_dir, 'model.json')
        model.save_model(model_path)
        print(f"  XGBoost model saved to: {model_path}")
    else:  # linear
        raise RuntimeError("Only Use XGBoost Train")

    # --- Save scaler if it exists ---
    scaler_path = None
    if scaler is not None:
        scaler_path = os.path.join(model_dir, 'scaler.joblib')
        joblib.dump(scaler, scaler_path)
        print(f"  Scaler saved to: {scaler_path}")
    
    # --- Save feature importance ---
    feature_importance_path = os.path.join(model_dir, 'feature_importance.csv')
    feature_importance.to_csv(feature_importance_path, index=False)
    print(f"  Feature importance saved to: {feature_importance_path}")
    
    # --- Save model metadata ---
    metadata = {
        'model_name': model_name,
        # Save relative filenames for portability
        'model_path': os.path.basename(model_path),
        'scaler_path': os.path.basename(scaler_path) if scaler_path else None,
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

def _parse_input(s):
    dataset, seq = s.split("@", 1)
    dataset = dataset.strip()
    if "_" not in seq:
        raise ValueError(f"Invalid --input '{s}', missing '_' between isl and osl")
    isl_str, osl_str = seq.split("_", 1)
    return {
        "dataset": dataset,
        "isl": (int(isl_str), int(isl_str)),
        "osl": (int(osl_str), int(osl_str)),
    }

def main():
    MACHINE_STATS_PREFIX = '/volume/ycao03/SiLLM-OP/test/modeling/machine_stats'

    parser = build_parser()

    parser.add_argument('--target', type=str, default='forward', help='Regression Target')
    parser.add_argument('--saved_model_name', type=str, default='saved_models', help='Directory Name of Saved Models')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for plots')
    
    parser.add_argument('--regressor', type=str, default='linear', choices=['linear', 'xgb', 'ftt'], help='Regression model to use')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Validation split ratio (0-1). Set 0 to disable splitting.')
    parser.add_argument('--group_split', action='store_true', help='If set, split train/val by model/parallel groups to test cross-model generalization.')
    # for target
    parser.add_argument('--stage', type=str, default='prefill', choices=['prefill', 'decode'], help='trainning target: prefill / decode')
    parser.add_argument('--disaggregation_mode', action="store_true", help='Disaggregation Mode')
    
    args = parser.parse_args()
    conf = BenchmarkConfig(args)
    
    util.echo_line(util.line_width, "-", "🔥 Benchmark Configuration")
    util.echo_info(conf.summary())
    
    logs = []
    for backend_opts in [('B200', '0.11.3.dev0+g275de3417.d20260113.precompiled'), ('H200', '0.11.2+precompiled')]:
        for model in conf.models:
            for parallel in conf.parallel:
                for batch in conf.batches:
                    for input in conf.inputs:
                        conf.set_curr('vllm', model, parallel, batch, input)
                        conf.input = input

                        hw_yaml = f"{MACHINE_STATS_PREFIX}/{backend_opts[0].lower()}/platform_features.yaml"
                        assert(os.path.exists(hw_yaml) == True)

                        conf.test_dst = conf.test_dst.replace(conf.gpuname, backend_opts[0])
                        conf.test_dst = conf.test_dst.replace(conf.backend_info[conf.backends[0]]['version'], backend_opts[1])
                        
                        rank_logs = []
                        if args.disaggregation_mode:
                            rank_log_prefix = args.stage + "_"
                        else:
                            rank_log_prefix = ""
                        for root, _, files in os.walk(conf.test_dst):
                            for f in files:
                                if f.startswith(f"{rank_log_prefix}rank") and f.endswith(".txt") and root.find('tmp') == -1:
                                    rank_logs.append(os.path.join(root, f)) 

                        logs.append({
                            "log_dir": conf.test_dst,
                            "model": model,
                            "hw_yaml": hw_yaml,
                            "parallel": parallel,
                            "batch_size": batch,
                            "iosl": f"{input['isl'][0]}@{input['osl'][0]}",
                            "rank_logs": rank_logs,
                        })
    assert(logs)

    failed_files, all_data = [], []
    for log_items in logs:
        print(log_items)
        model_uri = log_items["model"]
        platform_perf = PlatformPerf(log_items['hw_yaml'])

        for rank_log in log_items["rank_logs"]:
            data = parse_raw_features(rank_log, model_uri, platform_perf, args.target, args.stage, args.disaggregation_mode)
            if len(data) == 0:
                failed_files.append(rank_log)
            else:
                all_data.extend(data)
    
    # Process log data if we have any
    if all_data:
        df = pd.DataFrame([data['features'] for data in all_data])
    else:
        raise RuntimeError("Should have all data")

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
        
        X_tr, X_va, y_tr, y_va = split_train_val(df, existing_features, args.val_ratio, args.group_split, args.stage, args.disaggregation_mode)
        val_data_for_eval = (X_va, y_va)

        # Train/evaluate based on regressor
        if args.regressor == 'xgb':
            # Use the dedicated function for XGBoost
            results = perform_xgboost_regression(
                df=pd.DataFrame(X_tr, columns=existing_features), 
                feature_cols=existing_features, 
                val_data=val_data_for_eval,
                y_values=y_tr
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