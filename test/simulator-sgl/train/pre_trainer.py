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

sys.path.append("..")
TEST_ROOT = Path(__file__).resolve().parents[2]
COMMON_ROOT = TEST_ROOT / "common"
SGL_SIM_ROOT = TEST_ROOT / "simulator-sgl"
for _path in (TEST_ROOT, COMMON_ROOT, SGL_SIM_ROOT):
    path_str = str(_path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from parser import parse_log_file, find_dataset_logs, extract_params_from_path

from models import BaseModel, get_model
from config import InferenceConfig
from features import SGLRegressionFeatures, PlatformPerf
from print_summary import print_analysis_summary

from xgb_linear import perform_regression_analysis, perform_xgboost_regression

FEATURE_LIST = [
    # Workload
    "batch_size",
    "total_tokens",
    "avg_len",
    "max_len",
    
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

def parse_raw_features(log_file: str, model_uri: str, model_name: str, platform_perf: PlatformPerf, target: str, stage: str = '', disaggregation_mode: bool = False) -> List[Dict]:
    """
    Parse vLLM log file to extract profiler forward timing entries.
    """
    log_data = parse_log_file(log_file)
    data = []
    config_params = extract_params_from_path(log_file, stage, disaggregation_mode)
    if not config_params:
        return []
    inference_config = InferenceConfig(
        dp_size=config_params['dp'],
        pp_size=config_params['pp'],
        tp_size=config_params['tp'],
    )
    model = get_model(model_uri, inference_config)
    for timing_data in log_data:
        if timing_data.get('stage_type', None) != stage and target == "forward":
            continue
        # Extract required fields
        seq_lens = timing_data.get('seq_lens', [])
        time_ms = timing_data.get('forward_gpu_ms', 0.0) # forward time
        if target == 'post-forward':
            time_ms = timing_data.get('post_gpu_ms', 0.0)
        if target == 'pre-forward':
            time_ms = max(timing_data.get('pre_sched_ms', 0.0), 0) + max(timing_data.get('pre_process_ms', 0.0), 0)

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
            'seq_lens': timing_data.get('seq_lens', []),
            'source_file': log_file,
            'model_type': model_name,
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
                params = extract_params_from_path(p, stage, disaggregation_mode)
                parts = str(p).split(os.sep)
                for part in parts:
                    if part.startswith('vllm_') or part.startswith('sglang_'):
                        return part + f"_pp{params['pp']}_tp{params['tp']}"
            except Exception:
                pass
            print("Error occurs when finding model group")
            return 'unknown'
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
    import joblib
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
        # Use joblib instead of pickle for scikit-learn models
        model_path = os.path.join(model_dir, 'model.joblib')
        joblib.dump(model, model_path)
        print(f"  Linear model saved to: {model_path}")
    
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

def main():
    parser = argparse.ArgumentParser(description='Forward Time Regression Analysis')
    parser.add_argument('--dataset_name', type=str, required=True, help='Dataset name to filter (e.g., "100_500"), will search in fixed directory')
    parser.add_argument('--dataset_logs_root', type=str, required=True, help='Path to dataset root directory')
    
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
    DATASET_LOGS_ROOT = args.dataset_logs_root
    
    # Load data
    all_data = []
    failed_files = []

    # print(f"Dataset mode: collecting all models for dataset '{args.dataset_name}'")
    dataset_log_files = find_dataset_logs(DATASET_LOGS_ROOT, args.dataset_name)    
    assert(dataset_log_files)

    for log_file, host_name, model_name, iosl in dataset_log_files:
        # print(f"Processing: {log_file}, model_name={model_name}")
            
        # Get Hardware Features File
        #   from: root_dir/host_dir/model_dir
        hw_yaml = os.path.join(DATASET_LOGS_ROOT, host_name, "platform_features.yaml")
        assert(os.path.exists(hw_yaml) == True)
        platform_perf = PlatformPerf(hw_yaml)
        
        # Get Model Files
        model_uri = ""
        for root, _, files in os.walk("/models/preset"):
            if root.find(model_name) != -1:
                model_uri = root
                break
        model_uri = os.path.join(model_uri, "v1.0")
        data = parse_raw_features(log_file, model_uri, model_name, platform_perf, args.target, args.stage, args.disaggregation_mode)
        if len(data) == 0:
            failed_files.append(log_file)
        else:
            all_data.extend(data)
    
    # Process log data if we have any
    if all_data:
        # print(f"\nTotal entries collected: {len(all_data)}")
        
        # Count entries by model for better understanding of data distribution
        # print("\nData distribution by model:")
        model_counts = {}
        for entry in all_data:
            # Extract directory info from the source log file path
            assert('source_file' in entry)

            model_name = "unknown"
            path_parts = entry['source_file'].split(os.sep)
            for part in path_parts:
                if part.startswith("sglang_"):
                    model_name = part
                    model_counts[model_name] = model_counts.get(model_name, 0) + 1
                    break
            assert(model_name != "unknown")
            
        # Sort by count and display
        # sorted_models = sorted(model_counts.items(), key=lambda x: x[1], reverse=True)
        # for model_name, count in sorted_models:
        #     print(f"  {model_name}: {count:,} entries")
        
        df = pd.DataFrame([data['features'] for data in all_data])
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    model_dir = os.path.join(args.output_dir, args.saved_model_name)
    os.makedirs(model_dir, exist_ok=True)    
    
    # Save processed data
    # df.to_csv(f'{model_dir}/processed_timing_data.csv', index=False)
    # print(f"Processed data saved to {model_dir}/processed_timing_data.csv")
    
    # Debug: Show all available columns in the data
    # print(f"\nAll columns in processed data ({len(df.columns)} total):")
    # for i, col in enumerate(df.columns):
    #     print(f"  {i+1:2d}. {col}")
    # print(f"Data shape: {df.shape} (rows x columns)")
    
    feature_cols = []
    for col in FEATURE_LIST:
        if col in df.columns:
            # unique_vals = df[col].unique()
            # std_val = df[col].std() if pd.api.types.is_numeric_dtype(df[col]) else 'N/A'
            # print(f"  {col}: {len(unique_vals)} unique (std: {std_val})")
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
        elif args.regressor == 'ftt':
            # print("Using FT-Transformer regressor (train on train split, report train/val)")
            from ftt import perform_ft_transformer_regression, FTTConfig
            cfg = FTTConfig(
                d_tok=128, n_heads=4, n_layers=3, ffn_mult=2.0, dropout=0.1,
                batch_size=2048, lr=5e-4, weight_decay=2e-4,
                max_epochs=500, warmup_epochs=5, early_stop_patience=200,
                loss="huber", verbose=1
            )
            results = perform_ft_transformer_regression(
                df=pd.DataFrame(X_tr, columns=existing_features),
                feature_cols=existing_features,
                val_data=val_data_for_eval,
                y_values=y_tr,
                cfg=cfg,
            )
            if X_va is not None:
                y_va_pred = results['model'].predict(X_va)
                va_r2 = r2_score(y_va, y_va_pred)
                va_rmse = np.sqrt(mean_squared_error(y_va, y_va_pred))
                va_mae = np.mean(np.abs(y_va - y_va_pred))
                va_mape = np.mean(np.abs((y_va - y_va_pred) / (np.abs(y_va) + 1e-9))) * 100
                va_median_ape = np.median(np.abs((y_va - y_va_pred) / (np.abs(y_va) + 1e-9))) * 100
                results['val_metrics'] = {
                    'r2': va_r2,
                    'rmse': va_rmse,
                    'mae': va_mae,
                    'mape': va_mape,
                    'median_ape': va_median_ape
                }
        else:
            print("Using Linear regressor (train on train split, report train/val)")
            # Use the dedicated function for Linear Regression
            results = perform_regression_analysis(pd.DataFrame(X_tr, columns=existing_features), existing_features, y_tr)
            
            # Add validation metrics if available
            if X_va is not None:
                if results.get('scaler') is not None:
                    X_va_input = results['scaler'].transform(X_va)
                else:
                    X_va_input = X_va
                y_va_pred = results['model'].predict(X_va_input)
                y_va_safe = np.where(np.abs(y_va) < 1e-6, 1e-6, y_va) # prevent MAPE explosion
                va_r2 = r2_score(y_va, y_va_pred)
                va_rmse = np.sqrt(mean_squared_error(y_va, y_va_pred))
                va_mae = np.mean(np.abs(y_va - y_va_pred))
                va_mape = np.mean(np.abs((y_va - y_va_pred) / y_va_safe)) * 100
                va_median_ape = np.median(np.abs((y_va - y_va_pred) / y_va_safe)) * 100
                results['val_metrics'] = {
                    'r2': va_r2,
                    'rmse': va_rmse,
                    'mae': va_mae,
                    'mape': va_mape,
                    'median_ape': va_median_ape
                }
    else:
        assert(0)

    # Prepare data source info
    data_source_info = f"Dataset '{args.dataset_name}' from {DATASET_LOGS_ROOT} ({len(dataset_log_files)} files across models)"
    raw_data_for_summary = all_data if all_data else None
    print_analysis_summary(df, results, data_source_info, raw_data_for_summary)

    # Save trained model in a subdirectory
    save_trained_model(results, model_dir)
    
    print(f"\nAnalysis complete! Check {args.output_dir}/ for output files.")
    print(f"Trained model saved in: {model_dir}")

if __name__ == "__main__":
    main() 