import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import joblib

from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import r2_score, mean_squared_error

import os, sys
from pathlib import Path

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

def print_analysis_summary(df: pd.DataFrame, results: Dict, data_source_info: str = "", raw_data: List[Dict] = None):
    """
    Print summary of the analysis.
    """
    print("\n" + "="*80)
    print("TIME REGRESSION ANALYSIS SUMMARY")
    print("="*80)
    
    if data_source_info:
        print(f"\nData Source:")
        print(f"  {data_source_info}")
    
    # Model configuration information (only numerical features for generalization)
    print(f"\nModel Configuration:")
    numerical_config_cols = [
        'num_hidden_layers', 'hidden_size', 'num_attention_heads', 'num_key_value_heads',
        'intermediate_size', 'tensor_parallel_size', 'pipeline_parallel_size', 'data_parallel_size'
    ]
 
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {len(df)}")
    print(f"  Time range: {df['time_ms'].min():.2f} - {df['time_ms'].max():.2f} ms")
    print(f"  Average time: {df['time_ms'].mean():.2f} ± {df['time_ms'].std():.2f} ms")
    print(f"  Batch size range: {df['batch_size'].min()} - {df['batch_size'].max()}")
    # Calculate total tokens on the fly since we removed the total_seq_tokens feature
    total_tokens = df['total_tokens']
    print(f"  Total tokens range: {total_tokens.min()} - {total_tokens.max()}")

    print(f"\nRegression Results:")
    print(f"  R-squared: {results['r2']:.4f}")
    print(f"  RMSE: {results['rmse']:.2f} ms")
    print(f"  MAE: {results['mae']:.2f} ms")
    print(f"  MAPE: {results['mape']:.2f}%  <- mean absolute percentage error")
    print(f"  Median APE: {results['median_ape']:.2f}%  <- median absolute percentage error")
    if 'val_metrics' in results and results['val_metrics'] is not None:
        vm = results['val_metrics']
        print(f"  Validation R-squared: {vm['r2']:.4f}")
        print(f"  Validation RMSE: {vm['rmse']:.2f} ms")
        print(f"  Validation MAE: {vm['mae']:.2f} ms")
        print(f"  Validation MAPE: {vm['mape']:.2f}%")
        print(f"  Validation Median APE: {vm['median_ape']:.2f}%")
    intercept_val = results.get('intercept', None)
    if intercept_val is None:
        print(f"  Intercept: N/A (tree-based model)")
    else:
        print(f"  Intercept: {intercept_val:.2f} ms")
    
    if results.get('scale_needed', False):
        print(f"  Feature scaling: Applied (cross-model analysis)")
    else:
        print(f"  Feature scaling: Not needed (similar scale features)")

