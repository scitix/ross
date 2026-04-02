import pandas as pd
import numpy as np
import os, sys

from typing import List, Dict, Tuple

import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

def perform_regression_analysis(df: pd.DataFrame,
                                feature_cols: List[str],
                                y_values: np.ndarray = None) -> Dict:
    """
    Perform linear regression analysis with feature scaling for cross-model analysis.
    
    Args:
        df: DataFrame with features (or just feature columns if y_values is provided)
        feature_cols: List of feature column names
        y_values: Optional y values (if not provided, will extract from df['time_ms'])
    """
    # Prepare data
    X = df[feature_cols].copy()
    if y_values is not None:
        y = y_values
    else:
        y = df['time_ms'].copy()
    
    # Handle any NaN values
    X = X.fillna(0)
    
    # Check if we have features with very different scales (e.g., hidden_size vs batch_size)
    feature_ranges = {}
    scale_needed = False
    for col in feature_cols:
        if col in X.columns and X[col].std() > 0:
            # Skip boolean/dummy variables (they are already 0/1 scaled)
            if X[col].dtype == 'bool' or set(X[col].unique()).issubset({0, 1}):
                feature_ranges[col] = 1  # Boolean features have range 1
                continue
                
            col_range = X[col].max() - X[col].min()
            feature_ranges[col] = col_range
            if col_range > 1000:  # Large scale features like hidden_size, vocab_size
                scale_needed = True
    
    print(f"\nFeature scaling analysis:")
    print(f"  Scale normalization needed: {scale_needed}")
    if scale_needed:
        print("  Large-scale features detected:")
        for col, range_val in feature_ranges.items():
            if range_val > 1000:
                print(f"    {col}: range {range_val:.0f}")
    
    # Apply feature scaling if needed for cross-model analysis
    if scale_needed:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_for_model = X_scaled
        print("  Applied StandardScaler for cross-model features")
    else:
        X_for_model = X.values
        scaler = None
        print("  Using original scale (no large feature differences)")
    
    # Fit regression model
    model = LinearRegression()
    model.fit(X_for_model, y)
    
    # Predictions
    y_pred = model.predict(X_for_model)
    
    # Metrics
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    
    # Additional error metrics
    mae = np.mean(np.abs(y - y_pred))  # Mean Absolute Error
    mape = np.mean(np.abs((y - y_pred) / y)) * 100  # Mean Absolute Percentage Error
    median_ape = np.median(np.abs((y - y_pred) / y)) * 100  # Median Absolute Percentage Error
    
    # Feature importance (coefficients) - convert back to original scale if scaled
    if scaler is not None:
        # Convert standardized coefficients back to original scale for interpretation
        original_coefs = model.coef_ / scaler.scale_
        original_intercept = model.intercept_ - np.sum(original_coefs * scaler.mean_)
        
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'coefficient': original_coefs,
            'abs_coefficient': np.abs(original_coefs),
            'standardized_coef': model.coef_  # Also keep standardized for comparison
        }).sort_values('abs_coefficient', ascending=False)
        
        intercept_value = original_intercept
    else:
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'coefficient': model.coef_,
            'abs_coefficient': np.abs(model.coef_)
        }).sort_values('abs_coefficient', ascending=False)
        
        intercept_value = model.intercept_
    
    return {
        'model': model,
        'model_name': 'linear',
        'scaler': scaler,
        'X_original': X,
        'X_scaled': X_for_model if scaler else None,
        'y_true': y,
        'y_pred': y_pred,
        'r2': r2,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'median_ape': median_ape,
        'feature_importance': feature_importance,
        'intercept': intercept_value,
        'scale_needed': scale_needed
    }

# New: XGBoost regression (tree-based, robust to scaling)
def perform_xgboost_regression(df: pd.DataFrame,
                                feature_cols: List[str],
                                val_data: Tuple = None,
                                y_values: np.ndarray = None,
                                target: str = ''
                            ) -> Dict:
    """
    Perform XGBoost regression with sensible defaults and report metrics and importances.
    
    Args:
        df: DataFrame with features (or just feature columns if y_values is provided)
        feature_cols: List of feature column names
        y_values: Optional y values (if not provided, will extract from df['time_ms'])
    """
    X = df[feature_cols].copy().fillna(0)
    if y_values is not None:
        y = y_values
    else:
        y = df['time_ms'].astype(float).values

    if target == 'forward' or target == 'post-forward': # or target == 'pre-forward':
        model = xgb.XGBRegressor(
            n_estimators=30000,
            learning_rate=0.01,
                    
            max_depth=10,
            min_child_weight=1.0,
            gamma=0.0,                # 去掉分裂门槛
            
            # --- 正则化保持适度 ---
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,           # 不需要太强的正则，除非过拟合
            reg_alpha=0.1,
            
            objective='reg:squarederror', # 'reg:gamma',

            tree_method='hist',
            n_jobs=os.cpu_count() if os.cpu_count() else 32,
            random_state=42, 
            eval_metric='rmse', # 'mape',
            early_stopping_rounds=5000,
        )
    elif target == 'pre-forward':
        model = xgb.XGBRegressor(
            objective='reg:squarederror',  # 必须用这个！MAE 会低估总耗时
            eval_metric='rmse',
            
            n_estimators=10000,
            learning_rate=0.05,       # 稍微调大一点，避免陷入局部最优
            
            max_depth=10,             # 保持足够深度，区分大小 Batch
            min_child_weight=5.0,     # <--- 关键修改！降到1，允许模型为少数的大 Batch 建立专属分支
            
            # === 防止过拟合，但不要太激进 ===
            gamma=0.1,                # 轻微惩罚分裂，比 min_child_weight 更灵活
            subsample=0.8,            # 保持适度采样
            colsample_bytree=0.8,
            
            # === 正则化：适度即可 ===
            reg_lambda=1.0,           # 默认值即可，太大会压制高值的预测
            reg_alpha=0.0,            # 关掉 L1，因为我们需要精确拟合数值，不需要稀疏
            
            # === 系统参数 ===
            tree_method='hist',
            n_jobs=os.cpu_count() if os.cpu_count() else 32,
            random_state=42, 
            early_stopping_rounds=1000,
        )
    else: # other regression task
        model = xgb.XGBRegressor(
            n_estimators=30000,
            learning_rate=0.01,
                    
            max_depth=10,
            min_child_weight=1.0,
            gamma=0.0,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            reg_alpha=0.1,
            
            objective='reg:squarederror',

            tree_method='hist',
            n_jobs=os.cpu_count() if os.cpu_count() else 32,
            random_state=42, 
            eval_metric='rmse',
            early_stopping_rounds=5000,
        )
    eval_set = []
    if val_data:
        X_val, y_val = val_data
        eval_set.append((X_val, y_val))
        print("Early stopping enabled using validation set.")

    model.fit(X.values, y,
            eval_set=eval_set, 
            verbose=3000)
    
    y_pred = model.predict(X.values)
    
    # Metrics
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y - y_pred))
    mape = np.mean(np.abs((y - y_pred) / y)) * 100
    median_ape = np.median(np.abs((y - y_pred) / y)) * 100
    
    # Feature importance (gain preferred)
    try:
        booster = model.get_booster()
        score_map = booster.get_score(importance_type='gain')
        # Map feature indices back to names
        imp_rows = []
        for k, v in score_map.items():
            # k like 'f0', 'f12'
            if k.startswith('f') and k[1:].isdigit():
                idx = int(k[1:])
                if idx < len(feature_cols):
                    imp_rows.append((feature_cols[idx], float(v)))
        feature_importance = pd.DataFrame(imp_rows, columns=['feature', 'importance_gain']).sort_values('importance_gain', ascending=False)
    except Exception:
        # Fallback to built-in feature_importances_
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance_gain': getattr(model, 'feature_importances_', np.zeros(len(feature_cols)))
        }).sort_values('importance_gain', ascending=False)
    
    return {
        'model': model,
        'model_name': 'xgboost',
        'scaler': None,
        'X_original': X,
        'X_scaled': None,
        'y_true': y,
        'y_pred': y_pred,
        'r2': r2,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'median_ape': median_ape,
        'feature_importance': feature_importance,
        'intercept': None,
        'scale_needed': False
    }
