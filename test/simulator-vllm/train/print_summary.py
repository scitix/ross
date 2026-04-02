import pandas as pd
import numpy as np

from typing import List, Dict, Tuple

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
    
    # for col in numerical_config_cols:
    #     if col in df.columns:
    #         unique_vals = df[col].dropna().unique()
    #         if len(unique_vals) > 0:
    #             if len(unique_vals) == 1:
    #                 print(f"  {col}: {unique_vals[0]}")
    #             else:
    #                 print(f"  {col}: {list(unique_vals)} (multiple values)")

    # # Show categorical info separately for reference (but not used in regression)
    # categorical_info_cols = ['model_type', 'dtype', 'gpu_model', 'cpu_model', 'platform_name']
    # categorical_info_found = False
    # for col in categorical_info_cols:
    #     if col in df.columns:
    #         unique_vals = df[col].dropna().unique()
    #         if len(unique_vals) > 0:
    #             if not categorical_info_found:
    #                 print(f"\n  Categorical info (not used in regression):")
    #                 categorical_info_found = True
    #             if len(unique_vals) == 1:
    #                 print(f"    {col}: {unique_vals[0]}")
    #             else:
    #                 print(f"    {col}: {list(unique_vals)}")
    
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {len(df)}")
    print(f"  Time range: {df['time_ms'].min():.2f} - {df['time_ms'].max():.2f} ms")
    print(f"  Average time: {df['time_ms'].mean():.2f} ± {df['time_ms'].std():.2f} ms")
    print(f"  Batch size range: {df['batch_size'].min()} - {df['batch_size'].max()}")
    # Calculate total tokens on the fly since we removed the total_seq_tokens feature
    # total_tokens = df['prefill_total_tokens'] + df['decode_total_tokens']
    # print(f"  Total tokens range: {total_tokens.min()} - {total_tokens.max()}")

    # # Request distribution
    # if 'prefill_reqs' in df.columns and 'decode_reqs' in df.columns:
    #     print(f"\nRequest Distribution:")
    #     print(f"  Prefill requests range: {df['prefill_reqs'].min()} - {df['prefill_reqs'].max()}")
    #     print(f"  Decode requests range: {df['decode_reqs'].min()} - {df['decode_reqs'].max()}")
    #     total_reqs = df['prefill_reqs'] + df['decode_reqs']
    #     avg_prefill_ratio = (df['prefill_reqs'] / total_reqs).mean()
    #     print(f"  Average prefill/decode ratio: {avg_prefill_ratio:.2f} / {1-avg_prefill_ratio:.2f}")
    
    print(f"\nRegression Results:")
    print(f"  R-squared: {results['r2']:.4f}")
    print(f"  RMSE: {results['rmse']:.2f} ms")
    print(f"  MAE: {results['mae']:.2f} ms")
    print(f"  MAPE: {results['mape']:.2f}%  ← 平均误差百分比")
    print(f"  Median APE: {results['median_ape']:.2f}%  ← 中位数误差百分比")
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
    
    # if results.get('model_name') == 'xgboost':
    #     print(f"\nXGBoost Feature Importances (gain):")
    #     for idx, row in results['feature_importance'].iterrows():
    #         feature = row['feature']
    #         gain = row.get('importance_gain', 0.0)
    #         print(f"  {feature}: {gain:.6f}")
    # # else:
    # #     print(f"\nLinear Model Coefficients:")
    # #     print(f"  gpu_fw_ms = {results['intercept']:.4f}")
    # #     for idx, row in results['feature_importance'].iterrows():
    # #         sign = "+" if row['coefficient'] >= 0 else ""
    # #         feature = row['feature']
    # #         coeff = row['coefficient']
            
    # #         # Expand derived features to show original feature relationships
    # #         if feature == 'ffn_expansion_ratio':
    # #             print(f"             {sign} {coeff:.6f} × (intermediate_size / hidden_size)        # ffn_expansion_ratio")
    # #         elif feature == 'gqa_ratio':
    # #             print(f"             {sign} {coeff:.6f} × (num_attention_heads / num_key_value_heads) # gqa_ratio")
    # #         elif feature == 'head_dim':
    # #             print(f"             {sign} {coeff:.6f} × (hidden_size / num_attention_heads)      # head_dim")
    # #         elif feature == 'attn_flops_log':
    # #             print(f"             {sign} {coeff:.6f} × log₁₀(num_layers × batch_size × seq_len × hidden_size²) # attn_flops_log")
    # #         elif feature == 'mlp_flops_log':
    # #             print(f"             {sign} {coeff:.6f} × log₁₀(num_layers × batch_size × seq_len × hidden_size × intermediate_size) # mlp_flops_log")
    # #         elif feature == 'attn_flops_per_rank_log':
    # #             print(f"             {sign} {coeff:.6f} × log₁₀((num_layers/pp) × batch_size × seq_len × hidden_size² / tp) # attn_flops_per_rank_log")
    # #         elif feature == 'mlp_flops_per_rank_log':
    # #             print(f"             {sign} {coeff:.6f} × log₁₀((num_layers/pp) × batch_size × seq_len × hidden_size × intermediate_size / tp) # mlp_flops_per_rank_log")
    # #         elif feature == 'kv_cache_log':
    # #             print(f"             {sign} {coeff:.6f} × log₁₀(num_layers × num_kv_heads × head_dim × seq_len × batch_size) # kv_cache_log")
    # #         elif feature == 'kv_cache_per_rank_log':
    # #             print(f"             {sign} {coeff:.6f} × log₁₀((num_layers/pp) × num_kv_heads × head_dim × seq_len × batch_size / tp) # kv_cache_per_rank_log")
    # #         elif feature == 'attn_memory_log':
    # #             print(f"             {sign} {coeff:.6f} × log₁₀((num_layers/pp) × (batch_size × seq_len × hidden_size + 4×hidden_size²/tp)) # attn_memory_log")
    # #         elif feature == 'tp_comm_log':
    # #             print(f"             {sign} {coeff:.6f} × log₁₀((num_layers/pp) × batch_size × hidden_size) # tp_comm_log")
    # #         elif feature == 'pipeline_overhead_proxy':
    # #             print(f"             {sign} {coeff:.6f} × ((pp-1)/pp) # pipeline_overhead_proxy")
    # #         elif feature == 'total_params_log':
    # #             print(f"             {sign} {coeff:.6f} × log₁₀(num_layers × (4×hidden_size² + 2×hidden_size×intermediate_size)) # total_params_log")
    # #         elif feature == 'layers_x_hidden_sq':
    # #             print(f"             {sign} {coeff:.8f} × (num_layers × hidden_size²) # layers_x_hidden_sq")
    # #         elif feature == 'layers_x_hidden_x_ffn':
    # #             print(f"             {sign} {coeff:.8f} × (num_layers × hidden_size × intermediate_size) # layers_x_hidden_x_ffn")
    # #         elif feature == 'batch_x_seq_x_hidden':
    # #             print(f"             {sign} {coeff:.8f} × (batch_size × seq_len × hidden_size) # batch_x_seq_x_hidden")

    # #         else:
    # #             print(f"             {sign} {coeff:.6f} × {feature}")
    
    # if results.get('model_name') == 'xgboost':
    #     print(f"\nTop 10 Most Important Features (by gain):")
    #     for idx, row in results['feature_importance'].head(10).iterrows():
    #         gain_val = row.get('importance_gain', 0.0)
    #         print(f"  {row['feature']:25s}: {gain_val:10.6f}")
    #     # Skip linear interpretation for tree-based model
    # # else:
    #     print(f"\nTop 10 Most Important Features (by absolute coefficient):")
    #     for idx, row in results['feature_importance'].head(10).iterrows():
    #         print(f"  {row['feature']:25s}: {row['coefficient']:10.6f}")
        
    #     # Simple interpretation (linear model)
    #     print(f"\nInterpretation:")
    #     for idx, row in results['feature_importance'].head(5).iterrows():
    #         feature = row['feature']
    #         coef = row['coefficient']
    #         if 'batch_size' in feature:
    #             print(f"  Each additional request increases time by {coef:.4f} ms")
    #         elif 'total_tokens' in feature or 'prefill_total' in feature or 'decode_total' in feature:
    #             print(f"  Each additional token increases time by {coef:.6f} ms")
    #         elif 'avg_len' in feature:
    #             print(f"  Each unit increase in {feature} changes time by {coef:.6f} ms")
    #         elif 'ratio' in feature:
    #             print(f"  Each 0.1 increase in {feature} changes time by {coef*0.1:.4f} ms")
    #         else:
    #             print(f"  Each unit increase in {feature} changes time by {coef:.6f} ms")
