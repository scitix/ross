#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import json
import yaml
from typing import Dict, List, Tuple

class PlatformEfficiencyCalculator:
    """Calculate platform efficiency and extract key platform features."""
    
    def __init__(self):
        pass
    
    @staticmethod
    def _to_builtin(obj):
        """Recursively convert numpy types and containers to Python builtins for YAML/JSON."""
        import numpy as _np
        if isinstance(obj, dict):
            return {PlatformEfficiencyCalculator._to_builtin(k): PlatformEfficiencyCalculator._to_builtin(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [PlatformEfficiencyCalculator._to_builtin(x) for x in obj]
        if isinstance(obj, (_np.generic,)):
            return obj.item()
        return obj
    
    def load_platform_specs(self, yaml_path: str) -> Dict:
        """Load platform specifications from a YAML file."""
        try:
            with open(yaml_path, 'r') as f:
                specs = yaml.safe_load(f)
            print(f"Loaded platform specs from {yaml_path}")
            return specs
        except Exception as e:
            print(f"Warning: Failed to load {yaml_path}: {e}")
            return {}
    
    def _convert_yaml_specs_to_peaks(self, yaml_specs: Dict) -> Dict:
        """Convert YAML specifications into a theoretical peak format."""
        peaks = {}
        
        if 'gpu' in yaml_specs:
            gpu_specs = yaml_specs['gpu']
            
            # Convert memory bandwidth (bytes/s -> GB/s)
            mem_bw_gbps = gpu_specs.get('mem_bw', 0) / 1e9
            
            # Convert memory capacity (bytes -> GB)
            mem_capacity_gb = gpu_specs.get('mem_capacity', 0) / 1e9
            
            # Convert compute capability (FLOPS -> TFLOPS)
            fp16_tflops = gpu_specs.get('float16_tc_flops', 0) / 1e12
            int8_tflops = gpu_specs.get('int8_tc_flops', 0) / 1e12
            fp8_tflops = gpu_specs.get('fp8_tc_flops', 0) / 1e12
            
            # Power (Watt)
            power_w = gpu_specs.get('power', 0)
            
            peaks = {
                "fp16_tflops": fp16_tflops,
                "int8_tflops": int8_tflops,
                "fp8_tflops": fp8_tflops,
                "memory_bandwidth_gbps": mem_bw_gbps,
                "memory_gb": mem_capacity_gb,
                "power_w": power_w
            }
        
        return peaks
    
    def calculate_platform_efficiency(self, gemm_data: pd.DataFrame = None, attn_data: pd.DataFrame = None, nccl_data: pd.DataFrame = None, yaml_specs_path: str = None) -> Dict:
        """Calculate platform efficiency and extract key features."""
        
        # The YAML spec file must be provided
        if not yaml_specs_path:
            raise ValueError("YAML specs file is required. Please provide --platform_specs")
        
        # Load from the YAML file
        yaml_specs = self.load_platform_specs(yaml_specs_path)
        peaks = self._convert_yaml_specs_to_peaks(yaml_specs)
        
        if not peaks:
            raise ValueError(f"Failed to load valid platform specs from {yaml_specs_path}")
        
        # 1. Compute realized compute capability
        actual_capabilities = self._calculate_actual_capabilities(gemm_data, attn_data)
        
        
        # 3. Extract key platform features
        platform_features = self._extract_key_platform_features(actual_capabilities, peaks)
        
        # 4. If NCCL data is available, add communication features
        if nccl_data is not None:
            nccl_features = self._extract_nccl_features(nccl_data)
            platform_features.update(nccl_features)
        
        return platform_features
    
    def _calculate_actual_capabilities(self, gemm_data: pd.DataFrame = None, attn_data: pd.DataFrame = None) -> Dict:
        """Calculate realized compute capabilities."""
        capabilities = {}
        
        # Process GEMM data
        if gemm_data is not None and not gemm_data.empty:
            gemm_capabilities = self._analyze_gemm_capabilities(gemm_data)
            capabilities.update(gemm_capabilities)
        
        # Process attention data
        if attn_data is not None and not attn_data.empty:
            attn_capabilities = self._analyze_attention_capabilities(attn_data)
            capabilities.update(attn_capabilities)
        
        return capabilities
    
    def _analyze_gemm_capabilities(self, gemm_data: pd.DataFrame) -> Dict:
        """Analyze GEMM kernel capabilities."""
        capabilities = {}
        
        # Group by quantization type
        for quant_type, group in gemm_data.groupby('gemm_dtype'):
            group = group.copy()
            
            # Compute GEMM FLOPs: 2 * M * N * K - vectorized version
            m_values = group['m'].values
            n_values = group['n'].values
            k_values = group['k'].values
            latency_values = group['latency'].values
            
            # Vectorized FLOPs computation
            flops_values = 2 * m_values * n_values * k_values
            flops_per_ms_values = flops_values / (latency_values + 1e-8)
            
            # Assign the results back to the DataFrame
            group['flops'] = flops_values
            group['flops_per_ms'] = flops_per_ms_values
            
            # Compute memory access: M*N + M*K + N*K (input matrices + output matrix) - vectorized version
            memory_bytes_values = (m_values * n_values + m_values * k_values + n_values * k_values) * 2  # Assume FP16
            memory_gbps_values = memory_bytes_values / (latency_values * 1e6)
            
            # Assign the results back to the DataFrame
            group['memory_bytes'] = memory_bytes_values
            group['memory_gbps'] = memory_gbps_values
            
            # Record best observed performance
            capabilities[f"gemm_{quant_type}_max_flops_per_ms"] = group['flops_per_ms'].max()
            capabilities[f"gemm_{quant_type}_max_memory_gbps"] = group['memory_gbps'].max()
            capabilities[f"gemm_{quant_type}_avg_latency"] = group['latency'].mean()
            
            # Compute arithmetic intensity (FLOPs/Byte) - vectorized version
            arithmetic_intensity_values = flops_values / (memory_bytes_values + 1e-8)  # Avoid division by zero
            group['arithmetic_intensity'] = arithmetic_intensity_values
            capabilities[f"gemm_{quant_type}_avg_arithmetic_intensity"] = np.mean(arithmetic_intensity_values)
        
        # Compute aggregate GEMM performance - vectorized version
        if not gemm_data.empty:
            # Use numpy for vectorized computation
            m_all = gemm_data['m'].values
            n_all = gemm_data['n'].values
            k_all = gemm_data['k'].values
            latency_all = gemm_data['latency'].values
            
            all_flops = 2 * m_all * n_all * k_all
            all_flops_per_ms = all_flops / (latency_all + 1e-8)
            
            capabilities['gemm_overall_max_flops_per_ms'] = np.max(all_flops_per_ms)
            capabilities['gemm_overall_avg_latency'] = np.mean(latency_all)
        
        return capabilities
    
    def _analyze_attention_capabilities(self, attn_data: pd.DataFrame) -> Dict:
        """Analyze attention kernel capabilities."""
        capabilities = {}
        
        # Group by kernel type
        # for kernel_name, group in attn_data.groupby('kernel_source'):
        #     group = group.copy()
            
        #     if 'attention' in kernel_name.lower():
        #         group['flops'] = group.apply(self._compute_attention_flops, axis=1)
        #     elif 'mlp' in kernel_name.lower():
        #         continue
        #     else:
        #         group['flops'] = group['batch_size'] * group['isl'] * group['num_heads'] * group['head_dim']
            
        #     # Compute realized FLOPs per millisecond
        #     group['flops_per_ms'] = group['flops'] / (group['latency'] + 1e-8)
            
        #     # Compute memory bandwidth
        #     group['memory_bytes'] = group.apply(self._compute_memory_bytes, axis=1)
        #     group['gbps'] = group['memory_bytes'] / ((group['latency'] + 1e-8) * 1e6)
            
        #     # Record maximum values (best performance)
        #     capabilities[f"{kernel_name}_max_flops_per_ms"] = group['flops_per_ms'].max()
        #     capabilities[f"{kernel_name}_max_memory_gbps"] = group['gbps'].max()
        #     capabilities[f"{kernel_name}_avg_latency"] = group['latency'].mean()
        
        # Additionally, aggregate by op_name (e.g., context_attention)
        if 'op_name' in attn_data.columns:
            for op_name, op_group in attn_data.groupby('op_name'):
                op_group = op_group.copy()
                if 'attention' in op_name.lower():
                    op_group['flops'] = op_group.apply(self._compute_attention_flops, axis=1)
                elif 'mlp' in op_name.lower():
                    continue
                else:
                    op_group['flops'] = op_group['batch_size'] * op_group['isl'] * op_group['num_heads'] * op_group['head_dim']
                op_group['flops_per_ms'] = op_group['flops'] / (op_group['latency'] + 1e-8)
                op_group['memory_bytes'] = op_group.apply(self._compute_memory_bytes, axis=1)
                op_group['gbps'] = op_group['memory_bytes'] / (op_group['latency'] * 1e6)
                capabilities[f"{op_name}_max_flops_per_ms"] = op_group['flops_per_ms'].max()
                capabilities[f"{op_name}_max_memory_gbps"] = op_group['gbps'].max()
                capabilities[f"{op_name}_avg_latency"] = op_group['latency'].mean()
        
        # Group by data type
        if 'attn_dtype' in attn_data.columns:
            for dtype, group in attn_data.groupby('attn_dtype'):
                capabilities[f"{dtype}_avg_latency"] = group['latency'].mean()
                capabilities[f"{dtype}_relative_performance"] = group['latency'].mean() / attn_data['latency'].min()
        
        return capabilities
    
    def _extract_key_platform_features(self, actual_capabilities: Dict, theoretical_peaks: Dict) -> Dict:
        """Extract key platform features without simple transformations."""
        print(actual_capabilities)
        # Retrieve GEMM memory bandwidth
        gemm_mem_keys = [k for k in actual_capabilities.keys() if 'gemm' in k.lower() and k.endswith('_max_memory_gbps')]
        gemm_mem_bandwidth = max([actual_capabilities.get(k, 0) for k in gemm_mem_keys]) if gemm_mem_keys else 0
        
        # Retrieve attention memory bandwidth
        attn_mem_keys = [k for k in actual_capabilities.keys() if 'attention' in k.lower() and k.endswith('_max_memory_gbps')]
        attn_mem_bandwidth = max([actual_capabilities.get(k, 0) for k in attn_mem_keys]) if attn_mem_keys else 0
        
        # Merge memory bandwidth metrics by taking the maximum
        combined_memory_bandwidth = max(gemm_mem_bandwidth, attn_mem_bandwidth)
        # Core platform features (six independent and meaningful metrics)
        key_features = {
            # 1. Platform GEMM compute capability (measured)
            "platform_gemm_flops_per_ms": (actual_capabilities.get('gemm_overall_max_flops_per_ms') or 0) / 1024 ** 3,
            
            # 2. Platform attention compute capability (measured)
            "platform_attention_flops_per_ms": (actual_capabilities.get('context_attention_max_flops_per_ms') or 0) / 1024 ** 3,
            
            # 4. Platform memory bandwidth (measured)
            "platform_memory_bandwidth_gbps": combined_memory_bandwidth,
        }
        return {**key_features}
    
    def _extract_nccl_features(self, nccl_data: pd.DataFrame) -> Dict:
        """Extract NCCL communication features."""
        nccl_features = {}
        
        # Analyze by data type group
        for dtype, group in nccl_data.groupby('nccl_dtype'):
            # Compute communication performance for this data type
            group = group.copy()
            
            # Compute bandwidth (GB/s): GB/s = bytes / (latency_ms * 1e6)
            group['bandwidth_gbps'] = group['message_size'] / ((group['latency'] + 1e-8) * 1e6)
            
            # Compute latency (us)
            group['latency_us'] = group['latency'] * 1e6
            
            # Group by GPU count
            for num_gpus, gpu_group in group.groupby('num_gpus'):
                # Record best performance (maximum bandwidth)
                max_bandwidth = gpu_group['bandwidth_gbps'].max()
                min_latency = gpu_group['latency_us'].min()
                avg_latency = gpu_group['latency_us'].mean()
                
                # Feature naming: platform_nccl_{dtype}_{num_gpus}gpus_{metric}
                nccl_features[f'platform_nccl_{dtype}_{num_gpus}gpus_max_bandwidth_gbps'] = max_bandwidth
                nccl_features[f'platform_nccl_{dtype}_{num_gpus}gpus_min_latency_ms'] = min_latency / 1e3
                nccl_features[f'platform_nccl_{dtype}_{num_gpus}gpus_avg_latency_ms'] = avg_latency / 1e3
        
        # Compute aggregate communication features
        nccl_features.update(self._compute_nccl_summary_features(nccl_data))
        
        return nccl_features
    
    def _compute_nccl_summary_features(self, nccl_data: pd.DataFrame) -> Dict:
        """Compute aggregate NCCL features."""
        summary_features = {}
        
        # Compute overall best bandwidth (GB/s): GB/s = bytes / (latency_ms * 1e6)
        nccl_data['bandwidth_gbps'] = nccl_data['message_size'] / ((nccl_data['latency'] + 1e-8) * 1e6)
        max_overall_bandwidth = nccl_data['bandwidth_gbps'].max()
        summary_features['platform_nccl_max_bandwidth_gbps'] = max_overall_bandwidth
        
        # Compute overall average latency (ms)
        avg_overall_latency_ms = nccl_data['latency'].mean()  # Already in ms
        summary_features['platform_nccl_avg_latency_ms'] = avg_overall_latency_ms
        
        # Compute communication stability (latency coefficient of variation)
        latency_cv = nccl_data['latency'].std() / nccl_data['latency'].mean()
        comm_stability = 1 / (1 + latency_cv) if latency_cv > 0 else 1.0
        summary_features['platform_nccl_stability'] = comm_stability
        
        return summary_features
    
    def _compute_attention_flops_single(self, row: pd.Series) -> float:
        """Compute FLOPs for a single attention row."""
        seq_len = row['isl']
        batch_size = row['batch_size']
        num_heads = row['num_heads']
        head_dim = row['head_dim']
        
        # QKV projection plus attention computation
        qkv_proj = 3 * batch_size * seq_len * (num_heads * head_dim) ** 2
        attention = batch_size * num_heads * (seq_len ** 2) * head_dim
        
        return qkv_proj + attention
    
    def _compute_attention_flops(self, row: pd.Series) -> float:
        """Compute attention FLOPs."""
        return self._compute_attention_flops_single(row)
    
    def _compute_mlp_flops_single(self, row: pd.Series) -> float:
        """Compute FLOPs for a single MLP row."""
        seq_len = row['isl']
        batch_size = row['batch_size']
        hidden_size = row['head_dim'] * row['num_heads']
        intermediate_size = hidden_size * 4
        
        return 2 * batch_size * seq_len * hidden_size * intermediate_size
    
    def _compute_mlp_flops(self, row: pd.Series) -> float:
        """Compute MLP FLOPs."""
        return self._compute_mlp_flops_single(row)
    
    def _compute_memory_bytes(self, row: pd.Series) -> float:
        """Compute memory footprint in bytes."""
        seq_len = row['isl']
        batch_size = row['batch_size']
        num_heads = row['num_heads']
        head_dim = row['head_dim']
        
        # Size of the data type in bytes
        dtype_bytes = 2 if row['attn_dtype'] in ['float16', 'bfloat16'] else 4
        
        # Input and output data
        data_bytes = batch_size * seq_len * num_heads * head_dim * dtype_bytes
        
        # KV cache (when processing attention)
        if 'attention' in row['kernel_source'].lower():
            kv_cache_bytes = 2 * batch_size * seq_len * num_heads * head_dim * dtype_bytes
            data_bytes += kv_cache_bytes
        
        return data_bytes

def main():
    """Example entry point to compute platform efficiency."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Platform efficiency calculator')
    parser.add_argument('--gemm_data', help='Path to GEMM CSV data')
    parser.add_argument('--attn_data', help='Path to Context Attention CSV data')
    parser.add_argument('--nccl_data', help='Path to NCCL communication CSV data')
    parser.add_argument('--platform_specs', required=True, help='Path to platform specs YAML file')
    parser.add_argument('--output_dir', default='platform_analysis', help='Output directory')
    
    args = parser.parse_args()
    
    # Load GEMM data
    gemm_data = None
    if args.gemm_data:
        print("Loading GEMM data...")
        gemm_data = pd.read_csv(args.gemm_data) # TODO: add column head
        gemm_data.columns = ['framework', 'version', 'device', 'op_name', 'kernel_source', 'gemm_dtype', 'm', 'n', 'k', 'latency']

        print(f"Loaded {len(gemm_data)} GEMM records")
        print(f"GEMM quantization types: {gemm_data['gemm_dtype'].unique()}")
    
    # Load context attention data
    attn_data = None
    if args.attn_data:
        print("Loading Context Attention data...")
        attn_data = pd.read_csv(args.attn_data)
        print(f"Loaded {len(attn_data)} Context Attention records")
        print(f"Attention kernel types: {attn_data['kernel_source'].unique()}")
    
    # Load NCCL data when provided
    nccl_data = None
    if args.nccl_data:
        print("Loading NCCL communication data...")
        nccl_data = pd.read_csv(args.nccl_data)
        print(f"Loaded {len(nccl_data)} NCCL communication records")
        print(f"NCCL data types: {nccl_data['nccl_dtype'].unique()}")
        print(f"GPU counts: {sorted(nccl_data['num_gpus'].unique())}")
    
    # Ensure at least one dataset is provided
    if not any([gemm_data is not None, attn_data is not None, nccl_data is not None]):
        raise ValueError("At least one data file must be provided (--gemm_data, --attn_data, or --nccl_data)")
    
    # Compute platform efficiency
    calculator = PlatformEfficiencyCalculator()
    platform_features = calculator.calculate_platform_efficiency(
        gemm_data=gemm_data,
        attn_data=attn_data,
        nccl_data=nccl_data,
        yaml_specs_path=args.platform_specs
    )
    
    print("="*60)
    print("PLATFORM EFFICIENCY ANALYSIS")
    print("="*60)
    
    print("\nCore platform features (six independent, meaningful metrics):")
    print("1. platform_gemm: {:.2f} Tflops".format(platform_features['platform_gemm_flops_per_ms']))
    print("2. platform_attention: {:.2f} Tflops".format(platform_features['platform_attention_flops_per_ms']))
    # platform_memory_bandwidth_gbps is already in GB/s; do not scale again
    print("4. platform_memory_bandwidth_gbps: {:.1f} GB/s".format(platform_features['platform_memory_bandwidth_gbps']))
    
    # Display NCCL features when available
    nccl_features = [k for k in platform_features.keys() if k.startswith('platform_nccl_')]
    if nccl_features:
        print("\nNCCL communication features:")
        for feat in sorted(nccl_features):
            if 'bandwidth' in feat:
                print(f"  {feat}: {platform_features[feat]:.2f} GB/s")
            elif 'latency' in feat:
                print(f"  {feat}: {platform_features[feat]:.2f} us")
            else:
                print(f"  {feat}: {platform_features[feat]:.3f}")
                
    # Persist results
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    output_file = f"{args.output_dir}/platform_features.json"
    with open(output_file, 'w') as f:
        
        json.dump(platform_features, f, indent=2)
    
    # Additionally, write measured metrics and theoretical peaks to YAML for regression reuse
    # Reload theoretical peaks to ensure exact alignment with the input YAML
    try:
        specs_for_yaml = calculator.load_platform_specs(args.platform_specs)
        peaks_for_yaml = calculator._convert_yaml_specs_to_peaks(specs_for_yaml)
    except Exception:
        specs_for_yaml = {}
        peaks_for_yaml = {}
    combined_yaml = {
        'measured': {
            # Keep non-NCCL features as-is
            **{k: v for k, v in platform_features.items() if not k.startswith('platform_nccl_')},
            # Only retain two NCCL summary metrics, omitting per-GPU details
            **({
                'platform_nccl_max_bandwidth_gbps': platform_features.get('platform_nccl_max_bandwidth_gbps')
            } if 'platform_nccl_max_bandwidth_gbps' in platform_features else {}),
            **({
                'platform_nccl_avg_latency_ms': platform_features.get('platform_nccl_avg_latency_ms')
            } if 'platform_nccl_avg_latency_ms' in platform_features else {}),
        },           # Measured characteristics (retain units: FLOPs/ms and GB/s)
        'theoretical': peaks_for_yaml            # Theoretical peaks (TFLOPS and GB/s per conversion routine)
    }
    output_yaml = f"{args.output_dir}/platform_features.yaml"
    with open(output_yaml, 'w') as fy:
        yaml.safe_dump(PlatformEfficiencyCalculator._to_builtin(combined_yaml), fy, allow_unicode=True, sort_keys=True)
    
    print(f"\nResults saved to {output_file}")
    print(f"Combined YAML saved to {output_yaml}")

if __name__ == "__main__":
    main() 
