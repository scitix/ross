import numpy as np
from typing import List

from common.models import BaseModel
from common.config import InferenceConfig, get_yaml_config

class PlatformPerf:
    def __init__(self, platform_perf_yaml: str):
        perf_data = get_yaml_config(platform_perf_yaml)
        
        self.platform_gemm_flops_per_ms = perf_data['measured']['platform_gemm_flops_per_ms']
        self.platform_attention_flops_per_ms = perf_data['measured']['platform_attention_flops_per_ms']
        self.platform_memory_bandwidth_gbps = perf_data['measured']['platform_memory_bandwidth_gbps']

        self.theoretical_fp16_flops_per_ms = perf_data['theoretical']['fp16_tflops'] * 1e9 
        self.theoretical_memory_bandwidth_gbps = perf_data['theoretical']['memory_bandwidth_gbps']
        self.theoretical_memory_gb = perf_data['theoretical']['memory_gb']
        self.platform_nccl_max_bandwidth_gbps = perf_data['measured']['platform_nccl_max_bandwidth_gbps']
        self.platform_nccl_avg_latency_ms = perf_data['measured']['platform_nccl_avg_latency_ms']
    
class RegressionFeatures:
    def __init__(self,
                batch_size: int,
                prefill_seq_lens: List[int],
                decode_seq_lens: List[int],
                model: BaseModel,
                inference_config: InferenceConfig, 
                platform_perf: PlatformPerf):
        self.batch_size = batch_size
        self.model = model
        self.inference_config = inference_config
        self.platform_perf = platform_perf
        
        if prefill_seq_lens:
            self.prefill_reqs = len(prefill_seq_lens)
            self.prefill_avg_len = np.mean(prefill_seq_lens)
            self.prefill_total_tokens = sum(prefill_seq_lens)
            self.prefill_max_len = max(prefill_seq_lens)

            self.prefill_sq_sum = np.sum(np.square(prefill_seq_lens)) 
            self.prefill_sq_avg = np.mean(np.square(prefill_seq_lens))
        else:
            self.prefill_reqs = 0
            self.prefill_avg_len = 0
            self.prefill_total_tokens = 0
            self.prefill_max_len = 0
            self.prefill_sq_sum = 0
            self.prefill_sq_avg = 0
            
        if decode_seq_lens:
            self.decode_reqs = len(decode_seq_lens)
            self.decode_avg_len = np.mean(decode_seq_lens)
            self.decode_total_tokens = sum(decode_seq_lens)
            self.decode_max_len = max(decode_seq_lens)

            self.decode_sq_sum = np.sum(np.square(decode_seq_lens)) 
            self.decode_sq_avg = np.mean(np.square(decode_seq_lens))
        else:
            self.decode_reqs = 0
            self.decode_avg_len = 0
            self.decode_total_tokens = 0
            self.decode_max_len = 0
            self.decode_sq_sum = 0
            self.decode_sq_avg = 0
        
        self.num_attention_heads = self.model.get_num_heads()
        self.num_key_value_heads = self.model.get_num_kv_heads()
        self.hidden_size = self.model.get_hidden_size()
        self.num_hidden_layers = self.model.get_num_layers()
        self.intermediate_size = self.model.get_intermediate_size()
        self.vocab_size = self.model.get_vocab_size()
        
        self.pipeline_parallel_size = self.inference_config.pp_size
        self.tensor_parallel_size = self.inference_config.tp_size
        
        avg_seq_len = (self.prefill_total_tokens + self.decode_total_tokens) / self.batch_size
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.gqa_ratio = self.num_attention_heads // self.num_key_value_heads
        
        self.ffn_expansion_ratio = self.intermediate_size / self.hidden_size
        layers_per_rank = self.num_hidden_layers / self.pipeline_parallel_size
        
        # Attention FLOPs total: L * b * Sq * d^2
        attn_flops_raw = self.num_hidden_layers * self.batch_size * avg_seq_len * (self.hidden_size ** 2)
        self.attn_flops_log = np.log10(attn_flops_raw) if attn_flops_raw > 0 else 0
                    
        # MLP FLOPs total: L * b * Sq * d * dff  
        mlp_flops_raw = self.num_hidden_layers * self.batch_size * avg_seq_len * self.hidden_size * self.intermediate_size
        self.mlp_flops_log = np.log10(mlp_flops_raw) if mlp_flops_raw > 0 else 0

        # Per-rank compute (considering self.tensor_parallel_size, self.pipeline_parallel_size)
        attn_flops_per_rank_raw = layers_per_rank * self.batch_size * avg_seq_len * (self.hidden_size ** 2) / self.tensor_parallel_size
        mlp_flops_per_rank_raw = layers_per_rank * self.batch_size * avg_seq_len * self.hidden_size * self.intermediate_size / self.tensor_parallel_size
        self.attn_flops_per_rank_log = np.log10(attn_flops_per_rank_raw) if attn_flops_per_rank_raw > 0 else 0
        self.mlp_flops_per_rank_log = np.log10(mlp_flops_per_rank_raw) if mlp_flops_per_rank_raw > 0 else 0
                    
        # KV-cache memory per token per layer
        kv_head_dim = self.hidden_size // self.num_attention_heads
        kv_cache_raw = self.num_hidden_layers * self.num_key_value_heads * kv_head_dim * avg_seq_len * self.batch_size
        self.kv_cache_log = np.log10(kv_cache_raw) if kv_cache_raw > 0 else 0
                
        # Assume KV cache sharded across tensor parallel ranks (common in self.tensor_parallel_size attention)
        kv_cache_per_rank_raw = layers_per_rank * self.num_key_value_heads * kv_head_dim * avg_seq_len * self.batch_size / self.tensor_parallel_size
        self.kv_cache_per_rank_log = np.log10(kv_cache_per_rank_raw) if kv_cache_per_rank_raw > 0 else 0
                    
        # Attention memory bandwidth proxy per layer per rank
        # Previous proxy: b*Sq*d + 4*d^2/self.tensor_parallel_size. Make it per-rank and across its layers
        attn_mem_per_layer_per_rank = self.batch_size * avg_seq_len * self.hidden_size + 4 * (self.hidden_size ** 2) / self.tensor_parallel_size
        attn_mem_total_per_rank = layers_per_rank * attn_mem_per_layer_per_rank
        self.attn_memory_log = np.log10(attn_mem_total_per_rank) if attn_mem_total_per_rank > 0 else 0
                    
        # Total model parameters (architecture-scale, not per-rank)
        total_params_raw = self.num_hidden_layers * (4 * (self.hidden_size ** 2) + 2 * self.hidden_size * self.intermediate_size)
        self.total_params_log = np.log10(total_params_raw) if total_params_raw > 0 else 0
                    
        # Add simpler multiplicative features that are less correlated
        self.layers_x_hidden_sq = self.num_hidden_layers * (self.hidden_size ** 2)  # L × d²
        self.layers_x_hidden_x_ffn = self.num_hidden_layers * self.hidden_size * self.intermediate_size  # L × d × dff
        self.batch_x_seq_x_hidden = self.batch_size * avg_seq_len * self.hidden_size  # b × Sq × d
                    
        self.pipeline_overhead_proxy = (self.pipeline_parallel_size - 1) / self.pipeline_parallel_size
                
        tp_comm_raw = layers_per_rank * self.batch_size * self.hidden_size
        self.tp_comm_log = np.log10(tp_comm_raw)


class SGLRegressionFeatures(RegressionFeatures):
    def __init__(self,
                batch_size: int,
                seq_lens: List[int],
                model: BaseModel,
                inference_config: InferenceConfig, 
                platform_perf: PlatformPerf):
        super().__init__(batch_size, None, None, model, inference_config, platform_perf)

        self.avg_len = np.mean(seq_lens)
        self.total_tokens = sum(seq_lens)
        self.max_len = max(seq_lens)

        self.sq_sum = np.sum(np.square(seq_lens)) 
        self.sq_avg = np.mean(np.square(seq_lens))
