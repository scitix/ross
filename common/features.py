import numpy as np
from typing import List

from common.models import BaseModel
from common.config import InferenceConfig, get_yaml_config

COMMON_MODEL_FEATURES = [
    "num_hidden_layers",
    "hidden_size",
    "num_attention_heads",
    "num_key_value_heads",
    "intermediate_size",
    "tensor_parallel_size",
    "pipeline_parallel_size",
    "vocab_size",
    "num_experts",
    "topk",
    "moe_intermediate_size",
    "shared_expert_intermediate_size",
]

COMMON_DERIVED_FEATURES = [
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
    "active_expert_width",
    "router_flops_proxy",
    "expert_compute_proxy",
    "shared_ffn_proxy",

    "moe_ep_comm_proxy_log",       # EP Communication proxy: log(L × b × Sq × d × num_experts) (assuming all experts active, which is a worst-case for communication)
    "expert_sparsity_ratio",       # topk / num_experts
    "tokens_per_expert_log",       # log((b × Sq) / (num_experts / topk)) - Tokens per expert, considering sparsity
    "batch_active_expert_ratio",   # (batch_size * topk) / num_experts - Ratio of active experts per batch
    "expert_intensity_proxy",      # L × b × Sq × d × topk × moe_intermediate_size / (num_experts / topk) - Proxy for compute intensity per expert, considering sparsity
    "shared_expert_compute_ratio", # shared_ffn_proxy / expert_compute_proxy - Ratio of shared FFN compute to MoE compute, indicating how much the shared FFN contributes relative to the experts
    "decode_step_tokens",          # Current-step active tokens during decode, approximated as one token per scheduled request.
    "decode_tokens_per_expert_log",
    "decode_tokens_per_expert_per_rank_log",
    "decode_router_flops_proxy",
    "decode_expert_compute_proxy",
    "decode_moe_ep_comm_proxy_log",
]

COMMON_PLATFORM_FEATURES = [
    "platform_gemm_flops_per_ms",
    "platform_attention_flops_per_ms",
    "platform_memory_bandwidth_gbps",
    "theoretical_fp16_flops_per_ms",
    "theoretical_memory_bandwidth_gbps",
    "platform_nccl_max_bandwidth_gbps",
    "platform_nccl_avg_latency_ms",
]

COMMON_REGRESSION_FEATURES = (
    COMMON_MODEL_FEATURES + COMMON_DERIVED_FEATURES + COMMON_PLATFORM_FEATURES
)

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
        self.num_experts = self.model.get_num_experts()
        self.topk = self.model.get_topk()
        self.moe_intermediate_size = self.model.get_moe_intermediate_size()
        self.shared_expert_intermediate_size = self.model.get_shared_expert_intermediate_size()
        
        self.pipeline_parallel_size = self.inference_config.pp_size
        self.tensor_parallel_size = self.inference_config.tp_size
        
        avg_seq_len = (self.prefill_total_tokens + self.decode_total_tokens) / self.batch_size
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.gqa_ratio = self.num_attention_heads // self.num_key_value_heads
        
        self.ffn_expansion_ratio = self.intermediate_size / self.hidden_size
        layers_per_rank = self.num_hidden_layers / self.pipeline_parallel_size

        # 1. Params
        total_params_raw = self.model.get_total_params()
        num_tokens = self.batch_size * avg_seq_len
        self.total_params_log = np.log10(total_params_raw) if total_params_raw > 0 else 0

        # 2. MoE Communication Proxy
        # by default, EP size is the number of ranks that share the same expert weights, which is typically DP × TP size
        self.moe_ep_size = inference_config.dp_size * inference_config.tp_size
        ep_comm_raw = (num_tokens * self.hidden_size * self.topk) / self.moe_ep_size
        self.moe_ep_comm_proxy_log = np.log10(ep_comm_raw) if ep_comm_raw > 0 else 0

        # 3. Expert Load and Sparsity
        self.expert_sparsity_ratio = self.topk / self.num_experts if self.num_experts > 0 else 1.0
        tokens_per_expert = num_tokens / max(self.num_experts, 1)
        self.tokens_per_expert_log = np.log10(tokens_per_expert) if tokens_per_expert > 0 else 0

        # 4. Dynamic Activation Ratio (Important: Distinguish between Memory-Bound and Compute-Bound)
        if self.num_experts > 0 and num_tokens > 0:
            # Probabilistic Model: What proportion of experts are accessed at least once in a Batch
            self.batch_active_expert_ratio = 1.0 - np.power(1.0 - (1.0 / self.num_experts), num_tokens * self.topk)
        else:
            self.batch_active_expert_ratio = 1.0

        # 5. Compute Intensity
        self.expert_compute_proxy = (self.num_hidden_layers * num_tokens * self.hidden_size * self.topk * self.moe_intermediate_size)
        self.expert_intensity_proxy = self.expert_compute_proxy / (total_params_raw / self.tensor_parallel_size) if total_params_raw > 0 else 0

        # 6. Shared Expert Weights
        active_width = (self.topk * self.moe_intermediate_size) + self.shared_expert_intermediate_size
        self.shared_expert_compute_ratio = self.shared_expert_intermediate_size / active_width if active_width > 0 else 0

        if self.num_experts > 1 and self.topk > 0 and self.decode_reqs > 0:
            self.decode_step_tokens = self.decode_reqs
            decode_tokens_per_expert = (self.decode_step_tokens * self.topk) / self.num_experts
            experts_per_rank = self.num_experts / self.moe_ep_size if self.moe_ep_size > 0 else self.num_experts
            decode_tokens_per_expert_per_rank = (
                (self.decode_step_tokens * self.topk) / experts_per_rank
                if experts_per_rank > 0 else 0
            )
            decode_ep_comm_raw = (
                self.decode_step_tokens * self.hidden_size * self.topk
            ) / self.moe_ep_size if self.moe_ep_size > 0 else 0
            self.decode_tokens_per_expert_log = np.log10(decode_tokens_per_expert) if decode_tokens_per_expert > 0 else 0
            self.decode_tokens_per_expert_per_rank_log = (
                np.log10(decode_tokens_per_expert_per_rank) if decode_tokens_per_expert_per_rank > 0 else 0
            )
            self.decode_router_flops_proxy = (
                self.num_hidden_layers
                * self.decode_step_tokens
                * self.hidden_size
                * self.num_experts
            )
            self.decode_expert_compute_proxy = (
                self.num_hidden_layers
                * self.decode_step_tokens
                * self.hidden_size
                * self.topk
                * self.moe_intermediate_size
            )
            self.decode_moe_ep_comm_proxy_log = np.log10(decode_ep_comm_raw) if decode_ep_comm_raw > 0 else 0
        else:
            self.decode_step_tokens = 0
            self.decode_tokens_per_expert_log = 0
            self.decode_tokens_per_expert_per_rank_log = 0
            self.decode_router_flops_proxy = 0
            self.decode_expert_compute_proxy = 0
            self.decode_moe_ep_comm_proxy_log = 0
        
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
        self.active_expert_width = self.topk * self.moe_intermediate_size
        self.router_flops_proxy = self.num_hidden_layers * self.batch_size * avg_seq_len * self.hidden_size * self.num_experts
        self.expert_compute_proxy = (
            self.num_hidden_layers
            * self.batch_size
            * avg_seq_len
            * self.hidden_size
            * self.topk
            * self.moe_intermediate_size
        )
        self.shared_ffn_proxy = (
            self.num_hidden_layers
            * self.batch_size
            * avg_seq_len
            * self.hidden_size
            * self.shared_expert_intermediate_size
        )
                    
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
        # Keep dense SGL models on the legacy feature path for backward
        # compatibility, while MoE models use the seq-len-aware path.
        if model.get_num_experts() > 1:
            super().__init__(batch_size, seq_lens, [], model, inference_config, platform_perf)
        else:
            super().__init__(batch_size, None, None, model, inference_config, platform_perf)

        self.avg_len = np.mean(seq_lens)
        self.total_tokens = sum(seq_lens)
        self.max_len = max(seq_lens)

        self.sq_sum = np.sum(np.square(seq_lens)) 
        self.sq_avg = np.mean(np.square(seq_lens))


        if self.num_experts > 1 and self.topk > 0 and self.batch_size > 0:
            self.decode_step_tokens = self.batch_size
            decode_tokens_per_expert = (self.decode_step_tokens * self.topk) / self.num_experts
            experts_per_rank = self.num_experts / self.moe_ep_size if self.moe_ep_size > 0 else self.num_experts
            decode_tokens_per_expert_per_rank = (
                (self.decode_step_tokens * self.topk) / experts_per_rank
                if experts_per_rank > 0 else 0
            )
            decode_ep_comm_raw = (
                self.decode_step_tokens * self.hidden_size * self.topk
            ) / self.moe_ep_size if self.moe_ep_size > 0 else 0
            self.decode_tokens_per_expert_log = np.log10(decode_tokens_per_expert) if decode_tokens_per_expert > 0 else 0
            self.decode_tokens_per_expert_per_rank_log = (
                np.log10(decode_tokens_per_expert_per_rank) if decode_tokens_per_expert_per_rank > 0 else 0
            )
            self.decode_router_flops_proxy = (
                self.num_hidden_layers
                * self.decode_step_tokens
                * self.hidden_size
                * self.num_experts
            )
            self.decode_expert_compute_proxy = (
                self.num_hidden_layers
                * self.decode_step_tokens
                * self.hidden_size
                * self.topk
                * self.moe_intermediate_size
            )
            self.decode_moe_ep_comm_proxy_log = np.log10(decode_ep_comm_raw) if decode_ep_comm_raw > 0 else 0
        else:
            self.decode_step_tokens = 0
            self.decode_tokens_per_expert_log = 0
            self.decode_tokens_per_expert_per_rank_log = 0
            self.decode_router_flops_proxy = 0
            self.decode_expert_compute_proxy = 0
            self.decode_moe_ep_comm_proxy_log = 0
