# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
import os
import json
import logging
from pathlib import Path

from typing import Dict, Any

import common.operations as ops
from common.config import InferenceConfig, GEMMQuantMode

logger = logging.getLogger(__name__)

MODEL_WEIGHTS = {
    "DeepSeek-R1-Distill-Llama-70B": 131.68,
    "Meta-Llama-3.1-70B-Instruct": 131.56,
    
    "Qwen3-32B": 61.2,
    "Qwen2.5-32B": 61.56,
    "Qwen2.5-32B-Instruct": 61.56,
    "Qwen2.5-72B-Instruct": 136.0,
    "Qwen3-235B-A22B": 439.36,
    "Qwen3-30B-A3B": 55.8,
    
    "Llama-2-7b-chat-hf": 12.55,
    "Qwen2.5-7B-Instruct": 14.25,
    "Meta-Llama-3.1-8B-Instruct": 14.99,
    
    "gemma-2-27b-it": 50.92,
    "gpt-oss-20b": 14.32,
    "gpt-oss-120b": 69.28,

    "DeepSeek-V3-0324": 680 # 636.96
}

def get_model(model_uri: str,
            inference_config: InferenceConfig
            ) -> BaseModel:
    with open(os.path.join(model_uri, 'config.json'), 'r') as f:
            model_configs = json.load(f)

    # by default: using llama models
    weights_from_config = 0
    for k, v in MODEL_WEIGHTS.items():
        if k in model_uri:
            weights_from_config = v

    model_configs['model_uri'] = model_uri
    model = LLAMAModel(model_configs, inference_config, weights_from_config)
    return model

class BaseModel(object):
    """
    Base model class.
    """
    def __init__(self, 
                model_configs: Dict[str, Any], 
                inference_config: InferenceConfig,
                weights_from_config: float = 0) -> None:
        
        self.inference_config = inference_config
        self.weights_from_config = weights_from_config
        # internal only
        self._num_layers = model_configs["num_hidden_layers"]
        self._num_heads = model_configs["num_attention_heads"]
        self._num_kv_heads = model_configs["num_key_value_heads"]
        self._hidden_size = model_configs["hidden_size"]
        if "head_dim" in model_configs:
            self._head_size = model_configs["head_dim"]
        else:
            self._head_size = self._hidden_size // self._num_heads
        self._inter_size = model_configs["intermediate_size"]
        self._vocab_size = model_configs["vocab_size"]

        self.model_uri = model_configs["model_uri"]
        if model_configs["model_uri"].lower().find("deepseek") != -1:
            # DeepSeek
            self.kv_lora_rank = model_configs["kv_lora_rank"]
            self.qk_rope_head_dim = model_configs["qk_rope_head_dim"]

        # TODO: remaining configs from ai-configurator
        self.context_ops = []
        self.generation_ops = []
        self._nextn = inference_config.nextn
        self._nextn_accept_rates = inference_config.nextn_accept_rates

    def get_intermediate_size(self):
        return self._inter_size
        
    def get_num_layers(self):
        return self._num_layers

    def get_num_heads(self):
        return self._num_heads
    
    def get_hidden_size(self):
        return self._hidden_size
    
    def get_num_kv_heads(self):
        return self._num_kv_heads

    def get_head_size(self):
        return self._head_size
    
    def get_num_kv_heads_per_gpu(self):
        return (self._num_kv_heads + self.inference_config.tp_size - 1) // self.inference_config.tp_size
    
    def get_vocab_size(self):
        return self._vocab_size
    
    def _get_activation(self, num_tokens: int) -> int:
        ...


class GPTModel(BaseModel):
    """
    GPT series uses this model impl.
    Some rules to follow,
    Due to implementation, attn layer name needs to be context_attention or generation_attention, exact match is required. Same for logits_gemm.
    Other than DS V3, all other models don't support mtp
    """
    def __init__(self, *args) -> None:
        super().__init__(*args)
        assert self._nextn == 0, 'Only DS V3 supports mtp'

        h = self._hidden_size
        tp_size = self.inference_config.tp_size
        pp_size = self.inference_config.pp_size
        num_kv_heads_per_GPU = self.get_num_kv_heads_per_gpu()
        gemm_quant_mode = self.inference_config.gemm_quant_mode
        kvcache_quant_mode = self.inference_config.kvcache_quant_mode
        fmha_quant_mode = self.inference_config.fmha_quant_mode

        self.context_ops.extend([ops.Embedding(f'context_embedding', 1, self._vocab_size, h, 0.3),
                                ops.ElementWise(f'context_add_norm_1', self._num_layers, 2*h, 2*h, 0.8),
                                ops.GEMM(f'context_qkv_gemm', self._num_layers, self._num_heads*self._head_size//tp_size + self._head_size*num_kv_heads_per_GPU*2, h, gemm_quant_mode), 
                                ops.ContextAttention(f'context_attention', self._num_layers, self._num_heads//tp_size, num_kv_heads_per_GPU, kvcache_quant_mode, fmha_quant_mode),
                                ops.GEMM(f'context_proj_gemm', self._num_layers, h, self._num_heads*self._head_size//tp_size, gemm_quant_mode),
                                ops.ElementWise(f'context_add_norm_2', self._num_layers, 2*h, 2*h, 0.8),
                                ops.GEMM(f'context_ffn1_gemm', self._num_layers, self._inter_size//tp_size, h, gemm_quant_mode),
                                ops.ElementWise(f'context_act', self._num_layers, self._inter_size//tp_size, self._inter_size//tp_size, 0.8),
                                ops.GEMM(f'context_ffn2_gemm', self._num_layers, h, self._inter_size//tp_size, gemm_quant_mode),
                                ops.GEMM(f'context_logits_gemm', 1, self._vocab_size//tp_size, h, GEMMQuantMode.float16)])

        self.generation_ops.extend([ops.Embedding(f'generation_embedding', 1, self._vocab_size, h, 0.3),
                                ops.ElementWise(f'generation_add_norm_1', self._num_layers, 2*h, 2*h, 0.8),
                                ops.GEMM(f'generation_qkv_gemm', self._num_layers, self._num_heads*self._head_size//tp_size+self._head_size*num_kv_heads_per_GPU*2, h, gemm_quant_mode), 
                                ops.GenerationAttention(f'generation_attention', self._num_layers, self._num_heads//tp_size, num_kv_heads_per_GPU, kvcache_quant_mode),
                                ops.GEMM(f'generation_proj_gemm', self._num_layers, h, self._num_heads*self._head_size//tp_size, gemm_quant_mode),
                                ops.ElementWise(f'generation_add_norm_2', self._num_layers, 2*h, 2*h, 0.8),
                                ops.GEMM(f'generation_ffn1_gemm', self._num_layers, self._inter_size//tp_size, h, gemm_quant_mode),
                                ops.ElementWise(f'generation_act', self._num_layers, self._inter_size//tp_size, self._inter_size//tp_size, 0.8),
                                ops.GEMM(f'generation_ffn2_gemm', self._num_layers, h, self._inter_size//tp_size, gemm_quant_mode),
                                ops.GEMM(f'generation_logits_gemm', 1, self._vocab_size//tp_size, h, GEMMQuantMode.float16)])
        
        # when tp_size=0, the comm part will be 0
        self.context_ops.append(ops.AllReduce('context_ar_1', self._num_layers, h, tp_size))
        self.context_ops.append(ops.AllReduce('context_ar_2', self._num_layers, h, tp_size))
        self.generation_ops.append(ops.AllReduce('generation_ar_1', self._num_layers, h, tp_size))
        self.generation_ops.append(ops.AllReduce('generation_ar_2', self._num_layers, h, tp_size))

        # pp
        pp_scale_factor = pp_size-1
        self.context_ops.append(ops.P2P('context_p2p', pp_scale_factor, h, pp_size))
        self.generation_ops.append(ops.P2P('generation_p2p', pp_scale_factor, h, pp_size))

    def _get_activation(self, num_tokens: int) -> int:
        """
        Estimates the peak activation memory usage for a GPT-style model.
        The peak is dominated by the output of the FFN's up-projection layer.
        """
        # Assuming bf16/fp16, where each element takes 2 bytes.
        bytes_per_element = 2
        tp_size = self.inference_config.tp_size
        
        # The largest activation tensor is after 'context_ffn1_gemm'.
        # Its shape is (num_tokens, self._inter_size / tp_size).
        peak_tensor_size = num_tokens * (self._inter_size // tp_size) * bytes_per_element
        
        # A heuristic factor to account for other concurrent activations.
        heuristic_factor = 2.0
        return int(peak_tensor_size * heuristic_factor)


class LLAMAModel(BaseModel):
    """
    LLAMA series uses this model impl. Other variants without large difference can use this as well, e.g., only positional embedding or activation is different.
    Some rules to follow,
    Due to implementation, attn layer name needs to be context_attention or generation_attention, exact match is required. Same for logits_gemm.
    Other than DS V3, all other models don't support mtp
    """
    def __init__(self, *args) -> None:
        super().__init__(*args)
        assert self._nextn == 0, 'Only DS V3 supports mtp'

        h = self._hidden_size
        tp_size = self.inference_config.tp_size
        pp_size = self.inference_config.pp_size
        num_kv_heads_per_GPU = self.get_num_kv_heads_per_gpu()
        gemm_quant_mode = self.inference_config.gemm_quant_mode
        kvcache_quant_mode = self.inference_config.kvcache_quant_mode
        fmha_quant_mode = self.inference_config.fmha_quant_mode

        self.context_ops.extend([ops.Embedding(f'context_embedding', 1, self._vocab_size, h, 0.3),
                                ops.ElementWise(f'context_add_norm_1', self._num_layers, 2*h, 2*h, 0.8),
                                ops.GEMM(f'context_qkv_gemm', self._num_layers, self._num_heads*self._head_size//tp_size+self._head_size*num_kv_heads_per_GPU*2, h, gemm_quant_mode), 
                                ops.ContextAttention(f'context_attention', self._num_layers, self._num_heads//tp_size, num_kv_heads_per_GPU, kvcache_quant_mode, fmha_quant_mode),
                                ops.GEMM(f'context_proj_gemm', self._num_layers, h, self._num_heads*self._head_size//tp_size, gemm_quant_mode),
                                ops.ElementWise(f'context_add_norm_2', self._num_layers, 2*h, 2*h, 0.8),
                                ops.GEMM(f'context_gate_ffn1_gemm', self._num_layers, 2*self._inter_size//tp_size, h, gemm_quant_mode),
                                ops.ElementWise(f'context_act_gate', self._num_layers, 2*self._inter_size//tp_size, self._inter_size//tp_size, 0.8),
                                ops.GEMM(f'context_ffn2_gemm', self._num_layers, h, self._inter_size//tp_size, gemm_quant_mode),
                                ops.GEMM(f'context_logits_gemm', 1, self._vocab_size//tp_size, h, GEMMQuantMode.float16)])

        self.generation_ops.extend([ops.Embedding(f'generation_embedding', 1, self._vocab_size, h, 0.3),
                                ops.ElementWise(f'generation_add_nrom_1', self._num_layers, 2*h, 2*h, 0.8),
                                ops.GEMM(f'generation_qkv_gemm', self._num_layers, self._num_heads*self._head_size//tp_size+self._head_size*num_kv_heads_per_GPU*2, h, gemm_quant_mode), 
                                ops.GenerationAttention(f'generation_attention', self._num_layers, self._num_heads//tp_size, num_kv_heads_per_GPU, kvcache_quant_mode),
                                ops.GEMM(f'generation_proj_gemm', self._num_layers, h, self._num_heads*self._head_size//tp_size, gemm_quant_mode),
                                ops.ElementWise(f'generation_add_norm_2', self._num_layers, 2*h, 2*h, 0.8),
                                ops.GEMM(f'generation_gate_ffn1_gemm', self._num_layers, 2*self._inter_size//tp_size, h, gemm_quant_mode),
                                ops.ElementWise(f'generation_act_gate', self._num_layers, 2*self._inter_size//tp_size, self._inter_size//tp_size, 0.8),
                                ops.GEMM(f'generation_ffn2_gemm', self._num_layers, h, self._inter_size//tp_size, gemm_quant_mode),
                                ops.GEMM(f'generation_logits_gemm', 1, self._vocab_size//tp_size, h, GEMMQuantMode.float16)])
        
        # when tp_message_size=0, the comm part will be 0
        self.context_ops.append(ops.AllReduce('context_ar_1', self._num_layers, h, tp_size))
        self.context_ops.append(ops.AllReduce('context_ar_2', self._num_layers, h, tp_size))
        self.generation_ops.append(ops.AllReduce('generation_ar_1', self._num_layers, h, tp_size))
        self.generation_ops.append(ops.AllReduce('generation_ar_2', self._num_layers, h, tp_size))

        # pp
        pp_scale_factor = pp_size-1
        self.context_ops.append(ops.P2P('context_p2p', pp_scale_factor, h, pp_size))
        self.generation_ops.append(ops.P2P('generation_p2p', pp_scale_factor, h, pp_size))

    def _get_activation(self, num_tokens: int) -> int:
        """
        Estimates the peak activation memory usage for a LLAMA-style model.
        The peak is dominated by the output of the gated FFN's up-projection.
        """
        # Assuming bf16/fp16, where each element takes 2 bytes.
        bytes_per_element = 2
        # tp_size = self.inference_config.tp_size
        
        # In LLAMA models with SwiGLU, the FFN up-projection ('context_gate_ffn1_gemm')
        # creates a tensor of size (num_tokens, 2 * self._inter_size / tp_size).
        # This is the largest intermediate tensor.
        peak_tensor_size = num_tokens * (2 * self._inter_size) * bytes_per_element
        
        # A heuristic factor to account for other concurrent activations.
        heuristic_factor = 2.0
        
        return int(peak_tensor_size * heuristic_factor)
    
    
# mostly for mixtral models
class MOEModel(BaseModel):
    """
    Traditional MoE models uses this model impl: Mixtral, LLAMA4_MOE, etc.
    Some rules to follow,
    Due to implementation, attn layer name needs to be context_attention or generation_attention, exact match is required. Same for logits_gemm.
    Other than DS V3, all other models don't support mtp
    TODO: redesign shared moe part.
    """
    def __init__(self, topk: int, num_experts: int, moe_inter_size: int, *args) -> None:
        super().__init__(*args)
        assert self._nextn == 0, 'Only DS V3 supports mtp'

        # make sure the paralel width is same
        assert(self.inference_config.tp_size * self.inference_config.attention_dp_size == self.inference_config.moe_tp_size * self.inference_config.moe_ep_size), \
            f"tp_size ({self.inference_config.tp_size}) * attention_dp_size ({self.inference_config.attention_dp_size}) should be equal to moe_tp_size ({self.inference_config.moe_tp_size}) * moe_ep_size ({self.inference_config.moe_ep_size})"
        
        assert(num_experts >= self.inference_config.moe_ep_size), f"ep size cannot be larger than num_experts {num_experts}"
        assert(self.inference_config.tp_size * self.inference_config.attention_dp_size <= 256), f"moe ep size {self.inference_config.moe_ep_size} * moe tp size {self.inference_config.moe_tp_size} should not be larger than 256"

        self._topk = topk

        self._topk = topk
        self._num_experts = num_experts
        self._moe_inter_size = moe_inter_size

        moe_quant_mode = self.inference_config.moe_quant_mode

        h = self._hidden_size
        tp_size = self.inference_config.tp_size
        moe_tp_size = self.inference_config.moe_tp_size
        moe_ep_size = self.inference_config.moe_ep_size
        attention_dp_size = self.inference_config.attention_dp_size
        pp_size = self.inference_config.pp_size
        num_kv_heads_per_GPU = self.get_num_kv_heads_per_gpu()
        gemm_quant_mode = self.inference_config.gemm_quant_mode
        kvcache_quant_mode = self.inference_config.kvcache_quant_mode
        fmha_quant_mode = self.inference_config.fmha_quant_mode
        workload_distribution = self.inference_config.workload_distribution

        self.context_ops.extend([ops.Embedding(f'context_embedding', 1, self._vocab_size, h, 0.3),
                                ops.ElementWise(f'context_add_norm_1', self._num_layers, 2*h, 2*h, 0.8),
                                ops.GEMM(f'context_qkv_gemm', self._num_layers, self._num_heads*self._head_size//tp_size+self._head_size*num_kv_heads_per_GPU*2, h, gemm_quant_mode),
                                ops.ContextAttention(f'context_attention', self._num_layers, self._num_heads//tp_size, num_kv_heads_per_GPU, kvcache_quant_mode, fmha_quant_mode),
                                ops.GEMM(f'context_proj_gemm', self._num_layers, h, self._num_heads*self._head_size//tp_size, gemm_quant_mode),
                                ops.ElementWise(f'context_add_norm_2', self._num_layers, 2*h, 2*h, 0.8)])

        #router, only take it into account when num_experts >= 128
        if self._num_experts >= 128:
            self.context_ops.extend([
                            ops.GEMM(f'context_router_gemm', self._num_layers, self._num_experts, h, GEMMQuantMode.float16)
                            ])

        # dispatch tokens to experts, moe calc and get tokens back
        self.context_ops.extend([
                                ops.MoEDispatch(f'context_moe_pre_dispatch', self._num_layers, h, self._topk, self._num_experts, moe_tp_size, moe_ep_size, attention_dp_size, True),
                                ops.MoE(f'context_moe', self._num_layers, h, self._moe_inter_size, self._topk, self._num_experts, moe_tp_size, moe_ep_size, moe_quant_mode, workload_distribution, attention_dp_size),
                                ops.MoEDispatch(f'context_moe_post_dispatch', self._num_layers, h, self._topk, self._num_experts, moe_tp_size, moe_ep_size, attention_dp_size, False)])
        
        self.context_ops.extend([ops.GEMM(f'context_logits_gemm', 1, self._vocab_size//tp_size, h, GEMMQuantMode.float16)])

        self.generation_ops.extend([ops.Embedding(f'generation_embedding', 1, self._vocab_size, h, 0.3),
                                ops.ElementWise(f'generation_add_norm_1', self._num_layers, 2*h, 2*h, 0.8),
                                ops.GEMM(f'generation_qkv_gemm', self._num_layers, self._num_heads*self._head_size//tp_size+self._head_size*num_kv_heads_per_GPU*2, h, gemm_quant_mode),
                                ops.GenerationAttention(f'generation_attention', self._num_layers, self._num_heads//tp_size, num_kv_heads_per_GPU, kvcache_quant_mode),
                                ops.GEMM(f'generation_proj_gemm', self._num_layers, h, self._num_heads*self._head_size//tp_size, gemm_quant_mode),
                                ops.ElementWise(f'generation_add_norm_2', self._num_layers, 2*h, 2*h, 0.8)])

        #router, only take it into account when num_experts >= 128
        if self._num_experts >= 128:
            self.generation_ops.extend([
                            ops.GEMM(f'generation_router_gemm', self._num_layers, self._num_experts, h, GEMMQuantMode.float16)
                            ])

        # dispatch tokens to experts, moe calc and get tokens back
        self.generation_ops.extend([
                                ops.MoEDispatch(f'generation_moe_pre_dispatch', self._num_layers, h, self._topk, self._num_experts, moe_tp_size, moe_ep_size, attention_dp_size, True),
                                ops.MoE(f'generation_moe', self._num_layers, h, self._moe_inter_size, self._topk, self._num_experts, moe_tp_size, moe_ep_size, moe_quant_mode, workload_distribution, attention_dp_size),
                                ops.MoEDispatch(f'generation_moe_post_dispatch', self._num_layers, h, self._topk, self._num_experts, moe_tp_size, moe_ep_size, attention_dp_size, False)
                                ])
        # logits gemm
        self.generation_ops.extend([ops.GEMM(f'generation_logits_gemm', 1, self._vocab_size//tp_size, h, GEMMQuantMode.float16)])

        # # # when tp_size=0, the comm part will be 0
        # self.context_ops.append(ops.AllReduce('context_ar_1', self._num_layers, h, tp_size))
        # self.context_ops.append(ops.AllReduce('context_ar_2', self._num_layers, h, tp_size))
        # self.generation_ops.append(ops.AllReduce('generation_ar_1', self._num_layers, h, tp_size))
        # self.generation_ops.append(ops.AllReduce('generation_ar_2', self._num_layers, h, tp_size))

        # pp
        pp_scale_factor = pp_size-1
        self.context_ops.append(ops.P2P('context_p2p', pp_scale_factor, h, pp_size))
        self.generation_ops.append(ops.P2P('generation_p2p', pp_scale_factor, h, pp_size))
        
    def _get_activation(self, num_tokens: int) -> int:
        """
        Estimates the peak activation memory usage for a MoE model like Mixtral.
        Activation memory is complex due to token routing. A reasonable estimate
        is based on the MoE up-projection size for the routed tokens.
        """
        bytes_per_element = 2
        moe_tp_size = self.inference_config.moe_tp_size
        
        # The MoE layer is the FFN for this model. The largest activation occurs
        # within the selected experts. Each token is routed to `_topk` experts.
        # The up-projection inside each expert expands to `_moe_inter_size`.
        # The effective tensor size is based on the number of tokens processed by
        # the local experts on one GPU.
        # A simplification is to assume tokens are evenly distributed.
        peak_tensor_size = num_tokens * self._topk * (self._moe_inter_size // moe_tp_size) * bytes_per_element
        
        heuristic_factor = 2.0
        
        return int(peak_tensor_size * heuristic_factor)
    

def calc_expectation(nextn: int, nextn_accept_rates: list[float]) -> float:
    """
    Calculate expectation for mtp
    """
    prob = 1.
    if nextn == 0:
        return 0.0
    
    for i in range(nextn):
        prob *= nextn_accept_rates[i]
    if nextn > 1:
        return prob + calc_expectation(nextn-1, nextn_accept_rates)
    else:
        return prob
    
class DeepSeekModel(BaseModel):
    """
    DeepSeek V3/R1 uses this model impl.
    """
    def __init__(self, topk: int, num_experts: int, moe_inter_size: int, *args) -> None:
        super().__init__(*args)

        # make sure the paralel width is same
        assert(self.inference_config.tp_size * self.inference_config.attention_dp_size == self.inference_config.moe_tp_size * self.inference_config.moe_ep_size), \
            f"tp_size ({self.inference_config.tp_size}) * attention_dp_size ({self.inference_config.attention_dp_size}) should be equal to moe_tp_size ({self.inference_config.moe_tp_size}) * moe_ep_size ({self.inference_config.moe_ep_size})"
        
        assert(num_experts >= self.inference_config.moe_ep_size), f"ep size cannot be larger than num_experts {num_experts}"
        assert(self.inference_config.tp_size * self.inference_config.attention_dp_size <= 256), f"moe ep size {self.inference_config.moe_ep_size} * moe tp size {self.inference_config.moe_tp_size} should not be larger than 256"

        self._topk = topk
        self._num_experts = num_experts
        self._moe_inter_size = moe_inter_size

         # used to scale the tpot to reflect mtp effect: 
         # 1. mtp will reduce the overall time by expected_tokens_per_step 
         # 2. mtp module introduces nextn new transformer layers+linear layers (we ignore the linear layers for now)
         # 3. special correction in ifb step due to we leveraging ctx phase for gen tokens non-attn part
         # meanwhile, needs to scale the actual bs of generation by nextn, this is covered in inferencesession
        self._mtp_scale_factor = 1./(1+calc_expectation(self._nextn, self._nextn_accept_rates))*(self._nextn+self._num_layers)/self._num_layers

        gemm_quant_mode = self.inference_config.gemm_quant_mode
        moe_quant_mode = self.inference_config.moe_quant_mode

        mla_bmm_quant_mode = GEMMQuantMode.fp8 if gemm_quant_mode != GEMMQuantMode.float16 else GEMMQuantMode.float16

        h = self._hidden_size # 7168
        tp_size = self.inference_config.tp_size
        moe_tp_size = self.inference_config.moe_tp_size
        moe_ep_size = self.inference_config.moe_ep_size
        attention_dp_size = self.inference_config.attention_dp_size
        pp_size = self.inference_config.pp_size
        num_kv_heads_per_GPU = self.get_num_kv_heads_per_gpu()

        kvcache_quant_mode = self.inference_config.kvcache_quant_mode
        fmha_quant_mode = self.inference_config.fmha_quant_mode
        workload_distribution = self.inference_config.workload_distribution

        self.context_ops.extend([ops.Embedding(f'context_embedding', 1, self._vocab_size, h, 0.3),
                                ops.ElementWise(f'context_add_norm_1', self._num_layers, 2*h, 2*h, 0.8),
                                ops.GEMM(f'context_downscale_gemm', self._num_layers, 2112, h, gemm_quant_mode), # on every gpu, fused_a
                                ops.GEMM(f'context_q_b_proj_gemm', self._num_layers, 24576//tp_size, 1536, gemm_quant_mode),
                                ops.GEMM(f'context_kv_b_proj_gemm', self._num_layers, 32768//tp_size, 512, gemm_quant_mode), # ifb ctx attn part
                                ops.ContextMLA(f'context_attention', self._num_layers, tp_size, kvcache_quant_mode, fmha_quant_mode), # ifb ctx attn part
                                ops.GEMM(f'context_proj_gemm', self._num_layers, h, 128*128//tp_size, gemm_quant_mode), # ifb ctx attn part
                                ops.ElementWise(f'context_add_norm_2', self._num_layers, 2*h, 2*h, 0.8)])

        # shared moe
        self.context_ops.extend([
                                ops.GEMM(f'context_shared_gate_gemm', self._num_layers, self._moe_inter_size//tp_size, h, gemm_quant_mode),
                                ops.GEMM(f'context_shared_ffn1_gemm', self._num_layers, self._moe_inter_size//tp_size, h, gemm_quant_mode),
                                ops.ElementWise(f'context_shared_act_gate', self._num_layers, 2*self._moe_inter_size//tp_size, self._moe_inter_size//tp_size, 0.8),
                                ops.GEMM(f'context_shared_ffn2_gemm', self._num_layers, h, self._moe_inter_size//tp_size, gemm_quant_mode)
                                ])
        
        # router gemm, num_experts is large enough, cannot be ignored anymore.
        self.context_ops.extend([
                                ops.GEMM(f'context_router_gemm', self._num_layers, self._num_experts, h, GEMMQuantMode.float16)
                                ])

        # dispatch tokens to experts, pre-dispatch
        self.context_ops.extend([
                                ops.MoEDispatch(f'context_moe_pre_dispatch', self._num_layers, h, self._topk, self._num_experts, moe_tp_size, moe_ep_size, attention_dp_size, True)
                                ])
        
        # moe part
        self.context_ops.extend([ops.MoE(f'context_moe', self._num_layers, h, self._moe_inter_size, self._topk, self._num_experts, moe_tp_size, moe_ep_size, moe_quant_mode, workload_distribution, attention_dp_size)
                                ])

        # dispatch tokens to experts, post-dispatch
        self.context_ops.extend([
                                ops.MoEDispatch(f'context_moe_post_dispatch', self._num_layers, h, self._topk, self._num_experts, moe_tp_size, moe_ep_size, attention_dp_size, False)
                                ])

        self.context_ops.extend([ops.GEMM(f'context_logits_gemm', 1, self._vocab_size//tp_size, h, GEMMQuantMode.float16)])
        #####generation part, only generation part is scaled by mtp_scale_factor
        self.generation_ops.extend([ops.Embedding(f'generation_embedding', 1*self._mtp_scale_factor, self._vocab_size, h, 0.3),
                                ops.ElementWise(f'generation_add_norm_1', self._num_layers*self._mtp_scale_factor, 2*h, 2*h, 0.8),
                                ops.GEMM(f'generation_downscale_gemm', self._num_layers*self._mtp_scale_factor, 2112, h, gemm_quant_mode), # on every gpu
                                ops.GEMM(f'generation_q_b_proj_gemm', self._num_layers*self._mtp_scale_factor, 24576//tp_size, 1536, gemm_quant_mode),
                                ops.MLABmm(f'generation_bmm_pre', self._num_layers*self._mtp_scale_factor, self._num_heads//tp_size, mla_bmm_quant_mode, if_pre=True), # ifb gen attn part
                                ops.GenerationMLA(f'generation_attention', self._num_layers*self._mtp_scale_factor, tp_size, kvcache_quant_mode), # ifb gen attn part
                                ops.MLABmm(f'generation_bmm_post', self._num_layers*self._mtp_scale_factor, self._num_heads//tp_size, mla_bmm_quant_mode, if_pre=False), # ifb gen attn part
                                ops.GEMM(f'generation_proj_gemm', self._num_layers*self._mtp_scale_factor, h, h//tp_size, gemm_quant_mode),
                                ops.ElementWise(f'generation_add_norm_2', self._num_layers*self._mtp_scale_factor, 2*h, 2*h, 0.8)])

        # shared moe
        self.generation_ops.extend([
                                ops.GEMM(f'generation_shared_gate_gemm', self._num_layers*self._mtp_scale_factor, self._moe_inter_size//tp_size, h, gemm_quant_mode),
                                ops.GEMM(f'generation_shared_ffn1_gemm', self._num_layers*self._mtp_scale_factor, self._moe_inter_size//tp_size, h, gemm_quant_mode),
                                ops.ElementWise(f'generation_shared_act_gate', self._num_layers*self._mtp_scale_factor, 2*self._moe_inter_size//tp_size, self._moe_inter_size//tp_size, 0.8),
                                ops.GEMM(f'generation_shared_ffn2_gemm', self._num_layers*self._mtp_scale_factor, h, self._moe_inter_size//tp_size, gemm_quant_mode)
                                ])     
        
        # router gemm, num_experts is large enough, cannot be ignored anymore.
        self.generation_ops.extend([
                                ops.GEMM(f'generation_router_gemm', self._num_layers*self._mtp_scale_factor, self._num_experts, h, GEMMQuantMode.float16)
                                ])

        # dispatch tokens to experts, pre-dispatch
        self.generation_ops.extend([
                                ops.MoEDispatch(f'generation_moe_pre_dispatch', self._num_layers*self._mtp_scale_factor, h, self._topk, self._num_experts, moe_tp_size, moe_ep_size, attention_dp_size, True)
                                ])
   
        # moe part
        self.generation_ops.extend([ops.MoE(f'generation_moe', self._num_layers*self._mtp_scale_factor, h, self._moe_inter_size, self._topk, self._num_experts, moe_tp_size, moe_ep_size, moe_quant_mode, workload_distribution, attention_dp_size),
                                ])

        # dispatch tokens to experts, post-dispatch
        self.generation_ops.extend([
                                ops.MoEDispatch(f'generation_moe_post_dispatch', self._num_layers*self._mtp_scale_factor, h, self._topk, self._num_experts, moe_tp_size, moe_ep_size, attention_dp_size, False)
                                ])

        self.generation_ops.extend([ops.GEMM(f'generation_logits_gemm', 1*self._mtp_scale_factor, self._vocab_size//tp_size, h, GEMMQuantMode.float16)])

        # when tp_size=0, the comm part will be 0
        # self.context_ops.append(ops.AllReduce('context_ar_1', self._num_layers, h, tp_size))
        # self.context_ops.append(ops.AllReduce('context_ar_2', self._num_layers, h, tp_size))
        # self.generation_ops.append(ops.AllReduce('generation_ar_1', self._num_layers*self._mtp_scale_factor, h, tp_size))
        # self.generation_ops.append(ops.AllReduce('generation_ar_2', self._num_layers*self._mtp_scale_factor, h, tp_size))

        # pp
        pp_scale_factor = pp_size-1
        self.context_ops.append(ops.P2P('context_p2p', pp_scale_factor*self._mtp_scale_factor, h, pp_size))
        self.generation_ops.append(ops.P2P('generation_p2p', pp_scale_factor*self._mtp_scale_factor, h, pp_size))

        # TODO
        # a lot of quantization ops

    def _get_activation(self, num_tokens: int) -> int:
        """
        Estimates the peak activation memory usage for the DeepSeek model.
        This model has a complex structure with shared MoE and expert MoE blocks.
        The peak activation is likely dominated by the largest GEMM operation,
        which can be in the attention block or one of the MoE FFNs.
        """
        bytes_per_element = 2
        tp_size = self.inference_config.tp_size
        
        # DeepSeek has multiple large GEMM ops. We need to find the max output dimension.
        # 1. Attention q_b_proj_gemm: output dim = 24576 / tp_size
        # 2. Shared MoE gate_gemm: output dim = self._moe_inter_size / tp_size
        # 3. Expert MoE up-projection (within ops.MoE): also uses self._moe_inter_size
        
        attn_up_projection_dim = 24576 // tp_size
        shared_moe_dim = self._moe_inter_size // tp_size
        
        # The peak is determined by the maximum of these intermediate sizes.
        max_inter_dim = max(attn_up_projection_dim, shared_moe_dim)

        # The logic for expert MoE activation is similar to the MOEModel class,
        # considering topk and distribution, but the shared MoE and attention
        # projections are dense and simpler to estimate. We use the max dense
        # activation as a strong baseline.
        peak_tensor_size = num_tokens * max_inter_dim * bytes_per_element
        
        heuristic_factor = 2.0
        
        return int(peak_tensor_size * heuristic_factor)