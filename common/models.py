# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
import os
import json
import logging
from pathlib import Path

import yaml
from typing import Dict, Any

from common.config import InferenceConfig

logger = logging.getLogger(__name__)


def _first_defined(configs: Dict[str, Any], keys: list[str], default: int = 0) -> int:
    for key in keys:
        value = configs.get(key)
        if value is not None:
            return int(value)
    return default


def _load_model_weights() -> Dict[str, float]:
    """Load model weight table from model_weights.yaml next to this file."""
    yaml_path = Path(__file__).with_name("model_weights.yaml")
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return {k: float(v) for k, v in data.items()}


MODEL_WEIGHTS: Dict[str, float] = _load_model_weights()

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
        self._num_experts = _first_defined(
            model_configs,
            ["num_local_experts", "n_routed_experts", "num_experts"],
        )
        self._topk = _first_defined(
            model_configs,
            ["num_experts_per_tok", "num_experts_per_token", "moe_top_k", "top_k", "topk"],
        )
        self._moe_inter_size = _first_defined(
            model_configs,
            ["moe_intermediate_size", "expert_intermediate_size", "ffn_dim"],
        )
        self._shared_expert_inter_size = _first_defined(
            model_configs,
            [
                "shared_expert_intermediate_size",
                "shared_moe_intermediate_size",
                "shared_ffn_intermediate_size",
            ],
        )
        self.model_uri = model_configs["model_uri"]
        # print("new features: ", self.model_uri, self._num_experts, self._topk, self._moe_inter_size, self._shared_expert_inter_size)
        
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

    def get_num_experts(self):
        return self._num_experts

    def get_topk(self):
        return self._topk

    def get_moe_intermediate_size(self):
        return self._moe_inter_size

    def get_shared_expert_intermediate_size(self):
        return self._shared_expert_inter_size
    
    def _get_activation(self, num_tokens: int) -> int:
        ...

    def get_total_params(self) -> float:
        """
        Fix MOE params
        """
        h = self._hidden_size
        L = self._num_layers
        d_ff = self._inter_size
        d_moe = self._moe_inter_size
        num_experts = self._num_experts
        
        # 1. Attention
        attn_params = 4 * (h ** 2)
        
        # 2. FFN / MoE
        if num_experts > 1:
            routed_params = num_experts * (2 * h * d_moe)
            shared_params = 2 * h * self._shared_expert_inter_size
            ffn_params = routed_params + shared_params
        else:
            ffn_params = 2 * h * d_ff
            
        return L * (attn_params + ffn_params)



class LLAMAModel(BaseModel):
    """
    LLAMA series uses this model impl. Other variants without large difference can use this as well,
    e.g., only positional embedding or activation is different.
    Other than DS V3, all other models don't support mtp.
    """
    def __init__(self, *args) -> None:
        super().__init__(*args)
        assert self._nextn == 0, 'Only DS V3 supports mtp'

    def _get_activation(self, num_tokens: int) -> int:
        """
        Estimates the peak activation memory usage for a LLAMA-style model.
        The peak is dominated by the output of the gated FFN's up-projection.
        """
        # Assuming bf16/fp16, where each element takes 2 bytes.
        bytes_per_element = 2

        # In LLAMA models with SwiGLU, the FFN up-projection ('context_gate_ffn1_gemm')
        # creates a tensor of size (num_tokens, 2 * self._inter_size / tp_size).
        # This is the largest intermediate tensor.
        peak_tensor_size = num_tokens * (2 * self._inter_size) * bytes_per_element

        # A heuristic factor to account for other concurrent activations.
        heuristic_factor = 2.0

        return int(peak_tensor_size * heuristic_factor)
