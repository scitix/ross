# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from common.config import CommQuantMode, GEMMQuantMode, KVCacheQuantMode, MoEQuantMode, FMHAQuantMode

class Operation(object):
    """
    Base operation class.
    """
    def __init__(self, name: str, scale_factor: float) -> None:
        self._name = name
        self._scale_factor = scale_factor

    def get_weights(self, **kwargs):
        raise NotImplementedError

class AllReduce(Operation):
    """
    AllReduce operation. Now it's mapped to only trtllm custom allreduce.
    """
    def __init__(self, name: str, scale_factor: float, h: int, tp_size: int) -> None:
        super().__init__(name, scale_factor)
        self._h = h
        self._tp_size = tp_size
        self._weights = 0.0

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor


class P2P(Operation):
    """
    P2P operation.
    """
    def __init__(self, name: str, scale_factor: float, h: int, pp_size: int) -> None:
        super().__init__(name, scale_factor)
        self._h = h
        self._pp_size = pp_size
        self._bytes_per_element = 2
        #self._empirical_scaling_factor = 1.1
        self._weights = 0.0

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor

class NCCL(Operation):
    """
    NCCL operation.
    """
    def __init__(self, name: str, scale_factor: float, nccl_op: str, num_elements_per_token: int, num_gpus: int, comm_quant_mode: CommQuantMode) -> None:
        super().__init__(name, scale_factor)
        self._nccl_op = nccl_op
        self._num_elements_per_token = num_elements_per_token
        self._num_gpus = num_gpus
        self._comm_quant_mode = comm_quant_mode
        self._weights = 0.0

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor

class GEMM(Operation):
    """
    GEMM operation.
    """
    def __init__(self, name: str, scale_factor: float, n: int, k: int, quant_mode: GEMMQuantMode) -> None:
        super().__init__(name, scale_factor)
        self._n = n
        self._k = k
        self._quant_mode = quant_mode
        self._weights = self._n*self._k*quant_mode.value.memory 

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor     

class MoE(Operation):
    """
    MoE operation.
    """
    def __init__(self, 
                 name: str, 
                 scale_factor: float, 
                 hidden_size: int, 
                 inter_size: int, 
                 topk: int, 
                 num_experts: int, 
                 moe_tp_size: int, 
                 moe_ep_size: int, 
                 quant_mode: MoEQuantMode, 
                 workload_distribution: str, 
                 attention_dp_size: int) -> None:
        super().__init__(name, scale_factor)
        self._hidden_size = hidden_size
        self._inter_size = inter_size
        self._quant_mode = quant_mode
        self._topk = topk
        self._num_experts = num_experts
        self._moe_tp_size = moe_tp_size
        self._moe_ep_size = moe_ep_size
        self._attention_dp_size = attention_dp_size
        self._workload_distribution = workload_distribution
        self._weights = self._hidden_size*self._inter_size*self._num_experts*quant_mode.value.memory*3 // self._moe_ep_size // self._moe_tp_size # 3 for ffn1,gate,ffn2; 2 for float16

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor

# a comm op to deduce the communication cost of MoE
class MoEDispatch(Operation):
    """
    MoE dispatch operation. For fine grained moe dispatch
    """
    def __init__(self, 
                 name: str, 
                 scale_factor: float, 
                 hidden_size: int, 
                 topk: int, 
                 num_experts: int, 
                 moe_tp_size: int, 
                 moe_ep_size: int, 
                 attention_dp_size: int,
                 pre_dispatch: bool,
                 enable_fp4_all2all: bool = True) -> None:
        super().__init__(name, scale_factor)
        self._hidden_size = hidden_size
        self._topk = topk
        self._num_experts = num_experts
        self._moe_tp_size = moe_tp_size
        self._moe_ep_size = moe_ep_size
        self._attention_dp_size = attention_dp_size
        self._weights = 0.0
        self._enable_fp4_all2all = enable_fp4_all2all
        self._pre_dispatch = pre_dispatch
        self.num_gpus = self._moe_ep_size*self._moe_tp_size
        self._attention_tp_size = moe_tp_size*moe_ep_size // self._attention_dp_size

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor

class ContextAttention(Operation):
    """
    Context attention operation.
    """
    def __init__(self, 
                 name: str, 
                 scale_factor: float, 
                 n: int, 
                 n_kv: int, 
                 kvcache_quant_mode: KVCacheQuantMode, 
                 fmha_quant_mode: FMHAQuantMode) -> None:
        super().__init__(name, scale_factor)
        self._n = n
        self._weights = 0.0
        self._n_kv = n_kv
        self._kvcache_quant_mode = kvcache_quant_mode
        self._fmha_quant_mode = fmha_quant_mode

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor
class GenerationAttention(Operation):
    """
    Generation attention operation.
    """
    def __init__(self, 
                 name: str, 
                 scale_factor: float, 
                 n: int, 
                 n_kv: int, 
                 kv_cache_dtype: KVCacheQuantMode) -> None:
        super().__init__(name, scale_factor)
        self._n = n
        self._weights = 0.0
        self._n_kv = n_kv
        self._kv_cache_dtype = kv_cache_dtype

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor

class ContextMLA(Operation):
    """
    Context MLA operation. now only contains MHA part.
    """
    def __init__(self, 
                 name: str, 
                 scale_factor: float, 
                 tp_size: int, 
                 kvcache_quant_mode: KVCacheQuantMode, fmha_quant_mode: FMHAQuantMode) -> None:
        super().__init__(name, scale_factor)
        self._tp_size = tp_size
        self._weights = 0. #2*(1536*24576/tp_size + 128/tp_size*512*128+128/tp_size*512*128) # up q, up k, up v  float16 # 104MB / tpsize per layer
        self._kvcache_quant_mode = kvcache_quant_mode
        self._fmha_quant_mode = fmha_quant_mode

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor

class GenerationMLA(Operation):
    """
    Generation MLA operation. now only contains MQA part.
    """
    def __init__(self, 
                 name: str, 
                 scale_factor: float, 
                 tp_size: int, 
                 kv_cache_dtype: KVCacheQuantMode) -> None:
        super().__init__(name, scale_factor)
        self._tp_size = tp_size
        self._weights = 0. # 2*(1536*24576/tp_size + 128/tp_size*512*128+128/tp_size*512*128) # up q, up k, v up  float16
        self._kv_cache_dtype = kv_cache_dtype

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor

class MLABmm(Operation):
    """
    MLABmm operation. consider to be contained by mla op. for now, keep it as a separate op to show the cost of bmm
    """
    def __init__(self, 
                 name: str, 
                 scale_factor: float, 
                 num_heads: int, 
                 quant_mode: GEMMQuantMode, 
                 if_pre: bool=True) -> None:
        super().__init__(name, scale_factor)
        self._num_heads = num_heads
        self._weights = 0. 
        self._quant_mode = quant_mode
        self._if_pre = if_pre

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor
    
class Embedding(Operation):
    """
    Embedding operation.
    """
    def __init__(self, 
                 name: str, 
                 scale_factor: float, 
                 row_size: int, 
                 column_size: int, 
                 empirical_bw_scaling_factor: float=0.3) -> None:
        super().__init__(name, scale_factor)
        self._row_size = row_size
        self._column_size = column_size
        self._weights = row_size * column_size * 2
        self._empirical_bw_scaling_factor = empirical_bw_scaling_factor
        self._constant_latency = 5e-6 # 5us

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor
    
class ElementWise(Operation):
    """
    Element-wise operation.
    """
    def __init__(self, 
                 name: str, 
                 scale_factor: float, 
                 dim_in: int, 
                 dim_out: int, 
                 empirical_bw_scaling_factor: float=0.8) -> None:
        super().__init__(name, scale_factor)
        self._weights = 0.
        self._empirical_bw_scaling_factor = empirical_bw_scaling_factor
        self._constant_latency = 5e-6 # 5us
        self._dim_in = dim_in
        self._dim_out = dim_out

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor