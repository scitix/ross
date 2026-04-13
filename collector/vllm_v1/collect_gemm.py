# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from cuda import cuda
import torch
import vllm
from vllm.model_executor.layers.linear import (LinearBase,
                                               ReplicatedLinear,
                                               MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.awq import AWQConfig
from vllm.model_executor.layers.quantization.gptq import GPTQConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size, init_distributed_environment, initialize_model_parallel
from vllm.version import __version__ as vllm_version
import torch.distributed as dist
from helper import getSMVersion, log_perf

# If we want to use advanced linear implementations like MergedColumnParallelLinear and RowParallelLinear
# we need to unit and destroy TP and rank group before and after each test case
def get_gemm_test_cases(is_unit_test=False):
    x_list = [1, 16, 128, 1024, 4096, 8192]
    nk_list = [256, 1024, 3584, 8192, 16384]
    nk_list_ext = [16384, 65536]
    #gemm_list = ['float16']
    gemm_list = ['awq','gptq']
    if getSMVersion() >= 89:
        gemm_list += ['fp8','fp8_block']
    
    if is_unit_test:
        x_list = [1,2,4,8]
        nk_list = [128]
        nk_list_ext = []
        gemm_list = ['float16']

    test_cases=[]

    for gemm_type in gemm_list:
        # x_list_orig+add+ext  <==> nk_list+ext
        for x in sorted(x_list, reverse=True):
            for n in sorted(nk_list + nk_list_ext, reverse=True):
                for k in sorted(nk_list + nk_list_ext, reverse=True):
                    if n*k == 65536*65536:
                        continue
                    test_cases.append([gemm_type,x,n,k,'gemm_perf_vllm.txt'])
    return test_cases

def _ensure_distributed(device):
    """Initialize vLLM distributed env and TP group once per worker process."""
    if not torch.distributed.is_initialized():
        device_id = device.index if isinstance(device, torch.device) else int(str(device).split(':')[-1])
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(29500 + device_id)
        init_distributed_environment()
        initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)

def run_gemm(gemm_type, m, n, k, perf_filename, device='cuda:0'):
    torch.set_default_dtype(torch.float16)
    torch.cuda.set_device(device)
    _ensure_distributed(device)
    dtype = torch.float16
    x = torch.randn((m, k), dtype=dtype).to(torch.device(device))
    w = torch.randn((k, n), dtype=dtype).to(torch.device(device))

    if gemm_type == 'fp8':
        qc = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="static",
            ignored_layers=None,
            weight_block_size=None
        )
    elif gemm_type == 'fp8_block':
        qc = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic", 
            weight_block_size=[128, 128]
        )
    elif gemm_type == 'awq':
        qc = AWQConfig(
            weight_bits=4,
            group_size=128,
            zero_point=True,
            modules_to_not_convert=None
        )
    elif gemm_type == 'gptq':
        qc = GPTQConfig(
            weight_bits=8,
            group_size=128,
            desc_act=False,
            lm_head_quantized=False,
            dynamic={}
        )
    else:
        qc = None
    #print(f"dtype: {dtype}, type: {type(dtype)}")
    #print(f"qc: {qc}, type: {type(qc)}")
    gemm = ReplicatedLinear(
            input_size=k,
            output_size=n,
            bias=False,
            skip_bias_add=True,
            params_dtype=dtype,
            quant_config=qc,
            prefix="",
            return_bias=True
        )
    # TODO, to evaluate random weights impact
    gemm.to(torch.device(device))
    #print(dir(gemm)) # print all attributes of gemm
    #print(gemm.weight.data.stride())

    if gemm_type == 'fp8' and hasattr(gemm, 'weight'):
        new_weight = gemm.weight.data.t()
        #print("new_weight stride:", new_weight.stride())  mnk = 1,128,128   weight stride = (128,1)  - transpose to (1,128) for fp8 cutlass limit
        gemm.weight = torch.nn.Parameter(new_weight)
        #print("after fix, weight stride:", gemm.weight.data.stride())
    
    gemm.forward(x) # dry run to init

    num_warmups = 3
    num_runs = 6

    # capture
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(num_runs):
            gemm.forward(x)
    # warmup
    for i in range(num_warmups):
        g.replay()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for i in range(num_runs):
        g.replay()
    end_event.record()
    torch.cuda.synchronize()
    latency = start_event.elapsed_time(end_event)/(num_runs*num_runs)

    log_perf(item_list=[{ 
                'gemm_dtype': gemm_type,
                'm': m,
                'n': n,
                'k': k,
                'latency': latency
                }], 
    framework='VLLM', 
    version=vllm_version, 
    device_name=torch.cuda.get_device_name(device), 
    op_name='gemm', 
    kernel_source='vllm_default', 
    perf_filename=perf_filename)


