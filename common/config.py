import yaml
from enum import Enum
from collections import namedtuple
from dataclasses import dataclass
from typing import Union, Dict, Any

def get_yaml_config(yaml_file):
    # print(f"   Loading YAML Configuration from: {yaml_file}")
    try:
        with open(yaml_file, 'r') as f:
            config_data = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at '{yaml_file}'")
        exit(1)
    return config_data

QuantMapping = namedtuple('QuantMapping', ['memory', 'compute', 'name'])

class GEMMQuantMode(Enum):
    """
    GEMM quant mode.
    """
    float16 = QuantMapping(2, 1, 'float16') # w16a16
    int8_wo = QuantMapping(1, 1, 'int8_wo') # w8a16
    int4_wo = QuantMapping(0.5, 1, 'int4_wo') # w4a16
    fp8 = QuantMapping(1, 2, 'fp8') #w8fp8
    sq = QuantMapping(1, 2, 'sq') # w8int8
    fp8_block = QuantMapping(1, 2, 'fp8_block') # specific for trtllm torch ds fp8
    fp8_ootb = QuantMapping(1, 2, 'fp8_ootb') # in future, should deprecate this mode as it's specific for trtllm trt backend
    nvfp4 = QuantMapping(0.5, 4, 'nvfp4') # nvfp4 on blackwell

class MoEQuantMode(Enum):
    """
    MoE quant mode.
    """
    float16 = QuantMapping(2, 1, 'float16') # w16a16
    fp8 = QuantMapping(1, 2, 'fp8') # w8fp8
    int4_wo = QuantMapping(0.5, 1, 'int4_wo') # w4a16
    fp8_block = QuantMapping(1, 2, 'fp8_block') # specific for trtllm torch ds fp8
    w4afp8 = QuantMapping(0.5, 2, 'w4afp8') # specific for trtllm torch ds w4a8
    nvfp4 = QuantMapping(0.5, 4, 'nvfp4') # nvfp4 on blackwell

class FMHAQuantMode(Enum):
    """
    FMHA quant mode.
    """
    float16 = QuantMapping(0, 1, 'float16')
    fp8 = QuantMapping(0, 2, 'fp8')

class KVCacheQuantMode(Enum):
    """
    KVCache quant mode.
    """
    float16 = QuantMapping(2, 0, 'float16')
    int8 = QuantMapping(1, 0, 'int8')
    fp8 = QuantMapping(1, 0, 'fp8')

class CommQuantMode(Enum):
    """
    Comm quant mode.
    """
    half = QuantMapping(2, 0, 'half')
    int8 = QuantMapping(1, 0, 'int8')
    fp8 = QuantMapping(1, 0, 'fp8')


@dataclass
class InferenceConfig:
    """
    Inference configuration.
    """
    dp_size: int = 1
    tp_size: int = 1
    pp_size: int = 1
    gemm_quant_mode: GEMMQuantMode = GEMMQuantMode.float16
    moe_quant_mode: MoEQuantMode = MoEQuantMode.float16
    kvcache_quant_mode: KVCacheQuantMode = KVCacheQuantMode.float16
    fmha_quant_mode: FMHAQuantMode = FMHAQuantMode.float16
    comm_quant_mode: CommQuantMode = CommQuantMode.half
    moe_tp_size: int = None
    moe_ep_size: int = None
    attention_dp_size: int = 1
    workload_distribution: str = "uniform"
    nextn: int = 0 # at most mtp5
    nextn_accept_rates: list = None
    overwrite_num_layers: int = 0


@dataclass
class RuntimeConfig:
    """
    Runtime configuration.
    """
    batch_size: int = None
    isl: int = None
    osl: int = None
    rate: str = "inf"
    scheduler_config: Dict[str, Any] = None
    ttft: float = None
    tpot:  float = None
    request_latency: float = None
    arrival_path: str = None
    
