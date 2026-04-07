import json
import os
import numpy as np
import glob
from collections import defaultdict
import ast
import re
from transformers import PreTrainedTokenizerBase
from typing import Any, AsyncGenerator, Collection, Dict, List, Optional, Tuple

from common.models import get_model
from common.ross_model import ROSSModel

def load_ross_models(model_uri: str,
                    models_to_load: Dict[str, Any]):
    """
    models_to_load (Dict[str, Any]): {
            [phase]: {
                "platform_perf":,
                "inference_config":,
                "path":,
            }
        }
    """
    model_dict = dict()
    for phase, phase_config in models_to_load.items():
        model = get_model(model_uri, phase_config["inference_config"])
        model_dict[phase] = ROSSModel(
            saved_model_path=phase_config["path"],
            platform_perf=phase_config["platform_perf"],
            model=model,
            inference_config=phase_config["inference_config"],
            regressor="xgboost",
        )
    return model_dict

def load_arrive_time(log_file) -> List[float]:
    data = []
    with open(log_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            # Remove ANSI color codes (like ^[[36m and ^[[0m)
            clean_line = re.sub(r'\x1b\[[0-9;]*m', '', line)
            # Try new profiler format first
            profiler_match = re.search(r'\s*(\{.*\})', clean_line)
            if profiler_match:
                # Parse the dictionary string
                dict_str = profiler_match.group(1)
                arrive_time_json = ast.literal_eval(dict_str)
    
                f.close()
                return arrive_time_json
    return data


def to_paths(log_dir: str, candidates: List[str]) -> List[str]:
    return [os.path.join(log_dir, name) for name in candidates]

def pick(paths: List[str]) -> Optional[str]:
    for p in paths:
        if os.path.exists(p):
            return p
    return None

def resolve_vllm_log_paths(
    log_dir: str,
    gpu: str,
    isl: int,
    osl: int,
    num_prompt: int,
    dataset: str = "sharegpt",
    request_rate: str = "inf",
    disaggregation: bool = False
) -> Dict[str, Any]:
    gpu_upper = gpu.upper()
    gpu_lower = gpu.lower()

    client_candidates = [
        f"vllm_{gpu_upper}_main_{dataset}_isl_{isl}_osl_{osl}_prompt_{num_prompt}_rate_{request_rate}.log",
        f"vllm_{gpu_upper}_main_prompt_{num_prompt}_rate_{request_rate}.log"
    ]
    client_paths = to_paths(log_dir, client_candidates)
    client_log = pick(client_paths)

    if not disaggregation:
        server_candidates = [f"vllm_{gpu_upper}_main_server.log"]
        rank_patterns = [ f"vllm_{gpu.lower()}_rank_dp_*_tp_*.txt" ]
        rank_candidates = []
        for pat in rank_patterns:
            rank_candidates.extend(sorted(glob.glob(os.path.join(log_dir, pat))))
        
        server_paths = to_paths(log_dir, server_candidates)
        server_log = pick(server_paths)
        return {
            "log_dir": log_dir,
            "client_log": client_log,
            "server_log": server_log,
            "rank_logs": rank_candidates,
        }
    else:
        prefill_server_candidates = [f"vllm_{gpu_upper}_main_prefill_server.log"]
        decode_server_candidates  = [f"vllm_{gpu_upper}_main_decode_server.log"]
        prefill_rank_patterns = [ f"vllm_{gpu.lower()}_prefill_rank_dp_*_tp_*.txt" ]
        decode_rank_patterns  = [ f"vllm_{gpu.lower()}_decode_rank_dp_*_tp_*.txt" ]
        prefill_rank_candidates = []
        decode_rank_candidates = []
        for pat in prefill_rank_patterns:
            prefill_rank_candidates.extend(sorted(glob.glob(os.path.join(log_dir, pat))))
        for pat in decode_rank_patterns:
            decode_rank_candidates.extend(sorted(glob.glob(os.path.join(log_dir, pat))))
        
        prefill_server_paths = to_paths(log_dir, prefill_server_candidates)
        decode_server_paths  = to_paths(log_dir, decode_server_candidates)

        prefill_server_log = pick(prefill_server_paths)
        decode_server_log = pick(decode_server_paths)
        return {
            "log_dir": log_dir,
            "client_log": client_log,
            "prefill_server_log": prefill_server_log,
            "decode_server_log": decode_server_log,
            "prefill_rank_logs": prefill_rank_candidates,
            "decode_rank_logs": decode_rank_candidates,
        }


def resolve_sglang_log_paths(
    log_dir: str,
    gpu: str,
    isl: int,
    osl: int,
    num_prompt: int,
    dataset: str = "sharegpt",
    request_rate: str = "inf",
    disaggregation: bool = False,
) -> Dict[str, Any]:
    gpu_upper = gpu.upper()
    gpu_lower = gpu.lower()

    if not disaggregation:
        client_candidates = [ f"sglang_{gpu_upper}_main_{dataset}_isl_{isl}_osl_{osl}_prompt_{num_prompt}_rate_{request_rate}.log" ]
        server_candidates = [ f"sglang_{gpu_upper}_main_0.log"]
        rank_patterns = [ f"sgl_{gpu_lower}_rank_dp_*_tp_*.txt" ]

        client_paths = to_paths(log_dir, client_candidates)
        server_paths = to_paths(log_dir, server_candidates)
        rank_candidates = []
        for pat in rank_patterns:
            rank_candidates.extend(sorted(glob.glob(os.path.join(log_dir, pat))))

        client_log = pick(client_paths)
        server_log = pick(server_paths)
        return { "log_dir": log_dir, "client_log": client_log,  "server_log": server_log, "rank_logs": rank_candidates, }
    else:
        client_candidates = [ f"sglang_{gpu_upper}_main_{dataset}_isl_{isl}_osl_{osl}_prompt_{num_prompt}_rate_{request_rate}.log" ]
        prefill_server_candidates = [ f"sglang_{gpu_upper}_main_prefill_0.log" ]
        decode_server_candidates  = [ f"sglang_{gpu_upper}_main_decode_0.log" ]
        rank_patterns = [
            f"sgl_{gpu_lower}_prefill_rank_dp_*_tp_*.txt",
            f"sgl_{gpu_lower}_decode_rank_dp_*_tp_*.txt",
        ]

        client_paths = to_paths(log_dir, client_candidates)
        prefill_server_paths = to_paths(log_dir, prefill_server_candidates)
        decode_server_paths = to_paths(log_dir, decode_server_candidates)
        rank_candidates = []
        for pat in rank_patterns:
            rank_candidates.extend(sorted(glob.glob(os.path.join(log_dir, pat))))

        client_log = pick(client_paths)
        prefill_server_log = pick(prefill_server_paths)
        decode_server_log = pick(decode_server_paths)
        return { "log_dir": log_dir, "client_log": client_log, "rank_logs": rank_candidates, 
            "prefill_server_log": prefill_server_log, "decode_server_log": decode_server_log,
        }
