import argparse
import pandas as pd
import sys
import time
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass

from pareto.inference_summary import InferenceSummary, SummaryColumns, is_moe
from common.config import InferenceConfig, RuntimeConfig
from common.models import BaseModel

import logging
logger = logging.getLogger(__name__)


def _resolve_vllm_model_paths(modeling_dir: str, moe: bool) -> dict:
    """Build xgboost model path dict from a single modeling_dir root."""
    tag  = "moe" if moe else "dense"
    base = Path(modeling_dir) / "vllm"
    return {
        "pp_pre_forward_path": str(base / f"dense/pre_forward_trained_models/xgboost_model"),
        "pre_forward_path":    str(base / f"dense/pre_forward_trained_models/xgboost_model"),
        "forward_path":        str(base / f"{tag}/forward_trained_models/xgboost_model"),
        "post_forward_path":   str(base / f"dense/post_forward_trained_models/xgboost_model"),
    }

def bench_online(
    model_uri: str,
    runtime_config: RuntimeConfig,
    config_dict: Dict[str, Any],
):
    SIM_ROOT = str(Path(__file__).resolve().parent)
    for path in (SIM_ROOT,):
        if path not in sys.path:
            sys.path.insert(0, path)

    if config_dict.get("disaggregation"):
        from simulator_main_aligned import run_sim
    elif config_dict.get("fast"):
        from simulator_main_fast import run_sim
    else:
        from simulator_main import run_sim
    
    args = argparse.Namespace(**config_dict)
    result_dict = run_sim(args)

    result_dict.update({
        "model": model_uri,
        "batch_size": runtime_config.batch_size,
        "isl": runtime_config.isl,
        "osl": runtime_config.osl,
        "request_rate": runtime_config.rate,
        "oom": False
    })

    summary = InferenceSummary(runtime_config)
    summary.set_oom(result_dict["oom"])
    summary.set_result_dict(result_dict)
    return summary


_MEM_PROFILING_DIR = Path(__file__).resolve().parent / "mem_profiling"


def find_best_colocate_result_under_constraints(
        model_uri: str,
        inference_config: InferenceConfig,
        runtime_config: RuntimeConfig,
        modeling_dir: str,
        platform_perf_yaml: str | None = None,
        fast: bool = False,
        top_k: int = 1,
) -> InferenceSummary:
    results_df = pd.DataFrame(columns=SummaryColumns)
    results_dict_list = []

    gpu = runtime_config.scheduler_config.get("gpu", "")
    config_dict = _resolve_vllm_model_paths(modeling_dir, is_moe(model_uri))

    if platform_perf_yaml is not None:
        config_dict["platform_perf"] = platform_perf_yaml

    config_dict.update({
        "model_uri": model_uri,

        "dp_size": inference_config.dp_size,
        "pp_size": inference_config.pp_size,
        "tp_size": inference_config.tp_size,

        "batch_size": runtime_config.batch_size,
        "max_prompt_len": runtime_config.isl,
        "max_output_len": runtime_config.osl,
        "request_rate": runtime_config.rate,

        "frontend_path": runtime_config.arrival_path,
        "mem_profiling_path": str(_MEM_PROFILING_DIR / f"mem_profiling_{gpu}.csv"),
        "gpu": gpu,

        "max_num_batched_tokens": runtime_config.scheduler_config["max_num_batched_tokens"],
        "gpu_memory_utilization": runtime_config.scheduler_config["gpu_memory_utilization"],

        "disaggregation": False,
        "fast": fast,
        "debug": False,
    })

    try:
        summary = bench_online(model_uri, runtime_config, config_dict)
    except MemoryError as e:
        summary = InferenceSummary(runtime_config=runtime_config)
        summary.set_oom(True)
        summary.set_summary_df(pd.DataFrame())
        return summary
    result_dict = summary.get_result_dict()
    results_df = pd.DataFrame([result_dict], columns=SummaryColumns).round(3)

    summary = InferenceSummary(runtime_config)
    summary.set_summary_df(results_df)
    summary.set_result_dict(result_dict)
    return summary


def find_best_disagg_result_under_constraints(
        model_uri: str,
        prefill_inference_config: InferenceConfig,
        decode_inference_config: InferenceConfig,
        runtime_config: RuntimeConfig,
        modeling_dir: str,
        platform_perf_yaml: str | None = None,
        fast: bool = False,
        top_k: int = 1,
) -> InferenceSummary:
    results_df = pd.DataFrame(columns=SummaryColumns)
    results_dict_list = []

    gpu = runtime_config.scheduler_config.get("gpu", "")
    config_dict = _resolve_vllm_model_paths(modeling_dir, is_moe(model_uri))

    if platform_perf_yaml is not None:
        config_dict["platform_perf"] = platform_perf_yaml

    config_dict.update({
        "model_uri": model_uri,

        "dp_size": prefill_inference_config.dp_size,
        "prefill_tp_size": prefill_inference_config.tp_size,
        "decode_tp_size": decode_inference_config.tp_size,

        "batch_size": runtime_config.batch_size,
        "max_prompt_len": runtime_config.isl,
        "max_output_len": runtime_config.osl,
        "request_rate": runtime_config.rate,

        "frontend_path": runtime_config.arrival_path,
        "mem_profiling_path": str(_MEM_PROFILING_DIR / f"mem_profiling_{gpu}.csv"),
        "gpu": gpu,

        "max_num_batched_tokens": runtime_config.scheduler_config["max_num_batched_tokens"],
        "gpu_memory_utilization": runtime_config.scheduler_config["gpu_memory_utilization"],

        "disaggregation": True,
        "fast": fast,
        "debug": False,
    })
    
    try:
        summary = bench_online(model_uri, runtime_config, config_dict)
    except MemoryError as e:
        summary = InferenceSummary(runtime_config=runtime_config)
        summary.set_oom(True)
        summary.set_summary_df(pd.DataFrame())
        return summary

    result_dict = summary.get_result_dict()
    results_df = pd.DataFrame([result_dict], columns=SummaryColumns).round(3)

    summary = InferenceSummary(runtime_config)
    summary.set_summary_df(results_df)
    summary.set_result_dict(result_dict)
    return summary
