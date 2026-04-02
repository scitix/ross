import argparse
import pandas as pd
import json
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

def bench_online(
    model_uri: str,
    runtime_config: RuntimeConfig,
    config_dict: Dict[str, Any],
    trace_configs: Dict[str, Any] | None = None,
):
    SIM_ROOT = str(Path(__file__).resolve().parent)
    for path in (SIM_ROOT,):
        if path not in sys.path:
            sys.path.insert(0, path)

    if config_dict.get("disaggregation"):
        from simulator_main_aligned import run_sim
    else:
        from simulator_main import run_sim
    
    args = argparse.Namespace(**config_dict)
    result_dict = run_sim(args, trace_configs)

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


def find_best_colocate_result_under_constraints(
        model_uri: str,
        inference_config: InferenceConfig,
        runtime_config: RuntimeConfig,
        gpu: str,
        platform_perf_yaml: str | None = None,
        top_k: int = 1,
        trace_configs: Dict[str, Any] | None = None,
) -> InferenceSummary:
    results_df = pd.DataFrame(columns=SummaryColumns)
    results_dict_list = []

    with open("./ross_config.json", "r", encoding="utf-8") as f:
        ross_config = json.load(f)
        if is_moe(model_uri):
            config_dict = ross_config["vllm"]["moe"]
        else:
            config_dict = ross_config["vllm"]["dense"]

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
        "mem_profiling_path": f"./log/mem_profiling_{gpu}.csv",
        "gpu": gpu,

        "max_num_batched_tokens": runtime_config.scheduler_config["max_num_batched_tokens"],
        "gpu_memory_utilization": runtime_config.scheduler_config["gpu_memory_utilization"],

        "disaggregation": False,
        "debug": False,
    })

    try:
        summary = bench_online(model_uri, runtime_config, config_dict, trace_configs)
    except MemoryError as e:
        summary = InferenceSummary(runtime_config=runtime_config)
        summary.set_oom(True)
        summary.set_summary_df(pd.DataFrame())
        return summary
    result_dict = summary.get_result_dict()
    results_df = pd.DataFrame([result_dict], columns=SummaryColumns).round(3)

    summary = InferenceSummary(runtime_config)
    summary.set_summary_df(results_df)
    return summary


def find_best_disagg_result_under_constraints(
        model_uri: str,
        prefill_inference_config: InferenceConfig,
        decode_inference_config: InferenceConfig,
        runtime_config: RuntimeConfig,
        gpu: str,
        platform_perf_yaml: str | None = None,
        top_k: int = 1,        
        trace_configs: Dict[str, Any] | None = None,
) -> InferenceSummary:
    results_df = pd.DataFrame(columns=SummaryColumns)
    results_dict_list = []

    with open("./ross_config.json", "r", encoding="utf-8") as f:
        ross_config = json.load(f)
        if is_moe(model_uri):
            config_dict = ross_config["vllm"]["moe"]
        else:
            config_dict = ross_config["vllm"]["dense"]

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
        "mem_profiling_path": f"./log/mem_profiling_{gpu}.csv",
        "gpu": gpu,

        "max_num_batched_tokens": runtime_config.scheduler_config["max_num_batched_tokens"],
        "gpu_memory_utilization": runtime_config.scheduler_config["gpu_memory_utilization"],

        "disaggregation": True,
        "debug": False,
    })
    
    try:
        summary = bench_online(model_uri, runtime_config, config_dict, trace_configs)
    except MemoryError as e:
        summary = InferenceSummary(runtime_config=runtime_config)
        summary.set_oom(True)
        summary.set_summary_df(pd.DataFrame())
        return summary

    result_dict = summary.get_result_dict()
    results_df = pd.DataFrame([result_dict], columns=SummaryColumns).round(3)

    summary = InferenceSummary(runtime_config)
    summary.set_summary_df(results_df)
    return summary
