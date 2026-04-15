# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import pandas as pd

from common.config import RuntimeConfig
from common.models import is_moe_model

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column groups (building blocks)
# ---------------------------------------------------------------------------

# Identity / workload
ID_COLUMNS = ["model", "isl", "osl", "batch_size", "request_rate"]

# Parallelism
PARALLEL_COLUMNS_COLOCATE = ["dp", "pp", "tp"]
PARALLEL_COLUMNS_DISAGG   = ["prefill_dp", "prefill_pp", "prefill_tp",
                              "decode_dp",  "decode_pp",  "decode_tp"]

# Scheduler knobs (per backend)
SCHEDULER_COLUMNS_SGL  = ["mem_fraction_static", "chunked_prefill_size"]
SCHEDULER_COLUMNS_VLLM = ["gpu_memory_utilization", "max_num_batched_tokens"]

# Performance metrics
PERF_COLUMNS = [
    "duration",
    "tokens/s", "tokens/s/gpu", "tokens/s/user", "throughput",
    "mean_ttft_ms",       "median_ttft_ms",       "std_ttft_ms",       "p99_ttft_ms",
    "mean_tpot_ms",       "median_tpot_ms",       "std_tpot_ms",       "p99_tpot_ms",
    "mean_itl_ms",        "median_itl_ms",         "std_itl_ms",        "p99_itl_ms",
    "mean_e2e_latency_ms","median_e2e_latency_ms", "std_e2e_latency_ms","p99_e2e_latency_ms",
]

# Internal phase breakdown (stored in DataFrame but not printed)
PHASE_COLUMNS = ["timing_phases", "prefill_phases", "decode_phases"]

# ---------------------------------------------------------------------------
# Composed column lists
# ---------------------------------------------------------------------------

# Full simulator output (used to construct result DataFrames)
SummaryColumns = (
    ID_COLUMNS
    + PARALLEL_COLUMNS_COLOCATE
    + PARALLEL_COLUMNS_DISAGG
    + SCHEDULER_COLUMNS_SGL
    + PHASE_COLUMNS
    + PERF_COLUMNS
)

# Human-readable print views
ColocatePrintedColumns = ID_COLUMNS + PARALLEL_COLUMNS_COLOCATE + SCHEDULER_COLUMNS_SGL + PERF_COLUMNS
DisaggPrintedColumns   = ID_COLUMNS + PARALLEL_COLUMNS_DISAGG   + SCHEDULER_COLUMNS_SGL + PERF_COLUMNS

# ---------------------------------------------------------------------------
# ross_predict.py record columns
# ---------------------------------------------------------------------------

RECORD_ID_COLS_COMMON = ["Platform", "model", "parallel", "iosl", "rate"]

RECORD_COMPARISON_COLS = [
    "pe", "pe_mean_ttft", "pe_mean_tpot", "pe_mean_itl", "pe_mean_throughput",
    "ross_duration",      "framework_duration",
    "ross_mean_ttft",     "framework_mean_ttft",
    "ross_mean_tpot",     "framework_mean_tpot",
    "ross_mean_itl",      "framework_mean_itl",
    "ross_mean_throughput","framework_mean_throughput",
]

RECORD_NO_COMPARISON_COLS = ["duration", "mean_ttft_ms", "mean_tpot_ms", "mean_itl_ms", "throughput"]

def is_moe(model_uri: str):
    return is_moe_model(model_uri)
    
class InferenceSummary:
    """
    InferecneSummary to hold results of inference

    Attributes:
        runtime_config: runtime config
        memory: memory breakdown
        context_latency_dict: latency breakdown for context
        generation_latency_dict: latency breakdown for generation
        summary_df: summary dataframe

    Methods:
        set_memory_and_check_oom: set memory and check oom
        set_oom: set oom
        set_context_latency_dict: set context latency dict
        set_generation_latency_dict: set generation latency dict
        get_context_latency_dict: get context latency dict
        get_generation_latency_dict: get generation latency dict
        check_oom: check oom
        get_static_info: get static info for static mode print
        set_summary_df: set summary dataframe
        get_summary_df: get summary dataframe
    """

    def __init__(self, runtime_config: RuntimeConfig) -> None:
        """
        Initialize inference summary.
        """
        self._runtime_config = runtime_config

        # raw data dict
        self._memory = {}
        self._context_latency_dict = {}
        self._generation_latency_dict = {}
        self._is_oom = False

        # summary dataframe
        self._summary_df = None

        # cached result dict for efficient batch operations
        self._result_dict = None

    def set_memory_and_check_oom(self, memory_dict: dict, mem_capacity: int) -> None:
        """
        Set memory and check oom.
        """
        self._memory = memory_dict
        self._is_oom = self._memory["total"] >= (mem_capacity / (1 << 30))

    def set_oom(self, is_oom: bool) -> None:
        """
        Set oom.
        """
        self._is_oom = is_oom

    def set_context_latency_dict(self, context_latency_dict: dict) -> None:
        """
        Set context latency dict.
        """
        self._context_latency_dict = context_latency_dict

    def set_generation_latency_dict(self, generation_latency_dict: dict) -> None:
        """
        Set generation latency dict.
        """
        self._generation_latency_dict = generation_latency_dict

    def get_context_latency_dict(self) -> dict:
        """
        Get context latency dict.
        """
        return self._context_latency_dict

    def get_generation_latency_dict(self) -> dict:
        """
        Get generation latency dict.
        """
        return self._generation_latency_dict

    def check_oom(self) -> bool:
        """
        Check oom.
        """
        # if self._is_oom is None:
        #     logger.warning("WARNING: memory status is not set")
        return self._is_oom

    def get_static_info(self) -> tuple[str, str, str, str]:
        """
        Get static info.
        """

        def get_latency_and_breakdown_percentage_string_helper(metrics: dict) -> tuple[float, str]:
            breakdown_string = ""
            latency = 0
            for op, op_latency in metrics.items():
                latency += op_latency

            breakdown_string += f"total                      ({latency:>10.5f} ms)\n"
            for op, op_latency in metrics.items():
                breakdown_string += f"{op:<25}   {op_latency:>10.3f} ms {int(op_latency / latency * 100):>5}%\n"
            return latency, breakdown_string

        context_latency, context_latency_string = get_latency_and_breakdown_percentage_string_helper(
            self._context_latency_dict
        )
        generation_latency, generation_latency_string = get_latency_and_breakdown_percentage_string_helper(
            self._generation_latency_dict
        )

        assert self._summary_df is not None, "summary df is not set"

        # summary string for display
        perf_info = "Performance Summary:\n"
        perf_info += f"total latency        {(context_latency + generation_latency):>17.5f} ms\n"
        perf_info += f"context latency (ttft):{context_latency:>16.5f} ms\n"
        if generation_latency != 0:
            perf_info += f"generation latency:{generation_latency:>19.5f} ms\n"
            perf_info += (
                f"throughput {self._summary_df.loc[0, 'tokens/s']:.2f} tokens/s, tpot "
                f"{self._summary_df.loc[0, 'tpot']:.3f} ms\n"
            )
        context_info = "Context breakdown:\n" + context_latency_string
        generation_info = "Generation breakdown:\n" + generation_latency_string

        mem_info = "\nMemory Usage: \n"
        for item, memory_usage in self._memory.items():
            mem_info += f"{item:29} {memory_usage:>8.3f} GiB\n"

        return perf_info, mem_info, context_info, generation_info

    def set_summary_df(self, summary_df: pd.DataFrame) -> None:
        """
        Set summary dataframe.
        """
        self._summary_df = summary_df

    def get_summary_df(self) -> pd.DataFrame:
        """
        Get summary dataframe.
        """
        if self._summary_df is None:
            logger.warning("WARNING: summary df is not set")
        return self._summary_df

    def set_result_dict(self, result_dict: dict) -> None:
        """
        Set the cached result dict for efficient batch operations.
        """
        self._result_dict = result_dict

    def get_result_dict(self) -> dict | None:
        """
        Get the result as a dict. Returns cached dict if available,
        otherwise extracts from the first row of the summary DataFrame.
        """
        if self._result_dict is not None:
            return self._result_dict

        # Fallback: create from DataFrame if not cached
        if self._summary_df is not None and len(self._summary_df) > 0:
            return self._summary_df.iloc[0].to_dict()
        return None
