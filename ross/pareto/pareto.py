import copy
import pandas as pd
import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass
import plotext
import math
import os

from pareto.inference_session import InferenceSession, DisaggInferenceSession
from pareto.inference_summary import SummaryColumns, DisaggPrintedColumns, ColocatePrintedColumns
from common.config import InferenceConfig, RuntimeConfig
from common.models import get_model

logger = logging.getLogger(__name__)

def enumerate_ttft_tpot_constraints(
    osl: int,
    request_latency: float,
    ttft: float | None = None,
) -> list[tuple[float, float]]:
    """
    Enumerate ttft and tpot constraints if given request latency.
    """
    assert osl > 1
    if ttft is None:
        ttft = request_latency * 0.95

    # typical values for ttft
    base_values = [300, 400, 500, 600, 800, 1000, 1200, 1400, 1600, 2000, 3000, 5000, 8000]
    base_min, base_max = base_values[0], base_values[-1]

    # values based on request_latency, only supplement values outside the base range
    interval_values = [request_latency * p for p in [0.1, 0.2, 0.3, 0.5, 0.7]]
    extra_values = [v for v in interval_values if v < base_min or v > base_max]

    ttft_set = set(base_values + extra_values)
    ttft_set.add(ttft)
    ttft_list = sorted([t for t in ttft_set if t < request_latency])
    return [(t, (request_latency - t) / (osl - 1)) for t in ttft_list]

def coloation_pareto(
    model_uri: str,
    runtime_config: RuntimeConfig,
    backend: str,
    parallel_config: List[int],
    gpu: str,
) -> pd.DataFrame:
    """
    Find Pareto front for agg.
    We will first enumerate all the parallel configurations and then find the Pareto front for
    each parallel configuration.

    Args:
        model_name: name of the model
        runtime_config: runtime config. tpot is a list of tpot values to search over or a single
            tpot value
        backend: name of the backend
        parallel_config_list: list of parallel configurations

    Returns:
        results_df: dataframe of the results
    """
    results_df = pd.DataFrame(columns=SummaryColumns)
    if parallel_config.find("@") == -1: # pd correlation
        dp_size, pp_size, tp_size = parallel_config.split(":")
        
        inference_config = InferenceConfig( dp_size=int(dp_size), pp_size=int(pp_size), tp_size=int(tp_size) )
        sess = InferenceSession(model_uri=model_uri, backend=backend)
    else:
        p_config, d_config = parallel_config.split("@")
        prefill_dp_size, prefill_pp_size, prefill_tp_size = p_config.split(":")
        decode_dp_size,  decode_pp_size,  decode_tp_size = d_config.split(":")

        prefill_inference_config = InferenceConfig( dp_size=int(prefill_dp_size), pp_size=int(prefill_pp_size), tp_size=int(prefill_tp_size) )
        decode_inference_config = InferenceConfig( dp_size=int(decode_dp_size), pp_size=int(decode_pp_size), tp_size=int(decode_tp_size) )
        sess = DisaggInferenceSession(model_uri=model_uri, backend=backend)
    try:
        if parallel_config.find("@") == -1: # pd correlation
            summary = sess.find_best_result_under_constraints(
                inference_config=inference_config,
                runtime_config=runtime_config,
                top_k=10,
                gpu=gpu,
            )
        else:
            summary = sess.find_best_result_under_constraints(
                prefill_inference_config=prefill_inference_config,
                decode_inference_config=decode_inference_config,
                runtime_config=runtime_config,
                top_k=10,
                gpu=gpu,
            )
        result_df = summary.get_summary_df()
        print(result_df)
        if len(result_df) == 0:
            logger.debug(
                "No result found for constraints ttft=%s, tpot=%s, request_latency=%s in agg pareto.",
                runtime_config.ttft,
                runtime_config.tpot,
                runtime_config.request_latency,
            )
        else:
            if len(results_df) == 0:
                results_df = result_df
            else:
                results_df = pd.concat([results_df, result_df], axis=0, ignore_index=True)
    except Exception as e:
        raise RuntimeError(f"exception: {e}")

    dedup_cols = [c for c in results_df.columns if c not in ["timing_phases", "decode_phases"]]
    results_df = results_df.drop_duplicates(subset=dedup_cols, ignore_index=True)
    return results_df

def get_pareto_front(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    *,
    maximize_x: bool = True,
    maximize_y: bool = True,
) -> pd.DataFrame:
    """
    Get Pareto front from raw data points.

    Args:
        df: Source dataframe.
        x_col: Column name for x axis.
        y_col: Column name for y axis.
        maximize_x: Treat larger values on x axis as better if True, else minimize.
        maximize_y: Treat larger values on y axis as better if True, else minimize.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.sort_values(by=x_col)

    def is_pareto(costs: np.ndarray) -> np.ndarray:
        is_better = np.ones(costs.shape[0], dtype=bool)
        for i, c in enumerate(costs):
            if is_better[i]:
                # Keep any point with a lower cost
                is_better[is_better] = np.any(costs[is_better] > c, axis=1)  # Remove dominated points
                is_better[i] = True  # And keep self
        return is_better

    working = df[[x_col, y_col]].copy()
    if not maximize_x:
        working[x_col] = -working[x_col]
    if not maximize_y:
        working[y_col] = -working[y_col]

    # Convert DataFrame columns to numpy array
    costs = working[[x_col, y_col]].values
    is_pareto_front = is_pareto(costs)

    # Plot Pareto front
    pareto_front = df[is_pareto_front]

    # Select Printed Columns
    if pareto_front['decode_dp'].notna().any(): # disaggregation
        pareto_front = pareto_front[DisaggPrintedColumns]
    else:
        pareto_front = pareto_front[ColocatePrintedColumns]        
    return pareto_front.sort_values(by=x_col).reset_index(drop=True)


def draw_pareto_to_string(
    ttpote: str,
    series: list[dict],
    *,
    highlight: dict | None = None,
    x_label: str = "tokens/s/user",
    y_label: str = "tokens/s/gpu_cluster",
) -> str:
    """Render one or more Pareto series as ASCII plot text.

    Args:
        ttpote: Plot ttpote prefix.
        series: List of dictionaries describing the series to plot. Expected keys:
            - "df": pandas DataFrame containing the Pareto frontier.
            - "label": Series label (default: "series-{index}").
            - "color": plotext color (RGB tuple or name).
            - "marker": plotext marker (default: "dot").
        highlight: Optional dictionary describing a highlighted point set. Accepts
            keys "df", "label", "color", "marker" similar to ``series``.
    """

    plotext.plot_size(80, 30)
    plotext.theme("clear")

    palette = [
        (144, 238, 144),  # light green
        (200, 200, 200),  # gray
        (135, 206, 235),  # sky blue
        (255, 182, 193),  # light pink
        (255, 160, 122),  # light salmon
        (221, 160, 221),  # plum
    ]
    markers = ["dot", "fdot", "hdot", "ldot", "sdot", "xdot"]

    y_max = 0.0
    x_max = 0.0
    x_min = math.inf

    for idx, entry in enumerate(series):
        df = entry.get("df")
        if df is None or df.empty:
            continue
        color = entry.get("color") or palette[idx % len(palette)]
        marker = entry.get("marker") or markers[idx % len(markers)]
        label = entry.get("label") or f"series-{idx + 1}"
        plotext.plot(
            df[x_label],
            df[y_label],
            label=label,
            color=color,
            marker=marker,
        )
        y_max = max(df[y_label].max(), y_max)
        x_max = max(df[x_label].max(), x_max)
        x_min = min(df[x_label].min(), x_min)

    if highlight is not None:
        highlight_df = highlight.get("df")
        if highlight_df is not None and not highlight_df.empty:
            color = highlight.get("color") or (255, 215, 0)  # gold
            marker = highlight.get("marker") or "xdot"
            label = highlight.get("label") or "Best"
            plotext.plot(
                highlight_df[x_label],
                highlight_df[y_label],
                label=label,
                color=color,
                marker=marker,
            )
            y_max = max(highlight_df[y_label].max(), y_max)
            x_max = max(highlight_df[x_label].max(), x_max)
            x_min = min(highlight_df[x_label].min(), x_min)

    plotext.title(f"{ttpote}: {y_label} vs {x_label}")
    plotext.xlabel(x_label)
    plotext.ylabel(y_label)
    plotext.grid(False)

    if y_max > 0.0 and x_max > 0.0:
        y_max = ((y_max * 1.2) + 49) // 50 * 50
        x_limit = ((x_max * 1.1) + 19) // 20 * 20
        cap = 300.0
        has_points_within_cap = x_min <= cap
        effective_x_max = min(x_limit, cap) if has_points_within_cap else x_limit
        plotext.ylim(0.0, y_max)
        plotext.xlim(0.0, effective_x_max)

    try:
        buf = plotext.build()
    except Exception:
        logger.exception("failed to build plotext")
        buf = ""
    plotext.clear_data()
    return buf
