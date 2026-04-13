import argparse
import os
import subprocess
import sys
import time
import logging
import faulthandler
import traceback
import multiprocessing as mp
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import pandas as pd
from tqdm import tqdm
import tqdm as tqdm_module

REPO_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, ".")
sys.path.insert(0, str(Path(__file__).resolve().parent / "servers"))
import util
from bench_config import BenchmarkConfig, DEFAULT_RESERVE, parse_predict_args
from common.loader import setup_logging
from common.config import InferenceConfig, RuntimeConfig
from pareto.inference_summary import (
    RECORD_ID_COLS_COMMON, RECORD_NO_COMPARISON_COLS, RECORD_COMPARISON_COLS,
    SummaryColumns,
)
import pareto.pareto as pa
from pareto.draw_pareto import draw_pareto_fronts

# ---------------------------------------------------------------------------
# Backend lazy-load globals
# ---------------------------------------------------------------------------
_bench_mod = None   # bench_sglang or bench_vllm


def _call_bench_serving(*args, **kwargs):
    from api_server import call_bench_serving
    return call_bench_serving(*args, **kwargs)


def _load_backend_modules(backend: str):
    global _bench_mod
    if backend == "sglang":
        from sgl_sim import bench_sglang as bmod
    elif backend == "vllm":
        from vllm_sim import bench_vllm as bmod
    else:
        raise ValueError(f"Unsupported backend: {backend!r}. Choose 'sglang' or 'vllm'.")
    _bench_mod = bmod


# ---------------------------------------------------------------------------
# Thread / process setup
# ---------------------------------------------------------------------------

_debug_logging = False  # set to True in main() when args.debug is on

def _log(msg: str) -> None:
    if _debug_logging:
        print(msg, file=sys.stderr, flush=True)


def _task_brief(task: "SimulationTask") -> str:
    return (
        f"task_id={task.task_id} model={Path(task.model).name} parallel={task.parallel} "
        f"batch={task.runtime_config.batch_size} iosl={task.runtime_config.isl}@{task.runtime_config.osl} "
        f"rate={task.runtime_config.rate}"
    )


def _debug_phase_dump(task: "SimulationTask", phase_name: str, phase_data: Any) -> None:
    print(f"[eval-debug] {_task_brief(task)} {phase_name}={phase_data}", flush=True)


def _set_thread_env(threads_per_worker: int) -> None:
    value = str(max(1, threads_per_worker))
    os.environ["OMP_NUM_THREADS"] = value
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = value
    os.environ["OPENBLAS_NUM_THREADS"] = value


def _worker_init(threads_per_worker: int) -> None:
    _set_thread_env(threads_per_worker)
    faulthandler.enable(all_threads=True)
    _log(f"[worker-init] pid={os.getpid()} threads_per_worker={threads_per_worker}")


def _recommend_parallelism(backend: str, cpu_count: int | None) -> tuple[int, int]:
    cpu_total = max(1, cpu_count or 1)
    if cpu_total >= 16:
        return min(16, cpu_total), 1
    return min(8, cpu_total), 1

# ---------------------------------------------------------------------------
# Parallel config search (shared)
# ---------------------------------------------------------------------------

def search_parallel_configs(num_gpus: int) -> list[str]:
    parallel_configs = []
    for p_dp in (1, 2, 4, 8):
        for p_pp in (1,):
            for p_tp in (1, 2, 4, 8):
                for d_dp in (1, 2, 4, 8):
                    for d_tp in (1, 2, 4, 8):
                        if p_dp != d_dp:
                            continue
                        num_gpus_used = p_dp * p_pp * p_tp + d_dp * d_tp
                        if num_gpus // 2 <= num_gpus_used <= num_gpus:
                            parallel_configs.append(f"{p_dp}:{p_pp}:{p_tp}@{d_dp}:1:{d_tp}")
                num_gpus_used = p_dp * p_pp * p_tp
                if num_gpus // 2 <= num_gpus_used <= num_gpus:
                    parallel_configs.append(f"{p_dp}:{p_pp}:{p_tp}")
    return sorted(set(parallel_configs))


# ---------------------------------------------------------------------------
# SimulationTask dataclass
# ---------------------------------------------------------------------------

@dataclass
class SimulationTask:
    task_id: int
    backend: str
    model: str
    parallel: str
    runtime_config: RuntimeConfig
    hw_yaml: str
    gpu: str
    debug: bool
    fast: bool = False
    # optional fields
    dataset: str = ""
    num_prompt: int = 0
    modeling_dir: str = ""
    # comparison: path to framework trace log directory (empty = skip)
    trace_log_dir: str = ""


# ---------------------------------------------------------------------------
# Shared metric helpers
# ---------------------------------------------------------------------------

def extract_metrics(sim_result: Dict[str, Any], columns: List[str]) -> pd.DataFrame:
    sim_result.update({"framework": "ROSS"})
    row = [sim_result.get(col) for col in columns]
    return pd.DataFrame([row], columns=columns)


def _tokens_per_gpu_from_parallel(parallel: str, throughput: float) -> float:
    if "@" in parallel:
        _, decode_parallel = parallel.split("@", 1)
        decode_dp, decode_pp, decode_tp = map(int, decode_parallel.split(":"))
        total_gpu = max(1, decode_dp * decode_pp * decode_tp)
    else:
        dp, pp, tp = map(int, parallel.split(":"))
        total_gpu = max(1, dp * pp * tp)
    return throughput / total_gpu


def _tokens_per_user_from_tpot(mean_tpot_ms: float) -> float:
    if mean_tpot_ms is None or mean_tpot_ms <= 0:
        return 0.0
    return 1000.0 / mean_tpot_ms


# ---------------------------------------------------------------------------
# Core simulator dispatch
# ---------------------------------------------------------------------------

def run_sim_predict(backend: str, model: str, parallel: str,
                    runtime_config: RuntimeConfig,
                    platform_perf_yaml: str,
                    modeling_dir: str,
                    fast: bool = False):
    """Dispatch to the correct bench module based on backend."""
    is_disagg = "@" in parallel
    extra_kwargs = {}
    if backend == "vllm":
        extra_kwargs["fast"] = fast
    if not is_disagg:
        dp, pp, tp = parallel.split(":")
        summary = _bench_mod.find_best_colocate_result_under_constraints(
            model_uri=model,
            inference_config=InferenceConfig(dp_size=int(dp), pp_size=int(pp), tp_size=int(tp)),
            runtime_config=runtime_config,
            platform_perf_yaml=platform_perf_yaml,
            modeling_dir=modeling_dir,
            **extra_kwargs,
        )
    else:
        p_cfg, d_cfg = parallel.split("@")
        pdp, ppp, ptp = p_cfg.split(":")
        ddp, dpp, dtp = d_cfg.split(":")
        summary = _bench_mod.find_best_disagg_result_under_constraints(
            model_uri=model,
            prefill_inference_config=InferenceConfig(dp_size=int(pdp), pp_size=int(ppp), tp_size=int(ptp)),
            decode_inference_config=InferenceConfig(dp_size=int(ddp), pp_size=int(dpp), tp_size=int(dtp)),
            runtime_config=runtime_config,
            platform_perf_yaml=platform_perf_yaml,
            modeling_dir=modeling_dir,
            **extra_kwargs,
        )
    return summary.get_result_dict()


# ---------------------------------------------------------------------------
# Pareto row builder
# ---------------------------------------------------------------------------

def _build_pareto_row(sim_result_dict: Dict[str, Any], task: "SimulationTask") -> Dict[str, Any]:
    row = {col: sim_result_dict.get(col) for col in SummaryColumns}
    scheduler_config = task.runtime_config.scheduler_config

    row.update({
        "task_id": task.task_id,
        "model": task.model,
        "dataset": task.dataset,
        "isl": task.runtime_config.isl,
        "osl": task.runtime_config.osl,
        "batch_size": task.runtime_config.batch_size,
        "request_rate": task.runtime_config.rate,
        # SGL uses mem_fraction_static / chunked_prefill_size
        # vLLM uses gpu_memory_utilization / max_num_batched_tokens
        # Use .get() with fallbacks so both backends populate these columns
        "mem_fraction_static": scheduler_config.get(
            "mem_fraction_static",
            scheduler_config.get("gpu_memory_utilization"),
        ),
        "chunked_prefill_size": scheduler_config.get(
            "chunked_prefill_size",
            scheduler_config.get("max_num_batched_tokens"),
        ),
    })

    if "@" in task.parallel:
        prefill_parallel, decode_parallel = task.parallel.split("@", 1)
        prefill_dp, prefill_pp, prefill_tp = map(int, prefill_parallel.split(":"))
        decode_dp, decode_pp, decode_tp = map(int, decode_parallel.split(":"))
        row.update({
            "prefill_dp": prefill_dp,
            "prefill_pp": prefill_pp,
            "prefill_tp": prefill_tp,
            "decode_dp": decode_dp,
            "decode_pp": decode_pp,
            "decode_tp": decode_tp,
        })
    else:
        dp, pp, tp = map(int, task.parallel.split(":"))
        row.update({
            "dp": dp,
            "pp": pp,
            "tp": tp,
        })

    return row


# ---------------------------------------------------------------------------
# run_single_sim -- top-level worker function (must be picklable)
# ---------------------------------------------------------------------------

def run_single_sim(task: "SimulationTask") -> tuple:
    _log(f"[task-start] pid={os.getpid()} {_task_brief(task)}")
    try:
        _load_backend_modules(task.backend)

        sim_start_t = time.perf_counter()
        sim_result_dict = run_sim_predict(
            task.backend, task.model, task.parallel,
            task.runtime_config, task.hw_yaml,
            modeling_dir=task.modeling_dir,
            fast=task.fast,
        )
        sim_elapsed_s = time.perf_counter() - sim_start_t

        if sim_result_dict is None:
            _log(f"[task-done] pid={os.getpid()} {_task_brief(task)} sim_elapsed_s={sim_elapsed_s:.3f} result=None")
            return None, sim_elapsed_s, task.parallel, task.model, task.runtime_config.rate

        if task.debug:
            if "@" in task.parallel:
                if "prefill_phases" in sim_result_dict:
                    _debug_phase_dump(task, "prefill_phases", sim_result_dict["prefill_phases"])
                if "decode_phases" in sim_result_dict:
                    _debug_phase_dump(task, "decode_phases", sim_result_dict["decode_phases"])
            else:
                if "timing_phases" in sim_result_dict:
                    _debug_phase_dump(task, "timing_phases", sim_result_dict["timing_phases"])

        _log(
            f"[task-done] pid={os.getpid()} {_task_brief(task)} "
            f"sim_elapsed_s={sim_elapsed_s:.3f} duration={sim_result_dict.get('duration')}"
        )
        return sim_result_dict, sim_elapsed_s, task.parallel, task.model, task.runtime_config.rate
    except Exception:
        _log(f"[task-error] pid={os.getpid()} {_task_brief(task)}\n{traceback.format_exc()}")
        raise


# ---------------------------------------------------------------------------
# Shared result processing helper
# ---------------------------------------------------------------------------

def _load_trace(task: "SimulationTask", args: Any) -> Optional[Dict[str, Any]]:
    """Load real framework trace for comparison. Returns None on failure."""
    backend = task.backend
    trace_log_dir = task.trace_log_dir
    if not trace_log_dir:
        return None

    is_disagg = "@" in task.parallel
    osl = task.runtime_config.osl
    rate = task.runtime_config.rate
    debug = task.debug

    try:
        if backend == "sglang":
            sys.path.insert(0, str(Path(REPO_ROOT) / "test" / "simulator-sgl"))
            from common.utils import resolve_sglang_log_paths
            import tests.load_traces_lmdb as sgl_load
            import tests.load_disaggregation_traces_lmdb as sgl_disagg_load
            trace_args = resolve_sglang_log_paths(
                log_dir=trace_log_dir,
                gpu=task.gpu,
                isl=task.runtime_config.isl,
                osl=osl,
                num_prompt=task.num_prompt,
                dataset=task.dataset,
                request_rate=rate,
                disaggregation=is_disagg,
            )
            if not is_disagg:
                trace_result = sgl_load.load_traces(trace_args, debug)
            else:
                trace_result = sgl_disagg_load.load_traces(trace_args, debug)
        else:  # vllm
            sys.path.insert(0, str(Path(REPO_ROOT) / "test" / "simulator-vllm"))
            from common.utils import resolve_vllm_log_paths
            import tests.load_traces_lmdb as vllm_load
            import tests.load_disagg_traces_lmdb as vllm_disagg_load
            trace_args = resolve_vllm_log_paths(
                log_dir=trace_log_dir,
                gpu=task.gpu,
                isl=task.runtime_config.isl,
                osl=osl,
                num_prompt=task.num_prompt,
                dataset=task.dataset,
                request_rate=rate,
                disaggregation=is_disagg,
            )
            if not is_disagg:
                trace_result = vllm_load.load_traces(trace_args, debug)
            else:
                req_name = None
                prefill_results, req_name = vllm_disagg_load.load_traces(
                    trace_args, debug=debug, stage_name="prefill", req_name=req_name
                )
                trace_result, req_name = vllm_disagg_load.load_traces(
                    trace_args, debug=debug, stage_name="decode", req_name=req_name
                )
                if trace_result:
                    trace_result.update(
                        {"prefill_phase": prefill_results.get("staged_timing_data")}
                    )

        trace_result["throughput"] = trace_result.get("output_throughput", trace_result.get("throughput"))
        return trace_result
    except Exception:
        logging.warning(f"[trace-load-error] task_id={task.task_id}\n{traceback.format_exc()}")
        return None


def _compute_pe(sim: Dict[str, Any], ref: Dict[str, Any]) -> Dict[str, float]:
    """Compute prediction error (%) between sim and framework results."""
    def _pe(s, b):
        return 0.0 if (b is None or b == 0) else abs(s - b) / abs(b) * 100.0

    return {
        "pe":                  _pe(sim["duration"],      ref["duration"]),
        "pe_mean_ttft":        _pe(sim["mean_ttft_ms"],  ref["mean_ttft_ms"]),
        "pe_mean_tpot":        _pe(sim["mean_tpot_ms"],  ref["mean_tpot_ms"]),
        "pe_mean_itl":         _pe(sim["mean_itl_ms"],   ref["mean_itl_ms"]),
        "pe_mean_throughput":  _pe(sim["throughput"],    ref["throughput"]),
    }


# ---------------------------------------------------------------------------
# Shared result processing helper
# ---------------------------------------------------------------------------

def _process_result(
    sim_result_dict: Dict[str, Any],
    sim_elapsed_s: float,
    parallel: str,
    model: str,
    rate: float,
    task: "SimulationTask",
    backend: str,
    metric_list: List[str],
    args: Any,
    results: list,
    pareto_rows: list,
    pareto_task_map: Dict[int, "SimulationTask"],
):
    """Process a completed simulation result."""
    if not sim_result_dict:
        return

    if args.get_pareto_front:
        pareto_rows.append(_build_pareto_row(sim_result_dict, task))
        pareto_task_map[task.task_id] = task

    # ------------------------------------------------------------------
    # Optionally load framework trace for comparison
    # ------------------------------------------------------------------
    do_comparison = args.eval and bool(task.trace_log_dir)
    trace_result_dict = None
    pe_dict = {}

    if do_comparison:
        trace_result_dict = _load_trace(task, args)
        if trace_result_dict is None:
            logging.warning(f"[comparison-skip] task_id={task.task_id}: no trace found at {task.trace_log_dir!r}, skipping comparison")
            do_comparison = False
        else:
            pe_dict = _compute_pe(sim_result_dict, trace_result_dict)

    if task.debug:
        for phase_name in ("timing_phases", "prefill_phases", "decode_phases"):
            if phase_name in sim_result_dict:
                tqdm_module.tqdm.write(
                    f"[debug] {_task_brief(task)} {phase_name}={sim_result_dict.get(phase_name)}"
                )
        if "sgl_predict_stats" in sim_result_dict:
            tqdm_module.tqdm.write(
                f"[debug] {_task_brief(task)} sgl_predict_stats={sim_result_dict.get('sgl_predict_stats')}"
            )
        if trace_result_dict:
            trace_phase_map = {
                "staged_timing_data": "trace_staged_timing_data",
                "prefill_phase": "trace_prefill_phase",
                "timing_phases": "trace_timing_phases",
                "prefill_phases": "trace_prefill_phases",
                "decode_phases": "trace_decode_phases",
            }
            for src_name, debug_name in trace_phase_map.items():
                if src_name in trace_result_dict:
                    tqdm_module.tqdm.write(
                        f"[debug] {_task_brief(task)} {debug_name}={trace_result_dict.get(src_name)}"
                    )

    # ------------------------------------------------------------------
    # Build and print display DataFrame
    # ------------------------------------------------------------------
    if do_comparison:
        rows = [
            {"framework": "ROSS",   **{c: sim_result_dict.get(c)   for c in metric_list if c != "framework"}},
            {"framework": backend,  **{c: trace_result_dict.get(c)  for c in metric_list if c != "framework"}},
            {"framework": "PE(%)",  "duration": pe_dict["pe"],
             "mean_ttft_ms": pe_dict["pe_mean_ttft"],
             "mean_tpot_ms": pe_dict["pe_mean_tpot"],
             "mean_itl_ms":  pe_dict["pe_mean_itl"],
             "throughput":   pe_dict["pe_mean_throughput"]},
        ]
        df = pd.DataFrame(rows, columns=metric_list)
    else:
        df = extract_metrics(sim_result_dict, metric_list)
    tqdm_module.tqdm.write(f"DF (took {sim_elapsed_s:.3f}s):\n{df}")

    # ------------------------------------------------------------------
    # Record row for CSV
    # ------------------------------------------------------------------
    if args.record_path:
        id_vals = [task.gpu, model, parallel,
                   f"{task.runtime_config.isl}@{task.runtime_config.osl}", rate,
                   task.runtime_config.scheduler_config.get("cpu_count")]

        if do_comparison:
            results.append(id_vals + [
                pe_dict["pe"],
                pe_dict["pe_mean_ttft"],
                pe_dict["pe_mean_tpot"],
                pe_dict["pe_mean_itl"],
                pe_dict["pe_mean_throughput"],
                sim_result_dict["duration"],        trace_result_dict["duration"],
                sim_result_dict["mean_ttft_ms"],    trace_result_dict["mean_ttft_ms"],
                sim_result_dict["mean_tpot_ms"],    trace_result_dict["mean_tpot_ms"],
                sim_result_dict["mean_itl_ms"],     trace_result_dict["mean_itl_ms"],
                sim_result_dict["throughput"],      trace_result_dict["throughput"],
            ])
        else:
            results.append(id_vals + [
                sim_result_dict["duration"],
                sim_result_dict["mean_ttft_ms"],
                sim_result_dict["mean_tpot_ms"],
                sim_result_dict["mean_itl_ms"],
                sim_result_dict["throughput"],
            ])


# ---------------------------------------------------------------------------
# Backend-specific sweep helpers
# ---------------------------------------------------------------------------

def _get_sweep(conf: "BenchmarkConfig", backend: str, gpu: str) -> list[tuple]:
    """Build the inner scheduler parameter sweep for a given backend+GPU.

    Returns a list of (chunk_size, mem, mbt, cpu_count) tuples.
    Fields not used by a backend are set to None.

    Scheduler field mapping
    -----------------------
    sglang : mem_fraction_static   / chunked_prefill_size
    vllm   : gpu_memory_utilization / max_num_batched_tokens

    ``cpu_count`` is backend-agnostic metadata (recorded in CSV; not passed to
    the simulator directly, but kept in the tuple so callers can log it).
    """
    mem_list   = [DEFAULT_RESERVE]
    chunk_list = [8192]
    mbt_list   = [8192]
    cpu_list   = [None]

    conf_args = {}
    if conf.args.get(backend):
        conf_args = conf.args[backend][0]

    if backend == "sglang":
        if "mem_fraction_static"  in conf_args: mem_list   = conf_args["mem_fraction_static"]
        if "chunked_prefill_size" in conf_args: chunk_list = conf_args["chunked_prefill_size"]
        if "cpu_count"            in conf_args:
            raw = conf_args["cpu_count"]
            cpu_list = raw if isinstance(raw, list) else [raw]
        # B200 requires larger chunked-prefill window
        if gpu.upper() == "B200":
            chunk_list = [16384]
        return [
            (chunk, mem, None, cpu)
            for chunk in chunk_list
            for mem   in mem_list
            for cpu   in cpu_list
        ]
    else:  # vllm
        if "gpu_memory_utilization" in conf_args: mem_list   = conf_args["gpu_memory_utilization"]
        if "max_num_batched_tokens" in conf_args: mbt_list   = conf_args["max_num_batched_tokens"]
        if "cpu_count"              in conf_args:
            raw = conf_args["cpu_count"]
            cpu_list = raw if isinstance(raw, list) else [raw]
        return [
            (None, mem, mbt, cpu)
            for mem in mem_list
            for mbt in mbt_list
            for cpu in cpu_list
        ]


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

def main():
    MACHINE_STATS_PREFIX = "../collector"
    args = parse_predict_args()

    global _debug_logging
    _debug_logging = args.debug

    conf = BenchmarkConfig(args)
    backend = conf.backends[0]

    _load_backend_modules(backend)

    setup_logging(f"./log/ross_{backend}_predict.log", args.debug)

    # Override parallel configs for pareto search
    if args.get_pareto_front:
        conf.parallel = search_parallel_configs(conf.num_gpu)

    util.echo_line(util.line_width, "-", "Benchmark Configuration")
    util.echo_info(conf.summary())
    if args.get_pareto_front:
        util.echo_info(f"pareto search num_gpus: {conf.num_gpu}")
        util.echo_info(f"auto searched parallel configs: num={len(conf.parallel)}")

    # CPU / thread configuration
    cpu_total = os.cpu_count() or 1
    auto_workers, auto_threads = _recommend_parallelism(backend, cpu_total)
    max_workers = args.max_workers or auto_workers
    max_workers = max(
        1,
        min(max_workers, len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else cpu_total),
    )
    threads_per_worker = args.threads_per_worker or auto_threads
    _set_thread_env(threads_per_worker)
    print(
        f"CPU total={cpu_total}, workers={max_workers}, threads/worker={threads_per_worker}, "
        f"total requested threads={max_workers * threads_per_worker}"
    )

    # Build RECORD_COL based on whether comparison is enabled.
    # cpu_count is now meaningful for both backends, so always include it.
    if args.eval:
        RECORD_COL = RECORD_ID_COLS_COMMON + ["cpu_count"] + RECORD_COMPARISON_COLS
    else:
        RECORD_COL = RECORD_ID_COLS_COMMON + ["cpu_count"] + RECORD_NO_COMPARISON_COLS

    metric_list = ["framework", "duration", "mean_ttft_ms", "mean_tpot_ms", "mean_itl_ms", "throughput"]

    if args.record_path:
        record_dir = str(Path(__file__).resolve().parent)
        if not os.path.exists(record_dir):
            raise RuntimeError(f"{args.record_path} Not Exist!")

    # -----------------------------------------------------------------------
    # Build task list
    # -----------------------------------------------------------------------
    tasks: List[SimulationTask] = []

    for inp in conf.inputs:
        dataset, data_path, isl, osl = inp["dataset"], inp["path"], inp["isl"][0], inp["osl"][0]
        datatype = f"{dataset}_isl_{isl}_osl_{osl}"

        for platform in conf.platforms:
            gpu = platform["gpu"]
            inner_iter = _get_sweep(conf, backend, gpu)

            for model in conf.models:
                p = Path(model)
                model_name = p.parent.name if p.name.startswith("v") else p.name

                for parallel in conf.parallel:
                    for batch in conf.batches:
                        for rate in conf.rates:
                            conf.set_curr(backend, model, parallel, batch, inp, platform)

                            # Arrival file (shared)
                            arrival_dir = os.path.abspath(f"arrivals/{backend}")
                            os.makedirs(arrival_dir, exist_ok=True)
                            arrival_path = os.path.join(
                                arrival_dir,
                                f"arrivals_{model_name}_{datatype}"
                                f"_batch_{batch}_promptnum_{conf.num_prompt}_rate_{rate}.jsonl"
                            )
                            if not os.path.exists(arrival_path):
                                _call_bench_serving(
                                    model, backend, dataset, data_path,
                                    str(isl), str(osl), str(rate), str(conf.num_prompt),
                                    arrival_path, str(batch), conf.random_range_ratio,
                                )

                            # Inner scheduler param loop
                            for (chunk_size, mem_or_gpu, mbt, cpu_count) in inner_iter:
                                hw_yaml = f"{MACHINE_STATS_PREFIX}/{gpu.lower()}/platform_features.yaml"
                                assert os.path.exists(hw_yaml), f"Missing platform yaml: {hw_yaml}"

                                # Build RuntimeConfig
                                if backend == "sglang":
                                    sched_cfg = {
                                        "mem_fraction_static":  mem_or_gpu,
                                        "chunked_prefill_size": chunk_size,
                                    }
                                else:
                                    sched_cfg = {
                                        "gpu_memory_utilization": mem_or_gpu,
                                        "max_num_batched_tokens":  mbt,
                                        "gpu": gpu.lower(),
                                    }
                                if cpu_count is not None:
                                    sched_cfg["cpu_count"] = cpu_count
                                runtime_config = RuntimeConfig(
                                    batch_size=batch, isl=isl, osl=osl, rate=rate,
                                    scheduler_config=sched_cfg,
                                    arrival_path=arrival_path,
                                )

                                tasks.append(SimulationTask(
                                    task_id=len(tasks),
                                    backend=backend,
                                    model=model,
                                    parallel=parallel,
                                    runtime_config=runtime_config,
                                    hw_yaml=hw_yaml,
                                    gpu=gpu.lower(),
                                    debug=args.debug,
                                    fast=args.fast,
                                    dataset=dataset,
                                    num_prompt=conf.num_prompt,
                                    modeling_dir=conf.modeling_dir,
                                    # Pass trace log dir for comparison (empty string = skip)
                                    trace_log_dir=conf.test_dst if args.eval else "",
                                ))

    print(f"Total tasks: {len(tasks)}")

    # -----------------------------------------------------------------------
    # Execute tasks: sequential or multiprocessing
    # -----------------------------------------------------------------------
    results = []
    record_df = []
    pareto_rows = []
    pareto_task_map: Dict[int, SimulationTask] = {}
    ross_search_start_t = time.perf_counter()

    if max_workers == 1:
        # Sequential path
        for task in tasks:
            try:
                sim_result_dict, sim_elapsed_s, parallel, model, rate = run_single_sim(task)
                _process_result(
                    sim_result_dict, sim_elapsed_s, parallel, model, rate,
                    task, backend, metric_list, args, results, pareto_rows, pareto_task_map,
                )
            except Exception as e:
                print(f"[error] task_id={task.task_id}: {e}", file=sys.stderr)
    else:
        # Multiprocessing path
        try:
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=max_workers,
                mp_context=ctx,
                initializer=_worker_init,
                initargs=(threads_per_worker,),
            ) as executor:
                task_iter = iter(tasks)
                future_to_task = {}

                def _submit_until_full() -> None:
                    while len(future_to_task) < max_workers:
                        try:
                            task = next(task_iter)
                        except StopIteration:
                            return
                        _log(f"[submit] {_task_brief(task)}")
                        future_to_task[executor.submit(run_single_sim, task)] = task

                _submit_until_full()
                progress = tqdm(total=len(tasks), desc="Simulating", file=sys.stderr,
                                dynamic_ncols=True, leave=False)
                while future_to_task:
                    done, _ = wait(list(future_to_task.keys()), return_when=FIRST_COMPLETED)
                    for future in done:
                        task = future_to_task.pop(future)
                        progress.update(1)
                        _submit_until_full()
                        try:
                            sim_result_dict, sim_elapsed_s, parallel, model, rate = future.result()
                            _process_result(
                                sim_result_dict, sim_elapsed_s, parallel, model, rate,
                                task, backend, metric_list, args, results, pareto_rows, pareto_task_map,
                            )
                        except Exception as e:
                            print(f"[error] task_id={task.task_id}: {e}", file=sys.stderr)
                progress.close()
        except KeyboardInterrupt:
            print("\nUser interrupted. Exiting...", file=sys.stderr)
            sys.exit(1)

    ross_search_elapsed_s = time.perf_counter() - ross_search_start_t
    print(f"ROSS search took {ross_search_elapsed_s:.3f}s")

    # -----------------------------------------------------------------------
    # Save record CSV
    # -----------------------------------------------------------------------
    if args.record_path:
        for record_data in results:
            row_df = pd.DataFrame([record_data], columns=RECORD_COL)
            if not row_df.empty:
                record_df.append(row_df)
        result_record_df = (
            pd.concat(record_df, axis=0, ignore_index=True) if record_df else pd.DataFrame()
        )
        print(result_record_df)
        # Aggregate mean PE columns when comparison is active
        if args.eval and not result_record_df.empty:
            agg_cols = ["pe", "pe_mean_ttft", "pe_mean_tpot", "pe_mean_itl", "pe_mean_throughput"]
            present = [c for c in agg_cols if c in result_record_df.columns]
            if present:
                avg_cols = ["avg_" + c for c in present]
                result_record_df[avg_cols] = (
                    result_record_df.groupby(["Platform", "model"])[present].transform("mean")
                )
        result_record_df.to_csv(args.record_path)
    # -----------------------------------------------------------------------
    # Pareto front analysis
    # -----------------------------------------------------------------------
    if args.get_pareto_front and pareto_rows:
        pareto_df = pd.DataFrame(pareto_rows)
        dedup_cols = [
            c for c in pareto_df.columns
            if c not in ["timing_phases", "decode_phases", "prefill_phases", "task_id"]
        ]
        print("=== pareto_df_raw ===")
        print(pareto_df)

        pareto_fronts: Dict[str, pd.DataFrame] = {}
        pareto_candidates_by_mode = {
            "Colocate": pareto_df[pareto_df["decode_dp"].isna()].reset_index(drop=True),
            "Disagg":   pareto_df[pareto_df["decode_dp"].notna()].reset_index(drop=True),
        }
        for mode_label, mode_df in pareto_candidates_by_mode.items():
            if mode_df.empty:
                continue
            mode_df = mode_df.drop_duplicates(subset=dedup_cols, ignore_index=True)
            pareto_candidates_by_mode[mode_label] = mode_df
            print(f"=== {mode_label.lower()} pareto_df ===")
            print(mode_df)
            pareto_front_df = pa.get_pareto_front(mode_df, "tokens/s/user", "tokens/s/gpu").reset_index(drop=True)
            merge_cols = [c for c in pareto_front_df.columns if c in mode_df.columns and c != "task_id"]
            task_map_df = mode_df[["task_id"] + merge_cols].drop_duplicates(subset=merge_cols, ignore_index=True)
            pareto_front_df = pareto_front_df.merge(task_map_df, on=merge_cols, how="left")
            pareto_fronts[mode_label] = pareto_front_df
            print(f"=== {mode_label.lower()} pareto_front ===")
            print(pareto_front_df)

        draw_pareto_fronts(pareto_fronts)
        if args.record_path:
            record_path = Path(args.record_path)
            pareto_df.to_csv(
                record_path.with_name(f"{record_path.stem}_pareto_candidates{record_path.suffix}"),
                index=False,
            )
            for mode_label, pf_df in pareto_fronts.items():
                mode_suffix = mode_label.lower()
                pareto_candidates_by_mode[mode_label].to_csv(
                    record_path.with_name(
                        f"{record_path.stem}_pareto_candidates_{mode_suffix}{record_path.suffix}"
                    ),
                    index=False,
                )
                pf_df.to_csv(
                    record_path.with_name(
                        f"{record_path.stem}_pareto_front_{mode_suffix}{record_path.suffix}"
                    ),
                    index=False,
                )


if __name__ == "__main__":
    main()
