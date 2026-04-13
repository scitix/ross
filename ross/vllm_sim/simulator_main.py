#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

from pathlib import Path
import os, sys
import time
from collections import deque
import pandas as pd
import json
import numpy as np
from typing import List, Tuple, Dict, Any

TEST_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TEST_ROOT))

from common.features import PlatformPerf
from common.models import get_model, BaseModel
from common.config import InferenceConfig
from common.kvpool import KVCachePool
from common.ross_model import ROSSModel
from common.sim_http_perf import RequestStore, VirtualClientStore

from scheduler.request import Request, RequestStatus
from scheduler.scheduler import Scheduler, SchedulerOutput

import logging
logger = logging.getLogger(__name__)


def _warn_pp_pre_forward_disabled(args) -> None:
    has_explicit_deprecated_path = bool(getattr(args, "_explicit_pp_pre_forward_path", False))
    if getattr(args, "pp_size", 1) > 1:
        print(
            "[warning] vLLM pp>1 currently does not support pp_pre_model; "
            "pp_pre_forward is forced to 0 and --pp_pre_forward_path is ignored.",
            flush=True,
        )
    elif has_explicit_deprecated_path:
        print(
            "[warning] --pp_pre_forward_path is deprecated and ignored in vLLM simulation.",
            flush=True,
        )


def _get_gpu_memory_utilization(args) -> float:
    if hasattr(args, "gpu_memory_utilization"):
        return args.gpu_memory_utilization
    if hasattr(args, "gpu_model_utilization"):
        return args.gpu_model_utilization
    raise AttributeError("Missing gpu memory utilization on args")

def parse_args():
    ap = argparse.ArgumentParser("ROSS VLLM simulator")
    ap.add_argument("--model-uri", type=str, default="")
    # Parallel Config
    ap.add_argument("--dp-size", type=int, default=1)
    ap.add_argument("--pp-size", type=int, default=1)
    ap.add_argument("--tp-size", type=int, default=1)
    # ap.add_argument("--ep-size", type=int, default=1)

    # Disaggregation Config
    ap.add_argument("--disaggregation", action='store_true', default=False)
    ap.add_argument("--prefill-tp-size", type=int, default=1)    
    ap.add_argument("--decode-tp-size", type=int, default=1)


    # Scheduler Config
    ap.add_argument('--gpu-model-utilization', type=float, required=True, help='GPU Memory Utilization')
    ap.add_argument('--max-num-batched-tokens', type=int, required=True, help='Max Number of Batched Tokens')
    ap.add_argument('--mem-profiling-path', type=str, required=True, help='Path to the Memory Profiling Result.')
    ap.add_argument('--gpu', type=str, required=True)

    # Workload Config
    ap.add_argument("--batch-size", type=int, required=True)
    ap.add_argument("--max-prompt-len", type=int, required=True)
    ap.add_argument("--max-output-len", type=int, required=True)
    ap.add_argument("--frontend-path", type=str, required=True)
    ap.add_argument("--request-rate", type=str, required=True)

    # ROSS Config
    ap.add_argument('--platform-perf', type=str, required=True, help='Path to the Platform Performance file.')

    ap.add_argument('--pre_forward_path', type=str, help='Path to the Saved PRE-FORWARD Model file.')
    ap.add_argument(
        '--pp_pre_forward_path',
        type=str,
        help='Deprecated. Ignored in vLLM simulation because pp_pre_model is not supported.',
    )

    ap.add_argument('--forward_path', type=str, help='Path to the Saved [Prefill] FORWARD Model file.')
    ap.add_argument('--prefill_forward_path', type=str, help='Path to the Saved [Prefill] FORWARD Model file.')
    ap.add_argument('--decode_forward_path', type=str, help='Path to the Saved [Decode] FORWARD Model file.')
    ap.add_argument('--post_forward_path', type=str, help='Path to the Saved [Prefill] POST-FORWARD Model file.')

    args = ap.parse_args()
    args._explicit_pp_pre_forward_path = "--pp_pre_forward_path" in sys.argv
    if not hasattr(args, "gpu_memory_utilization"):
        args.gpu_memory_utilization = args.gpu_model_utilization
    return args

def get_ross_models(model: BaseModel,
                    platform_perf: PlatformPerf,
                    inf_config: InferenceConfig,
                    model_path: Dict[str, str]):
    model_keys = ['pre_forward', 'forward', 'prefill_forward', 'decode_forward', 'post_forward']
    model_dict = dict()
    for key in model_path.keys():
        assert(key in model_keys)

        model_dict[key] = ROSSModel(
            saved_model_path=model_path[key],
            platform_perf=platform_perf,
            model=model,
            inference_config=inf_config,
            regressor="xgboost",
        )
    return model_dict

def get_ross_model_paths(args) -> Dict[str, str]:
    model_paths = {
        "pre_forward": getattr(args, "pre_forward_path", None),
        "post_forward": getattr(args, "post_forward_path", None),
    }
    if getattr(args, "forward_path", None):
        model_paths["forward"] = args.forward_path
    if getattr(args, "prefill_forward_path", None):
        model_paths["prefill_forward"] = args.prefill_forward_path
    if getattr(args, "decode_forward_path", None):
        model_paths["decode_forward"] = args.decode_forward_path
    return {key: value for key, value in model_paths.items() if value}

def get_regression_model(ross_models: Dict[str, Any], name: str, phase: str = ""):
    if name != "forward":
        return ross_models[name]
    if phase == "prefill":
        return ross_models.get("prefill_forward", ross_models.get("forward"))
    if phase == "decode":
        return ross_models.get("decode_forward", ross_models.get("forward"))
    return (
        ross_models.get("forward")
        or ross_models.get("prefill_forward")
        or ross_models.get("decode_forward")
    )

def get_mixed_forward_phase(schedule_output: SchedulerOutput) -> str:
    return "prefill" if schedule_output.prefill_seq_lens else 'decode'

def update_metrics(req: Request, wall_time: float, arrive_time: float, step: int) -> bool:
    if req._last_token_time is None:
        req._last_token_time = wall_time
        req.itl = []
    else:
        req.itl.append(wall_time - req._last_token_time)
        req._last_token_time = wall_time

    if req.output_len == 1 and not req.ttft:
        req.ttft = wall_time - arrive_time
        logger.debug(f"req={req.request_id} wall_time={wall_time:.2f}, arrive_time={arrive_time:.2f}, ttft={req.ttft}")
        req.ttft_step = step
                
    if req.is_finished and not req.e2e_latency:
        req.e2e_latency = wall_time - arrive_time
        req.ttlt_step = step
        req.tpot = (wall_time - arrive_time - req.ttft) / (req.output_len - 1)
        logger.debug(f"req={req.request_id} finished and start calculating metrics.. wall_time={wall_time:.2f}, ttft={req.ttft}, tpot={req.tpot}")
        return True

    return False

def calulcate_benchmark_results(request_list: List[Request], itl_list: List[float], wall_time: float) -> Dict[str, float]:
    def _calc(name, data: List[float]) -> Dict[str, float]:
        return {
            f"mean_{name}_ms": float(np.mean(data)),
            f"median_{name}_ms": float(np.median(data)),
            f"std_{name}_ms": float(np.std(data)),
            f"p99_{name}_ms": float(np.percentile(data, 99))
        }

    ttft = [r.ttft * 1000 for r in request_list]
    tpot = [r.tpot * 1000 for r in request_list]
    itl_list = [i * 1000 for i in itl_list]
    e2e_latency = [r.e2e_latency * 1000 for r in request_list]

    num_tokens = sum([r.num_tokens - r.prompt_tokens for r in request_list])
    return {
        "throughput": num_tokens / wall_time,
        **_calc("ttft", ttft),
        **_calc("tpot", tpot),
        **_calc("itl", itl_list),
        **_calc("e2e_latency", e2e_latency)
    }

def check_pipeline_clear(pipeline: List[SchedulerOutput], step: int, pp: int):
    if not pipeline or step - pp >= len(pipeline):
        return True
    for i in range(1, pp + 1):
        if pipeline[len(pipeline) - i] is not None:
            return False
    return True

def run_simulation(
    model: BaseModel,
    batch_size: int,
    request_list: VirtualClientStore,
    ross_models: Dict[str, Any],
    scheduler_kwargs: Dict[str, Any],
    isl: int,
    osl: int,
    gpu_memory_utilization: float,
    memory_profiling: Dict[str, Any] = None,
    total_gpu_memory: int | None = None,
    dp: int = 1, # vllm: p & d have same dp_size
    pp: int = 1,
) -> Dict[str, Any]:
    schedulers : List[Scheduler] = []
    total_time_slices = [{ "pre_forward": 0, "forward": 0, "post_forward": 0, "pp_pre_forward": 0 } for i in range(dp)]

    tokens_per_block = 16 if model.model_uri.lower().find('deepseek') == -1 else 64
    for idx in range(dp):
        kv_pool = KVCachePool(
            model=model,
            num_reqs=batch_size,
            tokens_per_block=tokens_per_block,
            total_gpu_memory=total_gpu_memory,
            gpu_memory_utilization=gpu_memory_utilization,
            vllm_non_torch_increase=memory_profiling['non_torch_mem_increase']
        )
        schedulers.append(Scheduler(
            max_running_reqs=batch_size,
            kv_pool=kv_pool,
            **scheduler_kwargs
        ))

    current_status = [ { "wall_time": 0.0, "step": 0, "batch_pipeline": [], "complete": False } for i in range(dp)]
    finished_reqs_count = 0
    pending_by_rank = [deque() for _ in range(dp)]
    active_reqs_by_rank = [dict() for _ in range(dp)]  # rid -> Request, bounded by inflight concurrency
    
    while True:
        current_global_time = max(status["wall_time"] for status in current_status)
        new_reqs = request_list.refresh(current_global_time)
        for r in new_reqs:
            pending_by_rank[r.dp_rank].append(r)
            current_status[r.dp_rank]["complete"] = False

        # idle admission by arrive_time: if a rank is idle, pull the next future req
        for rank, sched in enumerate(schedulers):
            status = current_status[rank]
            status["step"] += 1
            pending = pending_by_rank[rank]
            while pending and pending[0].ready_time <= status["wall_time"]:
                req = pending.popleft()
                sched.add_request(req)
                active_reqs_by_rank[rank][req.request_id] = req
            if not status["complete"]:
                # no in-flight pipeline slots pending
                if check_pipeline_clear(status["batch_pipeline"], status["step"], pp) and sched.should_terminate():
                    if pending:
                        req_next = pending.popleft()
                        status["wall_time"] = max(status["wall_time"], req_next.ready_time)
                        sched.add_request(req_next)
                        active_reqs_by_rank[rank][req_next.request_id] = req_next
                    else:
                        status["complete"] = True
        
        for idx, scheduler in enumerate(schedulers):
            schedule_output = scheduler.schedule()
            logger.debug(f"add to batch_pipeline; len = {len(current_status[idx]['batch_pipeline'])}, output: {schedule_output}")
            current_status[idx]['batch_pipeline'].append(schedule_output)
            if schedule_output and (not scheduler.should_terminate()):
                scheduler.debug_print_schedule(schedule_output, idx, current_status[idx]['step'])

                # 2. ROSSModel Estimate Step N Finish Time
                mixed_forward_phase = get_mixed_forward_phase(schedule_output)
                for name in ['pre_forward', 'forward', 'post_forward']:
                    regression_model = get_regression_model(ross_models, name, mixed_forward_phase)
                    _time_step = regression_model.predict(
                        req_ids=schedule_output.scheduled_req_ids,
                        prefill_seq_lens=schedule_output.prefill_seq_lens,
                        decode_seq_lens=schedule_output.decode_seq_lens,
                        isl=isl, osl=osl,
                    )
                    assert(_time_step >= 0)
                    total_time_slices[idx][name] += _time_step / 1000
                    if name == 'pre_forward' and pp > 1: # pp > 1
                        continue
                    current_status[idx]['wall_time'] += _time_step / 1000          
                    logger.debug(f"[dp_{idx}]       {name} Time: {_time_step} ms")
                
            else:
                # pipeline cleared
                status = current_status[idx]
                if check_pipeline_clear(status["batch_pipeline"], status["step"], pp):
                    oom, reason = scheduler.check_oom()
                    if oom:
                        raise MemoryError(f"OOM detected on rank {idx}: {reason}")

        for idx, scheduler in enumerate(schedulers):
            step = current_status[idx]["step"]
            if step >= pp:
                if current_status[idx]['complete']:
                    continue
                current_output = current_status[idx]['batch_pipeline'][step - pp]
                # logger.debug(f"step={step}, pp={pp}, updating(len = {len(current_status[idx]['batch_pipeline'])}): {current_output}")
                scheduler.update_from_output(current_output)
                
        for rank in range(dp):
            active_items = list(active_reqs_by_rank[rank].items())
            for rid, req in active_items:
                if req.num_tokens - req.prompt_tokens <= 1:
                    active_reqs_by_rank[rank].pop(rid, None)
                    finished_reqs_count += 1
                    continue
                if update_metrics(req, current_status[rank]["wall_time"], req.arrive_time, current_status[rank]["step"]):
                    finished_reqs_count += 1
                    request_list.record_finish(rid, current_status[rank]["wall_time"])
                    active_reqs_by_rank[rank].pop(rid, None)

        all_idle = all(
            check_pipeline_clear(current_status[i]["batch_pipeline"], current_status[i]["step"], pp)
            and scheduler.should_terminate()
            for i in range(dp)
        )
        if all_idle and request_list.should_terminate_idle(finished_reqs_count):
            break

    itl_list = []
    for req in request_list:
        if not req.ttft or not req.e2e_latency:
            raise RuntimeError(f"req={req.request_id}, ttft={req.ttft}, e2e_latency={req.e2e_latency}")
        itl_list.extend(req.itl)

    # Calculate Pre-forward on PP>1
    max_wall_time = 0
    if pp == 1:
        max_wall_time = max([s["wall_time"] for s in current_status])
    else:
        for dpi, status in enumerate(current_status):        
            # PP pre-forward regression is temporarily disabled because there is
            # no supported pp>1 model for this path yet.
            total_time_slices[dpi]['pp_pre_forward'] = 0
            if pp > 1:
                max_wall_time = max(max_wall_time, status["wall_time"] + total_time_slices[dpi]['pp_pre_forward'])
        # Refine TTFT and TTLT
        for req in request_list:
            dpi = req.dp_rank
            pp_pre_forward = total_time_slices[dpi]['pp_pre_forward']

            ttft = req.ttft + pp_pre_forward / current_status[dpi]['step'] * req.ttft_step
            req.tpot = (req.e2e_latency - ttft) / (req.output_len - 1)
            req.e2e_latency += pp_pre_forward / current_status[dpi]['step'] * req.ttlt_step
            logger.debug(
                f"req={req.request_id} re-calculating metrics.. "
                f"ttft={req.ttft}, tpot={req.tpot}; step={current_status[dpi]['step']}, {req.ttft_step}, {req.ttlt_step}"
            )

    benchmarks = calulcate_benchmark_results(request_list, itl_list, max_wall_time)
    result_dict = {
        "duration": max_wall_time,
        **benchmarks,
        "timing_phases": total_time_slices,
    }
    result_dict.update({
        "dp": dp,
        "pp": pp,
        "tp": model.inference_config.tp_size,
        "tokens/s": result_dict['throughput'],
        "tokens/s/gpu": result_dict['throughput'] / dp / pp / model.inference_config.tp_size,
        "tokens/s/user": 1000.0 / result_dict['mean_tpot_ms'],
    })

    return result_dict

def load_memory_increase(path: str, filters: Dict[str, Any]) -> Dict[str, float]:
    """
    Finds torch and non-torch memory increase from a DataFrame based on filters.
    """
    query_df = pd.read_csv(path)
    for key, value in filters.items():
        if value is not None:
            query_df = query_df[query_df[key] == type(query_df[key].iloc[0])(value)]
    try:
        ret = max(query_df['non_torch_peak_increase'])
    except:
        ret = 0
    return {
        "non_torch_mem_increase": ret
    }


def run_sim(args):    
    _warn_pp_pre_forward_disabled(args)
    gpu_memory_utilization = _get_gpu_memory_utilization(args)
    scheduler_kwargs = {
        "max_num_batched_tokens": args.max_num_batched_tokens,
    }
    platform_perf = PlatformPerf(platform_perf_yaml=args.platform_perf)

    def _init_worker_config(args, phase = ""):
        def _get_config(dp_size: int, pp_size: int, tp_size: int, model_path: Dict[str, str]):
            inference_config = InferenceConfig( dp_size=dp_size, pp_size=pp_size, tp_size=tp_size )
            model = get_model(args.model_uri, inference_config)
            ross_model_dict = get_ross_models(model, platform_perf, inference_config,
                                model_path=model_path)
            return model, ross_model_dict, inference_config

        return _get_config(
            dp_size=args.dp_size,
            pp_size=getattr(args, f"{phase}pp_size") if phase == "" else 1,
            tp_size=getattr(args, f"{phase}tp_size"),
            model_path=get_ross_model_paths(args)
        )
    request_store = VirtualClientStore(
        args.frontend_path, args.request_rate, args.batch_size,
        args.dp_size, args.disaggregation,
    )

    if not args.disaggregation:
        memory_increase = load_memory_increase(args.mem_profiling_path, { "pp": args.pp_size, "tp": args.tp_size })
        model, ross_model_dict, _ = _init_worker_config(args)
        ret = run_simulation(
            model=model,
            batch_size=args.batch_size,
            request_list=request_store,
            scheduler_kwargs=scheduler_kwargs,

            memory_profiling=memory_increase,
            total_gpu_memory=platform_perf.theoretical_memory_gb * (1024 ** 3),
            gpu_memory_utilization=gpu_memory_utilization,

            ross_models=ross_model_dict,
            dp=args.dp_size, pp=args.pp_size,
            isl=args.max_prompt_len, osl=args.max_output_len,
        )
    else:
        prefill_memory_increase = load_memory_increase(args.mem_profiling_path, { "pp": 1, "tp": args.prefill_tp_size })
        decode_memory_increase  = load_memory_increase(args.mem_profiling_path, { "pp": 1, "tp": args.decode_tp_size })
        
        prefill_model, prefill_ross_model_dict, _ = _init_worker_config(args, "prefill_")
        decode_model, decode_ross_model_dict, _ = _init_worker_config(args, "decode_")

        ret = run_disagg_simulation(
            prefill_model=prefill_model,
            decode_model=decode_model,
            batch_size=args.batch_size,
            request_list=request_store,
            scheduler_kwargs=scheduler_kwargs,

            prefill_memory_profiling=prefill_memory_increase,
            decode_memory_profiling=decode_memory_increase,
            total_gpu_memory=platform_perf.theoretical_memory_gb * (1024 ** 3),
            gpu_memory_utilization=gpu_memory_utilization,

            prefill_ross_models=prefill_ross_model_dict,
            decode_ross_models=decode_ross_model_dict,
            dp=args.dp_size, pp=1,
            isl=args.max_prompt_len, osl=args.max_output_len,
        )

    ret.update({
        "gpu_memory_utilization": gpu_memory_utilization,
    })
    return ret
