#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

from pathlib import Path
import os, sys
import time
from collections import deque
import numpy as np
from typing import List, Tuple, Dict, Any

TEST_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TEST_ROOT))

from common.features import PlatformPerf
from common.models import get_model, BaseModel
from common.config import InferenceConfig
from common.kvpool import SGLKVCachePool
from common.sim_http_perf import RequestStore, VirtualClientStore

from scheduler.request import Request
from scheduler.scheduler import Scheduler, Batch
from common.ross_model import SGLROSSModel

import logging
logger = logging.getLogger(__name__)

_PLATFORM_PERF_CACHE: Dict[str, PlatformPerf] = {}
_WORKER_CONFIG_CACHE: Dict[Tuple[str, str, int, int, Tuple[Tuple[str, str], ...]], Tuple[BaseModel, Dict[str, Any], InferenceConfig]] = {}

def parse_args():
    ap = argparse.ArgumentParser("ROSS SGLANG simulator")
    ap.add_argument("--model-uri", type=str, default="")
    # Parallel Config
    ap.add_argument("--dp-size", type=int, default=1)
    ap.add_argument("--pp-size", type=int, default=1)
    ap.add_argument("--tp-size", type=int, default=1)
    # ap.add_argument("--ep-size", type=int, default=1)

    # Disaggregation Config
    ap.add_argument("--disaggregation", action='store_true', default=False)
    ap.add_argument("--prefill-tp-size", type=int, default=1)
    ap.add_argument("--prefill-pp-size", type=int, default=1)
    ap.add_argument("--decode-tp-size", type=int, default=1)

    ap.add_argument("--prefill-dp-size", type=int, default=1)
    ap.add_argument("--decode-dp-size", type=int, default=1)

    # Scheduler Config
    ap.add_argument("--schedule-policy", default="lpm", choices=["lpm","fcfs","dfs_weight","lof","random"])
    ap.add_argument("--mem-fraction-static", type=float, default=0.9)
    ap.add_argument("--chunked-prefill-size", type=int, default=16384)
    ap.add_argument("--reserved-decode-tokens", type=int, default=512)

    # Workload Config
    ap.add_argument("--batch-size", type=int, required=True)
    ap.add_argument("--max-prompt-len", type=int, required=True)
    ap.add_argument("--max-output-len", type=int, required=True)
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--dataset-path", type=str, required=True)
    ap.add_argument("--frontend-path", type=str, required=True)
    ap.add_argument("--request-rate", type=str, required=True)
    # ap.add_argument('--scheduler_config', type=str, help='Path to the Scheduler YAML configuration file.')

    # ROSS Config
    ap.add_argument('--platform-perf', type=str, required=True, help='Path to the Platform Performance file.')

    ap.add_argument('--prefill_pre_forward_path', type=str, required=True, help='Path to the Saved PRE-FORWARD Model file.')    
    ap.add_argument('--prefill_forward_path', type=str, required=True, help='Path to the Saved [Prefill] FORWARD Model file.')
    ap.add_argument('--prefill_post_forward_path', type=str, required=True, help='Path to the Saved [Prefill] POST-FORWARD Model file.')
    ap.add_argument('--decode_pre_forward_path', type=str, required=True, help='Path to the Saved PRE-FORWARD Model file.')
    ap.add_argument('--decode_forward_path', type=str, required=True, help='Path to the Saved [Decode] FORWARD Model file.')
    ap.add_argument('--decode_post_forward_path', type=str, required=True, help='Path to the Saved [Decode] POST-FORWARD Model file.')
    ap.add_argument('--cache-worker-config', action='store_true', default=False, help='Cache heavy simulator objects within each worker process.')

    return ap.parse_args()


def get_cached_platform_perf(platform_perf_yaml: str) -> PlatformPerf:
    platform_perf = _PLATFORM_PERF_CACHE.get(platform_perf_yaml)
    if platform_perf is None:
        platform_perf = PlatformPerf(platform_perf_yaml=platform_perf_yaml)
        _PLATFORM_PERF_CACHE[platform_perf_yaml] = platform_perf
    return platform_perf


def get_cached_worker_config(
    model_uri: str,
    platform_perf_yaml: str,
    tp_size: int,
    pp_size: int,
    model_path: Dict[str, str],
) -> Tuple[BaseModel, Dict[str, Any], InferenceConfig]:
    model_path_items = tuple(sorted(model_path.items()))
    cache_key = (model_uri, platform_perf_yaml, tp_size, pp_size, model_path_items)
    cached = _WORKER_CONFIG_CACHE.get(cache_key)
    if cached is not None:
        return cached

    platform_perf = get_cached_platform_perf(platform_perf_yaml)
    inference_config = InferenceConfig(tp_size=tp_size, pp_size=pp_size)
    model = get_model(model_uri, inference_config)
    ross_model_dict = get_ross_models(
        model,
        platform_perf,
        inference_config,
        model_path=model_path,
    )
    cached = (model, ross_model_dict, inference_config)
    _WORKER_CONFIG_CACHE[cache_key] = cached
    return cached

def get_ross_models(model: BaseModel,
                    platform_perf: PlatformPerf,
                    inf_config: InferenceConfig,
                    model_path: Dict[str, str]):
    model_keys = [ 'prefill_pre_forward', 'prefill_forward', 'prefill_post_forward', \
                'decode_pre_forward', "decode_forward", "decode_post_forward" ]
    model_dict = dict()
    for key in model_path.keys():
        assert(key in model_keys)

        model_dict[key] = SGLROSSModel(
            saved_model_path=model_path[key],
            platform_perf=platform_perf,
            model=model,
            inference_config=inf_config,
            regressor="xgboost" # if key.find('pre_forward') == -1 else 'linear',
        )
    return model_dict

def update_metrics(
    req: Request,
    wall_time: float,
    arrive_time: float,
    batch_size: int = 0,
    post_decode_overhead_s: float = 0.0,
) -> bool:
    if req._last_token_time is None:
        req._last_token_time = wall_time
        req.itl = []
    else:
        req.itl.append(wall_time - req._last_token_time)
        req._last_token_time = wall_time

    if req.chunk_offset >= req.prompt_tokens and not req.ttft:
        req.ttft = wall_time - arrive_time
        logger.debug(f"req={req.request_id} wall_time={wall_time:.2f}, arrive_time={arrive_time:.2f}, ttft={req.ttft}")

        # if (int(req.request_id[4:]) - 2) % 256 < 10:
        #     print(f"rid={req.request_id}, arr_time={req.arrive_time:.5f}, ttft={req.ttft:.5f}, batch_size={batch_size}")

    if req.finished and not req.e2e_latency:
        finish_wall_time = wall_time + post_decode_overhead_s
        req.e2e_latency = finish_wall_time - arrive_time
        req.tpot = (req.e2e_latency - req.ttft) / (req.decode_tokens - 1)
        return True

    return False

def calulcate_benchmark_results(request_list: List[Request], itl_list: List[float], wall_time: float) -> Dict[str, float]:
    def _calc(name, data: List[float]) -> Dict[str, float]:
        return {
            f"mean_{name}_ms": np.mean(data),
            f"median_{name}_ms": np.median(data),
            f"std_{name}_ms": np.std(data),
            f"p99_{name}_ms": np.percentile(data, 99)
        }

    ttft = [r.ttft * 1000 for r in request_list]
    # plot_and_save_distribution(ttft, 'plot/sim_ttft.png')
    tpot = [r.tpot * 1000 for r in request_list]
    itl_list = [i * 1000 for i in itl_list]
    e2e_latency = [r.e2e_latency * 1000 for r in request_list]

    num_tokens = sum([r.decode_tokens for r in request_list])
    return {
        "throughput": num_tokens / wall_time,
        **_calc("ttft", ttft),
        **_calc("tpot", tpot),
        **_calc("itl", itl_list),
        **_calc("e2e_latency", e2e_latency)
    }

def run_simulation(
    model: BaseModel,
    batch_size: int,
    request_list: VirtualClientStore,
    ross_models: Dict[str, Any],
    scheduler_kwargs: Dict[str, Any],
    total_gpu_memory: int | None = None,
    pp: int = 1,
    dp: int = 1,
    mem_fraction_static: float = 0.9,
    post_decode_overhead_ms: float = 0.0,
):
    schedulers : List[Scheduler] = []
    prefill_phases, decode_phases = [], []

    for idx in range(dp):
        kv_pool = SGLKVCachePool(
            model=model,
            num_reqs=batch_size,
            tokens_per_block=1, # assign tokens_per_block = 1
            total_gpu_memory=total_gpu_memory,
            gpu_memory_utilization=mem_fraction_static,
            framework='sglang',
        )
        schedulers.append(Scheduler(
            waiting_queue=[],
            kv_pool=kv_pool,
            **scheduler_kwargs,
            pp=pp
        ))
        prefill_phases.append({  "prefill_pre_forward": 0, "prefill_forward": 0, "prefill_post_forward": 0, })
        decode_phases.append({  "decode_pre_forward": 0, "decode_forward": 0, "decode_post_forward": 0, })

    current_status = [ { "wall_time": 0.0, "step": 0, "batch_pipeline": [], "complete": False } for i in range(dp)]
    pending_by_rank = [deque() for _ in range(dp)]
    finished_reqs_count = 0
    active_reqs_by_rank = [dict() for _ in range(dp)]  # rid -> Request, bounded by inflight concurrency
    post_decode_overhead_s = post_decode_overhead_ms / 1000.0

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
                sched.waiting_queue.append(req)
                active_reqs_by_rank[rank][req.request_id] = req
            if not status["complete"]:
                if check_sched_idle(sched, status["batch_pipeline"], status["step"], pp):
                    if pending:
                        req_next = pending.popleft()
                        status["wall_time"] = max(status["wall_time"], req_next.ready_time)
                        sched.waiting_queue.append(req_next)
                        active_reqs_by_rank[rank][req_next.request_id] = req_next

                        logger.debug(f"[dp_{rank}] step: {current_status[rank]['step']}, wall_time = {status['wall_time']:.2f}")
                    else:
                        status["complete"] = True
        for idx, scheduler in enumerate(schedulers):
            batch = scheduler.get_next_batch_to_run()
            current_status[idx]['batch_pipeline'].append((batch, scheduler.running_batch))
            cur_batch_size = 0 if not batch else len(batch.reqs)
            if batch is not None:
                logger.debug(f"[dp_{idx}] step: { current_status[idx]['step'] }, wall_time = {current_status[idx]['wall_time']:.2f}\n"
                            f"num_free_blocks={scheduler.kv_allocator.num_free_blocks} {batch}\n----")
                # 2. ROSSModel Estimate Step N Duration
                mode = batch.forward_mode
                req_ids = [r.request_id for r in batch.reqs]
                seq_lens = [r.num_computed_tokens for r in batch.reqs]

                for name in ['pre_forward', 'forward', 'post_forward']:
                    if mode == "prefill":
                        _time_step = ross_models["prefill_" + name].predict(req_ids=req_ids, seq_lens=seq_lens)
                        prefill_phases[idx]["prefill_" + name] += _time_step / 1000
                    elif mode == "decode":
                        _time_step = ross_models["decode_" + name].predict(req_ids=req_ids, seq_lens=seq_lens)
                        decode_phases[idx]["decode_" + name] += _time_step / 1000
                    else:
                        assert(0)
                    current_status[idx]['wall_time'] += _time_step / 1000
                    logger.debug(f"    stage={name}, time spent: {_time_step:.2f} ms")
            else:
                # pipeline cleared
                status = current_status[idx]
                if not status['batch_pipeline'] or status['step'] - pp >= len(status['batch_pipeline']):
                    oom, reason = scheduler.check_oom()
                    if oom:
                        raise MemoryError(f"[Prefill] OOM detected on rank {idx}: {reason}")

        for idx, scheduler in enumerate(schedulers):
            step = current_status[idx]["step"]
            if step >= pp:
                if current_status[idx]['complete']:
                    continue
                current_batch, cur_running_batch = current_status[idx]['batch_pipeline'][step - pp]
                if current_batch is not None:
                    # logger.debug(f"UPDATING current_batch {[r.request_id for r in current_batch.reqs]}")
                    current_mode = current_batch.forward_mode
                    if current_mode == "prefill":
                        scheduler.process_batch_result_prefill(current_batch, cur_running_batch)
                    else:
                        scheduler.process_batch_result_decode(current_batch)          

        for rank in range(dp):
            active_items = list(active_reqs_by_rank[rank].items())
            for rid, req in active_items:
                if req.max_new_tokens <= 1:
                    active_reqs_by_rank[rank].pop(rid, None)
                    finished_reqs_count += 1
                    continue
                if update_metrics(
                    req,
                    current_status[rank]["wall_time"],
                    req.arrive_time,
                    cur_batch_size,
                    post_decode_overhead_s,
                ):
                    finished_reqs_count += 1
                    request_list.record_finish(rid, current_status[rank]["wall_time"] + post_decode_overhead_s)
                    active_reqs_by_rank[rank].pop(rid, None)

        all_idle = all(
            check_sched_idle(schedulers[i], current_status[i]["batch_pipeline"], current_status[i]["step"], pp)
            for i in range(dp)
        )
        if all_idle and request_list.should_terminate_idle(finished_reqs_count):
            break

    itl_list = []
    for req in request_list:
        if not req.ttft or not req.e2e_latency:
            raise RuntimeError(f"req={req.request_id}, ttft={req.ttft}, e2e_latency={req.e2e_latency}")
        itl_list.extend(req.itl)

    max_wall_time = max([status["wall_time"] for status in current_status])
    benchmarks = calulcate_benchmark_results(request_list, itl_list, max_wall_time)
    result_dict = {
        "duration": max_wall_time,
        **benchmarks,
        "prefill_phases": prefill_phases,
        "decode_phases": decode_phases,
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

def check_sched_idle(sched: Scheduler, pipeline: List[Batch], step: int, pp: int):
    def _pipeline_clear(pipeline: List[Batch], step: int, pp: int):
        if not pipeline or step - pp >= len(pipeline):
            return True
        for i in range(1, min(pp + 1, len(pipeline))):
            pipeline_i = pipeline[len(pipeline) - i]
            if pipeline_i and pipeline_i[0] is not None and not pipeline_i[0].is_empty(): # not consider current running batch
                return False
        return True
    pipeline_clear = _pipeline_clear(pipeline, step, pp)
    ret = (
        sched.running_batch.is_empty()
        and not sched.running_queue
        and not sched.waiting_queue
        and not sched.pending_decode_queue
        and pipeline_clear
    )
    # logger.debug(f"len_pipeline = {len(pipeline)}, running_batch = {sched.running_batch.reqs}, pipeline_clear = {pipeline_clear}, ret={ret}")
    return ret

def run_sim(args):
    scheduler_kwargs = {
        "chunked_prefill_size": args.chunked_prefill_size,
        "reserved_decode_tokens": args.reserved_decode_tokens,
        "max_running_requests": args.batch_size,
    }
    if args.model_uri.lower().find("deepseek") != -1:
        scheduler_kwargs.update({"page_size": 64})

    use_cache = getattr(args, "cache_worker_config", False)
    platform_perf = (
        get_cached_platform_perf(args.platform_perf)
        if use_cache
        else PlatformPerf(platform_perf_yaml=args.platform_perf)
    )

    def _init_worker_config(tp_size: int, pp_size: int, model_path: Dict[str, str]):
        inference_config = InferenceConfig(tp_size=tp_size, pp_size=pp_size)
        model = get_model(args.model_uri, inference_config)
        ross_model_dict = get_ross_models(
            model,
            platform_perf,
            inference_config,
            model_path=model_path,
        )
        return model, ross_model_dict, inference_config

    dp_size = args.dp_size if not args.disaggregation else args.prefill_dp_size
    request_store = VirtualClientStore(
        args.frontend_path, args.request_rate, args.batch_size,
        dp_size, args.disaggregation,
    )
    if not args.disaggregation:
        model_path = {
            "prefill_pre_forward": args.prefill_pre_forward_path,
            "prefill_forward": args.prefill_forward_path,
            "prefill_post_forward": args.prefill_post_forward_path,
            "decode_pre_forward": args.decode_pre_forward_path,
            "decode_forward": args.decode_forward_path,
            "decode_post_forward": args.decode_post_forward_path,
        }
        if use_cache:
            model, ross_model_dict, _ = get_cached_worker_config(
                model_uri=args.model_uri,
                platform_perf_yaml=args.platform_perf,
                tp_size=args.tp_size,
                pp_size=args.pp_size,
                model_path=model_path,
            )
        else:
            model, ross_model_dict, _ = _init_worker_config(
                tp_size=args.tp_size,
                pp_size=args.pp_size,
                model_path=model_path,
            )
        ret = run_simulation(
            model=model,
            batch_size=args.batch_size,
            request_list=request_store,

            scheduler_kwargs=scheduler_kwargs,
            mem_fraction_static=args.mem_fraction_static,
            total_gpu_memory=platform_perf.theoretical_memory_gb * (1024 ** 3),

            ross_models=ross_model_dict,
            dp=args.dp_size, pp=args.pp_size,
            post_decode_overhead_ms=0,
        )
    else:
        raise RuntimeError("Run DISAGG in simulator_aligned.py")

    ret.update({
        "mem_fraction_static": args.mem_fraction_static,
        "chunked_prefill_size": args.chunked_prefill_size
    })
    return ret

if __name__ == "__main__":
    args = parse_args()
    ret = run_sim(args)
    print(f"[SIM] result={ret}")
