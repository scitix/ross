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

TEST_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TEST_ROOT))

from common.features import PlatformPerf
from common.models import get_model, BaseModel
from common.config import InferenceConfig
from common.kvpool import KVCachePool
from common.ross_model import ROSSModel
from common.sim_transport import RequestStore, VirtualClientStore

from scheduler.request import Request, RequestStatus
from scheduler.scheduler import Scheduler, SchedulerOutput

import logging
logger = logging.getLogger(__name__)

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
    ap.add_argument('--pp_pre_forward_path', type=str, help='Path to the Saved PRE-FORWARD Model file on PP>1.')    

    ap.add_argument('--forward_path', type=str, help='Path to the Saved [Prefill] FORWARD Model file.')
    ap.add_argument('--post_forward_path', type=str, help='Path to the Saved [Prefill] POST-FORWARD Model file.')

    # ap.add_argument('--prefill_pre_forward_path', type=str, help='Path to the Saved PRE-FORWARD Model file.')    
    # ap.add_argument('--decode_pre_forward_path', type=str, help='Path to the Saved PRE-FORWARD Model file.')    
    # ap.add_argument('--prefill_forward_path', type=str, help='Path to the Saved [Prefill] FORWARD Model file.')
    # ap.add_argument('--decode_forward_path', type=str, help='Path to the Saved [Decode] FORWARD Model file.')
    # ap.add_argument('--prefill_post_forward_path', type=str, help='Path to the Saved [Prefill] POST-FORWARD Model file.')
    # ap.add_argument('--decode_post_forward_path', type=str, help='Path to the Saved [Decode] POST-FORWARD Model file.')

    return ap.parse_args()

def get_ross_models(model: BaseModel,
                    platform_perf: PlatformPerf,
                    inf_config: InferenceConfig,
                    model_path: Dict[str, str]):
    model_keys = [ 'pre_forward', 'forward', 'post_forward', ]
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
    pp_pre_model: ROSSModel,
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
                for name in ['pre_forward', 'forward', 'post_forward']:
                    regression_model = ross_models[name]
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
            total_time_slices[dpi]['pp_pre_forward'] = pp_pre_model.predict(
                req_ids=["req_id"] * batch_size,
                prefill_seq_lens=[], decode_seq_lens=[0 for i in range(status['step'])],
                isl=isl, osl=osl,
            )
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

def run_decode_simulation(
    decode_model: BaseModel,
    batch_size: int,
    request_list: VirtualClientStore,
    ross_models: Dict[str, Any],
    pp_pre_model: ROSSModel,
    scheduler_kwargs: Dict[str, Any],
    isl: int,
    osl: int,
    gpu_memory_utilization: float,
    memory_profiling: Dict[str, Any] = None,
    total_gpu_memory: int | None = None,
    dp: int = 1, # vllm: p & d have same dp_size
    pp: int = 1,
    
) -> Dict[str, Any]:
    decode_schedulers : List[Scheduler] = []
    total_time_slices = [{ "pre_forward": 0, "forward": 0, "post_forward": 0, "pp_pre_forward": 0 } for i in range(dp)]

    tokens_per_block = 16 if decode_model.model_uri.lower().find('deepseek') == -1 else 64
    for idx in range(dp):
        kv_pool = KVCachePool(
            model=decode_model,
            num_reqs=batch_size,
            tokens_per_block=tokens_per_block,
            total_gpu_memory=total_gpu_memory,
            gpu_memory_utilization=gpu_memory_utilization,
            vllm_non_torch_increase=memory_profiling['non_torch_mem_increase']
        )
        decode_schedulers.append(Scheduler(
            max_running_reqs=batch_size,
            kv_pool=kv_pool,
            **scheduler_kwargs
        ))

    current_status = [ { "wall_time": 0.0, "step": 0, "batch_pipeline": [], "complete": False } for i in range(dp)]
    pending_by_rank = [deque() for _ in range(dp)]
    active_reqs_by_rank = [dict() for _ in range(dp)]
    finished_reqs_count = 0
    for req in request_list:
        pending_by_rank[req.decode_dp_rank].append(req)

    while True:
        for rank, sched in enumerate(decode_schedulers):
            status = current_status[rank]
            status["step"] += 1
            pending = pending_by_rank[rank]
            while pending and pending[0].prefill_end_time <= status["wall_time"]:
                req = pending.popleft()
                if req.num_tokens - req.prompt_tokens <= 1:
                    update_metrics(req, status["wall_time"], req.arrive_time, status["step"])
                    finished_reqs_count += 1
                    request_list.record_finish(req.request_id, req.prefill_end_time)
                    continue
                req.decode_init()
                sched.add_request(req)
                active_reqs_by_rank[rank][req.request_id] = req
            if not status["complete"]:
                if check_pipeline_clear(status["batch_pipeline"], status["step"], pp) and sched.should_terminate():
                    if pending:
                        while True:
                            req_next = pending.popleft()
                            status["wall_time"] = max(status["wall_time"], req_next.prefill_end_time)
                            if req_next.num_tokens - req_next.prompt_tokens <= 1:
                                update_metrics(req_next, status["wall_time"], req_next.arrive_time, status["step"])
                                finished_reqs_count += 1
                                request_list.record_finish(req_next.request_id, req_next.prefill_end_time)
                                continue
                            req_next.decode_init()
                            sched.add_request(req_next)
                            active_reqs_by_rank[rank][req_next.request_id] = req_next
                            break
                    else:
                        status["complete"] = True

        for idx, scheduler in enumerate(decode_schedulers):
            schedule_output = scheduler.schedule()
            logger.debug(
                f"dp_{idx}, running: {len(scheduler.running)}, waiting: {len(scheduler.waiting)}, wall_time = {[s['wall_time'] for s in current_status]}"
            )
            current_status[idx]['batch_pipeline'].append(schedule_output)

            if schedule_output and (not scheduler.should_terminate()):
                scheduler.debug_print_schedule(schedule_output, idx, current_status[idx]['step'])

                # 2. ROSSModel Estimate Step N Finish Time
                for name in ['pre_forward', 'forward', 'post_forward']:
                    regression_model = ross_models[name]
                    _time_step = regression_model.predict(
                        req_ids=schedule_output.scheduled_req_ids,
                        prefill_seq_lens=schedule_output.prefill_seq_lens,
                        decode_seq_lens=schedule_output.decode_seq_lens,
                        isl=isl, osl=osl,
                    )
                    current_status[idx]['wall_time'] += _time_step / 1000
                    total_time_slices[idx][name] += _time_step / 1000                    
                    logger.debug(f"[dp_{idx}]          {name} Time: {_time_step} ms")
            else:
                # pipeline cleared
                status = current_status[idx]
                if check_pipeline_clear(status["batch_pipeline"], status["step"], pp):
                    oom, reason = scheduler.check_oom()
                    if oom:
                        raise MemoryError(f"OOM detected on rank {idx}: {reason}")

        for idx, scheduler in enumerate(decode_schedulers):
            # 3. Update Step (N - PP + 1) Model's Output
            step = current_status[idx]["step"]
            if step >= pp:
                if current_status[idx]['complete']:
                    continue
                current_output = current_status[idx]['batch_pipeline'][step - pp]
                scheduler.update_from_output(current_output)

        for scheduler in decode_schedulers:
            scheduler.running = [req for req in scheduler.running if req.status != RequestStatus.FINISHED]

        for rank in range(dp):
            active_items = list(active_reqs_by_rank[rank].items())
            for rid, req in active_items:
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
            raise RuntimeError(f"req={req.request_id}, ttft={req.ttft}, ttlt={req.e2e_latency}")
        itl_list.extend(req.itl)

    max_wall_time = 0
    for dpi, status in enumerate(current_status):
        total_time_slices[dpi]['pp_pre_forward'] = pp_pre_model.predict(
            req_ids=["req_id"] * batch_size,
            prefill_seq_lens=[], decode_seq_lens=[0 for i in range(status['step'])],
            isl=isl, osl=osl,
        )
        if pp > 1:
            max_wall_time = max(max_wall_time, status["wall_time"] + total_time_slices[dpi]['pp_pre_forward'])
        else:
            max_wall_time = max(max_wall_time, status["wall_time"])

    benchmarks = calulcate_benchmark_results(request_list, itl_list, max_wall_time)
    result_dict = {
        "duration": max_wall_time,
        **benchmarks,
        "decode_phases": total_time_slices,
    }
    result_dict.update({
        "tokens/s": result_dict['throughput'],
        "tokens/s/gpu": result_dict['throughput'] / dp / pp / decode_model.inference_config.tp_size,
        "tokens/s/user": 1000.0 / result_dict['mean_tpot_ms'],
    })
    return result_dict


def run_disagg_simulation(
    prefill_model: BaseModel,
    decode_model: BaseModel,
    batch_size: int,
    request_list: VirtualClientStore,
    ross_models: Dict[str, Any],
    pp_pre_model: ROSSModel,
    scheduler_kwargs: Dict[str, Any],
    isl: int,
    osl: int,
    gpu_memory_utilization: float,
    prefill_memory_profiling: Dict[str, Any] = None,
    decode_memory_profiling: Dict[str, Any] = None,
    total_gpu_memory: int | None = None,
    dp: int = 1,
    pp: int = 1,
) -> Dict[str, Any]:
    prefill_schedulers: List[Scheduler] = []
    decode_schedulers: List[Scheduler] = []
    prefill_time_slices = [{"pre_forward": 0, "forward": 0, "post_forward": 0} for _ in range(dp)]
    decode_time_slices = [{"pre_forward": 0, "forward": 0, "post_forward": 0, "pp_pre_forward": 0} for _ in range(dp)]

    prefill_tokens_per_block = 16 if prefill_model.model_uri.lower().find('deepseek') == -1 else 64
    decode_tokens_per_block = 16 if decode_model.model_uri.lower().find('deepseek') == -1 else 64
    for idx in range(dp):
        prefill_kv_pool = KVCachePool(
            model=prefill_model,
            num_reqs=batch_size,
            tokens_per_block=prefill_tokens_per_block,
            total_gpu_memory=total_gpu_memory,
            gpu_memory_utilization=gpu_memory_utilization,
            vllm_non_torch_increase=prefill_memory_profiling['non_torch_mem_increase']
        )
        prefill_schedulers.append(Scheduler(
            max_running_reqs=batch_size,
            kv_pool=prefill_kv_pool,
            **scheduler_kwargs
        ))

        decode_kv_pool = KVCachePool(
            model=decode_model,
            num_reqs=batch_size,
            tokens_per_block=decode_tokens_per_block,
            total_gpu_memory=total_gpu_memory,
            gpu_memory_utilization=gpu_memory_utilization,
            vllm_non_torch_increase=decode_memory_profiling['non_torch_mem_increase']
        )
        decode_schedulers.append(Scheduler(
            max_running_reqs=batch_size,
            kv_pool=decode_kv_pool,
            **scheduler_kwargs
        ))

    prefill_status = [{"wall_time": 0.0, "step": 0, "batch_pipeline": [], "complete": False} for _ in range(dp)]
    decode_status  = [{"wall_time": 0.0, "step": 0, "batch_pipeline": [], "complete": False} for _ in range(dp)]
    pending_prefill_by_rank = [deque() for _ in range(dp)]
    pending_decode_by_rank  = [deque() for _ in range(dp)]
    active_prefill_by_rank  = [dict() for _ in range(dp)]
    active_decode_by_rank   = [dict() for _ in range(dp)]

    finished_reqs_count, req_batch, decode_assign_idx = 0, 3, 0
    while True:
        current_global_time = max(
            max(status["wall_time"] for status in prefill_status),
            max(status["wall_time"] for status in decode_status),
        )
        new_reqs = request_list.refresh(current_global_time, disaggregation=True)
        for req in new_reqs:
            pending_prefill_by_rank[req.prefill_dp_rank].append(req)
            prefill_status[req.prefill_dp_rank]["complete"] = False

        # --- Prefill RUN ---
        for rank, sched in enumerate(prefill_schedulers):
            status = prefill_status[rank]
            status["step"] += 1
            pending = pending_prefill_by_rank[rank]
            while pending and pending[0].ready_time <= status["wall_time"]:
                req = pending.popleft()
                sched.add_request(req)
                active_prefill_by_rank[rank][req.request_id] = req
            if not status["complete"]:
                if check_pipeline_clear(status["batch_pipeline"], status["step"], pp) and sched.should_terminate():
                    if pending:
                        req_next = pending.popleft()
                        status["wall_time"] = max(status["wall_time"], req_next.ready_time)
                        sched.add_request(req_next)
                        active_prefill_by_rank[rank][req_next.request_id] = req_next
                    else:
                        status["complete"] = True

        for idx, scheduler in enumerate(prefill_schedulers):
            schedule_output = scheduler.schedule()
            prefill_status[idx]["batch_pipeline"].append(schedule_output)

            if schedule_output and (not scheduler.should_terminate()):
                scheduler.debug_print_schedule(schedule_output, idx, prefill_status[idx]["step"])
                for name in ["pre_forward", "forward", "post_forward"]:
                    regression_model = ross_models[name]
                    _time_step = regression_model.predict(
                        req_ids=schedule_output.scheduled_req_ids,
                        prefill_seq_lens=schedule_output.prefill_seq_lens,
                        decode_seq_lens=schedule_output.decode_seq_lens,
                        isl=isl,
                        osl=osl,
                    )
                    prefill_status[idx]["wall_time"] += _time_step / 1000
                    prefill_time_slices[idx][name] += _time_step / 1000
            else:
                status = prefill_status[idx]
                if check_pipeline_clear(status["batch_pipeline"], status["step"], pp):
                    oom, reason = scheduler.check_oom()
                    if oom:
                        raise MemoryError(f"OOM detected on prefill rank {idx}: {reason}")

        for idx, scheduler in enumerate(prefill_schedulers):
            step = prefill_status[idx]["step"]
            if step >= pp:
                if prefill_status[idx]["complete"]:
                    continue
                current_output = prefill_status[idx]["batch_pipeline"][step - pp]
                scheduler.update_from_output(current_output)

        # --- Decode Add Request ---
        for rank in range(dp):
            active_items = list(active_prefill_by_rank[rank].items())
            for rid, req in active_items:
                if not req.prefill_end_time and req.output_len == 1:
                    req.prefill_end_time = prefill_status[rank]["wall_time"]
                    req.status = RequestStatus.FINISHED
                    prefill_schedulers[rank].kv_cache_manager.free(req)
                    active_prefill_by_rank[rank].pop(rid, None)
                    # TODO: TTFT should be calculated after prefill; not decode #1
                    if req.num_tokens - req.prompt_tokens <= 1:
                        update_metrics(req, status["wall_time"], req.arrive_time, status["step"])
                        finished_reqs_count += 1
                        request_list.record_finish(req.request_id, req.prefill_end_time)
                        continue
                    req.decode_dp_rank = (decode_assign_idx // req_batch) % dp
                    decode_assign_idx += 1
                    pending_decode_by_rank[req.decode_dp_rank].append(req)
                    decode_status[req.decode_dp_rank]["complete"] = False

        for scheduler in prefill_schedulers:
            scheduler.running = [req for req in scheduler.running if req.status != RequestStatus.FINISHED]

        # --- Decode RUN
        for rank, sched in enumerate(decode_schedulers):
            status = decode_status[rank]
            status["step"] += 1
            pending = pending_decode_by_rank[rank]
            while pending and pending[0].prefill_end_time <= status["wall_time"]:
                req = pending.popleft()
                req.decode_init()
                sched.add_request(req)
                active_decode_by_rank[rank][req.request_id] = req
            if not status["complete"]:
                if check_pipeline_clear(status["batch_pipeline"], status["step"], pp) and sched.should_terminate():
                    if pending:
                        req_next = pending.popleft()
                        status["wall_time"] = max(status["wall_time"], req_next.prefill_end_time)
                        req_next.decode_init()
                        sched.add_request(req_next)
                        active_decode_by_rank[rank][req_next.request_id] = req_next
                    else:
                        status["complete"] = True

        for idx, scheduler in enumerate(decode_schedulers):
            schedule_output = scheduler.schedule()
            decode_status[idx]["batch_pipeline"].append(schedule_output)

            if schedule_output and (not scheduler.should_terminate()):
                scheduler.debug_print_schedule(schedule_output, idx, decode_status[idx]["step"])
                for name in ["pre_forward", "forward", "post_forward"]:
                    regression_model = ross_models[name]
                    _time_step = regression_model.predict(
                        req_ids=schedule_output.scheduled_req_ids,
                        prefill_seq_lens=schedule_output.prefill_seq_lens,
                        decode_seq_lens=schedule_output.decode_seq_lens,
                        isl=isl, osl=osl,
                    )
                    decode_status[idx]["wall_time"] += _time_step / 1000
                    decode_time_slices[idx][name] += _time_step / 1000
            else:
                status = decode_status[idx]
                if check_pipeline_clear(status["batch_pipeline"], status["step"], pp):
                    oom, reason = scheduler.check_oom()
                    if oom:
                        raise MemoryError(f"OOM detected on decode rank {idx}: {reason}")

        for idx, scheduler in enumerate(decode_schedulers):
            step = decode_status[idx]["step"]
            if step >= pp:
                if decode_status[idx]["complete"]:
                    continue
                current_output = decode_status[idx]["batch_pipeline"][step - pp]
                scheduler.update_from_output(current_output)

        for scheduler in decode_schedulers:
            scheduler.running = [req for req in scheduler.running if req.status != RequestStatus.FINISHED]

        for rank in range(dp):
            active_items = list(active_decode_by_rank[rank].items())
            for rid, req in active_items:
                if update_metrics(req, decode_status[rank]["wall_time"], req.arrive_time, decode_status[rank]["step"]):
                    finished_reqs_count += 1
                    request_list.record_finish(rid, decode_status[rank]["wall_time"])
                    active_decode_by_rank[rank].pop(rid, None)

        all_decode_idle = all(
            check_pipeline_clear(decode_status[i]["batch_pipeline"], decode_status[i]["step"], pp)
            and scheduler.should_terminate()
            and not pending_decode_by_rank[i]
            and not active_decode_by_rank[i]
            for i, scheduler in enumerate(decode_schedulers)
        )
        if all_decode_idle and request_list.should_terminate_idle(finished_reqs_count):
            break

    itl_list = []
    for req in request_list:
        if not req.prefill_end_time or not req.ttft or not req.e2e_latency:
            raise RuntimeError(
                f"req={req.request_id}, prefill_end_time={req.prefill_end_time}, ttft={req.ttft}, e2e_latency={req.e2e_latency}"
            )
        itl_list.extend(req.itl)

    max_wall_time = 0
    for dpi, status in enumerate(decode_status):
        decode_time_slices[dpi]["pp_pre_forward"] = pp_pre_model.predict(
            req_ids=["req_id"] * batch_size,
            prefill_seq_lens=[], decode_seq_lens=[0 for _ in range(status["step"])],
            isl=isl, osl=osl,
        )
        if pp > 1:
            max_wall_time = max(max_wall_time, status["wall_time"] + decode_time_slices[dpi]["pp_pre_forward"])
        else:
            max_wall_time = max(max_wall_time, status["wall_time"])

    benchmarks = calulcate_benchmark_results(request_list, itl_list, max_wall_time)
    result_dict = {
        "duration": max_wall_time,
        **benchmarks,
        "decode_phases": decode_time_slices,
        "prefill_result": {
            "wall_time": max(status["wall_time"] for status in prefill_status),
            "timing_phases": prefill_time_slices,
        }
    }
    result_dict.update({
        "tokens/s": result_dict["throughput"],
        "tokens/s/gpu": result_dict["throughput"] / dp / pp / decode_model.inference_config.tp_size,
        "tokens/s/user": 1000.0 / result_dict["mean_tpot_ms"],
    })
    return result_dict


def run_prefill_simulation(
    prefill_model: BaseModel,
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

    prefill_schedulers : List[Scheduler] = []
    total_time_slices = [{ "pre_forward": 0, "forward": 0, "post_forward": 0 } for i in range(dp)]

    tokens_per_block = 16 if prefill_model.model_uri.lower().find('deepseek') == -1 else 64
    for idx in range(dp):
        kv_pool = KVCachePool(
            model=prefill_model,
            num_reqs=batch_size,
            tokens_per_block=tokens_per_block,
            total_gpu_memory=total_gpu_memory,
            gpu_memory_utilization=gpu_memory_utilization,
            vllm_non_torch_increase=memory_profiling['non_torch_mem_increase']
        )
        prefill_schedulers.append(Scheduler(
            max_running_reqs=batch_size,
            kv_pool=kv_pool,
            **scheduler_kwargs
        ))
    current_status = [ { "wall_time": 0.0, "step": 0, "batch_pipeline": [], "complete": False } for i in range(dp)]
    pending_by_rank = [deque() for _ in range(dp)]
    prefilled_count = 0
    active_reqs_by_rank = [dict() for _ in range(dp)]  # rid -> Request, bounded by inflight concurrency

    while True:
        current_global_time = max(status["wall_time"] for status in current_status)
        new_reqs = request_list.refresh(current_global_time, disaggregation=True)
        for r in new_reqs:
            pending_by_rank[r.prefill_dp_rank].append(r)
            current_status[r.prefill_dp_rank]["complete"] = False

        for rank, sched in enumerate(prefill_schedulers):
            status = current_status[rank]
            status["step"] += 1
            pending = pending_by_rank[rank]
            while pending and pending[0].ready_time <= status["wall_time"]:
                req = pending.popleft()
                sched.add_request(req)
                active_reqs_by_rank[rank][req.request_id] = req
            if not status["complete"]:
                if check_pipeline_clear(status["batch_pipeline"], status["step"], pp) and sched.should_terminate():
                    if pending:
                        req_next = pending.popleft()
                        status["wall_time"] = max(status["wall_time"], req_next.ready_time)
                        sched.add_request(req_next)
                        active_reqs_by_rank[rank][req_next.request_id] = req_next
                    else:
                        status["complete"] = True

    
        # 1. Schedule
        for idx, scheduler in enumerate(prefill_schedulers):
            schedule_output = scheduler.schedule()
            logger.debug(
                f"dp_{idx}, running: {[r.request_id for r in scheduler.running]},"
                f"waiting: {[r.request_id for r in scheduler.waiting]}"
            )
            current_status[idx]['batch_pipeline'].append(schedule_output)

            if schedule_output and (not scheduler.should_terminate()):
                scheduler.debug_print_schedule(schedule_output, idx, current_status[idx]['step'])

                # 2. ROSSModel Estimate Step N Finish Time
                for name in ['pre_forward', 'forward', 'post_forward']:
                    regression_model = ross_models[name]
                    _time_step = regression_model.predict(
                        req_ids=schedule_output.scheduled_req_ids,
                        prefill_seq_lens=schedule_output.prefill_seq_lens,
                        decode_seq_lens=schedule_output.decode_seq_lens,
                        isl=isl,
                        osl=osl,
                    )
                    current_status[idx]['wall_time'] += _time_step / 1000
                    total_time_slices[idx][name] += _time_step / 1000                    
                    logger.debug(f"[dp_{idx}]          {name} Time: {_time_step} ms")
            else:
                # pipeline cleared
                status = current_status[idx]
                if check_pipeline_clear(status["batch_pipeline"], status["step"], pp):
                    oom, reason = scheduler.check_oom()
                    if oom:
                        raise MemoryError(f"OOM detected on rank {idx}: {reason}")

        for idx, scheduler in enumerate(prefill_schedulers):
            # 3. Update Step (N - PP + 1) Model's Output
            step = current_status[idx]["step"]
            if step >= pp:
                if current_status[idx]['complete']:
                    continue
                current_output = current_status[idx]['batch_pipeline'][step - pp]
                scheduler.update_from_output(current_output)

        for rank in range(dp):
            active_items = list(active_reqs_by_rank[rank].items())
            for rid, req in active_items:
                if not req.prefill_end_time and req.output_len == 1:
                    req.prefill_end_time = current_status[req.prefill_dp_rank]['wall_time']
                    req.status = RequestStatus.FINISHED
                    prefill_schedulers[req.prefill_dp_rank].kv_cache_manager.free(req)
                    prefilled_count += 1
                    active_reqs_by_rank[rank].pop(rid, None)

        for scheduler in prefill_schedulers:
            scheduler.running = [req for req in scheduler.running if req.status != RequestStatus.FINISHED]

        all_idle = all(
            check_pipeline_clear(current_status[i]["batch_pipeline"], current_status[i]["step"], pp)
            and scheduler.should_terminate()
            for i, scheduler in enumerate(prefill_schedulers)
        )
        if all_idle and request_list.should_terminate_idle(prefilled_count):
            break

    for req in request_list:
        if not req.prefill_end_time or req.ttft is not None:
            raise RuntimeError(f"req={req.request_id}, prefill_time={req.prefill_end_time}, ttft={req.ttft}")

    max_wall_time = max([status["wall_time"] for status in current_status])
    result_dict = {
        "duration": max_wall_time,
        "total_time_slices": total_time_slices,
        "requests": list(request_list),
    }
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


def run_sim(args, trace_configs = None):    
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
            model_path={
                "pre_forward": getattr(args, f"pre_forward_path"),
                "forward": getattr(args, f"forward_path"),
                "post_forward": getattr(args, f"post_forward_path"),
            }
        )
    request_store = VirtualClientStore(
        args.frontend_path, args.request_rate, args.batch_size,
        args.dp_size, args.disaggregation,
    )

    if not args.disaggregation:
        memory_increase = load_memory_increase(args.mem_profiling_path, { "pp": args.pp_size, "tp": args.tp_size })
        model, ross_model_dict, _ = _init_worker_config(args)
        pp_pre_model = ROSSModel(
            saved_model_path=args.pp_pre_forward_path,
            platform_perf=platform_perf,
            model=model,
            inference_config=model.inference_config,
            regressor="xgboost",
        )
        ret = run_simulation(
            model=model,
            batch_size=args.batch_size,
            request_list=request_store,
            scheduler_kwargs=scheduler_kwargs,

            memory_profiling=memory_increase,
            total_gpu_memory=platform_perf.theoretical_memory_gb * (1024 ** 3),
            gpu_memory_utilization=args.gpu_memory_utilization,

            ross_models=ross_model_dict,
            pp_pre_model=pp_pre_model,
            dp=args.dp_size, pp=args.pp_size,
            isl=args.max_prompt_len, osl=args.max_output_len,
        )
    else:
        prefill_memory_increase = load_memory_increase(args.mem_profiling_path, { "pp": 1, "tp": args.prefill_tp_size })
        decode_memory_increase  = load_memory_increase(args.mem_profiling_path, { "pp": 1, "tp": args.decode_tp_size })
        
        prefill_model, ross_model_dict, _ = _init_worker_config(args, "prefill_")
        decode_model, _, _ = _init_worker_config(args, "decode_")
        
        pp_pre_model = ROSSModel(
            saved_model_path=args.pp_pre_forward_path,
            platform_perf=platform_perf,
            model=decode_model,
            inference_config=decode_model.inference_config,
            regressor="xgboost",
        )

        ret = run_disagg_simulation(
            prefill_model=prefill_model,
            decode_model=decode_model,
            batch_size=args.batch_size,
            request_list=request_store,
            scheduler_kwargs=scheduler_kwargs,

            prefill_memory_profiling=prefill_memory_increase,
            decode_memory_profiling=decode_memory_increase,
            total_gpu_memory=platform_perf.theoretical_memory_gb * (1024 ** 3),
            gpu_memory_utilization=args.gpu_memory_utilization,

            ross_models=ross_model_dict,
            pp_pre_model=pp_pre_model,
            dp=args.dp_size, pp=1,
            isl=args.max_prompt_len, osl=args.max_output_len,
        )

    ret.update({
        "gpu_memory_utilization": args.gpu_memory_utilization,
    })
    return ret
