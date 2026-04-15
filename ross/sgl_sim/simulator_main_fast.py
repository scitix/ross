#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import sys
from collections import deque
from typing import Dict, Any, List, Tuple

TEST_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TEST_ROOT))

from common.sim_http_perf import VirtualClientStore
from common.kvpool import SGLKVCachePool

from scheduler.request import Request
from scheduler.scheduler import Scheduler, Batch

from simulator_main import (
    parse_args,
    get_cached_platform_perf,
    get_cached_worker_config,
    get_ross_models,
    get_model,
    PlatformPerf,
    InferenceConfig,
    update_metrics,
    calulcate_benchmark_results,
    check_sched_idle,
    collect_sgl_predict_stats,
    reset_sgl_predict_stats,
)


def _can_fast_collapse(
    scheduler: Scheduler,
    batch: Batch | None,
    request_store: VirtualClientStore,
    pp: int,
) -> bool:
    if pp != 1 or batch is None or batch.forward_mode != "decode":
        return False
    if request_store.next_to_admit_idx < request_store.num_prompts:
        return False
    if scheduler.waiting_queue or scheduler.pending_decode_queue or scheduler.chunked_reqs:
        return False
    if len(scheduler.running_queue) != 1:
        return False

    next_batch = scheduler.running_queue[0]
    if next_batch.forward_mode != "decode" or next_batch.is_empty():
        return False

    batch_ids = [req.request_id for req in batch.reqs]
    next_ids = [req.request_id for req in next_batch.reqs]
    if batch_ids != next_ids:
        return False

    return True


def _advance_decode_window_exact(
    scheduler: Scheduler,
    active_reqs: Dict[str, Request],
    request_store: VirtualClientStore,
    decode_phases: Dict[str, float],
    ross_models: Dict[str, Any],
    status: Dict[str, Any],
    post_decode_overhead_s: float,
) -> Tuple[int, int]:
    if not scheduler.running_queue:
        return 0, 0

    next_batch = scheduler.running_queue[0]
    if next_batch.is_empty():
        return 0, 0

    extra_steps = min(req.max_new_tokens - req.decode_tokens for req in next_batch.reqs)
    if extra_steps <= 0:
        return 0, 0

    collapsed = 0
    finished_inc = 0
    for _ in range(extra_steps):
        if not scheduler.running_queue:
            break

        batch = scheduler.running_queue.pop(0)
        scheduler.running_batch = batch
        req_ids = [req.request_id for req in batch.reqs]
        seq_lens = [req.num_computed_tokens for req in batch.reqs]

        for name in ["pre_forward", "forward", "post_forward"]:
            time_ms = ross_models["decode_" + name].predict(req_ids=req_ids, seq_lens=seq_lens)
            decode_phases["decode_" + name] += time_ms / 1000
            status["wall_time"] += time_ms / 1000

        cur_batch_size = len(batch.reqs)
        scheduler.process_batch_result_decode(batch)

        active_items = list(active_reqs.items())
        for rid, req in active_items:
            if req.max_new_tokens <= 1:
                active_reqs.pop(rid, None)
                finished_inc += 1
                continue
            if update_metrics(
                req,
                status["wall_time"],
                req.arrive_time,
                cur_batch_size,
                post_decode_overhead_s,
            ):
                finished_inc += 1
                request_store.record_finish(rid, status["wall_time"] + post_decode_overhead_s)
                active_reqs.pop(rid, None)

        status["step"] += 1
        status["batch_pipeline"].append((batch, scheduler.running_batch))
        collapsed += 1

        if not scheduler.running_queue:
            break

    return collapsed, finished_inc


def run_simulation_fast(
    model,
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
    if pp != 1:
        raise ValueError("simulator_main_fast currently only supports colocate pp=1")

    reset_sgl_predict_stats(ross_models)
    schedulers: List[Scheduler] = []
    prefill_phases, decode_phases = [], []

    for _ in range(dp):
        kv_pool = SGLKVCachePool(
            model=model,
            num_reqs=batch_size,
            tokens_per_block=1,
            total_gpu_memory=total_gpu_memory,
            gpu_memory_utilization=mem_fraction_static,
            framework="sglang",
        )
        schedulers.append(Scheduler(
            waiting_queue=[],
            kv_pool=kv_pool,
            **scheduler_kwargs,
            pp=pp,
        ))
        prefill_phases.append({"prefill_pre_forward": 0, "prefill_forward": 0, "prefill_post_forward": 0})
        decode_phases.append({"decode_pre_forward": 0, "decode_forward": 0, "decode_post_forward": 0})

    current_status = [{"wall_time": 0.0, "step": 0, "batch_pipeline": [], "complete": False} for _ in range(dp)]
    pending_by_rank = [deque() for _ in range(dp)]
    finished_reqs_count = 0
    active_reqs_by_rank = [dict() for _ in range(dp)]
    post_decode_overhead_s = post_decode_overhead_ms / 1000.0

    while True:
        current_global_time = max(status["wall_time"] for status in current_status)
        new_reqs = request_list.refresh(current_global_time)
        for req in new_reqs:
            pending_by_rank[req.dp_rank].append(req)
            current_status[req.dp_rank]["complete"] = False

        last_batches: List[Batch | None] = [None for _ in range(dp)]
        last_batch_sizes = [0 for _ in range(dp)]

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
                    else:
                        status["complete"] = True

        for idx, scheduler in enumerate(schedulers):
            batch = scheduler.get_next_batch_to_run()
            last_batches[idx] = batch
            current_status[idx]["batch_pipeline"].append((batch, scheduler.running_batch))
            cur_batch_size = 0 if not batch else len(batch.reqs)
            last_batch_sizes[idx] = cur_batch_size

            if batch is not None:
                mode = batch.forward_mode
                req_ids = [req.request_id for req in batch.reqs]
                seq_lens = [req.num_computed_tokens for req in batch.reqs]
                phase_bucket = prefill_phases[idx] if mode == "prefill" else decode_phases[idx]
                prefix = "prefill_" if mode == "prefill" else "decode_"

                for name in ["pre_forward", "forward", "post_forward"]:
                    time_ms = ross_models[prefix + name].predict(req_ids=req_ids, seq_lens=seq_lens)
                    phase_bucket[prefix + name] += time_ms / 1000
                    current_status[idx]["wall_time"] += time_ms / 1000
            else:
                status = current_status[idx]
                if not status["batch_pipeline"] or status["step"] - pp >= len(status["batch_pipeline"]):
                    oom, reason = scheduler.check_oom()
                    if oom:
                        raise MemoryError(f"[Prefill] OOM detected on rank {idx}: {reason}")

        for idx, scheduler in enumerate(schedulers):
            step = current_status[idx]["step"]
            if step >= pp:
                if current_status[idx]["complete"]:
                    continue
                current_batch, cur_running_batch = current_status[idx]["batch_pipeline"][step - pp]
                if current_batch is not None:
                    if current_batch.forward_mode == "prefill":
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
                    last_batch_sizes[rank],
                    post_decode_overhead_s,
                ):
                    finished_reqs_count += 1
                    request_list.record_finish(rid, current_status[rank]["wall_time"] + post_decode_overhead_s)
                    active_reqs_by_rank[rank].pop(rid, None)

        for rank, scheduler in enumerate(schedulers):
            batch = last_batches[rank]
            if _can_fast_collapse(scheduler, batch, request_list, pp):
                _, finished_inc = _advance_decode_window_exact(
                    scheduler=scheduler,
                    active_reqs=active_reqs_by_rank[rank],
                    request_store=request_list,
                    decode_phases=decode_phases[rank],
                    ross_models=ross_models,
                    status=current_status[rank],
                    post_decode_overhead_s=post_decode_overhead_s,
                )
                finished_reqs_count += finished_inc

        all_idle = all(
            check_sched_idle(schedulers[i], current_status[i]["batch_pipeline"], current_status[i]["step"], pp)
            for i in range(dp)
        )
        source_drained = (
            request_list.next_to_admit_idx >= request_list.num_prompts
            and request_list.inflight == 0
        )
        if all_idle and (
            request_list.should_terminate_idle(finished_reqs_count)
            or source_drained
        ):
            break

    itl_list = []
    for req in request_list:
        if not req.ttft or not req.e2e_latency:
            raise RuntimeError(f"req={req.request_id}, ttft={req.ttft}, e2e_latency={req.e2e_latency}")
        itl_list.extend(req.itl)

    max_wall_time = max(status["wall_time"] for status in current_status)
    benchmarks = calulcate_benchmark_results(request_list, itl_list, max_wall_time)
    result_dict = {
        "duration": max_wall_time,
        **benchmarks,
        "prefill_phases": prefill_phases,
        "decode_phases": decode_phases,
        "sgl_predict_stats": collect_sgl_predict_stats(ross_models),
    }
    result_dict.update({
        "dp": dp,
        "pp": pp,
        "tp": model.inference_config.tp_size,
        "tokens/s": result_dict["throughput"],
        "tokens/s/gpu": result_dict["throughput"] / dp / pp / model.inference_config.tp_size,
        "tokens/s/user": 1000.0 / result_dict["mean_tpot_ms"],
    })
    return result_dict


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

    if args.disaggregation:
        raise RuntimeError("Run DISAGG in simulator_aligned.py")

    request_store = VirtualClientStore(
        args.frontend_path, args.request_rate, args.batch_size,
        args.dp_size, args.disaggregation,
    )
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

    ret = run_simulation_fast(
        model=model,
        batch_size=args.batch_size,
        request_list=request_store,
        scheduler_kwargs=scheduler_kwargs,
        mem_fraction_static=args.mem_fraction_static,
        total_gpu_memory=platform_perf.theoretical_memory_gb * (1024 ** 3),
        ross_models=ross_model_dict,
        dp=args.dp_size,
        pp=args.pp_size,
        post_decode_overhead_ms=0,
    )
    ret.update({
        "mem_fraction_static": args.mem_fraction_static,
        "chunked_prefill_size": args.chunked_prefill_size,
    })
    return ret


if __name__ == "__main__":
    args = parse_args()
    ret = run_sim(args)
    print(f"[SIM] result={ret}")
