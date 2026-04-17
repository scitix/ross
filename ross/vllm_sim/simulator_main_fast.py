#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import sys
from collections import deque
from typing import Dict, Any, List

TEST_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TEST_ROOT))

from common.features import PlatformPerf
from common.models import get_model
from common.config import InferenceConfig
from common.kvpool import KVCachePool
from common.sim_http_perf import VirtualClientStore

from scheduler.request import RequestStatus
from scheduler.scheduler import Scheduler, SchedulerOutput

from simulator_main import (
    _warn_pp_pre_forward_disabled,
    _get_gpu_memory_utilization,
    parse_args,
    get_ross_models,
    get_ross_model_paths,
    get_regression_model,
    get_mixed_forward_phase,
    update_metrics,
    calulcate_benchmark_results,
    check_pipeline_clear,
    load_memory_increase,
)


def _can_fast_collapse(
    scheduler: Scheduler,
    schedule_output: SchedulerOutput | None,
    request_store: VirtualClientStore,
) -> bool:
    if schedule_output is None:
        return False
    if schedule_output.prefill_seq_lens:
        return False
    if schedule_output.new_reqs or schedule_output.resumed_reqs or schedule_output.preempted_reqs:
        return False
    if scheduler.waiting:
        return False
    if request_store.next_to_admit_idx < request_store.num_prompts:
        return False
    if not scheduler.running:
        return False
    if len(schedule_output.scheduled_req_ids) != len(scheduler.running):
        return False
    if any(tokens != 1 for tokens in schedule_output.num_scheduled_tokens.values()):
        return False
    running_ids = [req.request_id for req in scheduler.running]
    if schedule_output.scheduled_req_ids != running_ids:
        return False
    return True


def _advance_decode_window_exact(
    scheduler: Scheduler,
    active_reqs: Dict[str, Any],
    request_store: VirtualClientStore,
    total_time_slices: Dict[str, float],
    ross_models: Dict[str, Any],
    status: Dict[str, Any],
    isl: int,
    osl: int,
) -> int:
    if not scheduler.running:
        return 0

    remaining_steps = min(req.num_tokens - req.prompt_tokens - req.output_len for req in scheduler.running)
    extra_steps = max(0, remaining_steps - 1)
    if extra_steps <= 0:
        return 0

    collapsed = 0
    for _ in range(extra_steps):
        running_reqs = list(scheduler.running)
        scheduled_req_ids = [req.request_id for req in running_reqs]
        decode_seq_lens = [int(req.num_computed_tokens + 1) for req in running_reqs]

        for name in ["pre_forward", "forward", "post_forward"]:
            regression_model = get_regression_model(ross_models, name, "decode")
            time_ms = regression_model.predict(
                req_ids=scheduled_req_ids,
                prefill_seq_lens=[],
                decode_seq_lens=decode_seq_lens,
                isl=isl,
                osl=osl,
            )
            total_time_slices[name] += time_ms / 1000
            status["wall_time"] += time_ms / 1000

        for req in running_reqs:
            req.num_computed_tokens += 1

        for req in list(scheduler.input_requests):
            if req.request_id not in active_reqs:
                continue
            all_tokens = req.output_len + req.prompt_tokens
            if req.num_computed_tokens >= all_tokens:
                req.output_len += 1
            if all_tokens >= req.num_tokens - 1 or all_tokens >= scheduler.max_model_len - 1:
                req.status = RequestStatus.FINISHED
                scheduler.kv_cache_manager.free(req.request_id)

        scheduler.running = [req for req in scheduler.running if req.status != RequestStatus.FINISHED]
        active_items = list(active_reqs.items())
        for rid, req in active_items:
            if req.num_tokens - req.prompt_tokens <= 1:
                active_reqs.pop(rid, None)
                continue
            if update_metrics(req, status["wall_time"], req.arrive_time, status["step"] + 1):
                request_store.record_finish(rid, status["wall_time"])
                active_reqs.pop(rid, None)

        status["step"] += 1
        status["batch_pipeline"].append(None)
        collapsed += 1

        if not scheduler.running:
            break

    return collapsed


def run_simulation_fast(
    model,
    batch_size: int,
    request_list: VirtualClientStore,
    ross_models: Dict[str, Any],
    scheduler_kwargs: Dict[str, Any],
    isl: int,
    osl: int,
    gpu_memory_utilization: float,
    memory_profiling: Dict[str, Any],
    total_gpu_memory: int,
    dp: int = 1,
    pp: int = 1,
) -> Dict[str, Any]:
    if pp != 1:
        raise ValueError("simulator_main_fast currently only supports colocate pp=1")

    schedulers: List[Scheduler] = []
    total_time_slices = [{"pre_forward": 0, "forward": 0, "post_forward": 0, "pp_pre_forward": 0} for _ in range(dp)]
    tokens_per_block = 16 if model.model_uri.lower().find("deepseek") == -1 else 64

    for _ in range(dp):
        kv_pool = KVCachePool(
            model=model,
            num_reqs=batch_size,
            tokens_per_block=tokens_per_block,
            total_gpu_memory=total_gpu_memory,
            gpu_memory_utilization=gpu_memory_utilization,
            vllm_non_torch_increase=memory_profiling["non_torch_mem_increase"],
        )
        schedulers.append(
            Scheduler(
                max_running_reqs=batch_size,
                kv_pool=kv_pool,
                **scheduler_kwargs,
            )
        )

    current_status = [{"wall_time": 0.0, "step": 0, "batch_pipeline": [], "complete": False} for _ in range(dp)]
    finished_reqs_count = 0
    pending_by_rank = [deque() for _ in range(dp)]
    active_reqs_by_rank = [dict() for _ in range(dp)]

    while True:
        current_global_time = max(status["wall_time"] for status in current_status)
        new_reqs = request_list.refresh(current_global_time)
        for req in new_reqs:
            pending_by_rank[req.dp_rank].append(req)
            current_status[req.dp_rank]["complete"] = False

        last_outputs: List[SchedulerOutput | None] = [None for _ in range(dp)]
        for rank, sched in enumerate(schedulers):
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

        for idx, scheduler in enumerate(schedulers):
            schedule_output = scheduler.schedule()
            last_outputs[idx] = schedule_output
            current_status[idx]["batch_pipeline"].append(schedule_output)
            if schedule_output and (not scheduler.should_terminate()):
                mixed_forward_phase = get_mixed_forward_phase(schedule_output)
                for name in ["pre_forward", "forward", "post_forward"]:
                    regression_model = get_regression_model(ross_models, name, mixed_forward_phase)
                    time_ms = regression_model.predict(
                        req_ids=schedule_output.scheduled_req_ids,
                        prefill_seq_lens=schedule_output.prefill_seq_lens,
                        decode_seq_lens=schedule_output.decode_seq_lens,
                        isl=isl,
                        osl=osl,
                    )
                    total_time_slices[idx][name] += time_ms / 1000
                    current_status[idx]["wall_time"] += time_ms / 1000
            else:
                status = current_status[idx]
                if check_pipeline_clear(status["batch_pipeline"], status["step"], pp):
                    oom, reason = scheduler.check_oom()
                    if oom:
                        raise MemoryError(f"OOM detected on rank {idx}: {reason}")

        for idx, scheduler in enumerate(schedulers):
            step = current_status[idx]["step"]
            if step >= pp and not current_status[idx]["complete"]:
                current_output = current_status[idx]["batch_pipeline"][step - pp]
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

        for rank, scheduler in enumerate(schedulers):
            output = last_outputs[rank]
            if _can_fast_collapse(scheduler, output, request_list):
                collapsed = _advance_decode_window_exact(
                    scheduler=scheduler,
                    active_reqs=active_reqs_by_rank[rank],
                    request_store=request_list,
                    total_time_slices=total_time_slices[rank],
                    ross_models=ross_models,
                    status=current_status[rank],
                    isl=isl,
                    osl=osl,
                )
                if collapsed > 0:
                    finished_reqs_count = sum(
                        1 for req in request_list.as_list()
                        if req.e2e_latency is not None or req.num_tokens - req.prompt_tokens <= 1
                    )

        all_idle = all(
            check_pipeline_clear(current_status[i]["batch_pipeline"], current_status[i]["step"], pp)
            and schedulers[i].should_terminate()
            for i in range(dp)
        )
        if all_idle and request_list.should_terminate_idle(finished_reqs_count):
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
        "timing_phases": total_time_slices,
    }
    result_dict.update(
        {
            "dp": dp,
            "pp": pp,
            "tp": model.inference_config.tp_size,
            "tokens/s": result_dict["throughput"],
            "tokens/s/gpu": result_dict["throughput"] / dp / pp / model.inference_config.tp_size,
            "tokens/s/user": 1000.0 / result_dict["mean_tpot_ms"],
        }
    )
    return result_dict


def run_sim(args):
    if args.disaggregation:
        raise ValueError("simulator_main_fast currently only supports colocate mode")
    if args.pp_size != 1:
        raise ValueError("simulator_main_fast currently only supports pp_size=1")

    _warn_pp_pre_forward_disabled(args)
    gpu_memory_utilization = _get_gpu_memory_utilization(args)
    scheduler_kwargs = {
        "max_num_batched_tokens": args.max_num_batched_tokens,
    }
    platform_perf = PlatformPerf(platform_perf_yaml=args.platform_perf)

    inference_config = InferenceConfig(dp_size=args.dp_size, pp_size=args.pp_size, tp_size=args.tp_size)
    model = get_model(args.model_uri, inference_config)
    ross_model_dict = get_ross_models(model, platform_perf, inference_config, get_ross_model_paths(args))
    request_store = VirtualClientStore(
        args.frontend_path,
        args.request_rate,
        args.batch_size,
        args.dp_size,
        False,
    )
    memory_increase = load_memory_increase(args.mem_profiling_path, {"pp": args.pp_size, "tp": args.tp_size})

    ret = run_simulation_fast(
        model=model,
        batch_size=args.batch_size,
        request_list=request_store,
        scheduler_kwargs=scheduler_kwargs,
        memory_profiling=memory_increase,
        total_gpu_memory=platform_perf.theoretical_memory_gb * (1024 ** 3),
        gpu_memory_utilization=gpu_memory_utilization,
        ross_models=ross_model_dict,
        dp=args.dp_size,
        pp=args.pp_size,
        isl=args.max_prompt_len,
        osl=args.max_output_len,
    )
    ret.update(scheduler_kwargs)
    ret.update({"gpu_memory_utilization": gpu_memory_utilization})
    return ret
