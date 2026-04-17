#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import deque
from typing import Any, Dict, List

from common.config import InferenceConfig
from common.features import PlatformPerf
from common.kvpool import KVCachePool
from common.models import BaseModel, get_model
from common.ross_model import ROSSModel
from common.sim_http_perf import RequestStore, VirtualClientStore
from dummy_sched import create_request as create_dummy_request, create_sidecar_scheduler, make_dummy_model_output
from scheduler.request import RequestStatus

from simulator_main import (
    _get_gpu_memory_utilization,
    _normalize_vllm_output,
    _warn_pp_pre_forward_disabled,
    calulcate_benchmark_results,
    get_mixed_forward_phase,
    get_regression_model,
    get_ross_model_paths,
    get_ross_models,
    load_memory_increase,
    parse_args,
    update_metrics,
)


def _timing_view_from_vllm_output(output: Any):
    norm = _normalize_vllm_output(output)

    class TimingView:
        scheduled_req_ids = norm["scheduled_req_ids"]
        num_scheduled_tokens = norm["num_scheduled_tokens"]
        prefill_seq_lens = norm["prefill_seq_lens"]
        decode_seq_lens = norm["decode_seq_lens"]

    return TimingView()


def run_simulation_with_prefix_cache(
    model: BaseModel,
    batch_size: int,
    request_list: RequestStore,
    scheduler_kwargs: Dict[str, Any],
    ross_models: Dict[str, ROSSModel],
    dp: int,
    pp: int,
    isl: int,
    osl: int,
    gpu_memory_utilization: float,
    memory_profiling: Dict[str, Any] | None = None,
    total_gpu_memory: int | None = None,
    vllm_src_root: str = "",
) -> Dict[str, Any]:
    if pp != 1:
        raise ValueError("enable_prefix_caching currently only supports colocate pp=1")

    tokens_per_block = 16 if model.model_uri.lower().find("deepseek") == -1 else 64
    total_time_slices = [{"pre_forward": 0, "forward": 0, "post_forward": 0, "pp_pre_forward": 0} for _ in range(dp)]
    current_status = [{"wall_time": 0.0, "step": 0, "complete": False} for _ in range(dp)]
    pending_by_rank = [deque() for _ in range(dp)]
    active_reqs_by_rank = [dict() for _ in range(dp)]
    sidecar_schedulers: List[Any] = []
    finished_reqs_count = 0

    for _ in range(dp):
        kv_pool = KVCachePool(
            model=model,
            num_reqs=batch_size,
            tokens_per_block=tokens_per_block,
            total_gpu_memory=total_gpu_memory,
            gpu_memory_utilization=gpu_memory_utilization,
            vllm_non_torch_increase=memory_profiling["non_torch_mem_increase"] if memory_profiling else 0,
        )
        sidecar_schedulers.append(
            create_sidecar_scheduler(
                scheduler_kwargs=scheduler_kwargs,
                num_blocks=kv_pool.num_blocks,
                block_size=tokens_per_block,
                max_model_len=131072,
                max_num_seqs=batch_size,
                model_uri=model.model_uri,
                vllm_src_root=vllm_src_root,
                enable_prefix_caching=True,
            )
        )

    while True:
        current_global_time = max(status["wall_time"] for status in current_status)
        new_reqs = request_list.refresh(current_global_time)
        for r in new_reqs:
            pending_by_rank[r.dp_rank].append(r)
            current_status[r.dp_rank]["complete"] = False

        for rank, scheduler in enumerate(sidecar_schedulers):
            status = current_status[rank]
            status["step"] += 1
            pending = pending_by_rank[rank]
            while pending and pending[0].ready_time <= status["wall_time"]:
                req = pending.popleft()
                scheduler.add_request(
                    create_dummy_request(
                        request_id=req.request_id,
                        prompt_token_len=req.prompt_tokens,
                        max_output_tokens=req.num_tokens - req.prompt_tokens,
                        block_size=tokens_per_block,
                        vllm_src_root=vllm_src_root,
                    )
                )
                active_reqs_by_rank[rank][req.request_id] = req

            if len(scheduler.waiting) == 0 and len(scheduler.running) == 0:
                if pending:
                    req_next = pending.popleft()
                    status["wall_time"] = max(status["wall_time"], req_next.ready_time)
                    scheduler.add_request(
                        create_dummy_request(
                            request_id=req_next.request_id,
                            prompt_token_len=req_next.prompt_tokens,
                            max_output_tokens=req_next.num_tokens - req_next.prompt_tokens,
                            block_size=tokens_per_block,
                            vllm_src_root=vllm_src_root,
                        )
                    )
                    active_reqs_by_rank[rank][req_next.request_id] = req_next
                else:
                    status["complete"] = True

        for idx, scheduler in enumerate(sidecar_schedulers):
            output = scheduler.schedule()
            if output is not None and output.total_num_scheduled_tokens > 0:
                timing_view = _timing_view_from_vllm_output(output)
                mixed_forward_phase = get_mixed_forward_phase(timing_view)
                for name in ["pre_forward", "forward", "post_forward"]:
                    regression_model = get_regression_model(ross_models, name, mixed_forward_phase)
                    time_step = regression_model.predict(
                        req_ids=timing_view.scheduled_req_ids,
                        prefill_seq_lens=timing_view.prefill_seq_lens,
                        decode_seq_lens=timing_view.decode_seq_lens,
                        isl=isl,
                        osl=osl,
                    )
                    total_time_slices[idx][name] += time_step / 1000
                    current_status[idx]["wall_time"] += time_step / 1000

                dummy_output = make_dummy_model_output(scheduler, output)
                engine_core_outputs = scheduler.update_from_output(output, dummy_output)

                finished_req_ids = set()
                new_token_counts: Dict[str, int] = {}
                for eco in engine_core_outputs.values():
                    finished_req_ids.update(getattr(eco, "finished_requests", set()) or set())
                    for item in getattr(eco, "outputs", []) or []:
                        new_token_counts[item.request_id] = len(item.new_token_ids or [])
                        if item.finish_reason is not None:
                            finished_req_ids.add(item.request_id)

                active_items = list(active_reqs_by_rank[idx].items())
                for rid, req in active_items:
                    side_req = scheduler.requests.get(rid)
                    if side_req is not None:
                        req.output_len = side_req.num_output_tokens
                    else:
                        req.output_len += new_token_counts.get(rid, 0)

                    if rid in finished_req_ids:
                        req.status = RequestStatus.FINISHED

                    if update_metrics(req, current_status[idx]["wall_time"], req.arrive_time, current_status[idx]["step"]):
                        request_list.record_finish(rid, current_status[idx]["wall_time"])
                        active_reqs_by_rank[idx].pop(rid, None)
                        finished_reqs_count += 1

        all_idle = all(
            len(sidecar_schedulers[i].waiting) == 0 and len(sidecar_schedulers[i].running) == 0
            for i in range(dp)
        )
        if all_idle and request_list.should_terminate_idle(finished_reqs_count):
            break

    itl_list = []
    for req in request_list:
        if not req.ttft or not req.e2e_latency:
            raise RuntimeError(f"req={req.request_id}, ttft={req.ttft}, e2e_latency={req.e2e_latency}")
        itl_list.extend(req.itl)

    max_wall_time = max([s["wall_time"] for s in current_status])
    benchmarks = calulcate_benchmark_results(request_list, itl_list, max_wall_time)
    result_dict = {
        "duration": max_wall_time,
        **benchmarks,
        "timing_phases": total_time_slices,
        "dp": dp,
        "pp": pp,
        "tp": model.inference_config.tp_size,
        "tokens/s": benchmarks["throughput"],
        "tokens/s/gpu": benchmarks["throughput"] / dp / pp / model.inference_config.tp_size,
        "tokens/s/user": 1000.0 / benchmarks["mean_tpot_ms"],
    }
    return result_dict


def run_sim(args):
    _warn_pp_pre_forward_disabled(args)
    if getattr(args, "disaggregation", False):
        raise ValueError("enable_prefix_caching currently only supports colocate mode")
    if getattr(args, "pp_size", 1) != 1:
        raise ValueError("enable_prefix_caching currently only supports pp_size=1")

    gpu_memory_utilization = _get_gpu_memory_utilization(args)
    scheduler_kwargs = {
        "max_num_batched_tokens": args.max_num_batched_tokens,
    }
    platform_perf = PlatformPerf(platform_perf_yaml=args.platform_perf)

    inference_config = InferenceConfig(dp_size=args.dp_size, pp_size=args.pp_size, tp_size=args.tp_size)
    model = get_model(args.model_uri, inference_config)
    ross_model_dict = get_ross_models(
        model,
        platform_perf,
        inference_config,
        model_path=get_ross_model_paths(args),
    )
    request_store = VirtualClientStore(
        args.frontend_path, args.request_rate, args.batch_size,
        args.dp_size, args.disaggregation,
    )
    memory_increase = load_memory_increase(args.mem_profiling_path, {"pp": args.pp_size, "tp": args.tp_size})
    ret = run_simulation_with_prefix_cache(
        model=model,
        batch_size=args.batch_size,
        request_list=request_store,
        scheduler_kwargs=scheduler_kwargs,
        ross_models=ross_model_dict,
        dp=args.dp_size,
        pp=args.pp_size,
        isl=args.max_prompt_len,
        osl=args.max_output_len,
        gpu_memory_utilization=gpu_memory_utilization,
        memory_profiling=memory_increase,
        total_gpu_memory=platform_perf.theoretical_memory_gb * (1024 ** 3),
        vllm_src_root=getattr(args, "vllm_src_root", ""),
    )
    ret.update({"gpu_memory_utilization": gpu_memory_utilization})
    return ret


if __name__ == "__main__":
    args = parse_args()
    ret = run_sim(args)
    print(f"[SIM] result={ret}")
