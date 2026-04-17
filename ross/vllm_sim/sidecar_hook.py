#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from typing import Any, Dict, List

from scheduler.scheduler import Scheduler, SchedulerOutput

logger = logging.getLogger(__name__)


@dataclass
class TimingScheduleView:
    scheduled_req_ids: List[str]
    num_scheduled_tokens: Dict[str, int]
    prefill_seq_lens: List[int]
    decode_seq_lens: List[int]


def ross_output_to_timing_view(output: SchedulerOutput | None) -> TimingScheduleView:
    if output is None:
        return TimingScheduleView([], {}, [], [])
    return TimingScheduleView(
        scheduled_req_ids=list(output.scheduled_req_ids),
        num_scheduled_tokens=dict(output.num_scheduled_tokens),
        prefill_seq_lens=list(output.prefill_seq_lens),
        decode_seq_lens=list(output.decode_seq_lens),
    )


def flush_log_handlers() -> None:
    seen = set()
    for candidate in (logger, logging.getLogger()):
        for handler in candidate.handlers:
            if id(handler) in seen:
                continue
            seen.add(id(handler))
            try:
                handler.flush()
            except Exception:
                pass


def ross_scheduler_state(scheduler: Scheduler) -> Dict[str, Any]:
    head = 8
    return {
        "num_waiting": len(scheduler.waiting),
        "num_running": len(scheduler.running),
        "waiting_head": [req.request_id for req in scheduler.waiting[:head]],
        "running_head": [req.request_id for req in scheduler.running[:head]],
    }


def vllm_scheduler_state(sidecar_scheduler: Any) -> Dict[str, Any]:
    waiting = list(sidecar_scheduler.waiting)
    running = list(sidecar_scheduler.running)
    head = 8
    return {
        "num_waiting": len(waiting),
        "num_running": len(running),
        "waiting_head": [req.request_id for req in waiting[:head]],
        "running_head": [req.request_id for req in running[:head]],
    }


def ross_request_state(scheduler: Scheduler, request_id: str) -> Dict[str, Any] | None:
    for req in itertools.chain(scheduler.running, scheduler.waiting):
        if req.request_id == request_id:
            return {
                "request_id": req.request_id,
                "status": str(req.status),
                "num_computed_tokens": req.num_computed_tokens,
                "prompt_tokens": req.prompt_tokens,
                "output_len": req.output_len,
                "num_tokens": req.num_tokens,
                "transfer_loaded": getattr(req, "transfer_loaded", False),
            }
    return None


def vllm_request_state(sidecar_scheduler: Any, request_id: str) -> Dict[str, Any] | None:
    req = sidecar_scheduler.requests.get(request_id)
    if req is None:
        return None
    return {
        "request_id": req.request_id,
        "status": str(req.status),
        "num_computed_tokens": req.num_computed_tokens,
        "num_prompt_tokens": getattr(req, "num_prompt_tokens", None),
        "num_output_tokens": getattr(req, "num_output_tokens", None),
        "num_cached_tokens": getattr(req, "num_cached_tokens", None),
        "max_tokens": getattr(req, "max_tokens", None),
        "num_tokens": getattr(req, "num_tokens", None),
    }


def vllm_output_to_timing_view(
    output: Any,
    align_transfer_loaded_to_sim: bool = False,
) -> TimingScheduleView:
    if output is None:
        return TimingScheduleView([], {}, [], [])

    scheduled_req_ids = [req.req_id for req in output.scheduled_new_reqs]
    scheduled_req_ids.extend(output.scheduled_cached_reqs.req_ids)
    adjusted_tokens = dict(output.num_scheduled_tokens)

    prefill_seq_lens: List[int] = []
    decode_seq_lens: List[int] = []

    for req in output.scheduled_new_reqs:
        num_scheduled = output.num_scheduled_tokens[req.req_id]
        total_seq_len = req.num_computed_tokens + num_scheduled
        is_pd_decode_handoff = (
            align_transfer_loaded_to_sim
            and num_scheduled <= 1
            and req.prompt_token_ids is not None
            and req.num_computed_tokens + 1 == len(req.prompt_token_ids)
            and total_seq_len == len(req.prompt_token_ids)
        )
        if is_pd_decode_handoff:
            adjusted_tokens[req.req_id] = int(total_seq_len)
            decode_seq_lens.append(1)
            continue
        if num_scheduled <= 1:
            decode_seq_lens.append(int(total_seq_len))
        else:
            prefill_seq_lens.append(int(total_seq_len))

    for req_id, num_computed in zip(
        output.scheduled_cached_reqs.req_ids,
        output.scheduled_cached_reqs.num_computed_tokens,
    ):
        num_scheduled = output.num_scheduled_tokens[req_id]
        total_seq_len = num_computed + num_scheduled
        is_pd_decode_handoff = (
            align_transfer_loaded_to_sim
            and num_scheduled <= 1
            and output.scheduled_cached_reqs.num_output_tokens[
                output.scheduled_cached_reqs.req_ids.index(req_id)
            ] == 0
        )
        if is_pd_decode_handoff:
            adjusted_tokens[req_id] = int(total_seq_len)
            decode_seq_lens.append(1)
            continue
        if num_scheduled <= 1:
            decode_seq_lens.append(int(total_seq_len))
        else:
            prefill_seq_lens.append(int(total_seq_len))

    return TimingScheduleView(
        scheduled_req_ids=scheduled_req_ids,
        num_scheduled_tokens=adjusted_tokens,
        prefill_seq_lens=prefill_seq_lens,
        decode_seq_lens=decode_seq_lens,
    )


def normalize_vllm_output(output: Any) -> Dict[str, Any]:
    timing = vllm_output_to_timing_view(output, align_transfer_loaded_to_sim=False)
    return {
        "scheduled_req_ids": sorted(timing.scheduled_req_ids),
        "num_scheduled_tokens": dict(timing.num_scheduled_tokens),
        "prefill_seq_lens": sorted(timing.prefill_seq_lens),
        "decode_seq_lens": sorted(timing.decode_seq_lens),
    }


def normalize_ross_output_with_options(
    output: SchedulerOutput | None,
    treat_transfer_loaded_as_decode_one: bool = False,
) -> Dict[str, Any]:
    if output is None:
        return {
            "scheduled_req_ids": [],
            "num_scheduled_tokens": {},
            "prefill_seq_lens": [],
            "decode_seq_lens": [],
        }
    adjusted_tokens = dict(output.num_scheduled_tokens)
    adjusted_decode_seq_lens = list(output.decode_seq_lens)
    if treat_transfer_loaded_as_decode_one:
        scheduled_objs = {}
        adjusted_decode_seq_lens = []
        for req in itertools.chain(
            output.running_reqs or [],
            output.resumed_reqs or [],
            output.new_reqs or [],
        ):
            scheduled_objs[req.request_id] = req
        for req_id, req in scheduled_objs.items():
            if getattr(req, "transfer_loaded", False):
                adjusted_tokens[req_id] = 1
                continue
            num_scheduled = output.num_scheduled_tokens.get(req_id, 0)
            if (
                num_scheduled > 1
                and getattr(req, "output_len", None) == 0
                and getattr(req, "num_computed_tokens", None) == num_scheduled
            ):
                adjusted_tokens[req_id] = 1
        for req in scheduled_objs.values():
            num_scheduled = adjusted_tokens.get(req.request_id, 0)
            total_seq_len = getattr(req, "num_computed_tokens", 0)
            if num_scheduled <= 1:
                adjusted_decode_seq_lens.append(int(total_seq_len))
    return {
        "scheduled_req_ids": sorted(output.scheduled_req_ids),
        "num_scheduled_tokens": adjusted_tokens,
        "prefill_seq_lens": sorted(output.prefill_seq_lens),
        "decode_seq_lens": sorted(adjusted_decode_seq_lens),
    }


def compare_schedule_outputs(
    step: int,
    rank: int,
    ross_output: SchedulerOutput | None,
    vllm_output: Any,
    ross_scheduler: Scheduler | None = None,
    sidecar_scheduler: Any = None,
    treat_transfer_loaded_as_decode_one: bool = False,
) -> None:
    lhs = normalize_ross_output_with_options(
        ross_output,
        treat_transfer_loaded_as_decode_one=treat_transfer_loaded_as_decode_one,
    )
    rhs = normalize_vllm_output(vllm_output)
    if lhs != rhs:
        lhs_ids = set(lhs["scheduled_req_ids"])
        rhs_ids = set(rhs["scheduled_req_ids"])
        extra_ross = sorted(lhs_ids - rhs_ids)
        extra_vllm = sorted(rhs_ids - lhs_ids)
        differing_keys = sorted([key for key in lhs.keys() if lhs.get(key) != rhs.get(key)])
        message = (
            f"Schedule mismatch at step={step}, rank={rank}\n"
            f"differing_keys={differing_keys}\n"
            f"ROSS-only reqs={extra_ross}\n"
            f"vLLM-only reqs={extra_vllm}\n"
            f"ROSS summary={{num_reqs: {len(lhs['scheduled_req_ids'])}, prefill: {len(lhs['prefill_seq_lens'])}, decode: {len(lhs['decode_seq_lens'])}}}\n"
            f"vLLM summary={{num_reqs: {len(rhs['scheduled_req_ids'])}, prefill: {len(rhs['prefill_seq_lens'])}, decode: {len(rhs['decode_seq_lens'])}}}\n"
            f"ROSS tokens sample={dict(list(sorted(lhs['num_scheduled_tokens'].items()))[:8])}\n"
            f"vLLM tokens sample={dict(list(sorted(rhs['num_scheduled_tokens'].items()))[:8])}\n"
            f"ROSS prefill_seq_lens={lhs['prefill_seq_lens'][:16]}\n"
            f"vLLM prefill_seq_lens={rhs['prefill_seq_lens'][:16]}\n"
            f"ROSS decode_seq_lens={lhs['decode_seq_lens'][:16]}\n"
            f"vLLM decode_seq_lens={rhs['decode_seq_lens'][:16]}"
        )
        logger.error(message)
        flush_log_handlers()
        raise AssertionError(message)


def log_sidecar_schedule(
    step: int,
    rank: int,
    vllm_output: Any,
    phase: str | None = None,
) -> None:
    prefix = f"[dp_{rank}] sidecar"
    if phase:
        prefix += f" {phase}"
    if vllm_output is None:
        logger.debug(f"{prefix} step={step} output=None")
        return
    scheduled_new = [req.req_id for req in vllm_output.scheduled_new_reqs]
    scheduled_cached = list(vllm_output.scheduled_cached_reqs.req_ids)
    logger.debug(
        f"{prefix} step={step} "
        f"new_count={len(scheduled_new)} new_head={scheduled_new[:8]} "
        f"cached_count={len(scheduled_cached)} cached_head={scheduled_cached[:8]}"
    )


def choose_timing_output(
    ross_output: SchedulerOutput | None,
    vllm_output: Any,
    result_source: str,
    align_transfer_loaded_to_sim: bool = False,
) -> SchedulerOutput | TimingScheduleView | None:
    if result_source == "vllm":
        return vllm_output_to_timing_view(
            vllm_output,
            align_transfer_loaded_to_sim=align_transfer_loaded_to_sim,
        )
    return ross_output
