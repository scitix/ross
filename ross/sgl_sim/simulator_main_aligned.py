#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

from pathlib import Path
import os, sys
from queue import PriorityQueue
from collections import deque
from typing import List, Tuple, Dict, Any

TEST_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TEST_ROOT))

from common.features import PlatformPerf
from common.models import get_model, BaseModel
from common.config import InferenceConfig
from common.kvpool import SGLKVCachePool
from common.sim_http_perf import VirtualClientStore

from scheduler.request import Request
from scheduler.scheduler import Scheduler, Batch

from simulator_main import (
    get_ross_models,
    reset_sgl_predict_stats,
    collect_sgl_predict_stats,
    update_metrics,
    calulcate_benchmark_results,
    check_sched_idle,
    _get_sim_prefix_cache_cls,
    _load_sim_prefix_cache,
)

import logging
logger = logging.getLogger(__name__)

def parse_args():
    ap = argparse.ArgumentParser("ROSS SGLANG simulator")
    ap.add_argument("--model-uri", type=str, default="")
    ap.add_argument("--tokenize-url", type=str, default="")
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
    ap.add_argument("--chunked-prefill-size", type=int, default=8192)
    ap.add_argument("--reserved-decode-tokens", type=int, default=512)
    ap.add_argument("--enable-prefix-caching", action="store_true", default=False)

    # Workload Config
    ap.add_argument("--batch-size", type=int, required=True)
    ap.add_argument("--max-prompt-len", type=int, required=True)
    ap.add_argument("--max-output-len", type=int, required=True)
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--dataset-path", type=str, required=True)
    ap.add_argument("--frontend-path", type=str, required=True)
    ap.add_argument("--request-rate", type=str, required=True)
    ap.add_argument("--post-decode-overhead-ms", type=float, default=0.0)
    # ap.add_argument('--scheduler_config', type=str, help='Path to the Scheduler YAML configuration file.')

    ap.add_argument('--platform-perf', type=str, required=True, help='Path to the Platform Performance file.')

    ap.add_argument('--prefill_pre_forward_path', type=str, required=True, help='Path to the Saved PRE-FORWARD Model file.')    
    ap.add_argument('--prefill_forward_path', type=str, required=True, help='Path to the Saved [Prefill] FORWARD Model file.')
    ap.add_argument('--prefill_post_forward_path', type=str, required=True, help='Path to the Saved [Prefill] POST-FORWARD Model file.')
    ap.add_argument('--decode_pre_forward_path', type=str, required=True, help='Path to the Saved PRE-FORWARD Model file.')
    ap.add_argument('--decode_forward_path', type=str, required=True, help='Path to the Saved [Decode] FORWARD Model file.')
    ap.add_argument('--decode_post_forward_path', type=str, required=True, help='Path to the Saved [Decode] POST-FORWARD Model file.')
    return ap.parse_args()

class BaseWorker:
    def __init__(self,
        dp_rank: str,
        rank_id: int,
        request_list: List[Request],
        model: BaseModel,
        batch_size: int,
        mem_fraction_static: float,
        total_gpu_memory: int,
        pp: int = 1,
        scheduler_kwargs: Dict[str, Any] | None = None,
    ):
        scheduler_kwargs = scheduler_kwargs or {}
        self.scheduler_kwargs = scheduler_kwargs
        self.dp_rank = dp_rank
        self.rank_id = rank_id
        self.kv_pool = SGLKVCachePool(
            model=model,
            num_reqs=batch_size,
            tokens_per_block=1,
            gpu_memory_utilization=mem_fraction_static,
            total_gpu_memory=total_gpu_memory,
            framework='sglang',
        )
        if scheduler_kwargs.get("enable_prefix_caching", False):
            self.kv_pool.enable_prefix_cache(
                _get_sim_prefix_cache_cls()(page_size=self.kv_pool.page_size)
            )
        self.pp = pp
        self.request_list = list(request_list)

        self.wall_time = 0.0
        self.step = 0
        self.batch_pipeline = []
        self.complete = False
        self.queued = False

    def add_requests(self, new_reqs: List[Request]):
        if not new_reqs:
            return
        self.request_list.extend(new_reqs)
        self.complete = False

    def __lt__(self, other):
        if not isinstance(other, BaseWorker):
            return NotImplemented
        if self.wall_time == other.wall_time:
            if self.rank_id == other.rank_id:
                self_is_prefill = "prefill" in self.dp_rank
                other_is_prefill = "prefill" in other.dp_rank
                if self_is_prefill != other_is_prefill:
                    return self_is_prefill
                return self.dp_rank < other.dp_rank
            return self.rank_id < other.rank_id
        return self.wall_time < other.wall_time

class PrefillWorker(BaseWorker):
    def __init__(self,
        dp_rank: str,
        rank_id: int,
        request_list: List[Request],
        model: BaseModel,
        batch_size: int,
        mem_fraction_static: float,
        total_gpu_memory: int,
        pp: int = 1,
        scheduler_kwargs: Dict[str, Any] = dict(),
    ):
        super().__init__(
            dp_rank,
            rank_id,
            request_list,
            model,
            batch_size,
            mem_fraction_static,
            total_gpu_memory,
            pp=pp,
            scheduler_kwargs=scheduler_kwargs,
        )
        self.scheduler = Scheduler(
            waiting_queue=[],
            disaggregation_mode="prefill",
            kv_pool=self.kv_pool,
            pp=pp,
            **self.scheduler_kwargs)
        self.pending_reqs = deque(request_list)
        self.timing_phases = { "pre_forward": 0, "forward": 0, "post_forward": 0 }

    def add_requests(self, new_reqs: List[Request]):
        if not new_reqs:
            return
        self.pending_reqs.extend(new_reqs)
        self.complete = False

    def fetch_new_request(self, decode_workers):
        retry_time = None
        retained_reqs = deque()
        while self.pending_reqs:
            req = self.pending_reqs.popleft()
            if req.ready_time > self.wall_time:
                retained_reqs.append(req)
                continue
            decode_worker = decode_workers[req.decode_dp_rank]
            if decode_worker.reserve_prefill_kv(req):
                self.scheduler.waiting_queue.append(req)
                continue
            retained_reqs.append(req)
            worker_retry_time = decode_worker.wall_time
            if worker_retry_time <= self.wall_time:
                worker_retry_time = self.wall_time + 1e-9
            retry_time = worker_retry_time if retry_time is None else min(retry_time, worker_retry_time)
        self.pending_reqs = retained_reqs

        # If idle, jump to the next queued request arrival or decode reservation retry.
        if check_sched_idle(self.scheduler, self.batch_pipeline, self.step, self.pp):
            next_ready_time = self.pending_reqs[0].ready_time if self.pending_reqs else None
            if next_ready_time is not None and next_ready_time <= self.wall_time:
                next_ready_time = self.wall_time + 1e-9
            candidates = [t for t in (next_ready_time, retry_time) if t is not None]
            if candidates:
                self.wall_time = max(self.wall_time, min(candidates))
                logger.debug(
                    f"[PrefillFetch] worker={self.dp_rank} jump_to_ready={self.wall_time:.5f}, "
                    f"pending={len(self.pending_reqs)}, waiting={len(self.scheduler.waiting_queue)}"
                )
            else:
                self.complete = True

    def forward(self, ross_models):
        self.step += 1
        batch = self.scheduler.get_next_batch_to_run()
        logger.debug(
            f"[Prefill] step: { self.step }; current time: {self.wall_time}; "
            f"waiting_queue_size=({len(self.scheduler.waiting_queue)}) complete={self.complete} "
            f"avail_tokens={self.scheduler.kv_allocator.available_tokens}\n{batch}"
        )
        self.batch_pipeline.append((batch, self.scheduler.running_batch))
        if batch is not None:
            # ROSSModel Estimate Step N Duration
            req_ids = [r.request_id for r in batch.reqs]
            seq_lens = [r.num_computed_tokens for r in batch.reqs]
            for name in ["pre_forward", "forward", "post_forward"]:
                _time_step = ross_models["prefill_" + name].predict(
                    req_ids=req_ids,
                    seq_lens=seq_lens
                )
                self.timing_phases[name] += _time_step / 1000
                self.wall_time += _time_step / 1000
                logger.debug(f"    mode=prefill, stage={name}, time spent: {_time_step:.2f} ms")
        else:
            # check pipeline cleared
            if not self.batch_pipeline or self.step - self.pp >= len(self.batch_pipeline):
                oom, reason = self.scheduler.check_oom()
                if oom:
                    raise MemoryError(f"[Prefill] OOM detected on rank {self.dp_rank}: {reason}")

    def update(self):
        ret = None
        if self.step >= self.pp:
            current_batch, cur_running_batch = self.batch_pipeline[self.step - self.pp]
            if current_batch is not None:
                self.scheduler.process_batch_result_prefill(current_batch, cur_running_batch)
                new_complete_reqs = [
                        req for req in current_batch.reqs
                        if current_batch.prefill_targets[req.request_id] >= req.prompt_tokens
                ]
                logger.debug(f"Complete events after PREFILL {[r.request_id for r in new_complete_reqs]}, dp={self.dp_rank}, time = {self.wall_time}")
                ret = (self.wall_time, new_complete_reqs, self.rank_id)
        return ret

class DecodeWorker(BaseWorker):
    def __init__(self,
        dp_rank: str,
        rank_id: int,
        request_list: List[Request],
        model: BaseModel,
        batch_size: int,
        mem_fraction_static: float,
        total_gpu_memory: int,
        pp: int = 1,
        scheduler_kwargs: Dict[str, Any] = dict(),
    ):
        super().__init__(
            dp_rank,
            rank_id,
            request_list,
            model,
            batch_size,
            mem_fraction_static,
            total_gpu_memory,
            pp=pp,
            scheduler_kwargs=scheduler_kwargs,
        )
        self.scheduler = Scheduler(
            waiting_queue=[],
            disaggregation_mode="decode",
            kv_pool=self.kv_pool,
            pp=pp,
            **self.scheduler_kwargs)
        self.timing_phases = { "pre_forward": 0, "forward": 0, "post_forward": 0 }
        self.complete_events = deque()
        self.finished_req_ids = set()
        self.prefill_reserved_reqs: Dict[str, Request] = {}

    def reserve_prefill_kv(self, req: Request) -> bool:
        if req.request_id in self.prefill_reserved_reqs:
            return True
        sched = self.scheduler
        total_reqs = (
            sched.running_batch.batch_size()
            + len(sched.waiting_queue)
            + len(self.prefill_reserved_reqs)
            + 1
        )
        headroom = sched.reserved_decode_tokens * total_reqs
        if sched.kv_allocator.available_tokens < headroom:
            return False
        if not sched.kv_allocator.try_allocate_blocks(req, req.prompt_tokens):
            return False
        self.prefill_reserved_reqs[req.request_id] = req
        self.complete = False
        return True

    def release_prefill_reservation(self, req: Request) -> None:
        if req.request_id in self.prefill_reserved_reqs:
            self.prefill_reserved_reqs.pop(req.request_id)
            self.scheduler.kv_allocator.free(req.request_id)

    def add_complete_event(self, ready_time: float, reqs: List[Request]):
        if not reqs:
            return
        self.complete_events.append((ready_time, reqs))
        self.complete = False

    def fetch_new_request(self):
        while self.complete_events and self.complete_events[0][0] <= self.wall_time:
            _, new_reqs = self.complete_events[0]
            for new_req in new_reqs:
                if new_req.start_decode:
                    continue
                if new_req.decode_tokens + 1 >= new_req.max_new_tokens:
                    self.release_prefill_reservation(new_req)
                    new_req.finished = True
                    new_req.start_decode = True
                    continue
                sched = self.scheduler

                if new_req.request_id in self.prefill_reserved_reqs:
                    logger.debug(
                        f"    move reserved request={new_req.request_id} to decode waiting_queue, chunk_offset={new_req.chunk_offset}"
                    )
                    new_req.start_decode = True
                    sched.waiting_queue.append(new_req)
                    self.request_list.append(new_req)
                if not new_req.start_decode :
                    logger.debug(
                        f"    add new requests failed={new_req.request_id}, reserved={new_req.request_id in self.prefill_reserved_reqs}, "
                        f"avail_tokens={sched.kv_allocator.available_tokens}, chunk_offset={new_req.chunk_offset}"
                    )

            # If not all successfully added, wait next round
            if sum([1 for r in new_reqs if r.start_decode]) == len(new_reqs):
                self.complete_events.popleft()
            else:
                break
    
    def forward(self, ross_models):
        self.step += 1
        batch = self.scheduler.get_next_batch_to_run()
        logger.debug(
            f"[Decode] step: { self.step }; current time: {self.wall_time}; "
            f"avail_tokens={self.scheduler.kv_allocator.available_tokens}\n{batch}"
        )
        self.batch_pipeline.append((batch, self.scheduler.running_batch))
        if batch is not None:
            # ROSSModel Estimate Step N Duration
            req_ids = [r.request_id for r in batch.reqs]
            seq_lens = [r.num_computed_tokens for r in batch.reqs]
            for name in ["pre_forward", "forward", "post_forward"]:
                _time_step = ross_models["decode_" + name].predict(
                    req_ids=req_ids,
                    seq_lens=seq_lens
                )
                self.timing_phases[name] += _time_step / 1000
                self.wall_time += _time_step / 1000
                logger.debug(f"    mode=decode, stage={name}, time spent: {_time_step:.2f} ms")
        else:
            if (
                self.scheduler.running_batch.is_empty()
                and (not self.complete_events)
                and len(self.scheduler.waiting_queue) == 0
            ):
                self.complete = True
                return None
            if self.scheduler.running_batch.is_empty() and self.complete_events:
                # advance time to next ready prefill if idle
                self.wall_time = max(self.wall_time, self.complete_events[0][0])
            else: # pipeline cleared
                if not self.batch_pipeline or self.step - self.pp >= len(self.batch_pipeline):
                    oom, reason = self.scheduler.check_oom()
                    if oom:
                        raise MemoryError(f"[Decode] OOM detected on rank {self.dp_rank}: {reason}")
        return batch

    def update(self, batch: Batch, post_decode_overhead_s: float = 0.0):
        self.scheduler.process_batch_result_decode(batch)
        finished_now = []
        for req_idx, req in enumerate(self.request_list):
            if req.max_new_tokens > 1 and req.request_id not in self.finished_req_ids:
                if update_metrics(req, self.wall_time, req.arrive_time, post_decode_overhead_s=post_decode_overhead_s):
                    self.finished_req_ids.add(req.request_id)
                    finished_now.append(req)
        return finished_now

def run_simulation_disagg_aligned(
    prefill_model: BaseModel,
    decode_model: BaseModel,
    batch_size: int,
    request_source: VirtualClientStore,
    ross_models: Dict[str, Any],
    scheduler_kwargs: Dict[str, Any],
    total_gpu_memory: int | None = None,
    pp: int = 1,
    dp: Tuple[int, int] = (1, 1),
    mem_fraction_static: float = 0.9,
    post_decode_overhead_ms: float = 0.0,
):
    """Two-phase sim: prefill then decode with ready-time arrivals."""
    reset_sgl_predict_stats(ross_models)
    workers_pq = PriorityQueue()
    prefill_workers: List[PrefillWorker] = []
    decode_workers: List[DecodeWorker] = []

    def enqueue_worker(w: BaseWorker):
        if not w.queued:
            workers_pq.put((w.wall_time, w))
            w.queued = True

    for idx in range(dp[0]):
        worker = PrefillWorker(
            dp_rank=f"prefill_dp_{idx}", rank_id=idx,
            request_list=[],
            model=prefill_model,
            batch_size=batch_size,
            mem_fraction_static=mem_fraction_static,
            total_gpu_memory=total_gpu_memory,
            pp=pp,
            scheduler_kwargs=scheduler_kwargs,
        )
        prefill_workers.append(worker)
        enqueue_worker(worker)

    for idx in range(dp[1]):
        worker = DecodeWorker(
            dp_rank=f"decode_dp_{idx}", rank_id=idx,
            request_list=[],
            model=decode_model,
            batch_size=batch_size,
            mem_fraction_static=mem_fraction_static,
            total_gpu_memory=total_gpu_memory,
            pp=1,
            scheduler_kwargs=scheduler_kwargs,
        )
        decode_workers.append(worker)
        enqueue_worker(worker)

    all_workers = prefill_workers + decode_workers
    finished_req_ids = set()
    post_decode_overhead_s = post_decode_overhead_ms / 1000.0
    while True:
        current_global_time = max([w.wall_time for w in all_workers]) if all_workers else 0.0
        new_reqs = request_source.refresh(current_global_time, disaggregation=True)
        for req in new_reqs:
            prefill_workers[req.prefill_dp_rank].add_requests([req])
            enqueue_worker(prefill_workers[req.prefill_dp_rank])

        if workers_pq.empty():
            all_idle = all(w.complete for w in all_workers)
            if all_idle and request_source.should_terminate_idle(len(finished_req_ids)):
                break
            continue

        wall_time, worker = workers_pq.get()
        worker.queued = False
        if worker.complete:
            continue
        if abs(worker.wall_time - wall_time) > 1e-9:
            logger.debug(
                f"[PQDrift] worker={worker.dp_rank}, pq_wall_time={wall_time:.5f}, "
                f"worker_wall_time={worker.wall_time:.5f}"
            )
        # if worker.step > 10000:
        #     raise RuntimeError(f"Worker exceeded step limit: dp_rank={worker.dp_rank}, step={worker.step}, wall_time={worker.wall_time:.5f}")
        logger.debug(f"Worker Rank: {worker.dp_rank}, Step: {worker.step}, Wall Time: {wall_time}")
        if worker.dp_rank.find("prefill") != -1: # Prefill
            worker.fetch_new_request(decode_workers)
            worker.forward(ross_models)
            current_complete = worker.update()
            if current_complete:
                complete_time, complete_reqs, _ = current_complete
                decode_reqs_by_rank: Dict[int, List[Request]] = {}
                for req in complete_reqs:
                    decode_reqs_by_rank.setdefault(req.decode_dp_rank, []).append(req)
                for rank, reqs in decode_reqs_by_rank.items():
                    decode_workers[rank].add_complete_event(complete_time, reqs)
                    enqueue_worker(decode_workers[rank])
        else:
            worker.fetch_new_request()
            batch = worker.forward(ross_models)
            if batch is not None:
                finished_now = worker.update(batch, post_decode_overhead_s=post_decode_overhead_s)
                for req in finished_now:
                    finished_req_ids.add(req.request_id)
                    request_source.record_finish(req.request_id, worker.wall_time + post_decode_overhead_s)

        if not worker.complete:
            enqueue_worker(worker)
            if worker.dp_rank.find("prefill") != -1:
                logger.debug(
                    f"[PrefillRequeue] worker={worker.dp_rank}, wall_time={worker.wall_time:.5f}, "
                    f"pending={len(worker.pending_reqs)}, "
                    f"waiting={len(worker.scheduler.waiting_queue)}"
                )
        else:
            logger.debug(f"Worker {worker.dp_rank} complete at wall_time={worker.wall_time:.2f}")

    request_list = request_source.as_list()
    itl_list = []
    for req in request_list:
        if not req.ttft or not req.e2e_latency:
            raise RuntimeError(f"req={req.request_id}, ttft={req.ttft}, e2e_latency={req.e2e_latency}")
        itl_list.extend(req.itl)

    max_wall_time = max([w.wall_time for w in all_workers]) if all_workers else 0.0
    benchmarks = calulcate_benchmark_results(request_list, itl_list, max_wall_time)
    result_dict = {
        "duration": max_wall_time,
        **benchmarks,
        "prefill_phases": [w.timing_phases for w in prefill_workers],
        "decode_phases": [w.timing_phases for w in decode_workers],
        "sgl_predict_stats": collect_sgl_predict_stats(ross_models),
    }
    result_dict.update({
        "tokens/s": result_dict['throughput'],
        "tokens/s/gpu": result_dict['throughput'] / dp[1] / pp / decode_model.inference_config.tp_size,
        "tokens/s/user": 1000.0 / result_dict['mean_tpot_ms'],
    })
    return result_dict

def run_simulation_disagg_sequential(
    prefill_model: BaseModel,
    decode_model: BaseModel,
    batch_size: int,
    request_list: List[Request],
    ross_models: Dict[str, Any],
    scheduler_kwargs: Dict[str, Any],
    isl: int,
    osl: int,
    total_gpu_memory: int | None = None,
    pp: int = 1,
    dp: Tuple[int, int] = (1, 1),
    mem_fraction_static: float = 0.9,
    warmup_offset: float = 0,
):
    """Two-phase sim: prefill then decode with ready-time arrivals."""
    decode_phases = []

    prefill_scheds : List[Scheduler] = []
    decode_scheds  : List[Scheduler] = []

    for idx in range(dp[0]):
        prefill_kv_pool = SGLKVCachePool(
            model=prefill_model,
            num_reqs=batch_size,
            tokens_per_block=1, # assign tokens_per_block = 1
            gpu_memory_utilization=mem_fraction_static,
            total_gpu_memory=total_gpu_memory,
            framework='sglang',
        )
        prefill_scheds.append(Scheduler(
                                waiting_queue=[],
                                disaggregation_mode="prefill",
                                kv_pool=prefill_kv_pool,
                                pp=pp,
                                **scheduler_kwargs))
    for idx in range(dp[1]):
        decode_kv_pool = SGLKVCachePool(
            model=decode_model,
            num_reqs=batch_size,
            tokens_per_block=1, # assign tokens_per_block = 1
            gpu_memory_utilization=mem_fraction_static,
            total_gpu_memory=total_gpu_memory,
            framework='sglang',
        )
        decode_scheds.append(Scheduler(
                                waiting_queue=[],
                                disaggregation_mode="decode",
                                kv_pool=decode_kv_pool,
                                pp=1,
                                **scheduler_kwargs))
        decode_phases.append({  "decode_pre_forward": 0, "decode_forward": 0, "decode_post_forward": 0, })

    # Phase 1: prefill timeline
    current_status = [ { "wall_time": 0.0, "step": 0, "batch_pipeline": [], "complete": False } for i in range(dp[0])]
    req_prefilled = [False] * len(request_list)
    completed_events: List[Tuple[float, List[Request], int]] = [] # CompleteTime, RequestList, DP_Rank
    ready_idx, decode_rr_assign = 0, 0
    while True:
        while ready_idx < len(request_list):
            if req_prefilled[ready_idx] == True:
                ready_idx += 1
                continue
            req = request_list[ready_idx]
            if req.arrive_time <= current_status[req.prefill_dp_rank]['wall_time']:
                req_prefilled[ready_idx] = True
                ready_idx += 1
                prefill_scheds[req.prefill_dp_rank].waiting_queue.append(req)
            else:
                break
        # idle admission by arrive_time: if a rank is idle, pull the next future req
        for rank, prefill_sched in enumerate(prefill_scheds):
            status = current_status[rank]
            status["step"] += 1
            if not status["complete"]:
                # no in-flight pipeline slots pending
                if check_sched_idle(prefill_sched, status["batch_pipeline"], status["step"], pp):
                    # find the earliest request assigned to this dp_rank with arrive_time > current wall_time
                    future = [
                        (i, r)
                        for i, r in enumerate(request_list)
                        if r.prefill_dp_rank == rank and r.chunk_offset == 0 and r.arrive_time > status["wall_time"]
                    ]
                    if future:
                        idx_next, req_next = min(future, key=lambda x: x[0])
                        status["wall_time"] = req_next.arrive_time
                        req_prefilled[idx_next] = True
                        prefill_sched.waiting_queue.append(req_next)
                    else:
                        status["complete"] = True

        for idx, prefill_sched in enumerate(prefill_scheds):
            batch = prefill_sched.get_next_batch_to_run()
            logger.debug(f"[Prefill] step: { current_status[idx]['step'] }; ready_idx = {ready_idx}\n"
                        f"num_free_blocks={prefill_sched.kv_allocator.num_free_blocks} {batch}\n----")
            current_status[idx]['batch_pipeline'].append((batch, prefill_sched.running_batch))
            if batch is not None:
                # 2. ROSSModel Estimate Step N Duration
                req_ids = [r.request_id for r in batch.reqs]
                seq_lens = [r.num_computed_tokens for r in batch.reqs]
                for name in ["pre_forward", "forward", "post_forward"]:
                    _time_step = ross_models["prefill_" + name].predict(
                        req_ids=req_ids,
                        seq_lens=seq_lens, isl=isl, osl=osl
                    )
                    current_status[idx]['wall_time'] += _time_step / 1000
                    logger.debug(f"    mode=prefill, stage={name}, time spent: {_time_step:.2f} ms")
            else:
                # pipeline cleared
                status = current_status[idx]
                if not status['batch_pipeline'] or status['step'] - pp >= len(status['batch_pipeline']):
                    oom, reason = prefill_sched.check_oom()
                    if oom:
                        raise MemoryError(f"[Prefill] OOM detected on rank {idx}: {reason}")

        for idx, prefill_sched in enumerate(prefill_scheds):
            step = current_status[idx]["step"]
            if step >= pp:
                # logger.debug(f"Batch Pipeline for dp_{idx}: {current_status[idx]['batch_pipeline']}")
                # if step - pp < len(current_status[idx]['batch_pipeline']):
                current_batch, cur_running_batch = current_status[idx]['batch_pipeline'][step - pp]
                if current_batch is not None:
                    prefill_sched.process_batch_result_prefill(current_batch, cur_running_batch)
                    new_complete_reqs = [
                            req for req in current_batch.reqs
                            if current_batch.prefill_targets[req.request_id] >= req.prompt_tokens
                    ]
                    # logger.debug(f"Complete events after PREFILL {[r.request_id for r in new_complete_reqs]}, dp={decode_rr_assign}, time = {current_status[idx]['wall_time']}")
                    # assign decode dp
                    completed_events.append((current_status[idx]['wall_time'], new_complete_reqs, decode_rr_assign))                
                    decode_rr_assign = (decode_rr_assign + 1) % dp[1]
                    # logger.debug(f"    new complete events: {len(new_complete_reqs)}\n"
                    #             f"complete events: ", completed_events)

        complete_cnt = sum([(r["complete"] == True) for r in current_status])
        if complete_cnt >= dp[0]:
            break

    completed_events.sort(key=lambda x: x[0])

    # Phase 2: decode timeline (fix pp = 1)
    current_status = [ { "wall_time": 0.0, "step": 0, "batch_pipeline": [] } for i in range(dp[1])]
    ready_idx = 0
    complete_workers = 0
    while complete_workers < dp[1]:
        complete_workers = 0
        # inject ready reqs by timex
        while ready_idx < len(completed_events) and completed_events[ready_idx][0] <= current_status[ completed_events[ready_idx][2] ]['wall_time']:
            new_reqs = completed_events[ready_idx][1]
            for new_req in new_reqs:
                if new_req.decode_tokens + 1 >= new_req.max_new_tokens:
                    new_req.finished = True
                    new_req.start_decode = True
                    continue
                if new_req.start_decode:
                    continue
                decode_sched = decode_scheds[completed_events[ready_idx][2]]

                # Admission headroom check similar to decode worker
                total_reqs = decode_sched.running_batch.batch_size() + 1 # len(decode_sched.waiting_queue) + 1
                headroom = decode_sched.reserved_decode_tokens * total_reqs
                if decode_sched.kv_allocator.available_tokens >= headroom:
                    if decode_sched.kv_allocator.try_allocate_blocks(new_req, new_req.chunk_offset):
                        logger.debug(f"    add new requests={new_req.request_id}, avail_tokens={decode_sched.kv_allocator.available_tokens}"
                                    f"head_room={headroom}, chunk_offset={new_req.chunk_offset}")
                        new_req.start_decode = True
                        decode_sched.waiting_queue.append(new_req)
                if not new_req.start_decode :
                    logger.debug(f"    add new requests failed={new_req.request_id}, avail_tokens={decode_sched.kv_allocator.available_tokens}"
                                f"head_room={headroom}, chunk_offset={new_req.chunk_offset}")

            # If not all successfully added, wait next round
            if sum([1 for r in new_reqs if r.start_decode]) == len(new_reqs):
                ready_idx += 1
            else:
                break

        for idx, decode_sched in enumerate(decode_scheds):
            current_status[idx]["step"] += 1
            batch = decode_sched.get_next_batch_to_run()
            logger.debug(f"[Decode] step: { current_status[idx]['step'] }; current time: { current_status[idx]['wall_time'] }; "
                    f"avail_tokens={decode_sched.kv_allocator.available_tokens}, ready_idx={ready_idx}")
            logger.debug(f"running_batch: {len(decode_sched.running_batch.reqs)}; waiting_queue = {[r.request_id for r in decode_sched.waiting_queue]}")
            logger.debug(f"batch: {batch}\n----")

            if batch is None:
                if decode_sched.running_batch.is_empty() and ready_idx >= len(completed_events) and not decode_sched.waiting_queue:
                    complete_workers += 1
                    continue
                # advance time to next ready prefill if idle
                if decode_sched.running_batch.is_empty() and ready_idx < len(completed_events):
                    current_status[idx]['wall_time'] = max(current_status[idx]['wall_time'], completed_events[ready_idx][0])
                    continue
                else:
                    # pipeline cleared
                    status = current_status[idx]
                    if not status['batch_pipeline'] or ['step'] - pp >= len(status['batch_pipeline']):
                        oom, reason = decode_sched.check_oom()
                        if oom:
                            raise MemoryError(f"[Decode] OOM detected on rank {idx}: {reason}")
                    continue

            req_ids = [r.request_id for r in batch.reqs]
            seq_lens = [r.num_computed_tokens for r in batch.reqs]
            for name in ["forward", "post_forward"]: # "pre_forward", 
                _time_step = ross_models["decode_" + name].predict(
                    req_ids=req_ids,
                    seq_lens=seq_lens, isl=isl, osl=osl
                )
                current_status[idx]['wall_time'] += _time_step / 1000
                decode_phases[idx]["decode_" + name] += _time_step / 1000
                logger.debug(f"    mode=decode, stage={name}, time spent: {_time_step:.2f} ms")

            decode_sched.process_batch_result_decode(batch)

        for req_idx, req in enumerate(request_list):
            if req.max_new_tokens > 1:
                update_metrics(req, current_status[ req.decode_dp_rank ]["wall_time"], req.arrive_time)

    for req in request_list:
        if not req.ttft or not req.ttlt:
            raise RuntimeError(f"req={req.request_id}, ttft={req.ttft}, ttlt={req.ttlt}")

    max_wall_time = max([status["wall_time"] for status in current_status])
    benchmarks = calulcate_benchmark_results(request_list[1:], max_wall_time - warmup_offset)
    result_dict = {
        "duration": max_wall_time - warmup_offset,
        **benchmarks,
        "decode_phases": decode_phases,
    }
    result_dict.update({
        "tokens/s": result_dict['throughput'],
        "tokens/s/gpu": result_dict['throughput'] / dp[1] / pp / decode_model.inference_config.tp_size,
        "tokens/s/user": 1000.0 / result_dict['mean_tpot_ms'],
    })
    return result_dict

def run_sim(args):
    scheduler_kwargs = {
        "chunked_prefill_size": args.chunked_prefill_size,
        "reserved_decode_tokens": args.reserved_decode_tokens,
        "max_running_requests": args.batch_size,
        "enable_prefix_caching": getattr(args, "enable_prefix_caching", False),
    }
    if scheduler_kwargs["enable_prefix_caching"]:
        _load_sim_prefix_cache()
    platform_perf = PlatformPerf(platform_perf_yaml=args.platform_perf)

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

    request_store = VirtualClientStore(
        frontend_path=args.frontend_path,
        request_rate=float(args.request_rate),
        max_concurrency=args.batch_size,
        dp_size=args.prefill_dp_size,
        disaggregation=args.disaggregation,
    )
    # logger.debug([(r.request_id, r.arrive_time) for r in requests])
    if not args.disaggregation:
        exit(1)
    else:
        prefill_model_path = {
            "prefill_pre_forward": args.prefill_pre_forward_path,
            "prefill_forward": args.prefill_forward_path,
            "prefill_post_forward": args.prefill_post_forward_path,
        }
        decode_model_path = {
            "decode_pre_forward": args.decode_pre_forward_path,
            "decode_forward": args.decode_forward_path,
            "decode_post_forward": args.decode_post_forward_path,
        }
        prefill_model, prefill_ross_model_dict, _ = _init_worker_config(
            tp_size=args.prefill_tp_size,
            pp_size=args.prefill_pp_size,
            model_path=prefill_model_path,
        )
        decode_model, decode_ross_model_dict, _ = _init_worker_config(
            tp_size=args.decode_tp_size,
            pp_size=1,
            model_path=decode_model_path,
        )
        ross_model_dict = { **prefill_ross_model_dict, **decode_ross_model_dict }

        ret = run_simulation_disagg_aligned(
            prefill_model=prefill_model,
            decode_model=decode_model,

            batch_size=args.batch_size,
            request_source=request_store,
            total_gpu_memory=platform_perf.theoretical_memory_gb * (1024 ** 3),

            scheduler_kwargs=scheduler_kwargs,
            mem_fraction_static=args.mem_fraction_static,

            ross_models=ross_model_dict,
            
            dp=(args.prefill_dp_size, args.decode_dp_size),
            pp=args.prefill_pp_size,
            post_decode_overhead_ms=getattr(args, "post_decode_overhead_ms", 0.0),
        )
        ret.update({
            "prefill_dp": args.prefill_dp_size,
            "prefill_pp": args.prefill_pp_size,
            "prefill_tp": args.prefill_tp_size,
            "decode_dp": args.decode_dp_size,
            "decode_pp": 1,
            "decode_tp": args.decode_tp_size,
        })
    
    ret.update(scheduler_kwargs)
    ret.update({
        "mem_fraction_static": args.mem_fraction_static,
    })
    return ret
