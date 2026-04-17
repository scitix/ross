#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import sys
from collections import deque
from pathlib import Path
from queue import PriorityQueue
from typing import Any, Dict, List
import itertools

TEST_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TEST_ROOT))
TEST_SIM_ROOT = TEST_ROOT / "test" / "simulator-vllm"
if str(TEST_SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_SIM_ROOT))

from common.config import InferenceConfig
from common.features import PlatformPerf
from common.kvpool import KVCachePool
from common.models import get_model
from common.ross_model import ROSSModel
from common.sim_http_perf import VirtualClientStore
from dummy_sched import (
    create_pd_decode_request,
    create_request as create_dummy_request,
    create_sidecar_scheduler,
    make_dummy_model_output,
)
from sidecar_hook import (
    choose_timing_output,
    compare_schedule_outputs,
    log_sidecar_schedule,
)

from scheduler.request import Request, RequestStatus
from scheduler.scheduler import Scheduler, SchedulerOutput

from simulator_main import (
    _get_gpu_memory_utilization,
    get_ross_model_paths,
    calulcate_benchmark_results,
    check_pipeline_clear,
    get_regression_model,
    get_ross_models,
    load_memory_increase,
    parse_args,
    run_sim as legacy_run_sim,
    update_metrics,
)

logger = logging.getLogger(__name__)


def _snapshot_worker_state(worker: "BaseWorker") -> tuple:
    return (
        worker.wall_time,
        len(worker.pending_reqs),
        len(worker.active_reqs),
        len(worker.scheduler.waiting),
        len(worker.scheduler.running),
        worker.scheduler.kv_cache_manager.num_free_blocks,
    )


class BaseWorker:
    def __init__(
        self,
        worker_type: str,
        rank_id: int,
        scheduler: Scheduler,
        pp: int,
        sidecar_scheduler: Any = None,
        vllm_src_root: str = "",
        vllm_result_source: str = "sim",
        compare_vllm_schedule: bool = False,
    ):
        self.worker_type = worker_type
        self.rank_id = rank_id
        self.scheduler = scheduler
        self.pp = pp
        self.sidecar_scheduler = sidecar_scheduler
        self.vllm_src_root = vllm_src_root
        self.vllm_result_source = vllm_result_source
        self.compare_vllm_schedule = compare_vllm_schedule
        self.wall_time = 0.0
        self.step = 0
        self.batch_pipeline: List[SchedulerOutput | None] = []
        self.sidecar_batch_pipeline: List[Any] = []
        self.complete = False
        self.queued = False

    def __lt__(self, other):
        if not isinstance(other, BaseWorker):
            return NotImplemented
        if self.wall_time == other.wall_time:
            if self.rank_id == other.rank_id:
                return self.worker_type < other.worker_type
            return self.rank_id < other.rank_id
        return self.wall_time < other.wall_time

    def next_wakeup_time(self) -> float | None:
        return None


class PrefillWorker(BaseWorker):
    def __init__(self, rank_id: int, scheduler: Scheduler, pp: int, sidecar_scheduler: Any = None, vllm_src_root: str = "", vllm_result_source: str = "sim", compare_vllm_schedule: bool = False):
        super().__init__("prefill", rank_id, scheduler, pp, sidecar_scheduler=sidecar_scheduler, vllm_src_root=vllm_src_root, vllm_result_source=vllm_result_source, compare_vllm_schedule=compare_vllm_schedule)
        self.pending_reqs = deque()
        self.active_reqs: Dict[str, Request] = {}
        self.decode_assign_idx = 0
        self.timing_phases = {"pre_forward": 0, "forward": 0, "post_forward": 0}

    def add_requests(self, reqs: List[Request]) -> None:
        if not reqs:
            return
        self.pending_reqs.extend(reqs)
        self.complete = False

    def next_wakeup_time(self) -> float | None:
        if not self.pending_reqs:
            return None
        return self.pending_reqs[0].ready_time

    def fetch_new_requests(self) -> None:
        while self.pending_reqs and self.pending_reqs[0].ready_time <= self.wall_time:
            req = self.pending_reqs.popleft()
            self.scheduler.add_request(req)
            self.active_reqs[req.request_id] = req
            if self.sidecar_scheduler is not None:
                self.sidecar_scheduler.add_request(
                    create_dummy_request(
                        request_id=req.request_id,
                        prompt_token_len=req.prompt_tokens,
                        max_output_tokens=req.num_tokens - req.prompt_tokens,
                        block_size=self.scheduler.kv_cache_manager.tokens_per_block,
                        vllm_src_root=self.vllm_src_root,
                    )
                )
        if (
            not self.complete
            and check_pipeline_clear(self.batch_pipeline, self.step, self.pp)
            and self.scheduler.should_terminate()
        ):
            if self.pending_reqs:
                req = self.pending_reqs.popleft()
                self.wall_time = max(self.wall_time, req.ready_time)
                self.scheduler.add_request(req)
                self.active_reqs[req.request_id] = req
                if self.sidecar_scheduler is not None:
                    self.sidecar_scheduler.add_request(
                        create_dummy_request(
                            request_id=req.request_id,
                            prompt_token_len=req.prompt_tokens,
                            max_output_tokens=req.num_tokens - req.prompt_tokens,
                            block_size=self.scheduler.kv_cache_manager.tokens_per_block,
                            vllm_src_root=self.vllm_src_root,
                        )
                    )
            else:
                self.complete = True

    def forward(self, ross_models: Dict[str, Any], isl: int, osl: int) -> None:
        self.step += 1
        schedule_output = self.scheduler.schedule()
        self.batch_pipeline.append(schedule_output)
        sidecar_output = None
        timing_output = schedule_output
        if self.sidecar_scheduler is not None:
            sidecar_output = self.sidecar_scheduler.schedule()
            log_sidecar_schedule(self.step, self.rank_id, sidecar_output, phase="prefill")
            if self.compare_vllm_schedule:
                compare_schedule_outputs(
                    self.step,
                    self.rank_id,
                    schedule_output,
                    sidecar_output,
                    ross_scheduler=self.scheduler,
                    sidecar_scheduler=self.sidecar_scheduler,
                )
            timing_output = choose_timing_output(
                schedule_output,
                sidecar_output,
                self.vllm_result_source,
                align_transfer_loaded_to_sim=False,
            )
            self.sidecar_batch_pipeline.append(sidecar_output)
        if schedule_output and (not self.scheduler.should_terminate()):
            self.scheduler.debug_print_schedule(schedule_output, self.rank_id, self.step)
            for name in ["pre_forward", "forward", "post_forward"]:
                regression_model = get_regression_model(ross_models, name, "prefill")
                _time_step = regression_model.predict(
                    req_ids=timing_output.scheduled_req_ids,
                    prefill_seq_lens=timing_output.prefill_seq_lens,
                    decode_seq_lens=timing_output.decode_seq_lens,
                    isl=isl,
                    osl=osl,
                )
                self.wall_time += _time_step / 1000
                self.timing_phases[name] += _time_step / 1000
                logger.debug(f"       {name} Time: {_time_step} ms")
        elif check_pipeline_clear(self.batch_pipeline, self.step, self.pp):
            oom, reason = self.scheduler.check_oom()
            if oom:
                raise MemoryError(f"OOM detected on prefill rank {self.rank_id}: {reason}")

    def update(self) -> List[Request]:
        if self.step < self.pp or self.complete:
            return []
        current_output = self.batch_pipeline[self.step - self.pp]
        if self.sidecar_scheduler is not None:
            current_sidecar_output = self.sidecar_batch_pipeline[self.step - self.pp]
            if current_sidecar_output is not None and current_sidecar_output.total_num_scheduled_tokens > 0:
                dummy_output = make_dummy_model_output(self.sidecar_scheduler, current_sidecar_output)
                self.sidecar_scheduler.update_from_output(current_sidecar_output, dummy_output)
        self.scheduler.update_from_output(current_output)
        completed = []
        for rid, req in list(self.active_reqs.items()):
            if not req.prefill_end_time and req.output_len == 1:
                req.prefill_end_time = self.wall_time
                req.status = RequestStatus.FINISHED
                self.scheduler.kv_cache_manager.free(req.request_id)
                if self.sidecar_scheduler is not None:
                    sidecar_req = self.sidecar_scheduler.requests.get(req.request_id)
                    if sidecar_req is not None:
                        self.sidecar_scheduler.finish_requests(
                            req.request_id,
                            type(sidecar_req.status).FINISHED_STOPPED,
                        )
                completed.append(req)
                self.active_reqs.pop(rid, None)
        self.scheduler.running = [req for req in self.scheduler.running if req.status != RequestStatus.FINISHED]
        return completed


class DecodeWorker(BaseWorker):
    def __init__(self, rank_id: int, scheduler: Scheduler, pp: int, sidecar_scheduler: Any = None, vllm_src_root: str = "", vllm_result_source: str = "sim", compare_vllm_schedule: bool = False):
        super().__init__("decode", rank_id, scheduler, pp, sidecar_scheduler=sidecar_scheduler, vllm_src_root=vllm_src_root, vllm_result_source=vllm_result_source, compare_vllm_schedule=compare_vllm_schedule)
        self.pending_reqs = deque()
        self.active_reqs: Dict[str, Request] = {}
        self.timing_phases = {"pre_forward": 0, "forward": 0, "post_forward": 0}

    def add_requests(self, reqs: List[Request]) -> None:
        if not reqs:
            return
        self.pending_reqs.extend(reqs)
        self.complete = False

    def next_wakeup_time(self) -> float | None:
        if not self.pending_reqs:
            return None
        return self.pending_reqs[0].prefill_end_time

    def fetch_new_requests(self) -> None:
        while self.pending_reqs and self.pending_reqs[0].prefill_end_time <= self.wall_time:
            req = self.pending_reqs.popleft()
            req.decode_init()
            self.scheduler.add_request(req)
            self.active_reqs[req.request_id] = req
        if (
            not self.complete
            and check_pipeline_clear(self.batch_pipeline, self.step, self.pp)
            and self.scheduler.should_terminate()
        ):
            if self.pending_reqs:
                req = self.pending_reqs.popleft()
                self.wall_time = max(self.wall_time, req.prefill_end_time)
                req.decode_init()
                self.scheduler.add_request(req)
                self.active_reqs[req.request_id] = req
            else:
                self.complete = True

    def forward(self, ross_models: Dict[str, Any], isl: int, osl: int) -> None:
        self.step += 1
        transfer_loaded_waiting = {
            req.request_id: req
            for req in self.scheduler.waiting
            if getattr(req, "transfer_loaded", False)
        }
        schedule_output = self.scheduler.schedule()
        timing_output = schedule_output
        if self.sidecar_scheduler is not None:
            if schedule_output is not None:
                for req in itertools.chain(
                    schedule_output.new_reqs or [],
                    schedule_output.resumed_reqs or [],
                    schedule_output.running_reqs or [],
                ):
                    if (
                        req.request_id in transfer_loaded_waiting
                        and req.request_id not in self.sidecar_scheduler.requests
                    ):
                        original_req = transfer_loaded_waiting[req.request_id]
                        self.sidecar_scheduler.add_request(
                            create_pd_decode_request(
                                request_id=original_req.request_id,
                                prompt_token_len=original_req.prompt_tokens,
                                max_output_tokens=original_req.num_tokens - original_req.prompt_tokens,
                                block_size=self.scheduler.kv_cache_manager.tokens_per_block,
                                vllm_src_root=self.vllm_src_root,
                            )
                        )
            sidecar_output = self.sidecar_scheduler.schedule()
            log_sidecar_schedule(self.step, self.rank_id, sidecar_output, phase="decode")
            if self.compare_vllm_schedule:
                compare_schedule_outputs(
                    self.step,
                    self.rank_id,
                    schedule_output,
                    sidecar_output,
                    ross_scheduler=self.scheduler,
                    sidecar_scheduler=self.sidecar_scheduler,
                    treat_transfer_loaded_as_decode_one=True,
                )
            timing_output = choose_timing_output(
                schedule_output,
                sidecar_output,
                self.vllm_result_source,
                align_transfer_loaded_to_sim=True,
            )
            self.sidecar_batch_pipeline.append(sidecar_output)
        self.batch_pipeline.append(schedule_output)
        if schedule_output and (not self.scheduler.should_terminate()):
            self.scheduler.debug_print_schedule(schedule_output, self.rank_id, self.step)
            for name in ["pre_forward", "forward", "post_forward"]:
                regression_model = get_regression_model(ross_models, name, "decode")
                _time_step = regression_model.predict(
                    req_ids=timing_output.scheduled_req_ids,
                    prefill_seq_lens=timing_output.prefill_seq_lens,
                    decode_seq_lens=timing_output.decode_seq_lens,
                    isl=isl,
                    osl=osl,
                )
                self.wall_time += _time_step / 1000
                self.timing_phases[name] += _time_step / 1000
                logger.debug(f"       {name} Time: {_time_step} ms")
        elif check_pipeline_clear(self.batch_pipeline, self.step, self.pp):
            oom, reason = self.scheduler.check_oom()
            if oom:
                raise MemoryError(f"OOM detected on decode rank {self.rank_id}: {reason}")

    def update(self) -> List[Request]:
        if self.step < self.pp or self.complete:
            return []
        current_output = self.batch_pipeline[self.step - self.pp]
        if self.sidecar_scheduler is not None:
            current_sidecar_output = self.sidecar_batch_pipeline[self.step - self.pp]
            if current_sidecar_output is not None and current_sidecar_output.total_num_scheduled_tokens > 0:
                dummy_output = make_dummy_model_output(self.sidecar_scheduler, current_sidecar_output)
                self.sidecar_scheduler.update_from_output(current_sidecar_output, dummy_output)
        self.scheduler.update_from_output(current_output)
        self.scheduler.running = [req for req in self.scheduler.running if req.status != RequestStatus.FINISHED]
        finished = []
        for rid, req in list(self.active_reqs.items()):
            if update_metrics(req, self.wall_time, req.arrive_time, self.step):
                finished.append(req)
                self.active_reqs.pop(rid, None)
        return finished


def run_simulation_disagg_aligned(
    prefill_model,
    decode_model,
    batch_size: int,
    request_source: VirtualClientStore,
    prefill_ross_models: Dict[str, Any],
    decode_ross_models: Dict[str, Any],
    scheduler_kwargs: Dict[str, Any],
    isl: int,
    osl: int,
    gpu_memory_utilization: float,
    prefill_memory_profiling: Dict[str, Any],
    decode_memory_profiling: Dict[str, Any],
    total_gpu_memory: int | None = None,
    dp: int = 1,
    pp: int = 1,
    validate_vllm_schedule: bool = False,
    compare_vllm_schedule: bool = False,
    vllm_src_root: str = "",
    vllm_result_source: str = "sim",
) -> Dict[str, Any]:
    if not validate_vllm_schedule and vllm_result_source != "sim":
        raise ValueError("vllm_result_source='vllm' requires validate_vllm_schedule to be enabled")

    workers_pq = PriorityQueue()
    prefill_workers: List[PrefillWorker] = []
    decode_workers: List[DecodeWorker] = []

    def enqueue_worker(worker: BaseWorker) -> None:
        if not worker.queued:
            workers_pq.put((worker.wall_time, worker))
            worker.queued = True

    tokens_per_block = 16 if prefill_model.model_uri.lower().find("deepseek") == -1 else 64
    for idx in range(dp):
        prefill_scheduler = Scheduler(
            max_running_reqs=batch_size,
            kv_pool=KVCachePool(
                model=prefill_model,
                num_reqs=batch_size,
                tokens_per_block=tokens_per_block,
                total_gpu_memory=total_gpu_memory,
                gpu_memory_utilization=gpu_memory_utilization,
                vllm_non_torch_increase=prefill_memory_profiling["non_torch_mem_increase"],
            ),
            **scheduler_kwargs,
        )
        prefill_sidecar = None
        if validate_vllm_schedule:
            prefill_sidecar = create_sidecar_scheduler(
                scheduler_kwargs=scheduler_kwargs,
                num_blocks=prefill_scheduler.kv_cache_manager.num_blocks,
                block_size=tokens_per_block,
                max_num_seqs=batch_size,
                max_model_len=prefill_scheduler.max_model_len,
                model_uri=prefill_model.model_uri,
                vllm_src_root=vllm_src_root,
            )
        worker = PrefillWorker(idx, prefill_scheduler, pp, sidecar_scheduler=prefill_sidecar, vllm_src_root=vllm_src_root, vllm_result_source=vllm_result_source, compare_vllm_schedule=compare_vllm_schedule)
        prefill_workers.append(worker)
        enqueue_worker(worker)

    for idx in range(dp):
        decode_scheduler = Scheduler(
            max_running_reqs=batch_size,
            kv_pool=KVCachePool(
                model=decode_model,
                num_reqs=batch_size,
                tokens_per_block=tokens_per_block,
                total_gpu_memory=total_gpu_memory,
                gpu_memory_utilization=gpu_memory_utilization,
                vllm_non_torch_increase=decode_memory_profiling["non_torch_mem_increase"],
            ),
            **scheduler_kwargs,
        )
        decode_sidecar = None
        if validate_vllm_schedule:
            decode_sidecar = create_sidecar_scheduler(
                scheduler_kwargs=scheduler_kwargs,
                num_blocks=decode_scheduler.kv_cache_manager.num_blocks,
                block_size=tokens_per_block,
                max_num_seqs=batch_size,
                max_model_len=decode_scheduler.max_model_len,
                model_uri=decode_model.model_uri,
                vllm_src_root=vllm_src_root,
            )
        worker = DecodeWorker(idx, decode_scheduler, 1, sidecar_scheduler=decode_sidecar, vllm_src_root=vllm_src_root, vllm_result_source=vllm_result_source, compare_vllm_schedule=compare_vllm_schedule)
        decode_workers.append(worker)
        enqueue_worker(worker)

    all_workers: List[BaseWorker] = prefill_workers + decode_workers
    finished_req_ids = set()
    req_batch = 1
    max_wall_time = 0.0
    global_dp_assign_idx = 0
    while True:
        current_global_time = max((w.wall_time for w in all_workers), default=0.0)
        new_reqs = request_source.refresh(current_global_time, disaggregation=True)
        for req in new_reqs:
            prefill_workers[req.prefill_dp_rank].add_requests([req])
            enqueue_worker(prefill_workers[req.prefill_dp_rank])

        if workers_pq.empty():
            if all(w.complete for w in all_workers) and request_source.should_terminate_idle(len(finished_req_ids)):
                break
            continue

        _, worker = workers_pq.get()
        worker.queued = False
        if worker.complete:
            continue

        prev_state = _snapshot_worker_state(worker)
        logger.debug(f"Worker Rank: [{worker.worker_type}] {worker.rank_id}, Step: {worker.step}, Wall Time: {worker.wall_time}")
        if worker.worker_type == "prefill":
            worker.fetch_new_requests()
            worker.forward(prefill_ross_models, isl, osl)
            completed = worker.update()
            decode_reqs_by_rank: Dict[int, List[Request]] = {}
            for req in completed:
                if req.num_tokens - req.prompt_tokens <= 1:
                    update_metrics(req, worker.wall_time, req.arrive_time, worker.step)
                    finished_req_ids.add(req.request_id)
                    request_source.record_finish(req.request_id, req.prefill_end_time)
                    continue
                req.decode_dp_rank = (worker.decode_assign_idx // req_batch) % dp
                worker.decode_assign_idx += 1
                decode_reqs_by_rank.setdefault(req.decode_dp_rank, []).append(req)
            for rank, reqs in decode_reqs_by_rank.items():
                decode_workers[rank].add_requests(reqs)
                enqueue_worker(decode_workers[rank])
        else:
            worker.fetch_new_requests()
            worker.forward(decode_ross_models, isl, osl)
            finished = worker.update()
            for req in finished:
                request_source.record_finish(req.request_id, worker.wall_time)
                finished_req_ids.add(req.request_id)

        curr_state = _snapshot_worker_state(worker)
        if (
            not worker.complete
            and worker.batch_pipeline
            and worker.batch_pipeline[-1] is None
            and curr_state == prev_state
        ):
            if curr_state[1:5] == (0, 0, 0, 0):
                worker.complete = True
            else:
                next_wakeup_time = worker.next_wakeup_time()
                if (
                    curr_state[2:5] == (0, 0, 0)
                    and next_wakeup_time is not None
                    and next_wakeup_time > worker.wall_time
                ):
                    worker.wall_time = next_wakeup_time
                else:
                    raise RuntimeError(
                        "Aligned vLLM simulator made no progress for worker "
                        f"{worker.worker_type}[{worker.rank_id}] at step={worker.step}. "
                        f"state=(wall_time={worker.wall_time}, pending={len(worker.pending_reqs)}, "
                        f"active={len(worker.active_reqs)}, waiting={len(worker.scheduler.waiting)}, "
                        f"running={len(worker.scheduler.running)}, "
                        f"free_blocks={worker.scheduler.kv_cache_manager.num_free_blocks}, "
                        f"next_wakeup_time={next_wakeup_time})"
                    )

        if not worker.complete:
            enqueue_worker(worker)
        else:
            max_wall_time = max(max_wall_time, worker.wall_time)

    request_list = request_source.as_list()
    itl_list = []
    for req in request_list:
        if req.ttft is None or req.e2e_latency is None:
            raise RuntimeError(f"req={req.request_id}, ttft={req.ttft}, e2e_latency={req.e2e_latency}")
        itl_list.extend(req.itl or [])

    for dpi, worker in enumerate(decode_workers):
        # worker.timing_phases["pp_pre_forward"] = decode_pp_pre_model.predict(
        #     req_ids=["req_id"] * batch_size,
        #     prefill_seq_lens=[],
        #     decode_seq_lens=[0 for _ in range(worker.step)],
        #     isl=isl,
        #     osl=osl,
        # )
        max_wall_time = max(max_wall_time, worker.wall_time)

    benchmarks = calulcate_benchmark_results(request_list, itl_list, max_wall_time)
    result_dict = {
        "duration": max_wall_time,
        **benchmarks,
        "prefill_phases": [w.timing_phases for w in prefill_workers],
        "decode_phases": [w.timing_phases for w in decode_workers],
    }
    result_dict.update({
        "tokens/s": result_dict["throughput"],
        "tokens/s/gpu": result_dict["throughput"] / dp / decode_model.inference_config.tp_size,
        "tokens/s/user": 1000.0 / result_dict["mean_tpot_ms"],
    })
    return result_dict


def run_sim(args):
    if not args.disaggregation:
        return legacy_run_sim(args)

    gpu_memory_utilization = _get_gpu_memory_utilization(args)
    scheduler_kwargs = {
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "enable_prefix_caching": getattr(args, "enable_prefix_caching", False),
    }
    platform_perf = PlatformPerf(platform_perf_yaml=args.platform_perf)

    def _init_worker_config(phase: str):
        infer_config = InferenceConfig(
            dp_size=args.dp_size,
            pp_size=1,
            tp_size=getattr(args, f"{phase}tp_size"),
        )
        model = get_model(args.model_uri, infer_config)
        ross_model_dict = get_ross_models(
            model,
            platform_perf,
            infer_config,
            model_path=get_ross_model_paths(args),
        )
        return model, ross_model_dict

    request_store = VirtualClientStore(
        args.frontend_path,
        args.request_rate,
        args.batch_size,
        args.dp_size,
        args.disaggregation,
    )

    prefill_memory_increase = load_memory_increase(args.mem_profiling_path, {"pp": 1, "tp": args.prefill_tp_size})
    decode_memory_increase = load_memory_increase(args.mem_profiling_path, {"pp": 1, "tp": args.decode_tp_size})
    prefill_model, prefill_ross_models = _init_worker_config("prefill_")
    decode_model, decode_ross_models = _init_worker_config("decode_")
    ret = run_simulation_disagg_aligned(
        prefill_model=prefill_model,
        decode_model=decode_model,
        batch_size=args.batch_size,
        request_source=request_store,
        prefill_ross_models=prefill_ross_models,
        decode_ross_models=decode_ross_models,
        scheduler_kwargs=scheduler_kwargs,
        isl=args.max_prompt_len,
        osl=args.max_output_len,
        gpu_memory_utilization=gpu_memory_utilization,
        prefill_memory_profiling=prefill_memory_increase,
        decode_memory_profiling=decode_memory_increase,
        total_gpu_memory=platform_perf.theoretical_memory_gb * (1024 ** 3),
        dp=args.dp_size,
        pp=1,
        validate_vllm_schedule=getattr(args, "validate_vllm_schedule", False),
        compare_vllm_schedule=getattr(args, "compare_vllm_schedule", False),
        vllm_src_root=getattr(args, "vllm_src_root", ""),
        vllm_result_source=getattr(args, "vllm_result_source", "sim"),
    )
    ret.update(scheduler_kwargs)
    ret.update({
        "gpu_memory_utilization": gpu_memory_utilization,
    })
    return ret
