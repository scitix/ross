#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Iterable, List
import heapq
import numpy as np
from scheduler.request import Request
from common.loader import load_online_requests_w_arrivals

import logging
logger = logging.getLogger(__name__)

class RequestStore:
    """A thin request container to replace bare List[Request] in simulator APIs."""

    def __init__(self, requests: Iterable[Request] | None = None):
        self._requests: List[Request] = list(requests or [])

    def __len__(self) -> int:
        return len(self._requests)

    def __iter__(self):
        return iter(self._requests)

    def __getitem__(self, idx: int) -> Request:
        return self._requests[idx]

    def append(self, req: Request) -> None:
        self._requests.append(req)

    def extend(self, reqs: Iterable[Request]) -> None:
        self._requests.extend(reqs)

    def as_list(self) -> List[Request]:
        return self._requests

    @property
    def is_online(self) -> bool:
        return False

    def refresh(self) -> int:
        return 0

    def should_terminate_idle(self, completed_count: int = 0) -> bool:
        return True

    def prepare_decode_requests(self, dp_size: int, req_batch: int) -> None:
        raise RuntimeError("prepare_decode_requests is only supported for VirtualClientStore")

class VirtualClientStore(RequestStore):
    def __init__(self, frontend_path: str, request_rate: str, max_concurrency: int,
        dp_size: int,
        disaggregation: bool,
    ):
        super().__init__([])
        self.max_concurrency = max_concurrency
        self.request_rate = request_rate
        self.disaggregation = disaggregation
        self.requests, _ = load_online_requests_w_arrivals(
            frontend_path, dp_size,
            disaggregation=disaggregation,
        )
        self.num_prompts = len(self.requests)

        if request_rate == "inf":
            self.arrival_times = np.zeros(self.num_prompts, dtype=float)
        else:
            # Frontend logs already encode the global arrival pattern before
            # client-side concurrency throttling. Preserve these absolute
            # arrivals; available slots only delay request readiness/admission.
            self.arrival_times = np.array(
                [max(0.0, float(req.arrive_time)) for req in self.requests],
                dtype=float,
            )

        self.dp_size = dp_size
        self.next_to_admit_idx = 0
        self.inflight = 0
        self.available_slots = None
        self.request_slots = {}
        self.last_wall_time = 0.0
        self.decode_req_batch = 3

    def record_finish(self, rid, finish_time):
        rid = str(rid)
        if rid not in self.request_slots:
            raise RuntimeError(f"record_finish called for unknown slot mapping (rid={rid})")
        slot_id = self.request_slots.pop(rid)
        heapq.heappush(self.available_slots, (finish_time, slot_id))
        if self.inflight > 0:
            self.inflight -= 1
        else:
            raise RuntimeError(f"record_finish called when inflight=0 (rid={rid})")

    def refresh(self, current_wall_time, disaggregation = None) -> int:
        if disaggregation is None:
            disaggregation = self.disaggregation
        if self.available_slots is None:
            self.available_slots = [(current_wall_time, slot_id) for slot_id in range(self.max_concurrency)]
            heapq.heapify(self.available_slots)

        effective_wall_time = current_wall_time
        if self.inflight == 0 and self.next_to_admit_idx < self.num_prompts and self.available_slots:
            next_req = self.requests[self.next_to_admit_idx]
            next_ingress_t = float(self.arrival_times[self.next_to_admit_idx])
            next_ready_t = next_ingress_t + next_req.tokenize_time
            effective_wall_time = max(effective_wall_time, next_ready_t)

        self.last_wall_time = effective_wall_time
        new_requests = []
        num_limit, iteration = 0, len(self._requests)
        if iteration == 0:
            num_limit = 1

        admit_t = effective_wall_time
        while self.next_to_admit_idx < self.num_prompts and self.available_slots:
            idx = self.next_to_admit_idx
            new_req = self.requests[idx]
            slot_ready_t, slot_id = self.available_slots[0]
            ingress_t = float(self.arrival_times[idx])
            ready_t = ingress_t + new_req.tokenize_time
            admit_t = max(ready_t, slot_ready_t)

            if admit_t <= effective_wall_time:
                heapq.heappop(self.available_slots)
                if disaggregation:
                    new_req.prefill_dp_rank = idx % self.dp_size
                    new_req.decode_dp_rank = new_req.prefill_dp_rank
                else:
                    new_req.dp_rank = idx % self.dp_size
                # Preserve the frontend-observed arrival separately. Engine-side
                # latency metrics should start once the request has both finished
                # client-side prep and acquired a concurrency slot.
                new_req.client_arrive_time = ingress_t
                new_req.arrive_time = admit_t
                new_req.ready_time = admit_t
                self.request_slots[new_req.request_id] = slot_id

                self._requests.append(new_req)
                new_requests.append(new_req)
                self.next_to_admit_idx += 1
                self.inflight += 1
            else:
                break
                
            if num_limit > 0 and len(self._requests) >= num_limit:
                break
        return new_requests
    
    def should_terminate_idle(self, completed_count = 0):
        return completed_count >= self.num_prompts

    def prepare_decode_requests(self, dp_size: int) -> None:
        self._requests.sort(key=lambda req: req.prefill_end_time)
        for idx, req in enumerate(self._requests):
            req.decode_dp_rank = (idx // self.decode_req_batch) % dp_size
