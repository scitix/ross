from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from scheduler.request import Request
from common.kvpool import KVCachePool
from scheduler.kv_alloc import KVAllocWrapper

import logging
logger = logging.getLogger(__name__)

@dataclass
class Batch:
    reqs: List[Request] = field(default_factory=list)
    forward_mode: str = "decode"  # or "prefill"
    chunked_reqs: List[Request] = field(default_factory=list)
    prefill_targets: dict = field(default_factory=dict)  # rid -> chunk_offset after this batch

    def batch_size(self) -> int:
        return len(self.reqs)

    def is_empty(self) -> bool:
        return len(self.reqs) == 0

    def merge_batch(self, other: "Batch"):
        seen = {r.request_id for r in self.reqs}
        self.reqs += [r for r in other.reqs if r.request_id not in seen]

    def filter_batch(self, chunked_to_exclude: List[Request]):
        self.reqs = [r for r in self.reqs if r not in chunked_to_exclude]
        
    def __str__(self):
        if self.forward_mode == 'prefill':
            return f"<Prefill>\nrequest_id (len={len(self.reqs)})={[r.request_id for r in self.reqs]}\n" + \
                f"commit_tokens (sum={sum([r.current_chunk for r in self.reqs])})={[r.current_chunk for r in self.reqs]}"
        else:
            return f"<Decode>\nrequest_id (len={len(self.reqs)})={[r.request_id for r in self.reqs]}\n" + \
                f"num_computed_tokens (sum={sum([r.num_computed_tokens for r in self.reqs])})={[r.num_computed_tokens for r in self.reqs]}"

class SchedulePolicy:
    """Placeholder - default FIFO"""
    def calc_priority(self, waiting_queue: List[Request]):
        # FIFO already sorted by arrival (list order), keep as-is
        return
    
class Scheduler:
    def __init__(
        self,
        waiting_queue: List[Request],
        max_running_requests: int,
        kv_pool: KVCachePool,
        chunked_prefill_size: int,
        reserved_decode_tokens: int,
        disaggregation_mode: str = None,
        pp: int = 1,
        page_size: int = 1,
        enable_prefix_caching: bool = False,
    ):
        self.waiting_queue = waiting_queue
        self.policy = SchedulePolicy()        
        # KV Block manager
        self.kv_allocator = KVAllocWrapper(
            kv_pool=kv_pool,
        )
        self.max_running_requests = max_running_requests
        self.max_reqs_per_mb = max_running_requests // pp
        # TODO: max_prefill_tokens functions?
        self.chunked_prefill_size = chunked_prefill_size
        self.page_size = max(1, page_size)
        self.enable_prefix_caching = enable_prefix_caching
        
        self.running_batch = Batch(forward_mode="decode")
        self.running_queue: List[Batch] = []

        self.chunked_reqs: List[Request] = []
        self.total_prefill_tokens = 0
        self.reserved_decode_tokens = reserved_decode_tokens

        self.disaggregation_mode = disaggregation_mode
        self.completed_prefill_reqs: List[Request] = []
        self.pending_decode_queue: List[Request] = []

    def _maybe_match_prefix(self, req: Request) -> None:
        if not self.enable_prefix_caching or req.prefix_match_checked:
            return
        if req.prompt_token_ids is None or req.chunk_offset > 0:
            req.prefix_match_checked = True
            return

        matched_tokens = self.kv_allocator.kv_pool.match_prefix(
            request_id=req.request_id,
            token_ids=req.prompt_token_ids,
            extra_key=req.prefix_extra_key,
            max_prefix_len=req.prompt_tokens,
        )
        req.prefix_match_checked = True
        req.prefix_matched_tokens = matched_tokens
        if matched_tokens > 0:
            req.chunk_offset = max(req.chunk_offset, matched_tokens)
            req.num_computed_tokens = max(req.num_computed_tokens, matched_tokens)

    def _commit_prefix(self, req: Request) -> None:
        if (
            not self.enable_prefix_caching
            or req.prompt_token_ids is None
            or req.chunk_offset <= req.prefix_committed_tokens
        ):
            return

        committed_tokens = self.kv_allocator.kv_pool.commit_prefix(
            request_id=req.request_id,
            token_ids=req.prompt_token_ids,
            computed_tokens=req.chunk_offset,
            extra_key=req.prefix_extra_key,
            chunked=req.chunk_offset < req.prompt_tokens,
        )
        req.prefix_committed_tokens = max(req.prefix_committed_tokens, committed_tokens)

    def _handle_zero_prefill_completion(self, req: Request, decode_batch: Batch) -> None:
        if self.disaggregation_mode == "prefill":
            self.completed_prefill_reqs.append(req)
            return

        total_reqs = decode_batch.batch_size() + 1
        headroom = self.reserved_decode_tokens * total_reqs
        if self.kv_allocator.available_tokens >= headroom:
            decode_batch.reqs.append(req)
        else:
            self.pending_decode_queue.append(req)

    def _promote_prefix_hits(self) -> None:
        if self.disaggregation_mode == "decode":
            return

        if self.running_batch.forward_mode != "decode":
            self.running_batch.forward_mode = "decode"

        kept_chunked = []
        for req in self.chunked_reqs:
            self._maybe_match_prefix(req)
            if req.needs_prefill() or req.finished:
                kept_chunked.append(req)
            else:
                self._handle_zero_prefill_completion(req, self.running_batch)
        self.chunked_reqs = kept_chunked

        kept_waiting = []
        for req in self.waiting_queue:
            self._maybe_match_prefix(req)
            if req.needs_prefill() or req.finished:
                kept_waiting.append(req)
            else:
                self._handle_zero_prefill_completion(req, self.running_batch)
        self.waiting_queue = kept_waiting

    def _tokens_needed_next_decode(self, requests: List[Request]):
        return len( [r for r in requests if not r.finished] )

    def _ceil_paged_tokens(self, tokens: int) -> int:
        if self.page_size <= 1:
            return tokens
        return -(-tokens // self.page_size) * self.page_size

    def _align_to_budget(self, tokens: int, budget: int) -> int:
        if tokens <= 0:
            return 0
        if self.page_size <= 1:
            return min(tokens, budget)
        if budget < self.page_size:
            return 0
        aligned_up = self._ceil_paged_tokens(tokens)
        if aligned_up <= budget:
            return aligned_up
        aligned_down = (budget // self.page_size) * self.page_size
        return max(0, aligned_down)

    def _ignore_eos_can_add(
        self,
        req: Request,
        req_states: List[Tuple[int, int]],
        tokens_to_alloc: int,
    ) -> Tuple[bool, List[Tuple[int, int]]]:
        cur_rem_tokens = self.kv_allocator.available_tokens - tokens_to_alloc
        if cur_rem_tokens <= 0:
            return False, req_states
        tokens_left = req.max_new_tokens - req.decode_tokens
        if tokens_left <= 0:
            return True, req_states
        tokens_occupied = req.prompt_tokens + req.decode_tokens

        # Insert into sorted req_states by tokens_left
        new_states = list(req_states)
        insert_idx = 0
        while insert_idx < len(new_states) and tokens_left > new_states[insert_idx][0]:
            insert_idx += 1
        new_states.insert(insert_idx, (tokens_left, tokens_occupied))

        tokens_freed = 0
        for i, (t_left, t_occ) in enumerate(new_states):
            bs = len(new_states) - i
            min_free_tokens = cur_rem_tokens + tokens_freed - t_left * bs
            if min_free_tokens <= bs:
                return False, req_states
            tokens_freed += t_occ

        return True, new_states

    def check_decode_mem(self, requests: List[Request]) -> bool:
        tokens_needed = self._tokens_needed_next_decode(requests)
        return tokens_needed <= self.kv_allocator.available_tokens

    def check_oom(self, requests: Optional[List[Request]] = None) -> Tuple[bool, str]:
        """Best-effort OOM/stall detector for the simulator.

        Returns (is_oom, reason). We treat OOM as: no available KV tokens and no
        possible progress via retract (e.g. only 1 active request that still needs KV),
        or pending prefill work but no KV space.
        """
        avail = self.kv_allocator.available_tokens
        if requests is None:
            requests = self.running_batch.reqs
            requests += [b.reqs for b in self.running_queue]

        in_decode_round = (
            self.disaggregation_mode == "decode"
            or (getattr(self.running_batch, "forward_mode", None) == "decode" and not self.running_batch.is_empty())
        )
        # Prefill side: no in-flight decode; just pending work but no space
        if not in_decode_round:
            pending = bool(self.waiting_queue) or bool(self.chunked_reqs)
            idle = (
                self.running_batch.is_empty()
                and not self.running_queue and not self.waiting_queue
                and not self.chunked_reqs
            )
            # If idle with pending requests, get_next_batch failed to schedule anything.
            # Even if avail > 0 (e.g., 1 token left), if it's insufficient for the 
            # head request, it results in an OOM deadlock.
            if pending and self.running_batch.is_empty() and not self.running_queue:
                next_req = self.chunked_reqs[0] if self.chunked_reqs else self.waiting_queue[0]
                needed_raw = next_req.next_chunk_tokens(self.chunked_prefill_size)
                needed = self._align_to_budget(needed_raw, self.chunked_prefill_size)
                
                if avail < needed:
                    return True, (f"prefill OOM: idle with pending reqs but insufficient space. "
                                    f"avail={avail}, needed={needed} (req={next_req.request_id})")
            return False, ""

        # Decode side
        active = [r for r in requests if not r.finished]
        if not active:
            if self.waiting_queue and avail <= 0:
                return True, "decode OOM: waiting requests but no KV space"
            return False, ""

        # Need at least prompt+decoded+next token capacity
        need_tokens = sum( max(0, r.max_new_tokens - r.decode_tokens) for r in active)
        if need_tokens <= avail:
            return False, ""

        # If multiple requests exist, retract can free space.
        if len(active) > 1:
            return False, "decode memory pressure: retractable"

        return True, f"decode OOM: need={need_tokens} avail={avail} active=1"

    def _free_decode_kv(self, req: Request):
        """Free decode KV; keep prefill KV in decode-only mode."""
        if self.disaggregation_mode == "decode":
            prompt_tokens = req.chunk_offset
            self.kv_allocator.free(req.request_id)
            if prompt_tokens > 0:
                ok = self.kv_allocator.try_allocate_blocks(req, prompt_tokens)
                if not ok:
                    raise RuntimeError( f"Failed to re-acquire prompt KV after free for req {req.request_id}" )
            return

        self.kv_allocator.free(req.request_id)
        released = min(req.chunk_offset, self.total_prefill_tokens)
        if released > 0:
            self.total_prefill_tokens -= released
            req.chunk_offset = 0

    def add_request(
        self,
        req: Request,
        chunk_budget: int,
        tokens_needed_raw: Optional[int] = None,
        tokens_to_alloc: Optional[int] = None,
    ) -> Tuple[int, int, bool]:
        if tokens_needed_raw is None:
            tokens_needed_raw = req.next_chunk_tokens(chunk_budget)
        # logger.debug(f"id={req.request_id}, chunk_offset={req.chunk_offset}, prompt={req.prompt_tokens}, budget={chunk_budget}\n"
                    # f"              tokens_needed={tokens_needed}, total_prefill_tokens={self.total_prefill_tokens}")
        if tokens_to_alloc is None:
            tokens_to_alloc = self._align_to_budget(tokens_needed_raw, chunk_budget)
        if tokens_to_alloc <= 0:
            return 0, 0, False
        can_allocate = self.kv_allocator.try_allocate_blocks(req, tokens_to_alloc)
        if not can_allocate:
            return 0, 0, False
        self.total_prefill_tokens += tokens_to_alloc
        req.commit_chunk(tokens_needed_raw)
        self._commit_prefix(req)
        return tokens_needed_raw, tokens_to_alloc, True
    
    def get_num_allocatable_reqs(self) -> int:
        return self.max_reqs_per_mb - self.running_batch.batch_size()

    def get_new_batch_prefill(self) -> Optional[Batch]:
        if self.disaggregation_mode == "decode":
            return None
        chunk_budget = self.chunked_prefill_size
        if (self.running_batch.batch_size() >= self.max_reqs_per_mb or len(self.waiting_queue) == 0) and not self.chunked_reqs:
            return None
        # running_bs = self.running_batch.batch_size()
        can_run_list = []
        if self.get_num_allocatable_reqs() <= 0 and not self.chunked_reqs:
            return None
        
        # policy reorder (FIFO placeholder, so no-op)
        self.policy.calc_priority(self.waiting_queue)

        req_states: List[Tuple[int, int]] = []
        if not self.running_batch.is_empty():
            for r in self.running_batch.reqs:
                tokens_left = r.max_new_tokens - r.decode_tokens
                if tokens_left <= 0:
                    continue
                tokens_occupied = r.prompt_tokens + r.decode_tokens
                req_states.append((tokens_left, tokens_occupied))
        req_states.sort(key=lambda x: x[0])

        next_chunked = []
        for req in self.chunked_reqs:
            self._maybe_match_prefix(req)
            if not req.needs_prefill():
                self._handle_zero_prefill_completion(req, self.running_batch)
                continue
            tokens_raw = req.next_chunk_tokens(chunk_budget)
            tokens_alloc = self._align_to_budget(tokens_raw, chunk_budget)
            tokens_raw, tokens_alloc, ok = self.add_request(
                req,
                chunk_budget,
                tokens_needed_raw=tokens_raw,
                tokens_to_alloc=tokens_alloc,
            )
            if ok:
                chunk_budget -= tokens_alloc
                can_run_list.append(req)
                next_chunked.append(req)
                # Update ignore_eos states for subsequent requests.
                tokens_left = req.max_new_tokens - req.decode_tokens
                if tokens_left > 0:
                    tokens_occupied = req.prompt_tokens + req.decode_tokens
                    insert_idx = 0
                    while (
                        insert_idx < len(req_states)
                        and tokens_left > req_states[insert_idx][0]
                    ):
                        insert_idx += 1
                    req_states.insert(insert_idx, (tokens_left, tokens_occupied))
            elif req.needs_prefill() and not req.finished:
                # Keep partially-prefilled requests queued even when they cannot
                # be scheduled in this round; otherwise they are dropped from
                # the scheduler and the simulator can spin with batch=None.
                next_chunked.append(req)

        self.chunked_reqs = [
            req for req in next_chunked if req.needs_prefill() and not req.finished
        ]

        while self.waiting_queue:
            req = self.waiting_queue[0]
            self._maybe_match_prefix(req)
            if not req.needs_prefill():
                self.waiting_queue.pop(0)
                self._handle_zero_prefill_completion(req, self.running_batch)
                continue
            total_running = self.running_batch.batch_size() + len(can_run_list)
            if total_running >= self.max_reqs_per_mb:
                break
            if len(can_run_list) >= self.get_num_allocatable_reqs():
                break
            tokens_raw = req.next_chunk_tokens(chunk_budget)
            tokens_alloc = self._align_to_budget(tokens_raw, chunk_budget)
            if tokens_alloc <= 0:
                break
            ok_ignore_eos, new_states = self._ignore_eos_can_add(
                req, req_states, tokens_alloc
            )
            if not ok_ignore_eos:
                break
            tokens_raw, tokens_alloc, ok = self.add_request(
                req,
                chunk_budget,
                tokens_needed_raw=tokens_raw,
                tokens_to_alloc=tokens_alloc,
            )
            if ok:
                req_states = new_states
                chunk_budget -= tokens_alloc
                can_run_list.append(req)
                self.waiting_queue.pop(0)
                
                if req.needs_prefill():
                    self.chunked_reqs.append(req)
            else:
                # cannot allocate for this req now
                break

        if len(can_run_list) == 0:
            return None

        batch = Batch(
            reqs=can_run_list,
            forward_mode="prefill",
            prefill_targets={r.request_id: r.chunk_offset for r in can_run_list},
        )
        batch.chunked_reqs = list(self.chunked_reqs)
        return batch

    def retract_decode(self, batch: Batch) -> List[Request]:
        """Free KV cache by temporarily retracting low-progress decode requests.

        Returns:
            Tuple[bool, int, List[Request]]: (whether allocation succeeds for target_req,
            how many tokens were allocated, list of retracted requests)
        """
        sorted_reqs = sorted([ req for req in batch.reqs if not req.finished ], key=lambda r: (r.decode_tokens, -r.prompt_tokens), reverse=True)
        candidates = sorted_reqs
        retracted: List[Request] = []
        first_iter = True
        while first_iter or not self.check_decode_mem(candidates):
            if not candidates:
                raise RuntimeError("No Space Left: Retracted all requests")
            first_iter = False
            victim = candidates.pop()  # smallest progress first
            self._free_decode_kv(victim)
            retracted.append(victim)

        if retracted:
            batch.reqs = [r for r in batch.reqs if r not in retracted]
            batch.reqs.sort(key=lambda r: (r.decode_tokens, -r.prompt_tokens), reverse=True)

            # Retracted requests always go back to waiting_queue to be rescheduled.
            self.waiting_queue = retracted + self.waiting_queue
            logger.debug(
                    f"[retract] freed {len(retracted)} reqs "
                    f"retracted={[r.request_id for r in retracted]}"
                )
        return retracted

    def update_running_batch(self, batch: Batch) -> Batch:
        still_running, retracted_reqs = [], []

        # Proactive check before stepping decode
        #     if batch as a whole cannot fit next decode step, retract first
        if not self.check_decode_mem(batch.reqs):
            newly_retracted = self.retract_decode(batch)
            retracted_reqs += newly_retracted

        for req in batch.reqs:
            if req.request_id not in retracted_reqs:
                tokens_to_alloc = 1
                if self.kv_allocator.try_allocate_blocks(req, tokens_to_alloc):
                    # update request outputs
                    req.decode_tokens += tokens_to_alloc
                    req.num_computed_tokens += tokens_to_alloc
                    if req.decode_tokens + 1 > req.max_new_tokens:
                        req.finished = True

                if req.finished:
                    self.kv_allocator.free(req.request_id)
                else:
                    still_running.append(req)
        batch.reqs = still_running
        self.chunked_reqs = [req for req in self.chunked_reqs if not req.finished]
        return batch

    def _try_admit_pending_decode(self) -> None:
        if self.disaggregation_mode == "decode":
            return
        if self.pending_decode_queue and self.running_batch.is_empty():
            self.running_batch.forward_mode = "decode"

        admitted = []
        for req in self.pending_decode_queue:
            total_reqs = self.running_batch.batch_size() + 1
            headroom = self.reserved_decode_tokens * total_reqs
            if self.kv_allocator.available_tokens < headroom:
                break
            self.running_batch.reqs.append(req)
            admitted.append(req)

        if admitted:
            admitted_ids = {req.request_id for req in admitted}
            self.pending_decode_queue = [
                req for req in self.pending_decode_queue
                if req.request_id not in admitted_ids
            ]

    def get_next_batch_to_run(self) -> Optional[Batch]:
        if self.running_queue:
            self.running_batch = self.running_queue.pop(0)
        else:
            self.running_batch = Batch(forward_mode="decode")

        self._promote_prefix_hits()
        self._try_admit_pending_decode()

        # logger.debug(f"[Scheduler] <disagg_mode={self.disaggregation_mode}> get_next_batch_to_run:\nrunning_batch={self.running_batch},\nself.waiting_queue={[r.request_id for r in self.waiting_queue]}")
        # Decode-only path for disaggregated decode worker
        if self.disaggregation_mode == "decode":
            if self.running_batch.is_empty():
                # If nothing is running, start from waiting_queue
                while self.waiting_queue and self.running_batch.batch_size() < self.max_reqs_per_mb:
                    cand = self.waiting_queue[0]
                    if not self.kv_allocator.try_allocate_blocks(cand, cand.num_computed_tokens):
                        break
                    self.running_batch.reqs.append(self.waiting_queue.pop(0))
                self.running_batch.forward_mode = "decode"
            else:
                # If there is capacity, merge waiting_queue into the running batch
                remaining = self.max_reqs_per_mb - self.running_batch.batch_size()
                while remaining > 0 and self.waiting_queue:
                    cand = self.waiting_queue[0]
                    if not self.kv_allocator.try_allocate_blocks(cand, cand.num_computed_tokens):
                        break
                    self.running_batch.reqs.append(self.waiting_queue.pop(0))
                    remaining -= 1

            if not self.running_batch.is_empty():
                return self.running_batch
            return None

        # If cannot send one req to decode; continue decoding
        running_size = self.running_batch.batch_size()
        if running_size > 0 and self.waiting_queue:
            future_headroom = self.reserved_decode_tokens * (running_size + 1)
            next_req = self.waiting_queue[0]
            estimated_prompt_cost = self._align_to_budget(next_req.prompt_tokens, self.chunked_prefill_size)

            if (self.kv_allocator.available_tokens - estimated_prompt_cost) < future_headroom:
                logger.debug(f"[Scheduler] Skip prefill: avail={self.kv_allocator.available_tokens} "
                            f"< headroom={future_headroom} (active={running_size})")
                
                if not self.running_batch.is_empty():
                    return self.running_batch
                return None

        new_batch = self.get_new_batch_prefill()
        if new_batch is not None:
            if not self.running_batch.is_empty():
                # Preserve the current decode batch when issuing a prefill batch
                # in the same scheduling round; otherwise active decode requests
                # are overwritten and disappear from future scheduling.
                self.running_queue = [self.running_batch] + self.running_queue
            ret = new_batch
        else:
            if not self.running_batch.is_empty():
                ret = self.running_batch
            else:
                ret = None

        return ret

    def process_batch_result_prefill(self, batch: Batch, running_batch: Batch):
        """Merge prefill results into running batch."""
        completed_prefill = [
                req for req in batch.reqs
                if batch.prefill_targets.get(req.request_id, req.chunk_offset) >= req.prompt_tokens
        ]
        if self.disaggregation_mode == "prefill":
            for req in completed_prefill:
                # In disaggregated mode, decode owns its own KV reservation. Once a
                # request finishes prefill on the prefill worker, release the prefill
                # worker's KV while keeping req.chunk_offset as the completed prompt
                # progress for decode admission.
                self.kv_allocator.free(req.request_id)
                released = min(req.chunk_offset, self.total_prefill_tokens)
                if released > 0:
                    self.total_prefill_tokens -= released
            self.completed_prefill_reqs.extend(completed_prefill)
            return
        if not completed_prefill:
            return

        # enqueue decode batch: allocate reserved decode tokens
        decode_batch = running_batch
        for req in completed_prefill:
            # Admission headroom check similar to decode worker
            total_reqs = decode_batch.batch_size() + 1
            headroom = self.reserved_decode_tokens * total_reqs

            if req.chunk_offset > 0 and self.kv_allocator.available_tokens >= headroom:
                decode_batch.reqs.append(req)
            else:
                # Keep prefill KV resident and wait for decode headroom instead
                # of restarting prefill from scratch.
                self.pending_decode_queue.append(req)
                logger.debug(
                    f"    [DecodePending] req {req.request_id} finished prefill but failed decode admission.\n"
                    f"                    avail_tokens={self.kv_allocator.available_tokens}, total_req={total_reqs}, head_room={headroom}"
                )

        if decode_batch.reqs:
            self.running_queue.append(decode_batch)

    def process_batch_result_decode(self, batch: Batch):
        """Advance decode progress and cleanup finished reqs."""
        new_running_batch = self.update_running_batch(batch)
        if not new_running_batch.is_empty():
            self.running_queue.append(new_running_batch)
