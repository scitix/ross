
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import itertools

from common.models import BaseModel
from scheduler.request import Request, RequestStatus
from common.kvpool import KVCachePool

import logging
logger = logging.getLogger(__name__)

@dataclass
class SchedulerOutput:
    scheduled_req_ids: List[str]
    
    num_scheduled_tokens: Dict[str, int] = None
    num_computed_tokens: Dict[str, int] = None

    prefill_seq_lens: List[int] = None
    decode_seq_lens: List[int] = None
    
    running_reqs: List[Request] = None
    new_reqs: List[Request] = None
    resumed_reqs: List[Request] = None
    preempted_reqs: List[Request] = None
    
    isl: int = 0
    osl: int = 0

    def __repr__(self) -> str:
        HEAD = 5
        def _list_head(xs):
            if not xs:
                return "[]"
            head = ", ".join(map(str, xs[:HEAD]))
            more = f", ... (+{len(xs)-HEAD} more)" if len(xs) > HEAD else ""
            return f"[{head}{more}]"

        def _dict_head(d):
            if not d:
                return "{}"
            items = sorted(d.items())[:HEAD]
            body = ", ".join(f"{k}:{v}" for k, v in items)
            more = f", ... (+{len(d)-HEAD} more)" if len(d) > HEAD else ""
            return f"{{{body}{more}}}"

        def _req_ids(rs):
            if not rs:
                return "[]"
            ids = [r.request_id for r in rs]
            return _list_head(ids)

        return (
            "SchedulerOutput("
            f"  scheduled_ids={_list_head(self.scheduled_req_ids)} (n={len(self.scheduled_req_ids) if self.scheduled_req_ids else 0}), \n"
            f"  scheduled_tokens={_dict_head(self.num_scheduled_tokens)}, \n"
            f"  computed_tokens={_dict_head(self.num_computed_tokens)}, \n"
            f"  prefill_seq_lens(n={len(self.prefill_seq_lens) if self.prefill_seq_lens else 0})={_list_head(self.prefill_seq_lens)}, \n"
            f"  decode_seq_lens(n={len(self.decode_seq_lens) if self.decode_seq_lens else 0})={_list_head(self.decode_seq_lens)}, \n"
            f"  running={_req_ids(self.running_reqs)}, \n"
            f"  new={_req_ids(self.new_reqs)}, \n"
            f"  resumed={_req_ids(self.resumed_reqs)}, \n"
            f"  preempted={_req_ids(self.preempted_reqs)}, \n"
            f"  isl={self.isl}, osl={self.osl}"
            ")"
        )

class Scheduler:
    def __init__(self,
                kv_pool: KVCachePool,
                max_running_reqs: int,
                max_num_batched_tokens: int,
                max_model_len: int = 131072,
                long_prefill_token_threshold: int = 8192,
                chunked_prefill_enabled: bool = True,                
                preempt_enabled: bool = True,
                enable_prefix_caching: bool = False,
        ):
        self.max_running_reqs = max_running_reqs
        self.max_model_len = max_model_len
        self.max_num_batched_tokens = max_num_batched_tokens
        self.long_prefill_token_threshold = long_prefill_token_threshold

        # KV Block manager
        self.kv_cache_manager = kv_pool

        self.waiting: List[Request] = []
        self.running: List[Request] = []
        
        self.input_requests: List[Request] = []
        self.seq_lens: Dict[str, int] = {}

        self.chunked_prefill_enabled = chunked_prefill_enabled
        self.preempt_enabled = preempt_enabled

    def clear(self):
        self.waiting, self.running, self.input_requests = []
        self.seq_lens = {}

    def add_request(self, request: Request):
        # logger.debug(f"add new request: {request.request_id}; arrive_time = {request.arrive_time}, prompt_tokens = {request.prompt_tokens}; total_tokens = {request.num_tokens}")
        self.waiting.append(request)
        self.input_requests.append(request)
        self.seq_lens[request.request_id] = 0

    def get_num_blocks(self):
        return self.kv_cache_manager.num_blocks
    
    def _kv_block_size(self) -> int:
        return getattr(self.kv_cache_manager, 'tokens_per_block')

    def _compute_kv_slack_tokens(self, request: Request) -> int:
        # Allocated capacity (in blocks) minus used tokens = available "tail slack" tokens
        block_size = self._kv_block_size()
        num_allocated_blocks = self.kv_cache_manager.get_allocated_block_count(request.request_id)
        capacity_tokens = num_allocated_blocks * block_size
        slack = max(0, capacity_tokens - request.num_computed_tokens)

        # logger.debug(f"    [ALLOC] req {request.request_id} num_allocated_blocks: {num_allocated_blocks}")
        # logger.debug(f"    [ALLOC] req {request.request_id} capacity_tokens: {capacity_tokens}; request_num_computed_tokens: {request.num_computed_tokens}")
        return slack

    def _try_allocate_blocks(self, request: Request, step_tokens: int) -> bool:
        # First consume tail slack in allocated blocks, then request from pool if needed
        slack = self._compute_kv_slack_tokens(request)
        need_tokens_from_pool = max(0, step_tokens - slack)
        # if need_tokens_from_pool > 0:
        #     logger.debug(f"    [ALLOC] req {request.request_id} step_tokens={step_tokens} slack={slack} -> need_from_pool={need_tokens_from_pool}")
        if need_tokens_from_pool == 0:
            return True
        allocated_blocks = self.kv_cache_manager.allocate_slots(request.request_id, need_tokens_from_pool)
        # logger.debug(f"    [ALLOC] req {request.request_id} took {allocated_blocks}  free: {getattr(self.kv_cache_manager, 'num_free_blocks', 'NA')}")
        return allocated_blocks is not None

    def check_oom(self) -> Tuple[bool, str]:
        """
        Check for Out-Of-Memory conditions.
        In a simulator, OOM typically means:
        1. Accounting error: free_blocks < 0
        2. Impossible request: A single request requires more blocks than total capacity.
        3. Unschedulable request: scheduler policy can never admit the request.
        """
        # 1. Accounting Consistency
        if self.kv_cache_manager.num_free_blocks < 0:
            return True, (f"System Logic Error: Negative free blocks detected "
                        f"({self.kv_cache_manager.num_free_blocks}).")

        # 2. Physical Limitation
        total_blocks_capacity = self.kv_cache_manager.num_blocks
        block_size = self._kv_block_size()

        # current running requests: whether over limit (double check)
        for req in self.running:
            total_needed_tokens = req.prompt_tokens + req.output_len
            total_needed_blocks = (total_needed_tokens + block_size - 1) // block_size
            
            if total_needed_blocks > total_blocks_capacity:
                return True, (f"Request {req.request_id} requires {total_needed_blocks} blocks, "
                            f"but total capacity is {total_blocks_capacity}.")

        # 3. Deadlock/Starvation due to OOM
        if not self.running and self.waiting:
            next_req = self.waiting[0]
            prompt_needed_blocks = (next_req.prompt_tokens + block_size - 1) // block_size
            
            if prompt_needed_blocks > total_blocks_capacity:
                    return True, (f"Request {next_req.request_id} prompt requires {prompt_needed_blocks} blocks, "
                                f"which exceeds total capacity {total_blocks_capacity}.")
            if next_req.transfer_loaded and self.max_num_batched_tokens < next_req.prompt_tokens:
                return True, (
                    f"Request {next_req.request_id} cannot be admitted on decode: "
                    f"transfer_loaded requires loading full prompt KV "
                    f"(prompt_tokens={next_req.prompt_tokens}), but "
                    f"max_num_batched_tokens={self.max_num_batched_tokens}."
                )

        return False, ""

    def _calculate_num_scheduled_tokens(self, request: Request, token_budget: int) -> int:
        """Calculate num_tokens to be scheduled
        """
        num_new_tokens = request.prompt_tokens + request.output_len - request.num_computed_tokens
        # logger.debug(f"request to sched: {request.request_id}, num_new_tokens: {request.prompt_tokens} + {request.output_len} - {request.num_computed_tokens} = {num_new_tokens}")
        if num_new_tokens <= 0:
            return 0
        if self.long_prefill_token_threshold > 0 and not request.transfer_loaded:
            num_new_tokens = min(num_new_tokens, self.long_prefill_token_threshold)
        # Whether to allow chunking by budget
        if not self.chunked_prefill_enabled and num_new_tokens > token_budget:
            return 0  # Don't schedule this prefill in this round
        num_new_tokens = min(num_new_tokens, token_budget)
        
        # Ensure not exceeding model length limit
        num_new_tokens = min(num_new_tokens,
                            self.max_model_len - 1 - request.num_computed_tokens)
        
        # 1st load in p->d cases: should allocate >= prompt
        if request.transfer_loaded and num_new_tokens < request.prompt_tokens:
            return 0
        return max(0, num_new_tokens)

    def _preempt_latest_request(self) -> Optional[Request]:
        """Preempt the latest request

        Returns:
            Optional[Request]: _description_
        """
        if not self.running:
            return None
        
        preempted_req = self.running[-1]
        # Debug before free
        logger.debug(f"[PREEMPT] progress={preempted_req.num_computed_tokens}/{preempted_req.num_tokens}\n"
                    f"          kv_before={self.kv_cache_manager}")
        self.running.remove(preempted_req)
        
        # Free its blocks
        self.kv_cache_manager.free(preempted_req.request_id)
        logger.debug(f"[PREEMPT] freed id={preempted_req.request_id}   kv_after={self.kv_cache_manager}")
        
        # Update status
        preempted_req.status = RequestStatus.PREEMPTED
        # update generated token lens
        preempted_req.num_computed_tokens = 0
        self.waiting.insert(0, preempted_req)
        self.seq_lens[preempted_req.request_id] = 0
        
        return preempted_req

    def _schedule_running_requests(self, token_budget: int) -> Tuple[List[Request], Dict[str, int], List[Request], int]:
        """schedule running requests

        Args:
            token_budget (int): _description_

        Returns:
            Tuple[List[Request], Dict[str, int], int]: scheduled requests, num_scheduled_tokens, available_token_budget
        """
        scheduled_reqs = []
        num_scheduled_tokens = {}
        preempted_reqs = []
        
        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]
            
            # Calculate number of tokens to schedule
            num_new_tokens = self._calculate_num_scheduled_tokens(request, token_budget)     
            if num_new_tokens == 0:
                req_index += 1
                continue
            
            can_sched = False
            while True:
                # Try to allocate KV blocks (reserve space for new tokens), use slack first
                can_allocate = self._try_allocate_blocks(request, num_new_tokens)
                if not can_allocate:
                    if not self.preempt_enabled:
                        break
                    # Cannot allocate blocks, try preemption
                    preempted_req = self._preempt_latest_request()
                    if preempted_req is None:
                        break
                    preempted_reqs.append(preempted_req)
                    if  preempted_req == request:
                        break
                else:
                    can_sched = True
                    break
            if not can_sched:
                break
            
            self.seq_lens[request.request_id] += num_new_tokens
            
            scheduled_reqs.append(request)
            num_scheduled_tokens[request.request_id] = num_new_tokens
            token_budget -= num_new_tokens
            
            req_index += 1
        
        return scheduled_reqs, num_scheduled_tokens, preempted_reqs, token_budget

    def _schedule_waiting_requests(self, token_budget: int) -> Tuple[List[Request], Dict[str, int], List[Request]]:
        """schedule waiting requests

        Args:
            token_budget (int): _description_

        Returns:
            Tuple[List[Request], Dict[str, int], int]: scheduled requests, num_scheduled_tokens, available_token_budget
        """
        scheduled_reqs = []
        resumed_reqs = []
        num_scheduled_tokens = {}
        while self.waiting and token_budget > 0 and len(self.running) < self.max_running_reqs:
            request = self.waiting[0]
            # Calculate number of tokens to schedule
            num_new_tokens = self._calculate_num_scheduled_tokens(request, token_budget)
            if num_new_tokens == 0:
                break
            # Try to allocate KV blocks (use slack first)
            can_allocate = self._try_allocate_blocks(request, num_new_tokens)
            if not can_allocate:
                break
            # Remove from waiting queue and add to running queue
            self.waiting.pop(0)
            self.running.append(request)
            
            if request.status == RequestStatus.PREEMPTED:
                resumed_reqs.append(request)
                logger.debug(f"[PREEMPT] resume req={request.request_id}\n"
                            f"          generated_len={request.output_len}  num_tokens={request.num_tokens}\n"
                            f"          num_new_tokens={num_new_tokens}  current_budget={token_budget}")
            else:
                # Record budget
                scheduled_reqs.append(request)
                
            request.status = RequestStatus.RUNNING
            self.seq_lens[request.request_id] += num_new_tokens
            num_scheduled_tokens[request.request_id] = num_new_tokens
            token_budget -= num_new_tokens

        return scheduled_reqs, num_scheduled_tokens, resumed_reqs

    def schedule(self) -> SchedulerOutput:
        """Execute one scheduling step"""
        # Clean up completed requests
        self.running = [req for req in self.running if req.status != RequestStatus.FINISHED]
        if self.should_terminate():
            return None
        # Initialize token budget
        token_budget = self.max_num_batched_tokens
        running_reqs, running_tokens, preempted_reqs, remaining_budget = self._schedule_running_requests(token_budget)
        new_reqs: List[Request] = []
        waiting_tokens: Dict[str, int] = {}
        resumed_reqs: List[Request] = []
        if not preempted_reqs:
            new_reqs, waiting_tokens, resumed_reqs = self._schedule_waiting_requests(remaining_budget)
        
        # Record scheduling results
        all_scheduled_reqs = running_reqs + resumed_reqs + new_reqs
        all_num_scheduled_tokens = {**running_tokens, **waiting_tokens}
        
        prefill_seq_lens, decode_seq_lens = [], []
        for idx, req in enumerate(all_scheduled_reqs):
            num_scheduled_tokens = all_num_scheduled_tokens[req.request_id]
            total_seq_len = req.num_computed_tokens + num_scheduled_tokens
            
            # p->d case: first decode will allocate kv_len=prompt_len
            if req.transfer_loaded:
                decode_seq_lens.append(1)
            elif num_scheduled_tokens <= 1:
                decode_seq_lens.append(int(total_seq_len))
            else:
                prefill_seq_lens.append(int(total_seq_len))
        
        # update num_computed_tokens
        self._update_after_sched(all_num_scheduled_tokens)        
        num_computed_tokens = {}
        for req in self.input_requests:
            num_computed_tokens.update({req.request_id: req.num_computed_tokens})

        output = SchedulerOutput(
            scheduled_req_ids=[r.request_id for r in all_scheduled_reqs],
            num_scheduled_tokens=all_num_scheduled_tokens,
                    
            num_computed_tokens=num_computed_tokens,
            prefill_seq_lens=prefill_seq_lens,
            decode_seq_lens=decode_seq_lens,
            
            new_reqs=new_reqs,
            running_reqs=running_reqs,
            resumed_reqs=resumed_reqs,
            preempted_reqs=preempted_reqs,
        )
        if not all_scheduled_reqs:
            return None
        return output
    
    def _update_after_sched(self, all_num_scheduled_tokens: Dict[str, int]):
        for req in self.input_requests:
            if req.request_id in all_num_scheduled_tokens:
                req.num_computed_tokens += all_num_scheduled_tokens[req.request_id]
                if req.transfer_loaded:
                    req.transfer_loaded = False
                    # logger.debug(f"request={req.request_id}, num_sched_tokens = {all_num_scheduled_tokens[req.request_id]}, num_computed_tokens = {req.num_computed_tokens}")
    
    def update_from_output(self, schedule_output: SchedulerOutput) -> None:
        """Process model outputs -> Update Requests & kv usage
        """
        if not schedule_output:
            return
        num_generated_tokens = dict(schedule_output.num_scheduled_tokens)
        num_computed_tokens = schedule_output.num_computed_tokens

        # Update each request based on execution results
        for target in self.input_requests:
            if target.request_id in num_generated_tokens and target.request_id in num_computed_tokens:
                all_tokens = target.output_len + target.prompt_tokens
                if num_computed_tokens[target.request_id] >= all_tokens:
                    target.output_len += 1
                # Check if completed
                if all_tokens >= target.num_tokens - 1 or all_tokens >= self.max_model_len - 1:
                    target.status = RequestStatus.FINISHED
                    self.kv_cache_manager.free(target.request_id)
                    
        # Clean up completed requests
        self.running = [req for req in self.running if req.status != RequestStatus.FINISHED]

    def should_terminate(self) -> bool:
        return not self.waiting and not self.running

    def debug_print_schedule(self, output: SchedulerOutput, dp_rank: int, round: int) -> None:
        # Classify requests: new requests vs cached requests
        cached_request_ids = []
        for req in itertools.chain(output.running_reqs, output.resumed_reqs):
            cached_request_ids.append(req.request_id)
        # Output formatted scheduling information
        logger.debug(
                    f"[dp_{dp_rank}] Scheduling Round {round}; num_free_blocks: {self.kv_cache_manager.num_free_blocks}\n"
                    f"  - New Request IDs (len={len(output.new_reqs)}): {[req.request_id for req in output.new_reqs]}\n"
                    f"  - Cached Request IDs (len={len(cached_request_ids)}): {cached_request_ids}\n"

                    f"  - num_computed_tokens: {[(rid, output.num_computed_tokens[rid]) for rid in output.scheduled_req_ids]}\n"
                    f"  - output_len: {[(req.request_id, req.output_len) for req in output.running_reqs]}\n"
                    # f"  - new_scheduled_tokens: {output.num_scheduled_tokens}\n"
        )
