import time
import enum

from typing import List, Optional, Tuple

class RequestStatus(enum.IntEnum):
    """Status of a request."""
    WAITING = 0
    RUNNING = 1
    PREEMPTED = 2
    FINISHED = 3
    
    def __str__(self):
        return self.name

    @staticmethod
    def is_finished(status: "RequestStatus") -> bool:
        return status == RequestStatus.FINISHED

class Request:
    def __init__(self, request_id: str,
                prompt: str = '',
                prompt_token_ids: Optional[List[int]] = None,
                prompt_tokens: int = 0,
                disaggregation: bool = False,
                dp_rank: Tuple[int, int] = None,
                arrive_time: float = 0,
                max_new_tokens: int = 1):
        
        self.request_id = request_id
        self.arrive_time = arrive_time
        self.ready_time = 0
        self.prompt_text = prompt
        if prompt_token_ids is not None:
            self.prompt_token_ids = prompt_token_ids
            self.prompt_tokens = len(self.prompt_token_ids)
        else:
            self.prompt_token_ids = None
            self.prompt_tokens = prompt_tokens

        if dp_rank is not None:
            if disaggregation:
                self.prefill_dp_rank = dp_rank[0]
                self.decode_dp_rank = dp_rank[1]
            else:
                self.dp_rank = dp_rank[0]

        self.num_tokens = self.prompt_tokens + max_new_tokens
        self.output_len = 0
        self.num_computed_tokens = 0

        self.status = RequestStatus.WAITING
        self.prefill_end_time = None
        self.transfer_loaded = False # p->d load kv-cache

        # metrics
        self.ttft = None
        self.ttft_step = None
        self.ttlt_step = None
        self.itl = None
        self.tpot = None
        self.e2e_latency = None
        self._last_token_time = None

    @property
    def is_finished(self) -> bool:
        return RequestStatus.is_finished(self.status)

    def decode_init(self):
        self.status = RequestStatus.WAITING
        self.output_len = 0
        self.num_computed_tokens = 0
        self.transfer_loaded = True

    def __lt__(self, other):
        return self.request_id < other.request_id

    def __str__(self) -> str:
        return f"Request ID: {self.request_id}, Status: {self.status}\n" \
            + f"Prompt Len = {self.prompt_tokens}, Output Len = {self.output_len}, Num Computed Tokens = {self.num_computed_tokens} \n"
    
    def __repr__(self) -> str:
        return self.__str__()
