from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Union, Tuple
from transformers import AutoTokenizer, PreTrainedTokenizer

class Request:
    def __init__(self,
                request_id: str,
                max_new_tokens: int,
                prompt_tokens: int = 0,
                prompt: str = '',
                disaggregation: bool = False,
                dp_rank: Tuple[int, int] = None,
                arrive_time: float = 0,
                tokenizer: Union[AutoTokenizer, PreTrainedTokenizer] = None):
        
        self.request_id = request_id
        if dp_rank is not None:
            if disaggregation:
                self.prefill_dp_rank = dp_rank[0]
                self.decode_dp_rank = dp_rank[1]
            else:
                self.dp_rank = dp_rank[0]

        self.max_new_tokens = max_new_tokens
        self.prompt = prompt
        self.prompt_tokens = prompt_tokens

        # arrivals
        self.arrive_time = arrive_time
        self.ready_time = 0
        self.tokenize_time = 0

        # runtime fields
        self.chunk_offset: int = 0
        self.decode_tokens: int = 0
        self.current_chunk: int = 0

        self.start_decode: bool = False
        self.num_computed_tokens: int = 0
        
        # metrics
        self.finished: bool = False
        self.ttft = None
        self.itl = None
        self._last_token_time = None
        self.e2e_latency = None

    def needs_prefill(self) -> bool:
        return self.chunk_offset < self.prompt_tokens

    def next_chunk_tokens(self, prefill_budget: int) -> int:
        if not self.needs_prefill():
            return 0
        return min(prefill_budget, self.prompt_tokens - self.chunk_offset)

    def commit_chunk(self, tokens: int):
        assert(tokens >= 0)
        self.current_chunk = tokens
        self.chunk_offset += tokens
        self.num_computed_tokens += tokens        
