from __future__ import annotations
from typing import List, Optional, Tuple, Dict

from scheduler.request import Request
from common.kvpool import SGLKVCachePool

import logging
logger = logging.getLogger(__name__)

class KVAllocWrapper:
    """Abstraction over KV allocation usable with a basic counter or SGLKVCachePool."""

    def __init__(self, kv_pool: SGLKVCachePool):
        self.kv_pool = kv_pool
        
    def get_num_blocks(self):
        return self.kv_pool.num_blocks
    
    @property
    def num_free_blocks(self) -> int:
        return self.kv_pool.num_free_blocks
    
    @property
    def available_tokens(self) -> int:
        """number of tokens can be placed with current free blocks."""
        return self.num_free_blocks * self._kv_block_size()
    
    def _kv_block_size(self) -> int:
        # Try to read from pool, otherwise use common default value 16
        return self.kv_pool.tokens_per_block
    
    def compute_kv_slack_tokens(self, request: Request) -> int:
        """
        Allocated capacity (in blocks) minus used tokens = available "tail slack" tokens
        """
        block_size = self._kv_block_size()
        num_allocated_blocks = self.kv_pool.get_allocated_block_count(request.request_id)
        capacity_tokens = num_allocated_blocks * block_size
        slack = max(0, capacity_tokens - request.num_computed_tokens)

        # logger.debug(f"    [ALLOC] req {request.request_id} num_allocated_blocks: {num_allocated_blocks}\n"
        #              f"    [ALLOC] req {request.request_id} capacity_tokens: {capacity_tokens}; request_num_computed_tokens: {request.num_computed_tokens}")
        return slack

    def try_allocate_blocks(self, request: Request, step_tokens: int) -> bool:
        """
        First consume tail slack in allocated blocks, then request from pool if needed
        """
        slack = self.compute_kv_slack_tokens(request)
        need_tokens_from_pool = max(0, step_tokens - slack)
        # if need_tokens_from_pool > 0:
        #     logger.debug(f"    [ALLOC] req {request.request_id} step_tokens={step_tokens} slack={slack} -> need_from_pool={need_tokens_from_pool}")
        
        if need_tokens_from_pool == 0:
            return True
        allocated_blocks = self.kv_pool.allocate_slots(request.request_id, need_tokens_from_pool)
        
        # logger.debug(f"    [ALLOC] req {request.request_id} took {allocated_blocks}  free: {getattr(self.kv_pool, 'num_free_blocks', 'NA')}")
        return allocated_blocks is not None

    def free(self, request_id: str):
        self.kv_pool.free(request_id)
