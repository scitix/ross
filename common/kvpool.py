# import torch
from typing import List, Dict, Optional, Tuple, Sequence
from collections import deque

from common.models import BaseModel
try:
    from common.prefix_cache import SimPrefixCache
except Exception:  # optional dependency for prefix-cache simulation only
    SimPrefixCache = None  # type: ignore[assignment]

import logging
logger = logging.getLogger(__name__)

class KVCachePool:
    def __init__(self,
                model: BaseModel,
                num_reqs: int,
                total_gpu_memory: int | None = None,
                vllm_non_torch_increase: float = 0,
                tokens_per_block : int = 16,
                page_size: int = 1,
                gpu_memory_utilization: float = 0.9,
                framework: str = 'vllm'):
        """
        Initializes the KV Cache pool.

        Args:
            model (BaseModel): _description_
            tokens_per_block (int): The number of tokens each block can hold.
        """
        self.model = model
        self.framework = framework
        self.gpu_memory_utilization = gpu_memory_utilization

        self.total_gpu_memory = total_gpu_memory
        self.tokens_per_block = tokens_per_block
        self.page_size = max(1, page_size)
        self.single_block_size = self.calculate_single_block_bytes()
        self.num_blocks = self.determine_num_available_blocks(num_reqs, vllm_non_torch_increase)
        if self.num_blocks <= 0:
            raise MemoryError("OOM on KVCachePool Init")
            
        # Use a deque as a stack for efficient retrieval and return of free block IDs
        self.free_blocks = deque(range(self.num_blocks))
        self.request_to_blocks: Dict[str, List[int]] = {}
        self.prefix_cache: Optional[SimPrefixCache] = None
        self.request_to_cached_tokens: Dict[str, int] = {}

    def enable_prefix_cache(self, prefix_cache: Optional[SimPrefixCache]) -> None:
        self.prefix_cache = prefix_cache

    def get_cached_token_count(self, request_id: str) -> int:
        return self.request_to_cached_tokens.get(request_id, 0)

    def set_cached_token_count(self, request_id: str, cached_tokens: int) -> None:
        self.request_to_cached_tokens[request_id] = max(0, cached_tokens)

    def match_prefix(
        self,
        request_id: str,
        token_ids: Sequence[int],
        extra_key: Optional[str] = None,
        max_prefix_len: Optional[int] = None,
    ) -> int:
        if self.prefix_cache is None:
            self.set_cached_token_count(request_id, 0)
            return 0
        lease = self.prefix_cache.acquire_request(
            request_id=request_id,
            token_ids=token_ids,
            extra_key=extra_key,
            max_prefix_len=max_prefix_len,
        )
        cached_tokens = max(lease.matched_tokens, lease.committed_tokens)
        self.set_cached_token_count(request_id, cached_tokens)
        return cached_tokens

    def commit_prefix(
        self,
        request_id: str,
        token_ids: Sequence[int],
        computed_tokens: int,
        extra_key: Optional[str] = None,
        priority: int = 0,
        chunked: bool = False,
    ) -> int:
        if self.prefix_cache is None:
            return self.get_cached_token_count(request_id)
        lease = self.prefix_cache.commit_request(
            request_id=request_id,
            token_ids=token_ids,
            computed_tokens=computed_tokens,
            extra_key=extra_key,
            priority=priority,
            chunked=chunked,
        )
        cached_tokens = max(lease.matched_tokens, lease.committed_tokens)
        self.set_cached_token_count(request_id, cached_tokens)
        return cached_tokens
    
    def get_allocated_blocks(self, request_id: str) -> List[int]:
        if request_id not in self.request_to_blocks:
            return []
        else:
            return [x + 1 for x in self.request_to_blocks[request_id]]

    def get_allocated_block_count(self, request_id: str) -> int:
        """Fast path for schedulers that only need allocated block count."""
        if request_id not in self.request_to_blocks:
            return 0
        return len(self.request_to_blocks[request_id])

    def _get_num_blocks_needed(self, num_tokens: int) -> int:
        """Calculates the number of blocks required for a given number of tokens."""
        return (num_tokens + self.tokens_per_block - 1) // self.tokens_per_block

    @property
    def num_free_blocks(self) -> int:
        """Returns the current number of free blFocks."""
        return len(self.free_blocks)

    def allocate_slots(self, request_id: str, num_tokens: int) -> Optional[List[int]]:
        """
        Actually allocates blocks for a new request (prefill stage).

        Args:
            request_id: The request to allocate for.
            num_tokens: 
        Return:
            Optional[List[int]]: allocated block ids
        """
        blocks_needed = self._get_num_blocks_needed(num_tokens)
        if not (self.num_free_blocks >= blocks_needed):
            # raise RuntimeError(f"Not enough space to allocate the required "
            #                    f"{blocks_needed} blocks for request {request_id}.")
            return None
        
        allocated_ids = []
        for _ in range(blocks_needed):
            allocated_ids.append(self.free_blocks.popleft())
        
        if request_id in self.request_to_blocks:
            self.request_to_blocks[request_id] += allocated_ids
        else:
            self.request_to_blocks[request_id] = allocated_ids
        return allocated_ids

    def free(self, request_id: str) -> None:
        """
        Frees all blocks occupied by a request.
        Typically called when a request is finished or preempted.
        """
        if request_id in self.request_to_blocks:
            blocks_to_return = self.request_to_blocks.pop(request_id)
            self.free_blocks.extend(blocks_to_return)
        self.request_to_cached_tokens.pop(request_id, None)
        if self.prefix_cache is not None:
            self.prefix_cache.release_request(request_id)

    def __repr__(self) -> str:
        return (f"KVCachePool(free={self.num_free_blocks}/{self.num_blocks}, "
                f"            allocations={len(self.request_to_blocks)})")

    def _get_non_kv_usage(self) -> float:
        """Get Model and Activation Memory Usage

        Returns:
            Tuple[float, float]: weights, activations
        """
        if self.model.weights_from_config > 0:
            weights = self.model.weights_from_config * (1024 ** 3)
            return weights / (self.model.inference_config.pp_size * self.model.inference_config.tp_size)

        weights = 0.
        for op in self.model.context_ops:
            weights += op.get_weights()

        # count weights on a single GPU
        weights /= self.model.inference_config.pp_size
        return weights

    def _get_torch_increase(self, batch_size, vocab_size) -> int:
        # LLAMA 70B: (256, 128256) -> 1.2G
        heuristic_base = int(1.2 * (1024 ** 3)) // (256 * 128256)
        return heuristic_base * batch_size * vocab_size
        
    def determine_num_available_blocks(self, num_reqs: int, vllm_non_torch_increase: float) -> int:
        """Profiles the peak memory usage of the model to determine how many
        KV blocks may be allocated without OOMs.

        The engine will first conduct a profiling of the existing memory usage.
        Then, it calculate the maximum possible number of GPU and CPU blocks
        that can be allocated with the remaining free memory.

        Tip:
            You may limit the usage of GPU memory
            by adjusting the `gpu_memory_utilization` parameter.
        """
        # Profile the memory usage of the model and get the maximum number of
        # cache blocks that can be allocated with the remaining free memory.
        if self.total_gpu_memory is not None:
            total_gpu_memory = self.total_gpu_memory
        else:
            raise RuntimeError("Not use cuda")
            # torch.cuda.empty_cache()
            # torch.cuda.reset_peak_memory_stats()

            # _, total_gpu_memory = torch.cuda.mem_get_info()
        weights = self._get_non_kv_usage()
        torch_increase = self._get_torch_increase(num_reqs, self.model.get_vocab_size())
        # activation add non-torch increase
        non_torch_increase = vllm_non_torch_increase * (1024 ** 3)
        
        memory_for_current_instance = total_gpu_memory * self.gpu_memory_utilization
        if self.framework == 'vllm':
            available_kv_cache_memory = memory_for_current_instance - (weights + torch_increase + non_torch_increase)
        else:
            available_kv_cache_memory = memory_for_current_instance - weights
        # only need 1/pp num_layers
        available_kv_cache_memory = available_kv_cache_memory
        num_gpu_blocks = int(available_kv_cache_memory // self.single_block_size)
        
        logger.debug(f"total_gpu_memory "
                    f"({(total_gpu_memory / 1024 ** 3):.2f}GiB)"
                    " x gpu_memory_utilization "
                    f"({self.gpu_memory_utilization:.2f})\n"
                    f"   = {(memory_for_current_instance / 1024 ** 3):.2f}GiB\n"
                    "model weights take "
                    f"{(weights / 1024 ** 3):.2f}GiB;\n"
                    " PyTorch mem increase takes "
                    f"{(torch_increase / 1024 ** 3):.2f}GiB.\n"
                    " non-torch mem increase takes "
                    f"{(non_torch_increase / 1024 ** 3):.2f}GiB.\n"
                    " Available KV memory = "
                    f"{(available_kv_cache_memory / 1024 ** 3):.2f}GiB;\n"
                    " GPU KV cache blocks = "
                    f"{num_gpu_blocks};"
                    "    Total Tokens = "
                    f"{num_gpu_blocks * self.tokens_per_block};"
                    "    Single Block Size = "
                    f"{(self.single_block_size / 1024 ** 2):.2f}MB;"
                    "    Page Size = "
                    f"{(self.single_block_size / (self.model.get_num_layers() // self.model.inference_config.pp_size)):.2f};")
        return max(num_gpu_blocks, 0)

    def calculate_single_block_bytes(self) -> int:
        """_summary_

        Args:
            tokens_per_block (int): _description_

        Returns:
            int: _description_
        """
        num_layers = self.model.get_num_layers() // self.model.inference_config.pp_size
        num_kv_heads = self.model.get_num_kv_heads_per_gpu()
        head_size = self.model.get_head_size()
        bytes_per_element = self.model.inference_config.kvcache_quant_mode.value.memory
        key_cache_entry_size = num_kv_heads * head_size
        value_cache_entry_size = key_cache_entry_size
        
        total_elements_per_token = num_layers * (key_cache_entry_size + value_cache_entry_size)
        block_bytes = self.tokens_per_block * total_elements_per_token * bytes_per_element
        
        if self.model.model_uri.lower().find('deepseek') != -1:
            d_c = self.model.kv_lora_rank # 512
            d_rope = self.model.qk_rope_head_dim # 64

            total_elements_per_token = num_layers * (d_c + d_rope)
            block_bytes = self.tokens_per_block * total_elements_per_token * bytes_per_element
        
        return block_bytes


class SGLKVCachePool(KVCachePool):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        assert(self.tokens_per_block == 1)
        assert(self.page_size >= 1)
        
        self.free_blocks = None 
        self.request_to_blocks = None
            
        # === Optimized ===
        # 1. count for free blocks
        self._free_block_cnt = self.num_blocks
            
        # 2. only numbers for allocated blocks：Dict[request_id, count]
        self._req_alloc_counts: Dict[str, int] = {}
            
    @property
    def num_free_blocks(self) -> int:
        return self._free_block_cnt

    def get_allocated_blocks(self, request_id: str) -> object:
        if request_id in self._req_alloc_counts:
            count = self._req_alloc_counts[request_id]
        else:
            count = 0
        return range(count)

    def get_allocated_block_count(self, request_id: str) -> int:
        if request_id not in self._req_alloc_counts:
            return 0
        return self._req_alloc_counts[request_id]

    def _get_num_blocks_needed(self, num_tokens: int) -> int:
        """SGL uses paged allocation; round up to page_size tokens."""
        if self.page_size <= 1:
            return num_tokens
        return ((num_tokens + self.page_size - 1) // self.page_size) * self.page_size

        
    def determine_num_available_blocks(self, num_reqs: int, vllm_non_torch_increase: float) -> int:
        """
        rest_memory = available_gpu_memory - total_gpu_memory * (1 - mem_fraction_static)
        cell_size = num_kv_heads * head_dim * num_layers * 2 * element_size(kv_dtype)
        max_num_token = int(rest_memory_bytes / cell_size)
        """
        # Profile the memory usage of the model and get the maximum number of
        # cache blocks that can be allocated with the remaining free memory.
        if self.total_gpu_memory is not None:
            total_gpu_memory = self.total_gpu_memory
        else:
            raise RuntimeError("Not use cuda")

        weights = self._get_non_kv_usage()
        
        memory_for_current_instance = total_gpu_memory * self.gpu_memory_utilization
        available_kv_cache_memory = memory_for_current_instance - weights
        # only need 1/pp num_layers
        available_kv_cache_memory = available_kv_cache_memory
        num_gpu_blocks = int(available_kv_cache_memory // self.single_block_size)
        
        logger.debug(f"total_gpu_memory "
                    f"({(total_gpu_memory / 1024 ** 3):.2f}GiB)"
                    " x gpu_memory_utilization "
                    f"({self.gpu_memory_utilization:.2f})\n"
                    f"   = {(memory_for_current_instance / 1024 ** 3):.2f}GiB\n"
                    "model weights take "
                    f"{(weights / 1024 ** 3):.2f}GiB;\n"
                    " Available KV memory = "
                    f"{(available_kv_cache_memory / 1024 ** 3):.2f}GiB;\n"
                    " GPU KV cache blocks = "
                    f"{num_gpu_blocks};"
                    "    Single Block Size = "
                    f"{(self.single_block_size / 1024 ** 2):.2f}MB;"
                    "    Page Size = "
                    f"{self.page_size};")
        return max(num_gpu_blocks, 0)

    def calculate_single_block_bytes(self) -> int:
        num_layers = self.model.get_num_layers() // self.model.inference_config.pp_size
        num_kv_heads = self.model.get_num_kv_heads_per_gpu()
        head_size = self.model.get_head_size()
        bytes_per_element = self.model.inference_config.kvcache_quant_mode.value.memory
        key_cache_entry_size = num_kv_heads * head_size
        value_cache_entry_size = key_cache_entry_size
        
        total_elements_per_token = num_layers * (key_cache_entry_size + value_cache_entry_size)
        block_bytes = self.tokens_per_block * total_elements_per_token * bytes_per_element

        if self.model.model_uri.lower().find('deepseek') != -1:
            d_c = self.model.kv_lora_rank # 512
            d_rope = self.model.qk_rope_head_dim # 64

            total_elements_per_token = num_layers * (d_c + d_rope)
            block_bytes = self.tokens_per_block * total_elements_per_token * bytes_per_element
        
        return block_bytes

    def allocate_slots(self, request_id: str, num_tokens: int) -> Optional[object]:
        blocks_needed = self._get_num_blocks_needed(num_tokens)
            
        if self._free_block_cnt < blocks_needed:
            return None
            
        self._free_block_cnt -= blocks_needed
            
        if request_id in self._req_alloc_counts:
            current_count = self._req_alloc_counts[request_id]
        else:
            current_count = 0
        self._req_alloc_counts[request_id] = current_count + blocks_needed
        return range(blocks_needed)

    def free(self, request_id: str) -> None:
        if request_id in self._req_alloc_counts:
            count = self._req_alloc_counts.pop(request_id)
            self._free_block_cnt += count
        self.request_to_cached_tokens.pop(request_id, None)
        if self.prefix_cache is not None:
            self.prefix_cache.release_request(request_id)

            
    def __repr__(self) -> str:
        return (f"SGLKVCachePool(opt=True, free={self._free_block_cnt}/{self.num_blocks}, "
                f"allocations={len(self._req_alloc_counts)})")
