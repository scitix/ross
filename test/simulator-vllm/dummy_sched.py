"""Utility script to run dummy scheduling loops for various vLLM v1 models.

This mirrors the structure of the scheduler unit tests but runs a lightweight
"dummy" schedule/update loop so we can profile scheduler behaviour without
loading a model. The script is intentionally self-contained so it can be used in
benchmarks similar to ``collect_attn.py``.
"""

from __future__ import annotations

import os
import sys
import numpy as np
import time
from dataclasses import dataclass, field
from pathlib import Path

from collections import defaultdict
from typing import Iterable, Optional, List, Dict, Tuple, Any

# Ensure both the repository root and the vendored vLLM package are importable.
REPO_ROOT = Path(__file__).resolve().parents[2]
THIRDPARTY_ROOT = REPO_ROOT / "thirdparty"
for _path in (REPO_ROOT, THIRDPARTY_ROOT):
    path_str = str(_path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

Request = Any
SchedulerOutput = Any
ModelRunnerOutput = Any
SamplingParams = None
get_request_block_hasher = None
init_none_hash = None
sha256 = None
CacheConfig = None
DeviceConfig = None
ModelConfig = None
SchedulerConfig = None
VllmConfig = None
FullAttentionSpec = None
KVCacheConfig = None
KVCacheGroupSpec = None
Scheduler = None
StructuredOutputManager = None
_vllm_imports_ready = False

@dataclass
class DummyScheduleCase:
    """Configuration for a dummy scheduling run."""
    prompt_tokens: list
    scheduler_kwargs: dict = field(default_factory=dict)
    
_none_hash_initialized = False


def _ensure_vllm_imports(vllm_src_root: str = "") -> None:
    global _vllm_imports_ready
    global Request, SchedulerOutput, ModelRunnerOutput
    global SamplingParams, get_request_block_hasher, init_none_hash, sha256
    global CacheConfig, DeviceConfig, ModelConfig, SchedulerConfig, VllmConfig
    global FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
    global Scheduler, StructuredOutputManager
    if _vllm_imports_ready:
        return

    candidate_roots = []
    if vllm_src_root:
        candidate_roots.append(Path(vllm_src_root).expanduser().resolve())
    for root in candidate_roots:
        root_str = str(root)
        if root.exists() and root_str not in sys.path:
            sys.path.insert(0, root_str)
        if root.exists():
            current_pythonpath = os.environ.get("PYTHONPATH", "")
            pythonpath_parts = [p for p in current_pythonpath.split(os.pathsep) if p]
            if root_str not in pythonpath_parts:
                os.environ["PYTHONPATH"] = os.pathsep.join([root_str, *pythonpath_parts])

    from vllm.config import (
        CacheConfig as _CacheConfig,
        DeviceConfig as _DeviceConfig,
        ModelConfig as _ModelConfig,
        SchedulerConfig as _SchedulerConfig,
        VllmConfig as _VllmConfig,
    )
    from vllm.sampling_params import SamplingParams as _SamplingParams
    from vllm.utils.hashing import sha256 as _sha256
    from vllm.v1.core.kv_cache_utils import (
        get_request_block_hasher as _get_request_block_hasher,
        init_none_hash as _init_none_hash,
    )
    from vllm.v1.core.sched.output import SchedulerOutput as _SchedulerOutput
    from vllm.v1.core.sched.scheduler import Scheduler as _Scheduler
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec as _FullAttentionSpec,
        KVCacheConfig as _KVCacheConfig,
        KVCacheGroupSpec as _KVCacheGroupSpec,
    )
    from vllm.v1.outputs import ModelRunnerOutput as _ModelRunnerOutput
    from vllm.v1.request import Request as _Request
    from vllm.v1.structured_output import StructuredOutputManager as _StructuredOutputManager

    Request = _Request
    SchedulerOutput = _SchedulerOutput
    ModelRunnerOutput = _ModelRunnerOutput
    SamplingParams = _SamplingParams
    get_request_block_hasher = _get_request_block_hasher
    init_none_hash = _init_none_hash
    sha256 = _sha256
    CacheConfig = _CacheConfig
    DeviceConfig = _DeviceConfig
    ModelConfig = _ModelConfig
    SchedulerConfig = _SchedulerConfig
    VllmConfig = _VllmConfig
    FullAttentionSpec = _FullAttentionSpec
    KVCacheConfig = _KVCacheConfig
    KVCacheGroupSpec = _KVCacheGroupSpec
    Scheduler = _Scheduler
    StructuredOutputManager = _StructuredOutputManager
    _vllm_imports_ready = True

def create_requests(
    prompt_token_lens: List[int],
    max_output_tokens: int,
    block_size: int = 16,
    vllm_src_root: str = "",
) -> list[Request]:
    _ensure_vllm_imports(vllm_src_root)
    global _none_hash_initialized
    if not _none_hash_initialized:
        init_none_hash(sha256)
        _none_hash_initialized = True

    block_hasher = get_request_block_hasher(block_size, sha256)

    requests = []
    for idx, token_lens in enumerate(prompt_token_lens, start=1):
        prompt_token_ids = ([idx] * token_lens)
        
        sampling_params = SamplingParams(ignore_eos=True,
                            max_tokens=max_output_tokens)
        
        request = Request(
            request_id=f"req_{idx}",
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
            pooling_params=None,
            eos_token_id=None,
            block_hasher=block_hasher,
        )
        requests.append(request)
    return requests


def create_request(
    request_id: str,
    prompt_token_len: int,
    max_output_tokens: int,
    block_size: int = 16,
    vllm_src_root: str = "",
) -> Request:
    _ensure_vllm_imports(vllm_src_root)
    global _none_hash_initialized
    if not _none_hash_initialized:
        init_none_hash(sha256)
        _none_hash_initialized = True

    block_hasher = get_request_block_hasher(block_size, sha256)
    return Request(
        request_id=request_id,
        prompt_token_ids=[0] * prompt_token_len,
        sampling_params=SamplingParams(ignore_eos=True, max_tokens=max_output_tokens),
        pooling_params=None,
        eos_token_id=None,
        block_hasher=block_hasher,
    )


def create_pd_decode_request(
    request_id: str,
    prompt_token_len: int,
    max_output_tokens: int,
    block_size: int = 16,
    vllm_src_root: str = "",
) -> Request:
    request = create_request(
        request_id=request_id,
        prompt_token_len=prompt_token_len,
        max_output_tokens=max_output_tokens,
        block_size=block_size,
        vllm_src_root=vllm_src_root,
    )
    # Emulate the decode-side handoff in PD mode: the prompt KV is already
    # available, and the next scheduler step should issue the first decode
    # token rather than re-prefill the whole prompt.
    request.num_computed_tokens = max(0, prompt_token_len - 1)
    request.num_cached_tokens = prompt_token_len
    return request

def create_sidecar_scheduler(
    scheduler_kwargs: Dict[str, Any],
    num_blocks: int,
    block_size: int = 16,
    max_num_seqs: int = 16,
    max_model_len: Optional[int] = None,
    model_uri: str = "",
    vllm_src_root: str = "",
):
    _ensure_vllm_imports(vllm_src_root)
    import torch
    os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")

    if max_model_len is None:
        max_model_len = scheduler_kwargs["max_num_batched_tokens"]

    scheduler_config = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=scheduler_kwargs["max_num_batched_tokens"],
        max_model_len=max_model_len,
        long_prefill_token_threshold=0,
        enable_chunked_prefill=True,
        async_scheduling=False,
    )
    model_config = ModelConfig(
        model=model_uri,
        trust_remote_code=True,
        dtype="float16",
        seed=42,
        max_model_len=max_model_len,
        skip_tokenizer_init=True,
    )
    cache_config = CacheConfig(
        block_size=block_size,
        gpu_memory_utilization=0.9,
        swap_space=0,
        cache_dtype="auto",
        enable_prefix_caching=bool(scheduler_kwargs.get("enable_prefix_caching", False)),
    )
    cache_config.num_gpu_blocks = num_blocks

    vllm_config = VllmConfig(
        scheduler_config=scheduler_config,
        model_config=model_config,
        cache_config=cache_config,
        device_config=DeviceConfig(device="cpu"),
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"], FullAttentionSpec(block_size, 1, 1, torch.float32)
            )
        ],
    )
    return Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        block_size=block_size,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(vllm_config),
    )


def make_dummy_model_output(scheduler, output: SchedulerOutput) -> ModelRunnerOutput:
    """Create a minimal ``ModelRunnerOutput`` compatible with ``update_from_output``.

    Requests scheduled for the first time (prefill) return empty ``sampled_token_ids``;
    cached requests produce a single dummy decode token so that the scheduler
    can advance their state.
    """
    req_ids = list(output.num_scheduled_tokens.keys())
    req_id_to_index = {req_id: idx for idx, req_id in enumerate(req_ids)}

    sampled_token_ids: list[np.ndarray] = []
    for req_id in req_ids:
        request = scheduler.requests[req_id]
        if request.num_computed_tokens >= request.num_prompt_tokens:
            sampled_token_ids.append(np.array([0], dtype=np.int64))
        else:
            sampled_token_ids.append(np.array([], dtype=np.int64))
        
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index=req_id_to_index,
        sampled_token_ids=sampled_token_ids,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )

def run_dummy_schedule(prompt_token_lens: List[int],
                        max_new_tokens: int,
                        num_blocks: int,
                        scheduler_kwargs: Dict[str, Any],
                        vllm_src_root: str = "") -> Tuple[List[Dict[str, float]], float]:
    """Execute the dummy scheduling loop and collect per-step timings."""

    scheduler = create_sidecar_scheduler(
        scheduler_kwargs=scheduler_kwargs,
        num_blocks=num_blocks,
        max_num_seqs=max(len(prompt_token_lens), 1),
        vllm_src_root=vllm_src_root,
    )
    requests = create_requests(prompt_token_lens=prompt_token_lens,
                                max_output_tokens=max_new_tokens,
                                vllm_src_root=vllm_src_root)
    for request in requests:
        scheduler.add_request(request)

    records: list[dict[str, float]] = []
    total_time = 0
    step = 0
    while True:
        start = time.perf_counter()
        output = scheduler.schedule()
        elapsed = time.perf_counter() - start

        records.append({
            "step": float(step),
            "scheduled_new": output.scheduled_new_reqs,
            "scheduled_cached": output.scheduled_cached_reqs.req_ids,
            "tokens": output.finished_req_ids,
            "new_blocks": output.scheduled_cached_reqs.new_block_ids,
            "waiting_status": [(r.request_id, r.status) for r in scheduler.waiting],
            "computed_tokens": output.scheduled_cached_reqs.num_computed_tokens,
            "elapsed_ms": elapsed * 1e3,
        })
        batch_size = len(output.scheduled_new_reqs) + len(output.scheduled_cached_reqs.req_ids)
        total_time += elapsed

        if output.total_num_scheduled_tokens == 0:
            break

        dummy_output = make_dummy_model_output(scheduler, output)
        scheduler.update_from_output(output, dummy_output)

        step += 1
        
    return records, total_time
