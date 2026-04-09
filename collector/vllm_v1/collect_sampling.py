#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
dummy_sampler_runner.py
-----------------------
Minimal dummy sampler run: take logits_shape, device, sampler
and run the same path as _dummy_sampler_run, except logits are
randomly generated instead of coming from model forward.
"""

import torch
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler
from vllm.utils import MemorySnapshot, memory_profiling

def dummy_sampler_run_with_shape(logits_shape, device, sampler, dtype=torch.bfloat16):
    """
    Run dummy sampler with random logits instead of model forward.

    Args:
        logits_shape (tuple): (batch_size, vocab_size)
        device (torch.device): target GPU device
        sampler: your sampler instance (with .sample method)
        dtype (torch.dtype): logits dtype, default bf16

    Returns:
        output of sampler.sample(logits, metadata)
    """
    R, V_eff = logits_shape
    logits = torch.randn((R, V_eff), dtype=dtype, device=device)
    dummy_tensors = lambda v: torch.full(
            (R, ), v, device=device)

    metadata = SamplingMetadata(
        temperature=dummy_tensors(0.5),
        all_greedy=False,
        all_random=False,
        top_p=dummy_tensors(0.9),
        top_k=dummy_tensors(logits.size(1) - 1),
        generators={},
        max_num_logprobs=None,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=dummy_tensors(0.1),
        presence_penalties=dummy_tensors(0.1),
        repetition_penalties=dummy_tensors(0.1),
        output_token_ids=[[] for _ in range(R)],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
    )

    return sampler.sample(logits, metadata)


if __name__ == "__main__":
    device = torch.device("cuda:0")
    sampler = Sampler()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    init_snapshot = MemorySnapshot()

    with memory_profiling(init_snapshot,
                weights_memory=0) as profile_result:
        out = dummy_sampler_run_with_shape(
            logits_shape=(256, 128256),
            device=device,
            sampler=sampler,
            dtype=torch.bfloat16,
        )
    print(f"only run forward: {profile_result}")
