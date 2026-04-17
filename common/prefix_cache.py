from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple

import torch

from common.sgl_prefix_tree import (
    InsertParams,
    MatchPrefixParams,
    RadixCache,
    RadixKey,
)


def _align_down(num_tokens: int, page_size: int) -> int:
    if page_size <= 1:
        return max(0, num_tokens)
    return max(0, num_tokens // page_size * page_size)


@dataclass
class PrefixMatch:
    matched_tokens: int
    matched_indices: torch.Tensor


@dataclass
class PrefixLease:
    request_id: str
    token_ids: Tuple[int, ...]
    extra_key: Optional[str]
    matched_tokens: int = 0
    committed_tokens: int = 0


class SimPrefixCache:
    """Thin simulation wrapper over SGLang's Python RadixCache.

    This class intentionally reuses only the prefix-tree logic (`match_prefix`,
    `insert`) while keeping lifecycle and memory accounting outside the cache.

    Notes:
    - It does not require a GPU. The underlying `RadixCache` runs on CPU.
    - `release_request()` only updates wrapper-side refcounts. It does not evict
      entries from the radix tree.
    - Inserted values are fake token indices because the simulator only needs
      prefix lengths, not real KV locations.
    """

    def __init__(
        self,
        page_size: int = 1,
        extra_key_fn: Optional[Callable[[object], Optional[str]]] = None,
    ):
        self.page_size = max(1, page_size)
        self.extra_key_fn = extra_key_fn
        self.tree = RadixCache.create_simulated(page_size=self.page_size)

        self._leases: Dict[str, PrefixLease] = {}
        self._refcnt: Dict[Tuple[Optional[str], Tuple[int, ...]], int] = {}

    def reset(self) -> None:
        self.tree.reset()
        self._leases.clear()
        self._refcnt.clear()

    def _normalize_tokens(
        self,
        token_ids: Sequence[int],
        limit: Optional[int] = None,
    ) -> Tuple[int, ...]:
        effective_len = len(token_ids) if limit is None else min(len(token_ids), limit)
        aligned_len = _align_down(effective_len, self.page_size)
        return tuple(token_ids[:aligned_len])

    def _make_key(
        self,
        token_ids: Sequence[int],
        extra_key: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> RadixKey:
        normalized = self._normalize_tokens(token_ids, limit=limit)
        return RadixKey(token_ids=list(normalized), extra_key=extra_key)

    def _make_ref_key(
        self,
        token_ids: Sequence[int],
        extra_key: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Tuple[Optional[str], Tuple[int, ...]]:
        return (extra_key, self._normalize_tokens(token_ids, limit=limit))

    def _adjust_refcnt(
        self,
        token_ids: Sequence[int],
        extra_key: Optional[str],
        limit: int,
        delta: int,
    ) -> None:
        if limit <= 0:
            return
        key = self._make_ref_key(token_ids, extra_key=extra_key, limit=limit)
        next_value = self._refcnt.get(key, 0) + delta
        if next_value > 0:
            self._refcnt[key] = next_value
        else:
            self._refcnt.pop(key, None)

    def match(
        self,
        token_ids: Sequence[int],
        extra_key: Optional[str] = None,
        max_prefix_len: Optional[int] = None,
    ) -> PrefixMatch:
        key = self._make_key(token_ids, extra_key=extra_key, limit=max_prefix_len)
        if len(key) == 0:
            return PrefixMatch(
                matched_tokens=0,
                matched_indices=torch.empty(0, dtype=torch.int64),
            )

        result = self.tree.match_prefix(MatchPrefixParams(key=key))
        return PrefixMatch(
            matched_tokens=len(result.device_indices),
            matched_indices=result.device_indices,
        )

    def insert(
        self,
        token_ids: Sequence[int],
        extra_key: Optional[str] = None,
        limit: Optional[int] = None,
        priority: int = 0,
        chunked: bool = False,
    ) -> int:
        key = self._make_key(token_ids, extra_key=extra_key, limit=limit)
        if len(key) == 0:
            return 0

        fake_value = torch.arange(len(key), dtype=torch.int64)
        result = self.tree.insert(
            InsertParams(
                key=key,
                value=fake_value,
                priority=priority,
                chunked=chunked,
            )
        )
        return result.prefix_len

    def acquire_request(
        self,
        request_id: str,
        token_ids: Sequence[int],
        extra_key: Optional[str] = None,
        max_prefix_len: Optional[int] = None,
    ) -> PrefixLease:
        match = self.match(token_ids, extra_key=extra_key, max_prefix_len=max_prefix_len)

        old_lease = self._leases.pop(request_id, None)
        if old_lease is not None:
            held_tokens = max(old_lease.matched_tokens, old_lease.committed_tokens)
            self._adjust_refcnt(
                old_lease.token_ids,
                old_lease.extra_key,
                held_tokens,
                -1,
            )

        normalized = self._normalize_tokens(token_ids)
        lease = PrefixLease(
            request_id=request_id,
            token_ids=normalized,
            extra_key=extra_key,
            matched_tokens=match.matched_tokens,
        )
        self._leases[request_id] = lease
        self._adjust_refcnt(normalized, extra_key, lease.matched_tokens, +1)
        return lease

    def commit_request(
        self,
        request_id: str,
        token_ids: Sequence[int],
        computed_tokens: int,
        extra_key: Optional[str] = None,
        priority: int = 0,
        chunked: bool = False,
    ) -> PrefixLease:
        committed_tokens = _align_down(computed_tokens, self.page_size)
        normalized = self._normalize_tokens(token_ids)

        lease = self._leases.get(request_id)
        if lease is None:
            lease = PrefixLease(
                request_id=request_id,
                token_ids=normalized,
                extra_key=extra_key,
            )
            self._leases[request_id] = lease

        if lease.extra_key != extra_key or lease.token_ids != normalized:
            raise ValueError(
                f"Request {request_id} changed prefix identity during commit."
            )

        if committed_tokens > 0:
            self.insert(
                normalized,
                extra_key=extra_key,
                limit=committed_tokens,
                priority=priority,
                chunked=chunked,
            )

        prev_held = max(lease.matched_tokens, lease.committed_tokens)
        lease.committed_tokens = max(lease.committed_tokens, committed_tokens)
        lease.matched_tokens = max(lease.matched_tokens, committed_tokens)
        next_held = max(lease.matched_tokens, lease.committed_tokens)

        if next_held > prev_held:
            self._adjust_refcnt(normalized, extra_key, prev_held, -1)
            self._adjust_refcnt(normalized, extra_key, next_held, +1)

        return lease

    def release_request(self, request_id: str) -> None:
        lease = self._leases.pop(request_id, None)
        if lease is None:
            return

        held_tokens = max(lease.matched_tokens, lease.committed_tokens)
        self._adjust_refcnt(lease.token_ids, lease.extra_key, held_tokens, -1)

    def get_request_lease(self, request_id: str) -> Optional[PrefixLease]:
        return self._leases.get(request_id)

    def get_refcnt(
        self,
        token_ids: Sequence[int],
        extra_key: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> int:
        key = self._make_ref_key(token_ids, extra_key=extra_key, limit=limit)
        return self._refcnt.get(key, 0)

    def match_request(self, req: object, max_prefix_len: Optional[int] = None) -> PrefixMatch:
        token_ids = getattr(req, "prompt_token_ids")
        extra_key = self.extra_key_fn(req) if self.extra_key_fn is not None else None
        return self.match(token_ids, extra_key=extra_key, max_prefix_len=max_prefix_len)

    def acquire_request_from_obj(
        self,
        req: object,
        max_prefix_len: Optional[int] = None,
    ) -> PrefixLease:
        token_ids = getattr(req, "prompt_token_ids")
        extra_key = self.extra_key_fn(req) if self.extra_key_fn is not None else None
        return self.acquire_request(
            getattr(req, "request_id"),
            token_ids,
            extra_key=extra_key,
            max_prefix_len=max_prefix_len,
        )

    def commit_request_from_obj(
        self,
        req: object,
        computed_tokens: int,
        priority: int = 0,
        chunked: bool = False,
    ) -> PrefixLease:
        token_ids = getattr(req, "prompt_token_ids")
        extra_key = self.extra_key_fn(req) if self.extra_key_fn is not None else None
        return self.commit_request(
            getattr(req, "request_id"),
            token_ids,
            computed_tokens=computed_tokens,
            extra_key=extra_key,
            priority=priority,
            chunked=chunked,
        )
