from __future__ import annotations

import dataclasses
import heapq
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from typing import Any, Iterator, List, NamedTuple, Optional, Sequence, Tuple, Union

import torch


@dataclass
class CacheInitParams:
    disable: bool
    page_size: int
    is_eagle: bool = False
    eviction_policy: str = "lru"


@dataclasses.dataclass
class MatchPrefixParams:
    key: "RadixKey"


@dataclasses.dataclass
class InsertParams:
    key: "RadixKey"
    value: Optional[torch.Tensor] = None
    chunked: bool = False
    priority: int = 0


@dataclasses.dataclass
class InsertResult:
    prefix_len: int


@dataclasses.dataclass
class EvictParams:
    num_tokens: int


@dataclasses.dataclass
class EvictResult:
    num_tokens_evicted: int = 0


class MatchResult(NamedTuple):
    device_indices: torch.Tensor
    last_device_node: Any
    last_host_node: Any
    host_hit_length: int = 0
    mamba_branching_seqlen: Optional[int] = None
    cache_protected_len: Optional[int] = None


class RadixKey:
    def __init__(
        self,
        token_ids: List[int],
        extra_key: Optional[str] = None,
        is_bigram: bool = False,
    ):
        self.token_ids = token_ids
        self.extra_key = extra_key
        self.is_bigram = is_bigram

    def __len__(self) -> int:
        return len(self.token_ids)

    def __iter__(self) -> Iterator[int]:
        return iter(self.token_ids)

    def __getitem__(self, idx: Union[int, slice]) -> "RadixKey":
        if isinstance(idx, slice):
            return RadixKey(self.token_ids[idx], self.extra_key, self.is_bigram)
        return RadixKey([self.token_ids[idx]], self.extra_key, self.is_bigram)

    def __repr__(self) -> str:
        preview = self.token_ids[:10]
        suffix = "..." if len(self.token_ids) > 10 else ""
        return (
            f"RadixKey(extra_key={self.extra_key!r}, "
            f"token_ids={preview}{suffix})"
        )


def convert_to_bigram_key(tokens: Sequence[int]) -> List[Tuple[int, int]]:
    if len(tokens) < 2:
        return []
    return [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]


def maybe_bigram_convert(
    is_eagle: bool,
    key: RadixKey,
    value: Optional[torch.Tensor] = None,
) -> Tuple[RadixKey, Optional[torch.Tensor]]:
    if is_eagle and not key.is_bigram:
        key.token_ids = convert_to_bigram_key(key.token_ids)
        key.is_bigram = True
        if value is not None:
            value = value[: len(key)]
    return key, value


class TreeNode:
    counter = 0

    def __init__(self, id: Optional[int] = None, priority: int = 0):
        self.children = defaultdict(TreeNode)
        self.parent: Optional["TreeNode"] = None
        self.key: Optional[RadixKey] = None
        self.value: Optional[torch.Tensor] = None
        self.lock_ref = 0
        self.last_access_time = time.monotonic()
        self.creation_time = time.monotonic()
        self.hit_count = 0
        self.priority = priority
        self.id = TreeNode.counter if id is None else id
        TreeNode.counter += 1

    @property
    def evicted(self):
        return self.value is None

    def __lt__(self, other: "TreeNode"):
        return self.last_access_time < other.last_access_time


def _check_extra_key(key0: RadixKey, key1: RadixKey):
    if key0.extra_key != key1.extra_key:
        raise ValueError(
            "Prefix match requires the same extra_key namespace: "
            f"{key0.extra_key!r} != {key1.extra_key!r}"
        )


def _key_match_page_size1(key0: RadixKey, key1: RadixKey):
    _check_extra_key(key0, key1)
    i = 0
    for k0, k1 in zip(key0.token_ids, key1.token_ids):
        if k0 != k1:
            break
        i += 1
    return i


def _key_match_paged(key0: RadixKey, key1: RadixKey, page_size: int):
    _check_extra_key(key0, key1)
    min_len = min(len(key0), len(key1))
    i = 0
    while i < min_len:
        if key0.token_ids[i : i + page_size] != key1.token_ids[i : i + page_size]:
            break
        i += page_size
    return i


def get_child_key(key: RadixKey, page_size: int = 1):
    if page_size == 1:
        plain_key = key.token_ids[0]
    else:
        plain_key = tuple(key.token_ids[:page_size])
    if key.extra_key is None:
        return plain_key
    return (key.extra_key, plain_key)


class EvictionStrategy:
    def get_priority(self, node: TreeNode):
        raise NotImplementedError()


class LRUStrategy(EvictionStrategy):
    def get_priority(self, node: TreeNode):
        return node.last_access_time


class LFUStrategy(EvictionStrategy):
    def get_priority(self, node: TreeNode):
        return (node.hit_count, node.last_access_time)


class FIFOStrategy(EvictionStrategy):
    def get_priority(self, node: TreeNode):
        return node.creation_time


class MRUStrategy(EvictionStrategy):
    def get_priority(self, node: TreeNode):
        return -node.last_access_time


class FILOStrategy(EvictionStrategy):
    def get_priority(self, node: TreeNode):
        return -node.creation_time


class PriorityStrategy(EvictionStrategy):
    def get_priority(self, node: TreeNode):
        return (node.priority, node.last_access_time)


class SLRUStrategy(EvictionStrategy):
    def __init__(self, protected_threshold: int = 2):
        self.protected_threshold = protected_threshold

    def get_priority(self, node: TreeNode):
        is_protected = 1 if node.hit_count >= self.protected_threshold else 0
        return (is_protected, node.last_access_time)


class RadixCache:
    def __init__(self, params: CacheInitParams):
        self.disable = params.disable
        self.page_size = params.page_size
        self.is_eagle = params.is_eagle
        self.eviction_policy = params.eviction_policy.lower()
        self.device = torch.device("cpu")

        if self.page_size == 1:
            self.key_match_fn = _key_match_page_size1
            self.get_child_key_fn = get_child_key
        else:
            self.key_match_fn = partial(_key_match_paged, page_size=self.page_size)
            self.get_child_key_fn = partial(get_child_key, page_size=self.page_size)

        if self.eviction_policy == "lru":
            self.eviction_strategy = LRUStrategy()
        elif self.eviction_policy == "lfu":
            self.eviction_strategy = LFUStrategy()
        elif self.eviction_policy == "fifo":
            self.eviction_strategy = FIFOStrategy()
        elif self.eviction_policy == "mru":
            self.eviction_strategy = MRUStrategy()
        elif self.eviction_policy == "filo":
            self.eviction_strategy = FILOStrategy()
        elif self.eviction_policy == "priority":
            self.eviction_strategy = PriorityStrategy()
        elif self.eviction_policy == "slru":
            self.eviction_strategy = SLRUStrategy()
        else:
            raise ValueError(f"Unknown eviction policy: {self.eviction_policy}")

        self.evictable_leaves = set()
        self.reset()

    @classmethod
    def create_simulated(
        cls,
        disable: bool = False,
        page_size: int = 1,
        is_eagle: bool = False,
        eviction_policy: str = "lru",
    ) -> "RadixCache":
        return cls(
            CacheInitParams(
                disable=disable,
                page_size=page_size,
                is_eagle=is_eagle,
                eviction_policy=eviction_policy,
            )
        )

    def reset(self):
        self.root_node = TreeNode(priority=-sys.maxsize)
        self.root_node.key = RadixKey(token_ids=[], extra_key=None)
        self.root_node.value = []
        self.root_node.lock_ref = 1
        self.evictable_size_ = 0
        self.protected_size_ = 0
        self.evictable_leaves.clear()

    def maybe_bigram_convert(
        self, key: RadixKey, value: Optional[torch.Tensor] = None
    ) -> Tuple[RadixKey, Optional[torch.Tensor]]:
        return maybe_bigram_convert(self.is_eagle, key, value)

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        key = params.key
        key, _ = self.maybe_bigram_convert(key)

        def empty():
            return MatchResult(
                device_indices=torch.empty((0,), dtype=torch.int64, device=self.device),
                last_device_node=self.root_node,
                last_host_node=self.root_node,
            )

        if self.disable or len(key) == 0:
            return empty()

        if self.page_size != 1:
            key = key[: len(key) // self.page_size * self.page_size]
        if len(key) == 0:
            return empty()

        value, last_node = self._match_prefix_helper(self.root_node, key)
        merged = (
            torch.cat(value)
            if value
            else torch.empty((0,), dtype=torch.int64, device=self.device)
        )
        return MatchResult(
            device_indices=merged,
            last_device_node=last_node,
            last_host_node=last_node,
        )

    def insert(self, params: InsertParams) -> InsertResult:
        if self.disable:
            return InsertResult(prefix_len=0)

        key = params.key
        value = params.value
        if value is None:
            value = torch.tensor(key.token_ids, dtype=torch.int64)

        key, value = self.maybe_bigram_convert(key, value)
        prefix_len = self._insert_helper(
            self.root_node,
            key,
            value,
            params.priority,
            params.chunked,
        )
        return InsertResult(prefix_len=prefix_len)

    def evict(self, params: EvictParams) -> EvictResult:
        if self.disable:
            return EvictResult()

        leaves = list(self.evictable_leaves)
        eviction_heap = [
            (self.eviction_strategy.get_priority(node), node) for node in leaves
        ]
        heapq.heapify(eviction_heap)

        num_evicted = 0
        while num_evicted < params.num_tokens and eviction_heap:
            _priority, node = heapq.heappop(eviction_heap)
            num_evicted += len(node.value)
            self._delete_leaf(node)
            if len(node.parent.children) == 0 and node.parent.lock_ref == 0:
                heapq.heappush(
                    eviction_heap,
                    (self.eviction_strategy.get_priority(node.parent), node.parent),
                )
        return EvictResult(num_tokens_evicted=num_evicted)

    def total_size(self):
        total = 0
        stack = [self.root_node]
        while stack:
            node = stack.pop()
            total += len(node.value)
            for child in node.children.values():
                if not child.evicted:
                    stack.append(child)
        return total

    def _match_prefix_helper(self, node: TreeNode, key: RadixKey):
        access_time = time.monotonic()
        node.last_access_time = access_time
        child_key = self.get_child_key_fn(key)
        value = []

        while len(key) > 0 and child_key in node.children:
            child = node.children[child_key]
            child.last_access_time = access_time
            prefix_len = self.key_match_fn(child.key, key)
            if prefix_len < len(child.key):
                new_node = self._split_node(child.key, child, prefix_len)
                value.append(new_node.value)
                node = new_node
                break

            value.append(child.value)
            node = child
            key = key[prefix_len:]
            if len(key):
                child_key = self.get_child_key_fn(key)

        return value, node

    def _split_node(self, key: RadixKey, child: TreeNode, split_len: int):
        new_node = TreeNode(priority=child.priority)
        new_node.hit_count = child.hit_count
        new_node.children = {self.get_child_key_fn(key[split_len:]): child}
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref
        new_node.key = child.key[:split_len]
        new_node.value = child.value[:split_len].clone()
        child.parent = new_node
        child.key = child.key[split_len:]
        child.value = child.value[split_len:].clone()
        new_node.parent.children[self.get_child_key_fn(key)] = new_node
        return new_node

    def _inc_hit_count(self, node: TreeNode, chunked: bool = False):
        if chunked:
            return
        node.hit_count += 1

    def _insert_helper(
        self,
        node: TreeNode,
        key: RadixKey,
        value: torch.Tensor,
        priority: int = 0,
        chunked: bool = False,
    ):
        access_time = time.monotonic()
        node.last_access_time = access_time
        node.priority = max(node.priority, priority or 0)
        if len(key) == 0:
            return 0

        child_key = self.get_child_key_fn(key)
        total_prefix_length = 0

        while len(key) > 0 and child_key in node.children:
            node = node.children[child_key]
            node.last_access_time = access_time
            prefix_len = self.key_match_fn(node.key, key)
            total_prefix_length += prefix_len
            key = key[prefix_len:]
            value = value[prefix_len:]

            if prefix_len < len(node.key):
                new_node = self._split_node(node.key, node, prefix_len)
                new_node.priority = max(new_node.priority, priority or 0)
                self._inc_hit_count(new_node, chunked)
                node = new_node
            else:
                node.priority = max(node.priority, priority or 0)
                self._inc_hit_count(node, chunked)

            if len(key):
                child_key = self.get_child_key_fn(key)

        if len(key):
            new_node = TreeNode(priority=priority or 0)
            new_node.parent = node
            new_node.key = key
            new_node.value = value.clone()
            self._inc_hit_count(new_node, chunked)
            node.children[child_key] = new_node
            self.evictable_size_ += len(key)
            self._update_leaf_status(node)
            self._update_leaf_status(new_node)

        return total_prefix_length

    def _delete_leaf(self, node: TreeNode):
        key = self.get_child_key_fn(node.key)
        removed = node.parent.children.pop(key, None)
        assert removed == node
        self.evictable_size_ -= len(node.key)
        self.evictable_leaves.discard(node)
        self._update_leaf_status(node.parent)

    def _update_leaf_status(self, node: TreeNode):
        if node.evicted or node.lock_ref > 0:
            self.evictable_leaves.discard(node)
            return

        for child in node.children.values():
            if not child.evicted:
                self.evictable_leaves.discard(node)
                return

        self.evictable_leaves.add(node)
