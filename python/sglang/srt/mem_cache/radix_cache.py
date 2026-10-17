from __future__ import annotations

from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.utils import convert_to_bigram_key

"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
The radix tree data structure for managing the KV cache.
"""

import heapq
import logging
import sys
import time
from collections import defaultdict
from functools import lru_cache, partial
from typing import TYPE_CHECKING, Any, Iterator, List, Optional, Tuple, Union

import torch

logger = logging.getLogger(__name__)

from sglang.srt.disaggregation.kv_events import (
    MEDIUM_GPU,
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
)
from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    EvictParams,
    EvictResult,
    InsertParams,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.evict_policy import (
    EvictionStrategy,
    FIFOStrategy,
    FILOStrategy,
    LFUStrategy,
    LRUStrategy,
    MRUStrategy,
    PriorityStrategy,
)
from sglang.srt.mem_cache.hicache_storage import get_hash_str, hash_str_to_int64

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


class RadixKey:
    def __init__(
        self,
        token_ids: List[int],
        extra_key: Optional[str] = None,
        is_bigram: bool = False,
    ):
        # token ids sequence
        self.token_ids = token_ids
        # extra key (e.g. lora_id, cache_salt)
        self.extra_key = extra_key
        # is bigram key
        self.is_bigram = is_bigram

    def __len__(self) -> int:
        return len(self.token_ids)

    def __iter__(self) -> Iterator[int]:
        return iter(self.token_ids)

    def __getitem__(self, idx: Union[int, slice]) -> "RadixKey":
        if isinstance(idx, slice):
            return RadixKey(self.token_ids[idx], self.extra_key)
        return RadixKey([self.token_ids[idx]], self.extra_key)

    def __repr__(self) -> str:
        preview = self.token_ids[:10]
        return f"RadixKey(extra_key={self.extra_key!r}, token_ids={preview}{'...' if len(self.token_ids) > 10 else ''})"


class TreeNode:

    counter = 0

    def __init__(self, id: Optional[int] = None, priority: int = 0):
        self.children = defaultdict(TreeNode)
        self.parent: TreeNode = None
        self.key: RadixKey = None
        self.value: Optional[torch.Tensor] = None
        self.lock_ref = 0
        self.last_access_time = time.monotonic()
        self.creation_time = time.monotonic()

        self.hit_count = 0
        # indicating the node is locked to protect from eviction
        # incremented when the node is referenced by a storage operation
        self.host_ref_counter = 0
        # store the host indices of KV cache
        self.host_value: Optional[torch.Tensor] = None
        # store hash values of each pages
        self.hash_value: Optional[List[str]] = None
        # priority for priority-aware eviction
        self.priority = priority

        # ── Chain-ID tracking (agent call-chain KV cache management) ────────
        # Chains that have previously finished using this node
        self.visited_chain_ids: set = set()
        # Chains currently holding an active lock_ref on this node
        self.active_chain_ids: set = set()
        # Chains predicted (by the prefetch controller) to access this node soon
        self.predicted_chain_ids: set = set()
        # Set of (chain_id, agent_name) tuples: records which chain was calling
        # which agent when this node was created or accessed.
        self.chain_agent_pairs: set = set()
        # ────────────────────────────────────────────────────────────────────

        # ── Prefetch tracking ────────────────────────────────────────────────
        self._prefetched: bool = False        # 该节点是否由预取 load_back 加载
        self._prefetch_accessed: bool = False  # 该节点是否被真实请求 match_prefix 命中
        # 已对哪些 chain 记过“预取命中”（同一 chain 重复命中不重复计数）
        self._prefetch_hit_chain_ids: set = set()
        # ────────────────────────────────────────────────────────────────────

        self.id = TreeNode.counter if id is None else id
        TreeNode.counter += 1

    def __lt__(self, other: "TreeNode") -> bool:
        """Tie-breaker for heapq when priority tuples are equal."""
        return self.id < other.id

    @property
    def evicted(self):
        return self.value is None

    @property
    def backuped(self):
        return self.host_value is not None

    def protect_host(self):
        """Protect the host value from eviction."""
        self.host_ref_counter += 1

    def release_host(self):
        """Release the host value, allowing it to be evicted."""
        if self.host_ref_counter > 0:
            self.host_ref_counter -= 1
        else:
            raise RuntimeError("Host reference counter is already zero.")

    def get_last_hash_value(self) -> Optional[str]:
        """Returns the hash value of the last page in this node."""
        if self.hash_value is None or len(self.hash_value) == 0:
            return None
        return self.hash_value[-1]

    @lru_cache(maxsize=1)
    def get_prefix_hash_values(self, node: TreeNode) -> List[str]:
        if node is None or node.hash_value is None:
            return []

        return node.get_prefix_hash_values(node.parent) + node.hash_value

    def __lt__(self, other: "TreeNode"):
        return self.last_access_time < other.last_access_time


def _check_extra_key(key0: RadixKey, key1: RadixKey):
    if key0.extra_key != key1.extra_key:
        raise ValueError(
            f"_key_match should be run on the same extra key, but got key0.extra_key={key0.extra_key} != key1.extra_key={key1.extra_key}"
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
    else:
        return (key.extra_key, plain_key)


def compute_node_hash_values(node: "TreeNode", page_size: int) -> List[str]:
    """Compute SHA256-based hash values for position-aware identification.

    Args:
        node: The TreeNode to compute hash values for
        page_size: The page size for chunking tokens

    Returns:
        List of SHA256 hex strings, one per page
    """
    hash_values = []

    # Get parent's last hash value if parent exists
    parent_hash = None
    if node.parent is not None and node.parent.hash_value is not None:
        # Check if parent is root by checking if it has empty key
        if len(node.parent.key) > 0 and len(node.parent.hash_value) > 0:
            parent_hash = node.parent.hash_value[-1]

    # Iterate through node's pages
    for start in range(0, len(node.key), page_size):
        page_tokens = node.key.token_ids[start : start + page_size]
        if not page_tokens:
            continue

        # Use SHA256-based chaining via get_hash_str
        hash_val = get_hash_str(page_tokens, prior_hash=parent_hash)
        hash_values.append(hash_val)
        parent_hash = hash_val

    return hash_values


def split_node_hash_value(
    child_hash_value: Optional[List[str]], split_len: int, page_size: int
) -> tuple[Optional[List[str]], Optional[List[str]]]:
    """Split hash_value between parent and child nodes during node splitting.

    Args:
        child_hash_value: The hash_value list from the child node being split
        split_len: The length at which to split (in tokens)
        page_size: The page size for calculating number of pages

    Returns:
        Tuple of (new_node_hash_value, updated_child_hash_value)
    """
    if child_hash_value is None:
        return None, None

    if page_size == 1:
        split_pages = split_len
    else:
        split_pages = split_len // page_size

    new_node_hash = child_hash_value[:split_pages]
    child_hash = child_hash_value[split_pages:]

    return new_node_hash, child_hash


class RadixCache(BasePrefixCache):
    def __init__(self, params: CacheInitParams):
        self.disable = params.disable
        self.req_to_token_pool = params.req_to_token_pool
        self.token_to_kv_pool_allocator = params.token_to_kv_pool_allocator
        self.page_size = params.page_size
        self.enable_kv_cache_events = params.enable_kv_cache_events
        self.is_eagle = params.is_eagle
        self.disable_finished_insert = params.disable_finished_insert
        self.eviction_policy = params.eviction_policy.lower()

        self.kv_event_queue = []

        if params.enable_metrics:
            self.init_metrics_collector()

        if self.token_to_kv_pool_allocator:
            self.device = self.token_to_kv_pool_allocator.device
        else:
            self.device = torch.device("cpu")

        if self.page_size == 1:
            self.key_match_fn = _key_match_page_size1
            self.get_child_key_fn = get_child_key
        else:
            self.key_match_fn = partial(_key_match_paged, page_size=self.page_size)
            self.get_child_key_fn = partial(get_child_key, page_size=self.page_size)

        if self.eviction_policy == "lru":
            self.eviction_strategy: EvictionStrategy = LRUStrategy()
        elif self.eviction_policy == "lfu":
            self.eviction_strategy: EvictionStrategy = LFUStrategy()
        elif self.eviction_policy == "fifo":
            self.eviction_strategy: EvictionStrategy = FIFOStrategy()
        elif self.eviction_policy == "mru":
            self.eviction_strategy: EvictionStrategy = MRUStrategy()
        elif self.eviction_policy == "filo":
            self.eviction_strategy: EvictionStrategy = FILOStrategy()
        elif self.eviction_policy == "priority":
            self.eviction_strategy: EvictionStrategy = PriorityStrategy()
        else:
            raise ValueError(
                f"Unknown eviction policy: {self.eviction_policy}. Supported policies: 'lru', 'lfu', 'fifo', 'mru', 'filo', 'priority'."
            )

        self.evictable_leaves = set()
        # Set of chain IDs that have been notified as terminated; populated by
        # the scheduler when it receives a MyChainTerminatedReqInput message.
        # Nodes exclusively used by chains in this set become priority eviction candidates.
        self.terminated_chain_ids: set = set()

        self._pbkv = None
        self.pred_evict_enable = False
        self.pred_evict_K = 3
        self.pred_evict_gamma = 0.7
        self.chain_predictions = {}

        # ── Prefetch hit-rate tracking ────────────────────────────────────────
        self._prefetch_total: int = 0   # 被标记为预取的节点总数（含 split 产生的）
        self._prefetch_hit: int = 0     # 被驱逐前被真实请求访问的预取节点数
        self._prefetch_miss: int = 0    # 被驱逐时未被访问的预取节点数
        self._prefetch_chain_hit: int = 0  # 不同 chain 命中同一预取节点时分别计数
        # ──────────────────────────────────────────────────────────────────────

        self.reset()

    @classmethod
    def create_simulated(
        self,
        disable: bool = False,
        mock_allocator: Optional[Any] = None,
        page_size: int = 1,
        enable_kv_cache_events: bool = False,
    ) -> RadixCache:
        """Init a radix cache without memory pools for simulation purpose."""
        params = CacheInitParams(
            disable=disable,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=mock_allocator,
            page_size=page_size,
            enable_kv_cache_events=enable_kv_cache_events,
        )
        return RadixCache(params)

    ##### Public API #####

    def reset(self):
        # Initialize root with minimum priority so any real priority overrides it
        self.root_node = TreeNode(priority=-sys.maxsize)
        self.root_node.key = RadixKey(token_ids=[], extra_key=None)
        self.root_node.value = []
        self.root_node.host_value = []
        self.root_node.lock_ref = 1
        self.root_node.hash_value = []
        self.evictable_size_ = 0
        self.protected_size_ = 0
        self.evictable_leaves.clear()
        self.terminated_chain_ids.clear()
        self.chain_predictions.clear()
        if self._pbkv is not None:
            self._pbkv.clear()
            self.terminated_chain_ids = self._pbkv.terminated
            self.chain_predictions = self._pbkv.forecasts
        self._record_all_cleared_event()

    def maybe_bigram_convert(
        self, key: RadixKey, value: Optional[torch.Tensor] = None
    ) -> Tuple[RadixKey, Optional[torch.Tensor]]:
        if self.is_eagle and not key.is_bigram:
            key.token_ids = convert_to_bigram_key(key.token_ids)
            if value is not None:
                value = value[: len(key)]

        return key, value

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        """Find the longest cached prefix of ``key`` in the radix tree.

        The logical namespace for prefix matching is determined by both the
        token id sequence and the optional ``extra_key`` carried by ``RadixKey``.
        Entries that share identical leading token ids but have *different*
        ``extra_key`` values are intentionally kept disjoint and never share
        prefix nodes. This is useful to:

        * Isolate KV cache lines for different LoRA / adapter IDs.
        * Separate requests that intentionally should not share state (e.g.,
          different sampling salt, cache version, or retrieval augmentation
          context) by supplying a distinct ``extra_key``.

        Args:
            params (MatchPrefixParams): Parameters containing the lookup key
                with a list of token ids and an optional ``extra_key`` namespace tag.
                If ``page_size > 1`` the length is internally truncated to a multiple
                of ``page_size`` before matching. Passing an empty key returns an
                empty result with the root as the last node.

        Returns:
            MatchResult: ``device_indices`` is a 1-D ``torch.int64`` tensor of
            the concatenated KV cache indices corresponding to the longest
            cached prefix (may be length 0). ``last_device_node`` and
            ``last_host_node`` (currently the same) are the tree node objects
            representing the terminal node of the matched prefix. This method
            may mutate internal structure by splitting an existing node if the
            match ends inside a stored segment.

        Internal updates:
            * Refreshes access metadata (timestamps) used by the
                configured eviction strategy.
            * If the lookup ends inside a stored segment the node is split once
                to expose a precise boundary; this structural refinement improves
                subsequent match efficiency and does not duplicate data.
        """
        key = params.key
        key, _ = self.maybe_bigram_convert(key)

        def empty_match_result():
            return MatchResult(
                device_indices=torch.empty(
                    (0,),
                    dtype=torch.int64,
                    device=self.device,
                ),
                last_device_node=self.root_node,
                last_host_node=self.root_node,
            )

        if self.disable or len(key) == 0:
            return empty_match_result()

        if self.page_size != 1:
            page_aligned_len = len(key) // self.page_size * self.page_size
            key = key[:page_aligned_len]

        if len(key) == 0:
            return empty_match_result()

        value, last_node = self._match_prefix_helper(self.root_node, key)
        if value:
            value = torch.cat(value)
        else:
            value = torch.empty((0,), dtype=torch.int64, device=self.device)
        return MatchResult(
            device_indices=value,
            last_device_node=last_node,
            last_host_node=last_node,
        )

    def insert(self, params: InsertParams) -> InsertResult:
        if self.disable:
            return InsertResult(prefix_len=0)

        key = params.key
        value = params.value
        priority = params.priority

        if value is None:
            value = torch.tensor(key.token_ids, dtype=torch.int64)

        key, value = self.maybe_bigram_convert(key, value)

        prefix_len = self._insert_helper(self.root_node, key, value, priority)
        return InsertResult(prefix_len=prefix_len)

    def _page_align_keys(self, key: list) -> list:
        if self.page_size == 1:
            return key
        page_aligned_len = len(key) // self.page_size * self.page_size
        return key[:page_aligned_len]

    def cache_finished_req(self, req: Req, is_insert: bool = True):
        """Cache request when it finishes."""
        # In deterministic mode, disable finished request insertion to radix cache
        if self.disable_finished_insert:
            is_insert = False

        kv_committed_len = req.pop_committed_kv_cache()
        if self.disable:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, :kv_committed_len
            ]
            self.token_to_kv_pool_allocator.free(kv_indices)
            return

        token_ids = (req.origin_input_ids + req.output_ids)[:kv_committed_len]
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        # Maybe convert to bigram keys for EAGLE
        keys = convert_to_bigram_key(token_ids) if self.is_eagle else token_ids
        keys = self._page_align_keys(keys)
        values = kv_indices[: len(keys)].to(dtype=torch.int64, copy=True)
        radix_key = RadixKey(keys, req.extra_key, is_bigram=self.is_eagle)

        # Radix Cache takes one ref in memory pool
        if is_insert:
            priority = getattr(req, "priority", 0) or 0
            result = self.insert(
                InsertParams(
                    key=radix_key,
                    value=values,
                    prev_prefix_len=req.cache_protected_len,
                    priority=priority,
                )
            )
            duplicate_indices = result.duplicate_indices
            if duplicate_indices is not None:
                # HiRadix may have non-contiguous duplicated prefix segments
                # (e.g., when evicted nodes are restored in-place).
                self.token_to_kv_pool_allocator.free(duplicate_indices)
            else:
                new_prefix_len = result.prefix_len
                # Free the duplicates that were already in the tree
                self.token_to_kv_pool_allocator.free(
                    kv_indices[req.cache_protected_len : new_prefix_len]
                )
        else:
            self.token_to_kv_pool_allocator.free(
                kv_indices[req.cache_protected_len : len(keys)]
            )

        # free the unaligned tail
        self.token_to_kv_pool_allocator.free(kv_indices[len(keys) :])

        # Remove req slot release the cache lock
        self.dec_lock_ref(req.last_node)
        # Chain tracking: move this chain from active→visited on all ancestors.
        _chain_id = getattr(req, "my_chain_id", None)
        if _chain_id:
            _agent_name = getattr(req, "my_current_agent", None)
            if is_insert:
                # req.last_node is the cache-hit endpoint from match_prefix at
                # request start time.  The insert() above may have added deeper
                # nodes for the newly generated tokens.  Find the actual
                # end-of-sequence node so that the entire path
                # (pre-existing prefix + newly inserted content) is marked as
                # visited for this chain.
                _final_match = self.match_prefix(MatchPrefixParams(key=radix_key))
                _release_node = _final_match.last_device_node
            else:
                _release_node = req.last_node
            self.release_chain_node_ref(_release_node, _chain_id, _agent_name)

    def cache_unfinished_req(self, req: Req, chunked=False):
        """Cache request when it is unfinished."""
        if self.disable:
            return

        token_ids = req.fill_ids
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        # Maybe convert to bigram keys for EAGLE
        keys = convert_to_bigram_key(token_ids) if self.is_eagle else token_ids
        keys = self._page_align_keys(keys)
        values = kv_indices[: len(keys)].to(dtype=torch.int64, copy=True)
        radix_key = RadixKey(keys, req.extra_key, is_bigram=self.is_eagle)

        # Radix Cache takes one ref in memory pool
        result = self.insert(
            InsertParams(
                key=radix_key,
                value=values,
                prev_prefix_len=req.cache_protected_len,
                chunked=chunked,
                priority=getattr(req, "priority", 0) or 0,
            )
        )
        duplicate_indices = result.duplicate_indices
        if duplicate_indices is not None:
            self.token_to_kv_pool_allocator.free(duplicate_indices)
        else:
            new_prefix_len = result.prefix_len
            self.token_to_kv_pool_allocator.free(
                kv_indices[req.cache_protected_len : new_prefix_len]
            )

        # The prefix indices could be updated, reuse it
        match_result = self.match_prefix(MatchPrefixParams(key=radix_key))
        new_indices, new_last_node = (
            match_result.device_indices,
            match_result.last_device_node,
        )
        assert len(new_indices) == len(keys), f"{len(new_indices)=}, {len(keys)=}"

        self.req_to_token_pool.write(
            (req.req_pool_idx, slice(req.cache_protected_len, len(new_indices))),
            new_indices[req.cache_protected_len :],
        )

        # The cache_protected_len is not always equal to len(req.prefix_indices)
        # since for page_size > 1, the partial part is added to req.prefix_indices, but that part of kv indices is not added to the tree.
        # It should be freed in the next cache_unfinished_req and final cache_finished_req to avoid memory leak.
        # So we introduce this `cache_protected_len` field to make sure the partial part can be freed correctly.
        req.cache_protected_len = len(new_indices)

        _chain_id_unfinished = getattr(req, "my_chain_id", None)
        _agent_name_unfinished = getattr(req, "my_current_agent", None)
        self.dec_lock_ref(req.last_node)
        if _chain_id_unfinished:
            self.release_chain_node_ref(req.last_node, _chain_id_unfinished, _agent_name_unfinished)
        self.inc_lock_ref(new_last_node)
        if _chain_id_unfinished:
            self.add_chain_node_ref(new_last_node, _chain_id_unfinished, _agent_name_unfinished)

        # `req.prefix_indices` will be used in `PrefillAdder::add_chunked_req` later
        # - page_size != 1: there is a partial page at the end, keep the full kv_indices
        # - eagle case: bigram keys will only cache len - 1 kv indices
        if len(new_indices) < len(kv_indices):
            req.prefix_indices = torch.cat(
                [new_indices, kv_indices[len(new_indices) :]]
            )
        else:
            req.prefix_indices = new_indices

        req.last_node = new_last_node

    def pretty_print(self):
        self._print_helper(self.root_node, 0)
        print(f"#tokens: {self.total_size()}")

    def total_size(self):
        return self._total_size_helper()

    def evict(self, params: EvictParams) -> EvictResult:
        if self.disable:
            return EvictResult()
        start_time = time.perf_counter()
        heap = (
            self._pbkv.gpu
            if self._pbkv is not None
            else [(self._get_evict_priority(n), n) for n in self.evictable_leaves]
        )
        if self._pbkv is None:
            heapq.heapify(heap)
        num_evicted = 0
        while num_evicted < params.num_tokens and heap:
            priority, node = (
                heap.pop() if self._pbkv is not None else heapq.heappop(heap)
            )
            if node.lock_ref > 0 or node.evicted or node is self.root_node:
                continue
            try:
                from sglang.srt.my_modules import kvcache_logger as kvcl

                if priority[0] == 0:
                    kvcl.log_chain_evict_node(len(node.value), node.visited_chain_ids)
                elif self.pred_evict_enable:
                    kvcl.log_pred_evict_node(
                        len(node.value),
                        self._pbkv.score(node),
                        {a for _, a in node.chain_agent_pairs},
                    )
                else:
                    kvcl.log_regular_evict_node(len(node.value))
            except ImportError:
                pass
            if node._prefetched and not node._prefetch_accessed:
                self._prefetch_miss += 1
            self.token_to_kv_pool_allocator.free(node.value)
            num_evicted += len(node.value)
            self._delete_leaf(node)
            if (
                self._pbkv is None
                and not node.parent.children
                and node.parent.lock_ref == 0
            ):
                heapq.heappush(
                    heap, (self._get_evict_priority(node.parent), node.parent)
                )
            self._record_remove_event(node)
        self.update_eviction_metrics(num_evicted, start_time)
        return EvictResult(num_tokens_evicted=num_evicted)

    def inc_lock_ref(self, node: TreeNode):
        if self.disable:
            return 0

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 0:
                self.evictable_size_ -= len(node.key)
                self.protected_size_ += len(node.key)
                delta -= len(node.key)
            node.lock_ref += 1
            self._update_leaf_status(node)
            node = node.parent
        return delta

    def dec_lock_ref(self, node: TreeNode):
        if self.disable:
            return 0

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 1:
                self.evictable_size_ += len(node.key)
                self.protected_size_ -= len(node.key)
                delta += len(node.key)
            node.lock_ref -= 1
            self._update_leaf_status(node)
            if node.parent is None:
                assert (
                    node is self.root_node
                ), f"This request holds the node from another tree"
            node = node.parent
        return delta

    def evictable_size(self):
        return self.evictable_size_

    def effective_evictable_size(self):
        """Actual evictable token count considering cascade from leaves.

        evictable_size_ counts ALL unlocked non-evicted nodes, but eviction
        can only proceed from leaf nodes upward.  A non-leaf internal node
        becomes evictable only after ALL its non-evicted children have been
        evicted first.  If any child is locked (lock_ref > 0), the parent
        can never cascade during this scheduling cycle.

        This method simulates the cascade: starting from current evictable
        leaves, it "virtually evicts" each leaf and checks whether the
        parent becomes a new virtual leaf, accumulating the true token count
        the scheduler can rely on.
        """
        total = 0
        virtually_evicted = set()
        # Use a list as a queue; new virtual leaves are appended during iteration
        queue = list(self.evictable_leaves)

        for node in queue:
            if node in virtually_evicted:
                continue
            total += len(node.key)
            virtually_evicted.add(node)

            # Check whether parent becomes a new virtual leaf
            parent = node.parent
            if parent is None or parent is self.root_node:
                continue
            if parent.lock_ref > 0 or parent.evicted:
                continue
            if parent in virtually_evicted:
                continue

            all_children_done = True
            for child in parent.children.values():
                if not child.evicted and child not in virtually_evicted:
                    all_children_done = False
                    break
            if all_children_done:
                queue.append(parent)

        return total

    def protected_size(self):
        # protected size refers to the size of the cache that is locked
        return self.protected_size_

    # ── Chain-ID tracking helpers ────────────────────────────────────────────

    def add_chain_node_ref(self, node: TreeNode, chain_id: str,
                           agent_name: str = None) -> None:
        """Mark chain_id as actively using *node* and all its ancestors."""
        cur = node
        while cur != self.root_node:
            cur.active_chain_ids.add(chain_id)
            cur.predicted_chain_ids.discard(chain_id)  # promote predicted→active
            if agent_name:
                cur.chain_agent_pairs.add((chain_id, agent_name))
            self._pbkv_sync(cur)
            cur = cur.parent

    def release_chain_node_ref(self, node: TreeNode, chain_id: str,
                               agent_name: str = None) -> None:
        """Move chain_id from active to visited for *node* and all ancestors."""
        cur = node
        while cur != self.root_node:
            cur.active_chain_ids.discard(chain_id)
            cur.visited_chain_ids.add(chain_id)
            if agent_name:
                cur.chain_agent_pairs.add((chain_id, agent_name))
            self._pbkv_sync(cur)
            cur = cur.parent

    def add_chain_predicted_ref(self, node: TreeNode, chain_id: str) -> None:
        """Record chain_id as predicted to access *node* and ancestors soon."""
        cur = node
        while cur != self.root_node:
            if chain_id not in cur.active_chain_ids:
                cur.predicted_chain_ids.add(chain_id)
            cur = cur.parent

    # ── Prefetch hit-rate tracking ────────────────────────────────────────

    def mark_prefetch_accessed(
        self,
        node: TreeNode,
        chain_id: Optional[str] = None,
    ) -> None:
        """Walk from *node* to root, mark any prefetched nodes as accessed.

        Called when a real request's match_prefix selects *node* as its
        last_node, indicating the prefetched KV cache was actually reused.
        """
        cur = node
        while cur != self.root_node:
            if cur._prefetched and not cur._prefetch_accessed:
                cur._prefetch_accessed = True
                self._prefetch_hit += 1
            if cur._prefetched and chain_id:
                if chain_id not in cur._prefetch_hit_chain_ids:
                    cur._prefetch_hit_chain_ids.add(chain_id)
                    self._prefetch_chain_hit += 1
            cur = cur.parent

    def _is_chain_exclusive_evictable(self, node: TreeNode) -> bool:
        chains = node.visited_chain_ids | node.active_chain_ids
        return (
            node.lock_ref == 0
            and bool(chains)
            and chains.issubset(self.terminated_chain_ids)
        )

    def _get_evict_priority(self, node: TreeNode):
        if self._pbkv is not None:
            return self._pbkv.priority(node)
        return (1, self.eviction_strategy.get_priority(node))

    def setup_lifecycle_evict(self):
        from sglang.srt.my_modules.pbkv_policy import PBKVPolicy

        if self._pbkv is None:
            self._pbkv = PBKVPolicy(lambda node: node.last_access_time)
            self.terminated_chain_ids = self._pbkv.terminated
            self.chain_predictions = self._pbkv.forecasts
            stack = list(self.root_node.children.values())
            while stack:
                node = stack.pop()
                self._pbkv.sync(node)
                stack.extend(node.children.values())

    def _pbkv_sync(self, node):
        if self._pbkv is not None:
            self._pbkv.sync(node)

    def _pbkv_forget(self, node):
        if self._pbkv is not None:
            self._pbkv.forget(node)

    def setup_pred_evict(self, agent_labels, K=3, gamma=0.7, end_id=None, noise=0.0):
        self.setup_lifecycle_evict()
        if end_id is None:
            end_id = next(
                (
                    i
                    for i, label in enumerate(agent_labels)
                    if label in ("Chain_End", "chain_end", "END", "<END>", "⟨END⟩")
                ),
                None,
            )
        if end_id is None:
            raise ValueError("PBKV requires the predictor's END class index")
        self._pbkv.configure(agent_labels, K, gamma, end_id, noise)
        self.pred_evict_enable = True
        self.pred_evict_K, self.pred_evict_gamma = K, gamma

    def update_chain_prediction(self, chain_id, probabilities):
        self._pbkv.update(chain_id, probabilities)

    def invalidate_chain_prediction(self, chain_id):
        if self._pbkv is not None:
            self._pbkv.invalidate(chain_id)

    def remove_chain_prediction(self, chain_id):
        if self._pbkv is not None:
            self._pbkv.terminate(chain_id)

    def _compute_pred_evict_value_fast(self, node):
        return self._pbkv.score(node) if self._pbkv is not None else 0.0

    def compute_cpu_node_value(self, node):
        return self._pbkv.value(node) if self._pbkv is not None else 0.0

    def count_dead_kv_volume(self):
        if self._pbkv is None:
            return 0, 0
        return self._pbkv.dead_tokens, self._pbkv.dead_nodes

    def all_values_flatten(self):
        values = []

        def _dfs_helper(node: TreeNode):
            for _, child in node.children.items():
                values.append(child.value)
                _dfs_helper(child)

        _dfs_helper(self.root_node)
        return torch.cat(values)

    ##### Internal Helper Functions #####

    def _match_prefix_helper(self, node: TreeNode, key: RadixKey):
        access_time = time.monotonic()
        node.last_access_time = access_time
        self._pbkv_sync(node)

        child_key = self.get_child_key_fn(key)

        value = []
        while len(key) > 0 and child_key in node.children.keys():
            child = node.children[child_key]
            child.last_access_time = access_time
            self._pbkv_sync(child)
            prefix_len = self.key_match_fn(child.key, key)
            if prefix_len < len(child.key):
                new_node = self._split_node(child.key, child, prefix_len)
                value.append(new_node.value)
                node = new_node
                break
            else:
                value.append(child.value)
                node = child
                key = key[prefix_len:]

                if len(key):
                    child_key = self.get_child_key_fn(key)

        return value, node

    def _split_node(self, key: RadixKey, child: TreeNode, split_len: int):
        # new_node -> child
        # New node inherits child's priority (represents shared prefix)
        new_node = TreeNode(priority=child.priority)
        new_node.children = {self.get_child_key_fn(key[split_len:]): child}
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref
        new_node.key = child.key[:split_len]
        new_node.value = child.value[:split_len].clone()
        child.parent = new_node
        child.key = child.key[split_len:]
        child.value = child.value[split_len:].clone()
        new_node.parent.children[self.get_child_key_fn(key)] = new_node

        # Split hash_value if it was already computed, otherwise leave as None
        new_node.hash_value, child.hash_value = split_node_hash_value(
            child.hash_value, split_len, self.page_size
        )

        # ── Inherit chain metadata ──────────────────────────────────────
        # All chains that accessed the original child also traversed the
        # prefix portion (new_node).  Both halves keep the same sets.
        new_node.visited_chain_ids = child.visited_chain_ids.copy()
        new_node.active_chain_ids = child.active_chain_ids.copy()
        new_node.predicted_chain_ids = child.predicted_chain_ids.copy()
        new_node.chain_agent_pairs = child.chain_agent_pairs.copy()

        # ── Inherit prefetch tracking ──────────────────────────────────
        new_node._prefetched = child._prefetched
        new_node._prefetch_accessed = child._prefetch_accessed
        new_node._prefetch_hit_chain_ids = child._prefetch_hit_chain_ids.copy()
        if child._prefetched:
            self._prefetch_total += 1  # one node became two

        self._update_leaf_status(child)
        self._update_leaf_status(new_node)
        self._update_leaf_status(new_node.parent)
        return new_node

    def _insert_helper(self, node: TreeNode, key: RadixKey, value, priority: int = 0):
        # Convert None priority to 0
        if priority is None:
            priority = 0
        access_time = time.monotonic()
        node.last_access_time = access_time
        # Update priority along the path (take max to propagate higher priority)
        node.priority = max(node.priority, priority)
        if len(key) == 0:
            return 0

        child_key = self.get_child_key_fn(key)

        total_prefix_length = 0
        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = access_time
            prefix_len = self.key_match_fn(node.key, key)
            total_prefix_length += prefix_len
            key = key[prefix_len:]
            value = value[prefix_len:]

            if prefix_len < len(node.key):
                new_node = self._split_node(node.key, node, prefix_len)
                new_node.priority = max(new_node.priority, priority)
                node = new_node
            else:
                node.priority = max(node.priority, priority)

            self._pbkv_sync(node)
            if len(key):
                child_key = self.get_child_key_fn(key)

        if len(key):
            new_node = TreeNode(priority=priority)
            new_node.parent = node
            new_node.key = key
            new_node.value = value.clone()
            node.children[child_key] = new_node
            self.evictable_size_ += len(key)
            self._update_leaf_status(node)
            self._update_leaf_status(new_node)
            # Hash will be computed lazily during event emission
            self._record_store_event(new_node)
        return total_prefix_length

    def _print_helper(self, node: TreeNode, indent: int):
        """Prints the radix tree in a human-readable format."""
        stack = [(node, indent)]
        while stack:
            current_node, current_indent = stack.pop()
            print(
                " " * current_indent,
                len(current_node.key),
                current_node.key.token_ids[:10],
                f"r={current_node.lock_ref}",
            )
            for key, child in current_node.children.items():
                stack.append((child, current_indent + 2))

                assert key == self.get_child_key_fn(
                    child.key
                ), f"{key=}, {self.get_child_key_fn(child.key)=}"

    def _delete_leaf(self, node):
        key = self.get_child_key_fn(node.key)
        v = node.parent.children.pop(key, None)
        assert v == node, f"parent does not have child key, {key}"

        self.evictable_size_ -= len(node.key)
        if node in self.evictable_leaves:
            self.evictable_leaves.remove(node)
        self._pbkv_forget(node)
        self._update_leaf_status(node.parent)

    def _update_leaf_status(self, node: TreeNode):
        self._pbkv_sync(node)
        if node.evicted or node.lock_ref > 0:
            if node in self.evictable_leaves:
                self.evictable_leaves.remove(node)
            return

        for child in node.children.values():
            if not child.evicted:
                if node in self.evictable_leaves:
                    self.evictable_leaves.remove(node)
                return

        if node not in self.evictable_leaves:
            self.evictable_leaves.add(node)

    def _total_size_helper(self):
        total_size = 0
        stack = [self.root_node]
        while stack:
            current_node = stack.pop()
            total_size += len(current_node.value)
            for child in current_node.children.values():
                if child.evicted:
                    continue
                stack.append(child)
        return total_size

    def _record_store_event(self, node: TreeNode):
        # One BlockStored per ``page_size`` chunk.
        if self.enable_kv_cache_events:
            # Compute hash_value lazily if not already set
            if node.hash_value is None:
                node.hash_value = compute_node_hash_values(node, self.page_size)

            # Get parent's last hash value for first page
            parent_block_hash = None
            if node.parent is not None and node.parent != self.root_node:
                if (
                    node.parent.hash_value is not None
                    and len(node.parent.hash_value) > 0
                ):
                    parent_block_hash = hash_str_to_int64(node.parent.hash_value[-1])

            page_index = 0
            for start in range(0, len(node.key), self.page_size):
                page_tokens = node.key.token_ids[start : start + self.page_size]
                if not page_tokens:
                    continue

                block_hash = hash_str_to_int64(node.hash_value[page_index])

                self.kv_event_queue.append(
                    BlockStored(
                        block_hashes=[block_hash],
                        parent_block_hash=parent_block_hash,
                        token_ids=page_tokens,
                        block_size=len(page_tokens),
                        lora_id=None,
                        medium=MEDIUM_GPU,
                    )
                )

                parent_block_hash = block_hash
                page_index += 1

    def _record_remove_event(self, node: TreeNode):
        # One BlockRemoved per chunk.
        if self.enable_kv_cache_events:
            # Compute hash_value lazily if not already set (must match what was stored)
            if node.hash_value is None:
                node.hash_value = compute_node_hash_values(node, self.page_size)

            page_index = 0
            for start in range(0, len(node.key), self.page_size):
                page_tokens = node.key.token_ids[start : start + self.page_size]
                if not page_tokens:
                    continue

                block_hash = hash_str_to_int64(node.hash_value[page_index])

                self.kv_event_queue.append(
                    BlockRemoved(block_hashes=[block_hash], medium=MEDIUM_GPU)
                )

                page_index += 1

    def _record_all_cleared_event(self):
        if self.enable_kv_cache_events:
            self.kv_event_queue.append(AllBlocksCleared())

    def take_events(self):
        """Atomically takes all events and clears the queue.

        Returns:
            A list of KV cache events.
        """
        if not self.enable_kv_cache_events:
            return []
        events = self.kv_event_queue
        self.kv_event_queue = []
        return events


if __name__ == "__main__":
    tree = RadixCache.create_simulated()

    # Example token id sequences (as lists of ints)
    tree.insert(InsertParams(key=RadixKey(token_ids=[1, 2, 3], extra_key=None)))
    tree.insert(InsertParams(key=RadixKey(token_ids=[1, 2, 3], extra_key=None)))
    tree.insert(InsertParams(key=RadixKey(token_ids=[1, 2, 4, 5], extra_key=None)))
    tree.insert(
        InsertParams(key=RadixKey(token_ids=[1, 2, 4, 5, 6, 7], extra_key=None))
    )
    tree.insert(
        InsertParams(key=RadixKey(token_ids=[8, 9, 10, 11, 12], extra_key=None))
    )
    tree.pretty_print()

    print(
        tree.match_prefix(
            MatchPrefixParams(key=RadixKey(token_ids=[1, 2, 3, 13, 14], extra_key=None))
        )
    )
