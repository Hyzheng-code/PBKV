"""Exercise real radix methods with an allocation-checking CPU/DMA harness.

Requires the SGLang runtime dependencies; no language-model weights are needed.
"""

import unittest
from collections import deque
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey, TreeNode


class Allocator:
    device = torch.device("cpu")

    def __init__(self, size):
        self.size = size
        self.used = set()

    def available_size(self):
        return self.size - len(self.used)

    def alloc(self, size):
        free = sorted(set(range(1, self.size + 1)) - self.used)
        if len(free) < size:
            return None
        indices = free[:size]
        self.used.update(indices)
        return torch.tensor(indices, dtype=torch.int64)

    def free(self, indices):
        values = indices.tolist()
        assert len(set(values)) == len(values)
        assert set(values) <= self.used, "double free or freeing an unallocated slot"
        self.used.difference_update(values)


class Controller:
    write_policy = "write_through"

    def __init__(self, alloc):
        self.mem_pool_device_allocator = alloc
        self.pbkv_transfer_samples = deque()
        self.ack_load_queue = []
        self.ack_write_queue = []
        self.loads = []
        self.pending = []
        self.host_dropped = []

    def load(self, host_indices, node_id):
        indices = self.mem_pool_device_allocator.alloc(len(host_indices))
        if indices is not None:
            self.loads.extend(host_indices.tolist())
            self.pending.append(node_id)
        return indices

    def start_loading(self):
        if not self.pending:
            return -1
        event = SimpleNamespace(query=lambda: True, synchronize=lambda: None)
        self.ack_load_queue.append((None, event, self.pending))
        self.pending = []
        return 0

    def evict_device(self, indices):
        self.mem_pool_device_allocator.free(indices)
        return len(indices)

    def evict_host(self, indices):
        self.host_dropped.extend(indices.tolist())
        return len(indices)


class CacheHarness(HiRadixCache):
    def __init__(self, size=64, bandwidth=1e6, step=1):
        allocator = Allocator(size)
        self.evictable_host_leaves = set()
        self.cache_controller = Controller(allocator)
        self.ongoing_load_back = {}
        self.ongoing_write_through = {}
        self.tp_world_size = 1
        self.enable_storage = False
        self.token_to_kv_pool_host = SimpleNamespace(get_size_per_token=lambda: 1)
        self.load_back_threshold = 10
        RadixCache.__init__(self, CacheInitParams(False, None, allocator, 1))
        self.setup_pred_evict(["<END>", "A", "B"])
        self.setup_prefetch_budget(bandwidth, step)

    def reset(self):
        self.evictable_host_leaves.clear()
        RadixCache.reset(self)

    def add(self, size, workflow, agent="A", parent=None, resident=True):
        parent = parent or self.root_node
        node = TreeNode()
        node.parent = parent
        node.key = RadixKey([node.id + 100] * size)
        node.host_value = torch.arange(node.id * 1000, node.id * 1000 + size)
        node.value = self.token_to_kv_pool_allocator.alloc(size) if resident else None
        node.chain_agent_pairs.add((workflow, agent))
        node.visited_chain_ids.add(workflow)
        parent.children[self.get_child_key_fn(node.key)] = node
        if resident:
            self.evictable_size_ += size
        self._update_leaf_status(node)
        self._update_host_leaf_status(node)
        self._update_leaf_status(parent)
        self._update_host_leaf_status(parent)
        return node

    def forecast(self, workflow, a=0.8):
        self.update_chain_prediction(workflow, [[0, a, 1 - a]] * 3)


class CacheTests(unittest.TestCase):
    def assert_no_leaks(self, cache):
        nodes = [n for n in cache._pbkv.nodes.values() if not n.evicted]
        indices = [i for n in nodes for i in n.value.tolist()]
        self.assertEqual(len(indices), len(set(indices)))
        self.assertEqual(set(indices), cache.token_to_kv_pool_allocator.used)
        self.assertEqual(cache.evictable_size_ + cache.protected_size_, len(indices))

    def test_shared_host_prefix_allocated_once_and_unlocked_after_dma(self):
        c = CacheHarness()
        parent = c.add(4, "w", resident=False)
        a = c.add(4, "w", parent=parent, resident=False)
        b = c.add(4, "w", parent=parent, resident=False)
        c.forecast("w")
        stats = c.try_batch_prefetch()
        self.assertEqual(stats["loaded_tokens"], 12)
        self.assertEqual(len(c.cache_controller.loads), 12)
        self.assertEqual(len(set(c.cache_controller.loads)), 12)
        self.assert_no_leaks(c)
        c.loading_check()
        self.assertTrue(all(n.lock_ref == 0 for n in [parent, a, b]))
        self.assert_no_leaks(c)

    def test_flush_clears_inflight_prefetch_and_policy_indexes(self):
        c = CacheHarness()
        c.add(4, "w", resident=False)
        c.forecast("w")
        c.try_batch_prefetch()
        c.ongoing_prefetch = {}
        c.ongoing_backup = {}
        c.prefetch_loaded_tokens_by_reqid = {}
        c.cache_controller.reset = lambda: None
        c.token_to_kv_pool_host.clear = lambda: None
        HiRadixCache.reset(c)
        self.assertFalse(c.ongoing_load_back)
        self.assertFalse(c._pbkv.nodes)
        self.assertFalse(c.chain_predictions)
        self.assertEqual(c._pbkv_prefetch_consumer_index, -1)

    def test_controller_flush_waits_for_dma_before_releasing_indices(self):
        import threading

        from sglang.srt.managers.cache_controller import HiCacheController

        completed = []

        class Buffer:
            def clear(self):
                self_ready = completed == ["write", "load"]
                assert self_ready, "Indices released before DMA completed"

        buffers = {
            key: Buffer()
            for key in [
                "write_queue",
                "load_queue",
                "write_buffer",
                "load_buffer",
                "ack_write_queue",
                "ack_load_queue",
                "pbkv_transfer_samples",
            ]
        }
        controller = SimpleNamespace(
            **buffers,
            stop_event=threading.Event(),
            storage_stop_event=threading.Event(),
            write_stream=SimpleNamespace(synchronize=lambda: completed.append("write")),
            load_stream=SimpleNamespace(synchronize=lambda: completed.append("load")),
            enable_storage=False,
        )
        HiCacheController.reset(controller)
        self.assertEqual(completed, ["write", "load"])

    def test_request_waits_for_already_launched_prefetch_dma(self):
        c = CacheHarness()
        c.add(4, "w", resident=False)
        c.forecast("w")
        c.try_batch_prefetch()
        self.assertEqual(c.ready_to_load_host_cache(), 0)
        c.loading_check()
        self.assertEqual(c.ready_to_load_host_cache(), -1)

    def test_conservative_prefetch_reclaims_only_retired(self):
        c = CacheHarness(size=16)
        dead = c.add(8, "dead")
        active = c.add(8, "active")
        candidate = c.add(8, "future", resident=False)
        c.remove_chain_prediction("dead")
        c.forecast("future")
        stats = c.try_batch_prefetch()
        self.assertEqual(stats["active_evicted_tokens"], 0)
        self.assertTrue(dead.evicted)
        self.assertFalse(active.evicted)
        self.assertFalse(candidate.evicted)
        self.assert_no_leaks(c)

    def test_default_never_evicts_active_when_full(self):
        c = CacheHarness(size=16)
        active = c.add(16, "active")
        candidate = c.add(4, "future", resident=False)
        c.forecast("future")
        self.assertTrue(c.try_batch_prefetch()["skipped"])
        self.assertFalse(active.evicted)
        self.assertTrue(candidate.evicted)
        self.assert_no_leaks(c)

    def test_aggressive_cap_and_k_step_victim_order(self):
        c = CacheHarness(size=20)
        low = c.add(4, "w", "A")
        high = c.add(4, "w", "B")
        locked = c.add(12, "locked")
        c.inc_lock_ref(locked)
        c.update_chain_prediction("w", [[0, 0.6, 0.4], [0, 0, 1], [0, 0, 1]])
        candidate = c.add(4, "future", resident=False)
        c.forecast("future")
        c.add(4, "future2", resident=False)
        c.forecast("future2")
        stats = c.try_batch_prefetch(prefetch_evict_live=True, active_fraction=0.2)
        self.assertEqual(stats["active_evicted_tokens"], 4)
        self.assertTrue(low.evicted)
        self.assertFalse(high.evicted)
        self.assertFalse(candidate.evicted)
        self.assert_no_leaks(c)

    def test_bandwidth_budget_no_reserve_or_small_node_threshold(self):
        c = CacheHarness(size=8, bandwidth=2, step=1)
        c.add(2, "w", resident=False)
        c.add(2, "w", resident=False)
        c.forecast("w")
        stats = c.try_batch_prefetch()
        self.assertEqual(stats["loaded_tokens"], 2)
        self.assertEqual(stats["s_min"], 2)
        self.assert_no_leaks(c)

    def test_locked_prefix_not_reclaimed_for_its_own_prefetch(self):
        c = CacheHarness(size=8)
        parent = c.add(8, "dead")
        child = c.add(4, "future", parent=parent, resident=False)
        c.remove_chain_prediction("dead")
        c.forecast("future")
        self.assertTrue(c.try_batch_prefetch()["skipped"])
        self.assertFalse(parent.evicted)
        self.assertTrue(child.evicted)
        self.assertEqual(parent.lock_ref, 0)
        self.assert_no_leaks(c)

    def test_host_protection_then_deletion_updates_indexes(self):
        c = CacheHarness()
        node = c.add(4, "w", resident=False)
        c.forecast("w")
        node.protect_host()
        c.evict_host(4)
        self.assertIn(node.id, c._pbkv.nodes)
        node.release_host()
        c.evict_host(4)
        self.assertNotIn(node.id, c._pbkv.nodes)
        self.assertEqual(len(c._pbkv.host), 0)
        self.assertEqual(len(c._pbkv.host_drop), 0)
        self.assertEqual(c.cache_controller.host_dropped, node.host_value.tolist())

    def test_radix_split_inherits_membership_scores_and_dead_volume(self):
        alloc = Allocator(16)
        c = RadixCache.create_simulated(mock_allocator=alloc)
        c.setup_pred_evict(["<END>", "A", "B"])
        c.insert(InsertParams(RadixKey([1, 2, 3, 4]), alloc.alloc(4)))
        node = next(iter(c.root_node.children.values()))
        c.release_chain_node_ref(node, "w", "A")
        c.update_chain_prediction("w", [[0, 0.8, 0.2]] * 3)
        c.match_prefix(MatchPrefixParams(RadixKey([1, 2])))
        parent = node.parent
        self.assertAlmostEqual(c.compute_cpu_node_value(parent), 0.8)
        self.assertAlmostEqual(c.compute_cpu_node_value(node), 0.8)
        c.remove_chain_prediction("w")
        self.assertEqual(c.count_dead_kv_volume(), (4, 2))
        c.evict(EvictParams(4))
        self.assertEqual(alloc.used, set())
        self.assertEqual(c.count_dead_kv_volume(), (0, 0))
        self.assertEqual(len(c._pbkv.nodes), 0)

    def test_hiradix_split_then_eviction_keeps_accounting(self):
        c = CacheHarness()
        node = c.add(8, "w")
        c.forecast("w")
        c.remove_chain_prediction("w")
        parent = c._split_node(node.key, node, 4)
        self.assertEqual(c.count_dead_kv_volume(), (8, 2))
        c.evict(EvictParams(8))
        self.assertTrue(parent.evicted)
        self.assertTrue(node.evicted)
        self.assertEqual(c.count_dead_kv_volume(), (0, 0))
        self.assert_no_leaks(c)


if __name__ == "__main__":
    unittest.main()
