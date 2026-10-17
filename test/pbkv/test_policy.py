"""Equation and data-structure tests; run with PYTHONPATH=python python -m unittest discover -s test/pbkv."""

import random
import unittest
from types import SimpleNamespace

import numpy as np

from sglang.srt.my_modules.pbkv_policy import IndexedHeap, PBKVPolicy, RuntimeAverages


class Node:
    counter = 0

    def __init__(self, parent=None, size=4, pairs=(), resident=True, stamp=0):
        self.id = Node.counter
        Node.counter += 1
        self.parent = parent
        self.children = {}
        if parent is not None:
            parent.children[self.id] = self
        self.value = list(range(size)) if resident else None
        self.host_value = list(range(size))
        self.lock_ref = 0
        self.chain_agent_pairs = set(pairs)
        self.visited_chain_ids = {w for w, _ in pairs}
        self.active_chain_ids = set()
        self.last_access_time = stamp

    @property
    def evicted(self):
        return self.value is None

    @property
    def backuped(self):
        return self.host_value is not None


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.policy = PBKVPolicy(lambda n: n.last_access_time)
        self.policy.configure(["<END>", "A", "B"], 3, 0.7, 0)
        self.root = Node()
        self.p = np.array([[0.2, 0.3, 0.5], [0.4, 0.5, 0.1], [0.1, 0.2, 0.7]])

    def node(self, **kwargs):
        n = Node(self.root, **kwargs)
        self.policy.sync(n)
        return n

    def test_equation_2_per_workflow_and_survival(self):
        n = self.node(pairs=[("w1", "A"), ("w2", "B")])
        p2 = np.array([[0.8, 0.1, 0.1], [0.1, 0.2, 0.7], [0.3, 0.6, 0.1]])
        self.policy.update("w1", self.p)
        self.policy.update("w2", p2)
        expected = (
            0.3
            + 0.7 * 0.8 * 0.5
            + 0.7**2 * 0.8 * 0.6 * 0.2
            + 0.1
            + 0.7 * 0.2 * 0.7
            + 0.7**2 * 0.2 * 0.9 * 0.1
        )
        self.assertAlmostEqual(self.policy.score(n), expected)
        self.assertAlmostEqual(self.policy.value(n), 0.4)
        self.assertEqual(len(self.policy.forecasts["w1"]), 3)

    def test_binary_access_and_no_workflow_cross_product(self):
        n = self.node(pairs=[("w1", "A")])
        self.policy.update("w1", self.p)
        self.policy.update("unrelated", self.p)
        before = self.policy.score(n)
        for _ in range(4):
            self.policy.sync(n)
        self.assertAlmostEqual(before, self.policy.score(n))
        n.chain_agent_pairs.add(("w1", "B"))
        self.policy.sync(n)
        self.assertAlmostEqual(self.policy.value(n), 0.8)

    def test_refresh_replace_invalidate_terminate(self):
        n = self.node(pairs=[("w", "A")], resident=False)
        for _ in range(10):
            self.policy.update("w", self.p)
        self.assertAlmostEqual(self.policy.value(n), 0.3)
        self.assertEqual(len(self.policy.host), 1)
        self.policy.invalidate("w")
        self.assertAlmostEqual(self.policy.score(n), 0)
        self.assertEqual(len(self.policy.host), 0)
        self.policy.update("w", self.p)
        self.policy.terminate("w")
        self.assertTrue(self.policy.retired(n))
        self.policy.update("w", self.p)
        self.assertAlmostEqual(self.policy.score(n), 0)

    def test_end_stops_future_heads(self):
        n = self.node(pairs=[("w", "A"), ("w", "<END>")])
        self.policy.update("w", [[1, 0, 0], [0, 1, 0], [0, 1, 0]])
        self.assertEqual(self.policy.score(n), 0)
        self.assertEqual(self.policy.value(n), 0)

    def test_noise_shared_by_prefetch_and_eviction(self):
        n = self.node(pairs=[("w", "A")])
        self.policy.configure(["<END>", "A", "B"], 3, 0.7, 0, noise=1)
        self.policy.update("w", self.p)
        self.assertAlmostEqual(self.policy.value(n), 0.5)
        self.assertAlmostEqual(self.policy.score(n), 0.5 * (1 + 0.7 + 0.7**2))

    def test_retired_order_distinct_workflows_then_lru(self):
        active = self.node(pairs=[("live", "A")], stamp=-99)
        shared = self.node(pairs=[("x", "A"), ("y", "A")], stamp=-10)
        newer = self.node(pairs=[("x", "A"), ("x", "B")], stamp=2)
        older = self.node(pairs=[("y", "A")], stamp=1)
        self.policy.terminate("x")
        self.assertFalse(self.policy.retired(shared))
        self.policy.terminate("y")
        self.assertEqual(
            [self.policy.gpu.pop()[1] for _ in range(4)], [older, newer, shared, active]
        )

    def test_lae_uses_same_retired_order(self):
        self.policy.predictive = False
        many = self.node(pairs=[("x", "A"), ("y", "B")], stamp=0)
        one = self.node(pairs=[("x", "A")], stamp=2)
        self.policy.terminate("x")
        self.policy.terminate("y")
        self.assertIs(self.policy.gpu.pop()[1], one)
        self.assertIs(self.policy.gpu.pop()[1], many)

    def test_dead_volume_tracks_reclaimable_subtrees_and_locks(self):
        parent = self.node(pairs=[("dead", "A")], size=3)
        child = Node(parent, pairs=[("live", "A")], size=5)
        self.policy.sync(child)
        self.policy.sync(parent)
        self.policy.terminate("dead")
        self.assertEqual(self.policy.dead_tokens, 0)
        self.policy.terminate("live")
        self.assertEqual(self.policy.dead_tokens, 8)
        child.lock_ref = 1
        self.policy.sync(child)
        self.assertEqual(self.policy.dead_tokens, 0)
        child.lock_ref = 0
        self.policy.sync(child)
        self.assertEqual(self.policy.dead_tokens, 8)
        child.value = None
        self.policy.sync(child)
        self.assertEqual(self.policy.dead_tokens, 3)
        parent.children.clear()
        self.policy.forget(child)
        self.assertEqual(self.policy.dead_tokens, 3)

    def test_host_value_and_gpu_score_have_different_order(self):
        a = self.node(pairs=[("w", "A")])
        b = self.node(pairs=[("w", "B")])
        self.policy.update("w", [[0, 0.6, 0.4], [0, 0, 1], [0, 0, 1]])
        self.assertIs(self.policy.gpu.peek()[1], a)
        a.value = b.value = None
        self.policy.sync(a)
        self.policy.sync(b)
        self.assertIs(self.policy.host.peek()[1], a)

    def test_reject_partial_and_invalid_distributions(self):
        for bad in (
            [[0, 1, 0]],
            [[0, 1]] * 3,
            [[0, 1, 1]] * 3,
            [[float("nan"), 0, 1]] * 3,
        ):
            with self.assertRaises(ValueError):
                self.policy.update("w", bad)

    def test_heap_random_mutations_match_sorted_oracle(self):
        rng = random.Random(3)
        heap = IndexedHeap()
        oracle = {}
        nodes = [SimpleNamespace(id=i) for i in range(50)]
        for _ in range(3000):
            node = rng.choice(nodes)
            if rng.random() < 0.7:
                priority = (rng.randint(0, 4), rng.randint(0, 100))
                heap.set(node, priority)
                oracle[node.id] = priority
            else:
                heap.discard(node)
                oracle.pop(node.id, None)
            self.assertEqual(len(heap), len(oracle))
            if oracle:
                expected = min(oracle, key=lambda i: (oracle[i], i))
                self.assertEqual(heap.peek()[1].id, expected)
        order = sorted(oracle, key=lambda i: (oracle[i], i))
        self.assertEqual([heap.pop()[1].id for _ in order], order)

    def test_measurements_replace_bootstrap_and_align_pages(self):
        avg = RuntimeAverages(1000, 0.1)
        self.assertEqual(avg.token_budget(2, 4), 48)
        avg.observe_decode(0.02)
        avg.observe_decode(0.04)
        avg.observe_transfer(1000, 0.5)
        avg.observe_transfer(2000, 0.5)
        self.assertAlmostEqual(avg.bandwidth, 3000)
        self.assertAlmostEqual(avg.step_seconds, 0.03)
        self.assertEqual(avg.token_budget(2, 4), 44)
        avg.observe_decode(float("nan"))
        avg.observe_transfer(99, 0)
        self.assertEqual(avg.decode_count, 2)


if __name__ == "__main__":
    unittest.main()
