"""PBKV equations (1)/(2) and incremental cache indexes (paper §4.2–4.3).

This module has no CUDA or SGLang runtime dependencies. Cache nodes are owned by
the radix tree; the policy only indexes them and their observed workflow access.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np


class IndexedHeap:
    """Min heap with O(log n) replacement/removal and no stale-entry buildup."""

    def __init__(self):
        self.items = []
        self.positions = {}

    def __len__(self):
        return len(self.items)

    def clear(self):
        self.items.clear()
        self.positions.clear()

    def _swap(self, a, b):
        self.items[a], self.items[b] = self.items[b], self.items[a]
        self.positions[self.items[a][1]] = a
        self.positions[self.items[b][1]] = b

    def _up(self, i):
        while i and self.items[i][:2] < self.items[(i - 1) // 2][:2]:
            parent = (i - 1) // 2
            self._swap(i, parent)
            i = parent
        return i

    def _down(self, i):
        while 2 * i + 1 < len(self.items):
            j = 2 * i + 1
            if j + 1 < len(self.items) and self.items[j + 1][:2] < self.items[j][:2]:
                j += 1
            if self.items[i][:2] <= self.items[j][:2]:
                break
            self._swap(i, j)
            i = j

    def set(self, node, priority):
        if node.id in self.positions:
            i = self.positions[node.id]
            self.items[i] = (priority, node.id, node)
        else:
            i = len(self.items)
            self.positions[node.id] = i
            self.items.append((priority, node.id, node))
        self._down(self._up(i))

    def discard(self, node):
        i = self.positions.pop(node.id, None)
        if i is None:
            return
        last = self.items.pop()
        if i < len(self.items):
            self.items[i] = last
            self.positions[last[1]] = i
            self._down(self._up(i))

    def peek(self):
        priority, _, node = self.items[0]
        return priority, node

    def pop(self):
        priority, node = self.peek()
        self.discard(node)
        return priority, node


class PBKVPolicy:
    """Workflow-specific scores and persistent GPU/host candidate heaps.

    ``sync`` follows a radix mutation or an access. A forecast update visits
    only nodes associated with that workflow, updates cached contributions,
    and replaces their heap entries. It never scans unrelated workflows/nodes.
    """

    def __init__(self, fallback_priority):
        self.fallback_priority = fallback_priority
        self.labels = []
        self.agent_index = {}
        self.end_id = None
        self.k = 1
        self.gamma = 0.7
        self.noise = 0.0
        self.predictive = False
        self.clear()

    def clear(self):
        self.nodes = {}
        self.terminated = set()
        self.forecasts = {}
        self.vectors = {}
        self.memberships = {}
        self.pairs = {}
        self.by_workflow = defaultdict(dict)
        self.scores = {}
        self.gpu = IndexedHeap()
        self.host = IndexedHeap()
        self.host_drop = IndexedHeap()
        self.dead_subtree = {}
        self.dead_size = {}
        self.dead_tokens = 0
        self.dead_nodes = 0

    def configure(self, labels, k, gamma, end_id, noise=0.0):
        if k < 1 or not 0 <= gamma < 1 or not 0 <= noise <= 1:
            raise ValueError("PBKV requires K >= 1, 0 <= gamma < 1, 0 <= noise <= 1")
        if (
            len(labels) < 2
            or len(set(labels)) != len(labels)
            or not 0 <= end_id < len(labels)
        ):
            raise ValueError(
                "PBKV requires unique class labels and an explicit END class"
            )
        if self.forecasts:
            raise RuntimeError("Configure PBKV before submitting forecasts")
        self.labels = list(labels)
        self.agent_index = {label: i for i, label in enumerate(labels)}
        self.end_id, self.k, self.gamma, self.noise = end_id, k, gamma, noise
        self.predictive = True
        for node in self.nodes.values():
            self.refresh(node)

    def _vectors(self, probabilities):
        p = np.array(probabilities, dtype=np.float64, copy=True)
        if p.ndim != 2 or p.shape[0] < self.k or p.shape[1] != len(self.labels):
            raise ValueError(
                f"Expected at least {self.k} complete distributions over {len(self.labels)} classes"
            )
        p = p[: self.k]
        if (
            not np.isfinite(p).all()
            or (p < 0).any()
            or not np.allclose(p.sum(axis=1), 1, atol=1e-5)
        ):
            raise ValueError(
                "PBKV forecasts must be finite, nonnegative, normalized distributions"
            )
        if self.noise:
            uniform = np.ones(len(self.labels), dtype=np.float64)
            uniform[self.end_id] = 0
            uniform /= uniform.sum()
            p = (1 - self.noise) * p + self.noise * uniform
        survival = np.ones(self.k)
        survival[1:] = np.cumprod(1 - p[:-1, self.end_id])
        weights = survival * self.gamma ** np.arange(self.k)
        discounted = (weights[:, None] * p).sum(axis=0)
        one_step = p[0].copy()
        # A_w(c) is implicitly zero on END, including unknown-agent handling.
        discounted[self.end_id] = one_step[self.end_id] = 0
        return p, np.stack((discounted, one_step))

    def update(self, workflow, probabilities):
        if workflow in self.terminated:
            return
        p, vectors = self._vectors(probabilities)
        previous = self.vectors.get(workflow, np.zeros_like(vectors))
        delta = vectors - previous
        self.forecasts[workflow], self.vectors[workflow] = p, vectors
        for node_id, agents in self.by_workflow.get(workflow, {}).items():
            columns = [
                self.agent_index[a] for a in sorted(agents) if a in self.agent_index
            ]
            if columns:
                self.scores[node_id] += delta[:, columns].sum(axis=1)
            self.refresh(self.nodes[node_id])

    def invalidate(self, workflow):
        """Remove a workflow's forecast and its cached score contributions."""
        previous = self.vectors.pop(workflow, None)
        self.forecasts.pop(workflow, None)
        if previous is not None:
            for node_id, agents in self.by_workflow.get(workflow, {}).items():
                columns = [
                    self.agent_index[a] for a in sorted(agents) if a in self.agent_index
                ]
                if columns:
                    self.scores[node_id] -= previous[:, columns].sum(axis=1)
                self.refresh(self.nodes[node_id])

    def terminate(self, workflow):
        self.invalidate(workflow)
        self.terminated.add(workflow)
        for node_id in self.by_workflow.get(workflow, {}):
            self.refresh(self.nodes[node_id])

    def sync(self, node):
        if node.parent is None:
            return
        node_id = node.id
        if node_id not in self.nodes:
            self.nodes[node_id] = node
            self.scores[node_id] = np.zeros(2)
            self.memberships[node_id] = set()
            self.pairs[node_id] = set()
        chains = node.visited_chain_ids | node.active_chain_ids
        for workflow in sorted(chains - self.memberships[node_id]):
            self.by_workflow[workflow].setdefault(node_id, set())
        self.memberships[node_id].update(chains)
        for workflow, agent in sorted(node.chain_agent_pairs - self.pairs[node_id]):
            self.by_workflow[workflow].setdefault(node_id, set()).add(agent)
            self.memberships[node_id].add(workflow)
            column = self.agent_index.get(agent)
            if column is not None and workflow in self.vectors:
                self.scores[node_id] += self.vectors[workflow][:, column]
        self.pairs[node_id].update(node.chain_agent_pairs)
        self.refresh(node)

    def forget(self, node):
        self.gpu.discard(node)
        self.host.discard(node)
        self.host_drop.discard(node)
        for workflow in self.memberships.pop(node.id, ()):
            entries = self.by_workflow[workflow]
            entries.pop(node.id, None)
            if not entries:
                del self.by_workflow[workflow]
        self.nodes.pop(node.id, None)
        self.pairs.pop(node.id, None)
        self.scores.pop(node.id, None)
        self.dead_subtree.pop(node.id, None)
        old = self.dead_size.pop(node.id, 0)
        self.dead_tokens -= old
        self.dead_nodes -= int(old > 0)
        self._refresh_dead_ancestors(node.parent)

    def retired(self, node):
        chains = self.memberships.get(node.id, set())
        return bool(chains) and chains.issubset(self.terminated)

    def score(self, node):
        return max(0.0, float(self.scores.get(node.id, (0, 0))[0]))

    def value(self, node):
        return max(0.0, float(self.scores.get(node.id, (0, 0))[1]))

    def priority(self, node):
        fallback = self.fallback_priority(node)
        if self.retired(node):
            return (0, len(self.memberships[node.id]), fallback)
        return (1, self.score(node) if self.predictive else 0, fallback)

    def refresh(self, node):
        if node.parent is None or node.id not in self.nodes:
            return
        if (
            not node.evicted
            and node.lock_ref == 0
            and all(c.evicted for c in node.children.values())
        ):
            self.gpu.set(node, self.priority(node))
        else:
            self.gpu.discard(node)
        # Host protections may end asynchronously; recheck them when popping.
        if (
            node.evicted
            and node.backuped
            and node.lock_ref == 0
            and self.value(node) > 0
        ):
            self.host.set(node, (-self.value(node),))
        else:
            self.host.discard(node)
        if (
            node.evicted
            and node.backuped
            and node.lock_ref == 0
            and all(not c.evicted for c in node.children.values())
        ):
            self.host_drop.set(node, self.priority(node))
        else:
            self.host_drop.discard(node)
        self._refresh_dead_ancestors(node)

    def _refresh_dead_ancestors(self, node):
        while node is not None and node.parent is not None:
            eligible = (
                not node.evicted
                and node.lock_ref == 0
                and self.retired(node)
                and all(
                    c.evicted or self.dead_subtree.get(c.id, False)
                    for c in node.children.values()
                )
            )
            size = len(node.value) if eligible else 0
            old = self.dead_size.get(node.id, 0)
            prior_eligible = self.dead_subtree.get(node.id, False)
            self.dead_subtree[node.id] = eligible
            self.dead_size[node.id] = size
            self.dead_tokens += size - old
            self.dead_nodes += int(size > 0) - int(old > 0)
            if eligible == prior_eligible:
                break
            node = node.parent


class RuntimeAverages:
    """Measured mean decode time and transfer throughput; defaults bootstrap only."""

    def __init__(self, bandwidth=20e9, step_seconds=0.030):
        if (
            not math.isfinite(bandwidth)
            or not math.isfinite(step_seconds)
            or bandwidth <= 0
            or step_seconds <= 0
        ):
            raise ValueError(
                "Positive bootstrap bandwidth and decode duration required"
            )
        self.initial_bandwidth = bandwidth
        self.initial_step_seconds = step_seconds
        self.decode_seconds = 0.0
        self.decode_count = 0
        self.transfer_bytes = 0
        self.transfer_seconds = 0.0

    def observe_decode(self, seconds):
        if math.isfinite(seconds) and seconds > 0:
            self.decode_seconds += seconds
            self.decode_count += 1

    def observe_transfer(self, num_bytes, seconds):
        if num_bytes > 0 and math.isfinite(seconds) and seconds > 0:
            self.transfer_bytes += num_bytes
            self.transfer_seconds += seconds

    @property
    def bandwidth(self):
        return (
            self.transfer_bytes / self.transfer_seconds
            if self.transfer_seconds
            else self.initial_bandwidth
        )

    @property
    def step_seconds(self):
        return (
            self.decode_seconds / self.decode_count
            if self.decode_count
            else self.initial_step_seconds
        )

    def token_budget(self, bytes_per_token, page_size):
        tokens = int(self.bandwidth * self.step_seconds / bytes_per_token)
        return tokens // page_size * page_size
