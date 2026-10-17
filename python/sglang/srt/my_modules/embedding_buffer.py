"""FIFO prefill snapshots containing only the last token of each request.

The forward hook keeps GPU tensors until commit() selects one row per request
on the producing stream. Serving never copies the full prefill tensor to CPU.
"""

from __future__ import annotations

from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass

import torch

_staging_layers = {}
_queue = deque()


@dataclass
class EmbeddingBatch:
    layers: dict
    lengths: tuple
    ready_event: object = None


def put(layer_key, hidden):
    _staging_layers[layer_key] = hidden.detach()


def put_last(hidden):
    put("last", hidden)


def put_4th(hidden):
    put("4th", hidden)


def commit(lengths, stream=None, to_cpu=False):
    global _staging_layers
    lengths = tuple(int(n) for n in lengths)
    if not _staging_layers:
        _queue.append(EmbeddingBatch({}, lengths))
        return
    indices, offset = [], 0
    for length in lengths:
        if length < 0:
            raise ValueError("Prefill lengths must be nonnegative")
        offset += length
        indices.append(max(0, offset - 1))
    layers = {}
    ready = None
    context = torch.cuda.stream(stream) if stream is not None else nullcontext()
    with context:
        for key, tensor in _staging_layers.items():
            if offset > tensor.shape[0]:
                raise ValueError("Prefill lengths exceed captured hidden-state rows")
            if offset == 0:
                layers[key] = tensor.new_zeros((len(lengths), *tensor.shape[1:]))
            else:
                positions = torch.tensor(
                    indices, dtype=torch.long, device=tensor.device
                )
                layers[key] = tensor.index_select(0, positions)
        if any(t.is_cuda for t in layers.values()):
            ready = torch.cuda.Event()
            ready.record(stream)
    if to_cpu:
        if ready is not None:
            ready.synchronize()
        layers = {key: tensor.float().cpu() for key, tensor in layers.items()}
        ready = None
    _queue.append(EmbeddingBatch(layers, lengths, ready))
    _staging_layers = {}


def get():
    return _queue.popleft() if _queue else None


def queue_len():
    return len(_queue)
