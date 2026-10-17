"""
Shared in-process buffer for hidden-state transfer from model hook to scheduler.

Design
------
The model forward pass runs inside ``tp_worker`` and fires a ``register_forward_hook``
callback.  The scheduler calls ``process_batch_result_prefill`` right afterward.
Both happen sequentially in the same Python process, so a plain module-level variable
is race-free (no threading involved).

The stored tensor has shape ``[num_new_tokens, hidden_dim]`` — it covers only the
tokens actually computed during EXTEND mode (newly added tokens, excluding the
cached prefix).  The scheduler slices it per-request using ``extend_input_len_per_req``
from the batch result.
"""

from __future__ import annotations

from typing import Optional

import torch

# Module-level storage (one per TP-rank process — intentional)
_buffer: Optional[torch.Tensor] = None


def put(hidden: torch.Tensor) -> None:
    """
    Called by the forward hook during ``run_batch``.

    Stores the full [N, hidden_dim] tensor on CPU to avoid holding extra GPU
    memory.  Overwrites any previously uncollected state (only the most recent
    prefill matters).
    """
    global _buffer
    # Always detach and move to CPU; use float32 for numerical stability.
    _buffer = hidden.detach().float().cpu()


def get_and_clear() -> Optional[torch.Tensor]:
    """
    Called by the scheduler after ``process_batch_result_prefill``.

    Returns the captured tensor and clears the buffer.  Returns ``None`` if
    no hook has fired since the last call (e.g. decode-only forward).
    """
    global _buffer
    h = _buffer
    _buffer = None
    return h
