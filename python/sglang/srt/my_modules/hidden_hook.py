"""Hooks for post-norm hidden-state capture.

PBKV registers embedding_save_hook_factory("last") on model.norm. The hook
stages a GPU tensor; embedding_buffer.commit selects each request's last
prefill token on its producing stream. The scheduler submits these vectors to
the existing GraphSAGE predictor. The legacy hidden_state_hook_factory remains
available for callers of the older tool-prefix interface.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import torch

from sglang.srt.my_modules import hidden_state_buffer as _buf

logger = logging.getLogger(__name__)


def hidden_state_hook_factory(config: Dict[str, Any]):
    """
    Hook factory compatible with ``hook_manager.register_forward_hooks``.

    Returns a ``register_forward_hook``-compatible callable that writes the
    output hidden states of the hooked module into ``hidden_state_buffer``.

    Parameters
    ----------
    config:
        Unused in the current implementation; kept for interface compatibility
        with the hook_manager convention.
    """

    def hook(module: torch.nn.Module, input: Any, output: Any) -> None:
        # Some modules return tuples (e.g. attention layers return (out, kv)).
        hidden = output[0] if isinstance(output, tuple) else output

        if not isinstance(hidden, torch.Tensor):
            logger.debug(
                "[hidden_hook] Unexpected output type %s — skipping", type(hidden)
            )
            return

        # Normalise to 2-D [N, D].  In some configurations the model runner
        # may produce a 3-D [batch, seq, D] tensor.
        if hidden.dim() == 3:
            hidden = hidden.reshape(-1, hidden.shape[-1])
        elif hidden.dim() != 2:
            logger.debug(
                "[hidden_hook] Unexpected hidden shape %s — skipping", hidden.shape
            )
            return

        _buf.put(hidden)  # [num_new_tokens, hidden_dim] on CPU after put()
        logger.debug(
            "[Hook] Captured hidden state: shape=%s dtype=%s device=%s",
            list(hidden.shape),
            hidden.dtype,
            str(hidden.device),
        )

    return hook


def embedding_save_hook_factory(layer_key: str):
    """
    Hook factory for the ``--my-embedding-save`` feature.

    Returns a forward-hook callable that writes hidden states into
    :mod:`embedding_buffer` under the given *layer_key* (for example
    ``"last"`` or ``"layer_17"``).

    The tensor handling logic (tuple unwrap, dim normalisation) is
    identical to :func:`hidden_state_hook_factory`.
    """
    from sglang.srt.my_modules import embedding_buffer as _ebuf

    def hook(module: torch.nn.Module, input: Any, output: Any) -> None:
        hidden = output[0] if isinstance(output, tuple) else output

        if not isinstance(hidden, torch.Tensor):
            return

        if hidden.dim() == 3:
            hidden = hidden.reshape(-1, hidden.shape[-1])
        elif hidden.dim() != 2:
            return

        _ebuf.put(layer_key, hidden)

    return hook
