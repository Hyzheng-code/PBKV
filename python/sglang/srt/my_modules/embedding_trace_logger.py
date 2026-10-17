"""
JSONL trace logger for the ``--my-embedding-save`` feature.

Each completed request is serialised as a single JSON line containing:
- chain_id, step_id, chain_history
- embedding vectors (last layer and 4th-to-last layer) of the last input token
- input and output text

Usage::

    from sglang.srt.my_modules import embedding_trace_logger as etl
    etl.setup("/tmp/embedding_trace.jsonl")
    ...
    etl.log_trace(chain_id="abc", step_id=2, ...)

Reading the trace file::

    import json
    records = [json.loads(line) for line in open("trace.jsonl")]
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Dict, IO, List, Optional

import torch

logger = logging.getLogger(__name__)

_file: Optional[IO[str]] = None


def setup(trace_path: str) -> None:
    """Open the trace file for appending.  Idempotent if already set up."""
    global _file
    if _file is not None:
        return
    os.makedirs(os.path.dirname(trace_path) or ".", exist_ok=True)
    _file = open(trace_path, "a", encoding="utf-8")
    logger.info("[EmbeddingTraceLogger] Trace file opened: %s", trace_path)


def is_enabled() -> bool:
    return _file is not None


def log_trace(
    *,
    chain_id: Optional[str],
    step_id: Optional[int],
    chain_history: Optional[List[str]],
    embeddings_by_layer: Optional[Dict[str, torch.Tensor]] = None,
    hidden_last: Optional[torch.Tensor] = None,
    hidden_4th: Optional[torch.Tensor] = None,
    input_text: str,
    output_text: str,
    debug_info: Optional[dict] = None,
) -> None:
    """
    Write one JSONL record for a completed request.

    Parameters
    ----------
    embeddings_by_layer:
        Mapping ``layer_key -> 1-D tensor [hidden_dim]`` (CPU, float32).
    hidden_last, hidden_4th:
        Optional 1-D tensors of shape ``[hidden_dim]`` (CPU, float32),
        kept for backward compatibility with older analysis scripts.
    debug_info:
        Optional dict of diagnostic fields to include in the record.
    """
    if _file is None:
        return

    if embeddings_by_layer is None:
        embeddings_by_layer = {}

    record = {
        "timestamp": datetime.now().isoformat(timespec="milliseconds"),
        "chain_id": chain_id,
        "step_id": step_id,
        "chain_history": chain_history,
        "embeddings_by_layer": {
            key: emb.tolist() for key, emb in embeddings_by_layer.items()
        },
        "embedding_last_layer": (
            hidden_last.tolist() if hidden_last is not None else None
        ),
        "embedding_4th_to_last_layer": (
            hidden_4th.tolist() if hidden_4th is not None else None
        ),
        "input_text": input_text,
        "output_text": output_text,
    }

    if debug_info:
        record["_debug"] = debug_info

    _file.write(json.dumps(record, ensure_ascii=False))
    _file.write("\n")
    _file.flush()


def close() -> None:
    """Flush and close the trace file."""
    global _file
    if _file is not None:
        _file.flush()
        _file.close()
        _file = None
