"""
KV-Cache event logger — records evictions, prefetches, and cache flushes to a
dedicated log file for offline analysis.

Design goals
------------
* **Zero overhead when disabled** — all public functions are no-ops if
  :func:`setup` has not been called (or was called with an empty path).
* **Works for both servers** — the default SGLang server and the prefetch
  server can both use this module via ``--my-kvcache-log-file``.
* **Non-intrusive** — only log statements are added to the original sglang
  files; no logic is changed.

Log format
----------
Each line looks like::

    2026-03-09 14:00:01 | EVICT    | ▼ EVICT   GPU→CPU   512 tok  need=300 avail_before=1024 | kind=LRU
    2026-03-09 14:00:01 | LOAD-BCK | ▲ LOAD    CPU→GPU   410 tok  elapsed=3.2 ms            |
    2026-03-09 14:00:01 | FLUSH    | !! FLUSH  all KV cache cleared                          |

Usage
-----
Call :func:`setup` once at server startup (inside ``Scheduler.__init__``).
After that, call the ``log_*`` helpers from any cache-management module —
they are safe to import at module load time even before :func:`setup` is
called, because the module falls back to a silent no-op.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Sequence

# ── Private state ─────────────────────────────────────────────────────────────

_LOG_NAME   = "sglang.srt.kvcache_events"
_LOG_FORMAT = "%(asctime)s |%(levelname)s| %(message)s"
_DATE_FORMAT = "%y-%m-%d %H:%M:%S"

_file_handler: Optional[logging.FileHandler] = None
_debug_enabled: bool = False
_kv_logger:    Optional[logging.Logger]       = None

# Continuation-line indent for "★ REQUEST" multi-line log entries.
# Must match the log-format prefix length so the second line aligns
# with the start of the message on the first line.
# Prefix = 17 chars (%y date+time) + 8 chars (" |INFO| ") = 25.
_REQUEST_CONT_INDENT = " " * 25


def _fmt_rid(rid: str) -> str:
    """Return 'xxx...yyy' for long IDs (>9 chars), or the full ID otherwise."""
    if len(rid) > 9:
        return f"{rid[:3]}...{rid[-3:]}"
    return rid

# Cumulative eviction counters (reset when the server (re)starts).
_evict_count: int = 0
_evict_total_tokens: int = 0

# Cumulative chain-priority eviction counters.
_chain_evict_count: int = 0
_chain_evict_tokens: int = 0

# Cumulative regular (tier-1) eviction counters.
_regular_evict_count: int = 0
_regular_evict_tokens: int = 0

# Cumulative CPU-drop (host eviction) counters.
_cpu_drop_count: int = 0
_cpu_drop_tokens: int = 0

# Cumulative request hit-rate counters (for historical average).
_req_count: int = 0
_total_input_tokens: int = 0
_total_device_tokens: int = 0

# Cumulative prefill time counters (for historical average prefill latency).
_prefill_req_count: int = 0
_total_prefill_ms: float = 0.0

# Cumulative load_back time counters (CPU→GPU transfer).
_load_back_count: int = 0
_total_load_back_ms: float = 0.0


# ── Setup ─────────────────────────────────────────────────────────────────────

def setup(log_file: str, tp_rank: int = 0, debug: bool = False) -> None:
    """
    Attach a ``FileHandler`` to the KV-cache event logger.

    Parameters
    ----------
    log_file:
        Absolute or relative path to the target log file.  The parent
        directory is created automatically.  If empty, this is a no-op.
    tp_rank:
        Only rank-0 writes the file (other ranks skip to avoid duplicates).
    debug:
        If True, enable DEBUG-level phase markers (``log_debug_phase``).
        Otherwise those calls are silently dropped.
    """
    global _file_handler, _kv_logger, _debug_enabled
    _debug_enabled = debug

    if not log_file:
        return
    if tp_rank != 0:
        return
    if _file_handler is not None:
        return  # idempotent

    parent = os.path.dirname(os.path.abspath(log_file))
    os.makedirs(parent, exist_ok=True)

    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))

    lg = logging.getLogger(_LOG_NAME)
    lg.setLevel(logging.DEBUG)
    lg.addHandler(fh)
    # Prevent log records from propagating to the root logger (console).
    lg.propagate = False

    _file_handler = fh
    _kv_logger    = lg

    _kv_logger.info(
        "[KVCache-Log] ── File logging enabled → %s ──",
        os.path.abspath(log_file),
    )


def is_enabled() -> bool:
    """Return ``True`` if the logger has been configured."""
    return _kv_logger is not None


def log_debug_phase(phase: str, extra: str = "") -> None:
    """Lightweight scheduler-loop phase marker for detailed diagnosis.

    Only emits output when ``--my-debug-log`` is enabled.
    """
    if _kv_logger is None or not _debug_enabled:
        return
    msg = f"🔧 DEBUG  phase={phase}"
    if extra:
        msg += f"  {extra}"
    _kv_logger.info(msg)


def log_forward_begin(fwd_ct: int, mode, batch_size: int, extend_tokens: int, extra: str = "") -> None:
    """Always-on marker before run_batch — helps locate watchdog-timeout hangs."""
    if _kv_logger is None:
        return
    msg = f"⏩ FORWARD_BEGIN  fwd#%d  mode=%s  bs=%d  extend_tokens=%d"
    args = [fwd_ct, mode, batch_size, extend_tokens]
    if extra:
        msg += "  %s"
        args.append(extra)
    _kv_logger.info(msg, *args)


def log_forward_end(fwd_ct: int, elapsed_ms: float) -> None:
    """Always-on marker after run_batch returns."""
    if _kv_logger is None:
        return
    _kv_logger.info(
        "⏪ FORWARD_END    fwd#%d  elapsed=%.1fms",
        fwd_ct, elapsed_ms,
    )


# ── Log helpers ───────────────────────────────────────────────────────────────

def log_evict_header(
    num_tokens_requested: int,
    gpu_avail_before: int,
    gpu_max: int,
    cpu_avail_before: Optional[int] = None,
    cpu_max: Optional[int] = None,
    gpu_avail_swa_before: Optional[int] = None,
    gpu_max_swa: Optional[int] = None,
    kind: str = "LRU",
    evict_type: str = "",
) -> None:
    """
    Log one GPU-eviction header line BEFORE the eviction executes.

    This is the outer "parent" log entry; per-node detail logs
    (log_chain_evict_node / log_regular_evict_node) follow immediately
    after, indented by 4 spaces to show they belong to this event.

    Parameters
    ----------
    num_tokens_requested:
        Number of token slots the allocator was asked to free.
    gpu_avail_before:
        Free GPU KV-cache slots *before* the eviction call.
    gpu_max:
        Total GPU KV-cache capacity (slots).
    cpu_avail_before:
        Free CPU KV-cache slots *before* eviction (HiRadix only).
    cpu_max:
        Total CPU KV-cache capacity (HiRadix only).
    gpu_avail_swa_before / gpu_max_swa:
        Same but for the sliding-window attention layer (SWA hybrid cache only).
    kind:
        Eviction policy / reason label.
    evict_type:
        Direction label shown in brackets after "▼ EVICT", e.g. "GPU→CPU"
        (HiCache write_back), "GPU-drop" (direct eviction / already backuped).
    """
    global _evict_count, _evict_total_tokens
    if _kv_logger is None:
        return

    _evict_count += 1

    # GPU capacity string (pre-eviction free space)
    gpu_pct = 100.0 * gpu_avail_before / gpu_max if gpu_max > 0 else 0.0
    gpu_str = f"{gpu_avail_before}/{gpu_max}({gpu_pct:.1f}%)"
    if gpu_avail_swa_before is not None and gpu_max_swa is not None:
        swa_pct = 100.0 * gpu_avail_swa_before / gpu_max_swa if gpu_max_swa > 0 else 0.0
        gpu_str += f"  SWA:{gpu_avail_swa_before}/{gpu_max_swa}({swa_pct:.1f}%)"

    # CPU capacity string
    if cpu_avail_before is not None and cpu_max is not None:
        cpu_pct = 100.0 * cpu_avail_before / cpu_max if cpu_max > 0 else 0.0
        cpu_str = f"{cpu_avail_before}/{cpu_max}({cpu_pct:.1f}%)"
    else:
        cpu_str = "N/A"

    type_tag = f"({evict_type})" if evict_type else ""
    _kv_logger.info(
        "▼ EVICT%s #%d  need=%d tok"
        "  GPU-free:%s  CPU-free:%s"
        "  policy=%s",
        type_tag, _evict_count, num_tokens_requested,
        gpu_str, cpu_str,
        kind,
    )


def log_evict(
    num_tokens_requested: int,
    available_before: int,
    kind: str = "LRU",
) -> None:
    """
    Log a GPU→CPU eviction event.

    Parameters
    ----------
    num_tokens_requested:
        Number of token slots that the allocator was asked to free.
    available_before:
        Free slots in the allocator *before* the eviction call.
    kind:
        Eviction policy / reason label (e.g. ``"LRU"``, ``"SWA"``).
    """
    if _kv_logger is None:
        return
    need = num_tokens_requested
    _kv_logger.info(
        "▼ EVICT   GPU→CPU  %6d tok  |  need=%d  avail_before=%d  policy=%s",
        need, need, available_before, kind,
    )


def log_load_back(
    num_tokens: int,
    elapsed_ms: float,
    success: bool = True,
    reason: str = "",
) -> None:
    """
    Log a CPU→GPU load-back (prefetch) event.

    Parameters
    ----------
    num_tokens:
        Number of token slots being moved from CPU to GPU.
    elapsed_ms:
        Wall-clock time for the load operation in milliseconds.
    success:
        ``False`` if the load was skipped or failed.
    reason:
        Short reason string for a skipped/failed load (e.g. ``"quota exceeded"``).
    """
    if _kv_logger is None:
        return
    if success:
        _kv_logger.info(
            "▲ LOAD    CPU→GPU  %6d tok  |  elapsed=%.1f ms",
            num_tokens, elapsed_ms,
        )
    else:
        _kv_logger.info(
            "▲ LOAD    CPU→GPU  %6d tok  |  SKIPPED  reason=%s",
            num_tokens, reason or "unknown",
        )


def log_flush(success: bool = True, reason: str = "") -> None:
    """
    Log a full KV-cache flush (``/flush_cache`` endpoint).

    Parameters
    ----------
    success:
        ``False`` if flush was rejected due to active requests.
    reason:
        Details for a rejected flush.
    """
    if _kv_logger is None:
        return
    if success:
        _kv_logger.info(
            "!! FLUSH  ──── all KV cache cleared ────────────────────────────────",
        )
    else:
        _kv_logger.info(
            "!! FLUSH  ──── REJECTED  reason=%s", reason or "pending requests",
        )


def log_predict(
    session_id: str,
    feat_source: str,
    predictions: list,
    history: list,
) -> None:
    """
    Log a prediction event from PrefetchController.on_new_request().

    Parameters
    ----------
    session_id:
        Current session / request ID (first 8 chars are shown).
    feat_source:
        Human-readable description of where the text feature came from,
        e.g. ``"cold(285tok)"``, ``"cached-prompt(warm,1tok)"``.
    predictions:
        List of ``(tool_name, probability)`` pairs returned by the predictor.
    history:
        Session tool-call history at the time of prediction.
    """
    if _kv_logger is None:
        return
    top_str = ", ".join(f"{n}:{p:.3f}" for n, p in predictions[:3])
    _kv_logger.info(
        "◆ PREDICT  session=%.8s  feat=%-30s  top3=[%s]  history=%s",
        session_id, feat_source, top_str, history if history else "[]",
    )


def log_request_metrics(
    rid: str,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int,
    prefill_ms: Optional[float],
    decode_ms: Optional[float],
    total_forward_ms: Optional[float],
    hit_rate_pct: float,
    cached_device: int = 0,
    cached_host: int = 0,
    cached_storage: int = 0,
    pool_used: int = 0,
    pool_total: int = 0,
    running_reqs: int = 0,
    waiting_reqs: int = 0,
    prefill_breakdown: Optional[dict] = None,
    load_back_ms: Optional[float] = None,
    prefetch_total: int = 0,
    prefetch_hit: int = 0,
    prefetch_miss: int = 0,
    prefetch_chain_hit: int = 0,
) -> None:
    """
    Log per-request inference metrics when a request finishes.

    Parameters
    ----------
    rid:
        Request ID (first 16 chars are shown).
    input_tokens:
        Number of prompt tokens (``len(origin_input_ids)``).
    output_tokens:
        Number of generated tokens.
    cached_tokens:
        Total number of tokens served from KV-cache (prefix cache hit).
    prefill_ms:
        Prefill computation time in milliseconds, or ``None`` if unavailable.
    decode_ms:
        Decode computation time in milliseconds, or ``None`` if unavailable.
    total_forward_ms:
        Total forward time (prefill + decode) in milliseconds.
    hit_rate_pct:
        GPU cache hit rate as a percentage (``cached_device / input_tokens * 100``).
    cached_device:
        Cached tokens from GPU device (HiCache L1).
    cached_host:
        Cached tokens from CPU host (HiCache L2).
    cached_storage:
        Cached tokens from storage backend (HiCache L3).
    pool_used:
        Number of GPU KV-cache token slots currently in use.
    pool_total:
        Total GPU KV-cache token slots (capacity).
    running_reqs:
        Number of requests currently in the running batch (including the one
        just finished, before the scheduler removes it).
    waiting_reqs:
        Number of requests currently in the waiting queue.
    prefill_breakdown:
        Optional dict of fine-grained prefill sub-phase timings in ms.
        Keys: ``alloc``, ``init_batch``, ``model_forward``, ``sample``.
    load_back_ms:
        Time spent loading KV-cache from CPU host back to GPU (HiCache
        load_back) in milliseconds, or ``None`` if no load_back occurred.
    prefetch_total / prefetch_hit / prefetch_miss:
        Node-level prefetch tracking counters from tree cache.
    prefetch_chain_hit:
        Number of unique ``(prefetched node, chain_id)`` hit events.
    """
    global _req_count, _total_input_tokens, _total_device_tokens
    global _prefill_req_count, _total_prefill_ms
    global _load_back_count, _total_load_back_ms
    if _kv_logger is None:
        return

    # Update cumulative counters for historical average hit rate.
    _req_count += 1
    _total_input_tokens += input_tokens
    _total_device_tokens += cached_device
    avg_hit_rate_pct = (
        100.0 * _total_device_tokens / _total_input_tokens
        if _total_input_tokens > 0
        else 0.0
    )

    # Update cumulative prefill time counters for historical average.
    if prefill_ms is not None:
        _prefill_req_count += 1
        _total_prefill_ms += prefill_ms
    avg_prefill_ms = (
        _total_prefill_ms / _prefill_req_count
        if _prefill_req_count > 0 else None
    )

    # Update cumulative load_back time counters.
    if load_back_ms is not None and load_back_ms > 0:
        _load_back_count += 1
        _total_load_back_ms += load_back_ms
    avg_load_back_ms = (
        _total_load_back_ms / _load_back_count
        if _load_back_count > 0 else None
    )

    # Compute prefill_total = load_back + prefill (GPU compute).
    prefill_total_ms = None
    if prefill_ms is not None:
        prefill_total_ms = (load_back_ms or 0.0) + prefill_ms

    prefill_str = f"{prefill_ms:.1f}ms"       if prefill_ms       is not None else "N/A"
    decode_str  = f"{decode_ms:.1f}ms"        if decode_ms        is not None else "N/A"
    total_str   = f"{total_forward_ms:.1f}ms" if total_forward_ms is not None else "N/A"
    avg_prefill_str = f"{avg_prefill_ms:.1f}ms" if avg_prefill_ms is not None else "N/A"
    load_back_str = f"{load_back_ms:.1f}ms" if (load_back_ms is not None and load_back_ms > 0) else "0ms"
    avg_load_back_str = f"{avg_load_back_ms:.1f}ms" if avg_load_back_ms is not None else "0ms"
    prefill_total_str = f"{prefill_total_ms:.1f}ms" if prefill_total_ms is not None else "N/A"

    cache_detail = f"dev={cached_device} host={cached_host}"
    if cached_storage:
        cache_detail += f" stor={cached_storage}"

    # KV-cache pool usage: "22.2k/88.8k(25%)"
    if pool_total > 0:
        pool_pct = 100.0 * pool_used / pool_total
        pool_str = f"{pool_used/1000:.1f}k/{pool_total/1000:.1f}k({pool_pct:.0f}%)"
    else:
        pool_str = "N/A"

    # Build prefill breakdown line (indented)
    breakdown_line = ""
    if prefill_breakdown:
        parts = []
        # Display order: load_back → alloc → init_batch → model_forward → sample
        if load_back_ms is not None and load_back_ms > 0:
            parts.append(f"load_back={load_back_ms:.1f}ms")
        for key in ("alloc", "init_batch", "model_forward", "sample"):
            val = prefill_breakdown.get(key)
            if val is not None:
                parts.append(f"{key}={val:.1f}ms")
        if parts:
            breakdown_line = (
                f"\n{_REQUEST_CONT_INDENT}prefill_breakdown: {' | '.join(parts)}"
            )

    # Build prefetch metrics lines (only when prefetch tracking is active)
    prefetch_line = ""
    if prefetch_total > 0:
        pf_hit_rate = 100.0 * prefetch_hit / prefetch_total
        prefetch_line = (
            f"\n{_REQUEST_CONT_INDENT}prefetch: hit_rate={pf_hit_rate:.2f}%"
            f"(hit/total={prefetch_hit}/{prefetch_total} miss={prefetch_miss}) | unique_hit_rate={100.0 * prefetch_chain_hit / prefetch_total:.2f}%"
        )

    # Line 1: request ID + timing summary.
    # Line 2: cache-hit breakdown + pool usage + historical avg + queue state.
    # Line 3: prefill_total (load_back + GPU compute) + averages.
    # Line 4 (optional): prefill sub-phase breakdown.
    # Line 5 (optional): prefetch hit rate.
    _kv_logger.info(
        "★ REQUEST (%s) finish:   in=%d   out=%d   cached_total=%d   gpu_hit=%.1f%%\n"
        "%sprefill=%s   decode=%s   total=%s\n"
        "%s[%s]   pool=%s   avg_gpu_hit=%.2f%%(%d reqs)\n"
        "%sprefill_total=%s(load_back=%s + compute=%s)"
        "   avg_prefill=%s(%d reqs)   avg_load_back=%s(%d reqs)"
        "   this_req: cache/input=%d/%d   running=%d waiting=%d%s%s",
        _fmt_rid(rid),
        input_tokens,
        output_tokens,
        cached_tokens,
        hit_rate_pct,
        _REQUEST_CONT_INDENT,
        prefill_str,
        decode_str,
        total_str,
        _REQUEST_CONT_INDENT,
        cache_detail,
        pool_str,
        avg_hit_rate_pct,
        _req_count,
        _REQUEST_CONT_INDENT,
        prefill_total_str,
        load_back_str,
        prefill_str,
        avg_prefill_str,
        _prefill_req_count,
        avg_load_back_str,
        _load_back_count,
        cached_tokens,
        input_tokens,
        running_reqs,
        waiting_reqs,
        breakdown_line,
        prefetch_line,
    )


def log_prefetch_controller(
    tool_name: str,
    num_tokens: int,
    triggered: bool,
    prob: float,
    reason: str = "",
) -> None:
    """
    Log a prefetch decision from the PrefetchController (for the prefetch server).

    Parameters
    ----------
    tool_name:
        Name of the predicted agent/tool.
    num_tokens:
        Token count of the agent's system-prompt KV prefix.
    triggered:
        ``True`` if ``load_back`` was called.
    prob:
        Predictor probability for this candidate.
    reason:
        Skip reason if ``triggered`` is ``False``.
    """
    if _kv_logger is None:
        return
    if triggered:
        _kv_logger.info(
            "▶ PREFETCH  %-32s  p=%.3f  %d tok  →  TRIGGERED",
            f"'{tool_name}'", prob, num_tokens,
        )
    else:
        _kv_logger.info(
            "▷ PREFETCH  %-32s  p=%.3f  %d tok  →  SKIP  %s",
            f"'{tool_name}'", prob, num_tokens, reason,
        )


def log_chain_info(
    rid: str,
    chain_id: str,
    step: int,
    history: list,
    current_agent: Optional[str] = None,
) -> None:
    """
    Log one Agent call-chain step when a request arrives at the scheduler.

    Parameters
    ----------
    rid:
        Request ID (first 16 chars shown).
    chain_id:
        UUID shared by all requests belonging to the same call chain.
    step:
        0-based index of the current step (= len(history) before this call).
    history:
        Ordered list of tool/agent names already completed in this chain
        (does NOT include the current agent being called).
    current_agent:
        Name of the agent being invoked in this request.  When provided the
        displayed sequence includes this agent so the log reflects the chain
        state *after* this call is dispatched.
    """
    if _kv_logger is None:
        return
    display_history = list(history) + [current_agent] if current_agent else history
    _kv_logger.info(
        "⛓ CHAIN (%s) update for new Req (%s): step=%d  history=%s",
        _fmt_rid(chain_id), _fmt_rid(rid), step, display_history if display_history else "[]",
    )


def log_chain_evict_node(num_tokens: int, chain_ids: set) -> None:
    """Log one chain-exclusive KV-cache node eviction (tier-0 priority).

    Called from RadixCache.evict() / HiRadixCache.evict() whenever a node
    whose visited_chain_ids ⊆ terminated_chain_ids is popped from the
    eviction heap with tier=0, i.e. it was preferentially evicted because
    all chains that used it have already terminated.

    Indented by 4 spaces to appear as a child entry under the ▼ EVICT header
    logged by log_evict_header() just before the eviction loop started.

    Parameters
    ----------
    num_tokens:
        Number of KV token slots freed by evicting this node.
    chain_ids:
        The node's visited_chain_ids — the terminated chains that used it.
    """
    global _chain_evict_count, _chain_evict_tokens
    if _kv_logger is None:
        return
    _chain_evict_count += 1
    _chain_evict_tokens += num_tokens
    _chain_evict_tokens_k_str = f"{_chain_evict_tokens/1000:.1f}k"
    # Show up to 5 full chain IDs.
    ids_list = list(chain_ids)
    ids_preview = ", ".join(ids_list[:5])
    if len(ids_list) > 5:
        ids_preview += f" +{len(ids_list) - 5}more"
    _kv_logger.info(
        "    🗑 CHAIN_EVICT  tok=%d"
        "  (cum: #%d  total=%s tok)"
        "  chains=[%s]",
        num_tokens,
        _chain_evict_count, _chain_evict_tokens_k_str,
        ids_preview,
    )


def log_regular_evict_node(num_tokens: int) -> None:
    """Log one ordinary (tier-1) KV-cache node eviction.

    Called from RadixCache.evict() / HiRadixCache.evict() for nodes that
    are evicted by the normal strategy (LRU etc.) rather than because they
    are chain-exclusive dead nodes.  A tier-1 node may have no chain
    tracking at all, or may still have chains that are active/predicted
    (i.e. not all visited chains have terminated).

    Indented by 4 spaces to appear as a child entry under the ▼ EVICT header.

    Parameters
    ----------
    num_tokens:
        Number of KV token slots freed by evicting this node.
    """
    global _regular_evict_count, _regular_evict_tokens
    if _kv_logger is None:
        return
    _regular_evict_count += 1
    _regular_evict_tokens += num_tokens
    _regular_evict_tokens_k_str = f"{_regular_evict_tokens/1000:.1f}k"
    _kv_logger.info(
        "    ▽ node  tok=%d  (cum: #%d  total=%s tok)",
        num_tokens,
        _regular_evict_count, _regular_evict_tokens_k_str,
    )


def log_cpu_drop_node(num_tokens: int) -> None:
    """Log one CPU host-cache node eviction (CPU-drop).

    Called from HiRadixCache.evict_host() for nodes evicted from the host
    memory pool when CPU capacity is exhausted.

    Parameters
    ----------
    num_tokens:
        Number of host token slots freed by evicting this node.
    """
    global _cpu_drop_count, _cpu_drop_tokens
    if _kv_logger is None:
        return
    _cpu_drop_count += 1
    _cpu_drop_tokens += num_tokens
    _cpu_drop_tokens_k_str = f"{_cpu_drop_tokens/1000:.1f}k"
    _kv_logger.info(
        "    ▽ cpu_node  tok=%d  (cum: #%d  total=%s tok)",
        num_tokens,
        _cpu_drop_count, _cpu_drop_tokens_k_str,
    )


def log_pred_evict_node(
    num_tokens: int,
    value_score: float,
    node_agents: set,
) -> None:
    """Log one prediction-scored KV-cache node eviction.

    Called from HiRadixCache.evict() when prediction-based eviction is
    enabled.  Shows the Value(v) score and associated agents.

    Parameters
    ----------
    num_tokens:
        Number of KV token slots freed by evicting this node.
    value_score:
        The computed Value(v) for this node at eviction time.
    node_agents:
        Set of agent names associated with this node.
    """
    global _regular_evict_count, _regular_evict_tokens
    if _kv_logger is None:
        return
    _regular_evict_count += 1
    _regular_evict_tokens += num_tokens
    _regular_evict_tokens_k_str = f"{_regular_evict_tokens/1000:.1f}k"
    agents_str = ",".join(sorted(node_agents)) if node_agents else "none"
    _kv_logger.info(
        "    📊 PRED_EVICT  tok=%d  value=%.4f  agents=[%s]"
        "  (cum: #%d  total=%s tok)",
        num_tokens, value_score, agents_str,
        _regular_evict_count, _regular_evict_tokens_k_str,
    )


def log_chain_terminated(
    chain_id: str,
    dead_tokens: int = 0,
    dead_nodes: int = 0,
    total_gpu_tokens: int = 0,
    history: Optional[Sequence[str]] = None,
) -> None:
    """Log the termination of an agent call-chain and the resulting dead KV volume.

    Called when the scheduler receives a MyChainTerminatedReqInput message
    (i.e. the client sent POST /my/chain_terminated).

    Parameters
    ----------
    chain_id:
        UUID of the terminated call-chain.
    dead_tokens:
        Total GPU KV-cache token slots now held exclusively by fully dead
        subtrees (all chains in those subtrees are terminated).  This is the
        upper-bound of what can be freed by the next chain-priority eviction.
    dead_nodes:
        Number of tree nodes that make up those dead subtrees.
    total_gpu_tokens:
        Total GPU KV-cache capacity (token slots), used to compute the
        dead-ratio percentage.  Pass 0 to omit the percentage.
    history:
        Ordered list of tool names called in this chain (most-recent last).
        Logged before the "marked for priority eviction" note.
    """
    if _kv_logger is None:
        return
    hist_str = " → ".join(history) if history else "(none)"
    if dead_tokens > 0:
        pct_str = ""
        dead_tokens_k_str = f"{dead_tokens/1000:.1f}k"
        if total_gpu_tokens > 0:
            pct = 100.0 * dead_tokens / total_gpu_tokens
            pct_str = f"  ({pct:.1f}% of GPU pool)"
        _kv_logger.info(
            "✅ CHAIN_DONE  dead_kv=%s tok & %d nodes%s"
            "  chain_id=%s"
            "  final_history=[%s]"
            "  → marked for priority eviction",
            dead_tokens_k_str, dead_nodes,
            pct_str,
            _fmt_rid(chain_id),
            hist_str,
        )
    else:
        _kv_logger.info(
            "✅ CHAIN_DONE  chain_id=%.8s  final_history=[%s]  dead_kv=0 tok"
            "  → KV nodes marked for priority eviction",
            chain_id,
            hist_str,
        )


def log_chain_predict(
    chain_id: str,
    current_agent: str,
    predicted_agent: str,
    confidence: float,
    all_probs: list,
    history: Optional[Sequence[str]] = None,
    step_id: Optional[int] = None,
) -> None:
    """Log a chain-aware next-agent prediction.

    Parameters
    ----------
    chain_id:
        UUID of the call-chain.
    current_agent:
        Name of the agent that just finished.
    predicted_agent:
        The model's top-1 prediction for the next agent.
    confidence:
        Softmax probability of the top-1 prediction.
    all_probs:
        List of ``(agent_name, probability)`` sorted descending.
    history:
        Ordered list of agents called so far in this chain.
    step_id:
        Step index within the chain.
    """
    if _kv_logger is None:
        return
    hist_str = " → ".join(history) if history else "(none)"
    probs_str = ", ".join(f"{name}:{prob:.3f}" for name, prob in all_probs)
    _kv_logger.info(
        "🔮 CHAIN_PRED  chain_id=%s  step=%s  current=%s"
        "  → predicted=%s (%.1f%%)"
        "  probs=[%s]"
        "  history=[%s]",
        _fmt_rid(chain_id),
        step_id if step_id is not None else "?",
        current_agent,
        predicted_agent,
        confidence * 100,
        probs_str,
        hist_str,
    )


def log_batch_prefetch_skip(reason: str) -> None:
    """Log that batch prefetch was skipped (含prefill batch or other reason)."""
    if _kv_logger is None:
        return
    _kv_logger.info("▷ BATCH_PREFETCH  skip  reason=%s", reason)


def log_batch_prefetch_attempt(stats: dict) -> None:
    """Log the actual capacity/transfer budgets and Appendix G active cap."""
    if _kv_logger is None:
        return
    _kv_logger.info(
        "%s BATCH_PREFETCH  loaded=%d nodes (%d tok)  evicted=%d tok"
        "  active_evicted=%d active_budget=%d"
        "  S_space=%d S_bw=%d S_min=%d candidates=%d"
        "  bandwidth=%.0f B/s decode_step=%.6f s elapsed=%.2f ms reason=%s",
        "▷" if stats.get("skipped") else "▶",
        stats.get("loaded_count", 0),
        stats.get("loaded_tokens", 0),
        stats.get("evicted_for_prefetch", 0),
        stats.get("active_evicted_tokens", 0),
        stats.get("active_budget", 0),
        stats.get("s_space", 0),
        stats.get("s_bw", 0),
        stats.get("s_min", 0),
        stats.get("candidates", 0),
        stats.get("bandwidth_bytes_per_sec", 0),
        stats.get("decode_step_seconds", 0),
        stats.get("elapsed_ms", 0),
        stats.get("reason", ""),
    )
