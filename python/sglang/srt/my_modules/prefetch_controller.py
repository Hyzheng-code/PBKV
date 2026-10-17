"""
Prefetch Controller — ties together the Tool Predictor and the KV-cache.

Responsibilities
----------------
1. **Session history management**: tracks which tools have been called in each
   session so that the predictor can use the call history as a feature.

2. **Tool token registration**: tools must be registered with their token-ID
   prefixes so that the controller can locate the corresponding Radix-Tree
   nodes and trigger CPU→GPU prefetching.  Registration can be done either
   manually via :meth:`register_tool_tokens` or automatically from a JSON
   file via :meth:`load_tool_defs`.

3. **Prediction + prefetch trigger**: called for every completed prefill batch:
   a) Runs the :class:`ToolPredictorManager` with the LLM hidden state (from
      ``hidden_state_buffer``) + session history.
   b) For each candidate, searches the Radix Tree for the tool's prefix.
   c) If the node is resident in CPU memory (backuped), calls
      ``tree_cache.load_back()`` to prefetch it to GPU.

Integration with the Scheduler
-------------------------------
The scheduler calls::

    controller.on_new_request(session_id, tree_cache, text_feat)
        # called inside _my_post_prefill_predict(), after each prefill batch

    controller.on_request_complete(session_id, tool_name)
        # called by the business layer when a tool call is confirmed

Both calls are guarded by ``server_args.my_prefetch_enable`` in the scheduler
so the default code path is completely untouched.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# Warm-start detection threshold: if a prefill batch has fewer than this many
# new tokens, the hidden state is dominated by structural boundary tokens
# (e.g. "<|im_start|>user\n") rather than the agent's system-prompt semantics.
# In that case we reuse the cached hidden state from the most recent cold-start
# call for the same session.
_MIN_COLD_TOKENS = 20


class PrefetchController:
    """
    Orchestrates tool prediction and KV-cache prefetching.

    Parameters
    ----------
    predictor:
        A loaded :class:`~sglang.srt.my_modules.tool_predictor.ToolPredictorManager`.
    top_k:
        How many top-predicted tools to attempt to prefetch.
    max_history_per_session:
        Maximum number of past tool calls to keep in memory per session
        (older entries are dropped).
    """

    def __init__(
        self,
        predictor: Any = None,  # ToolPredictorManager (optional; None when using external chain predictor)
        top_k: int = 3,
        max_history_per_session: int = 32,
        min_prob: float = 0.9,
    ):
        self.predictor = predictor
        self.top_k = top_k
        self.max_history = max_history_per_session
        self.min_prob = min_prob

        # session_id → ordered list of tool names called so far
        self._session_history: Dict[str, List[str]] = defaultdict(list)

        # prompt_key (tuple of first-N token IDs) → cached text feature from
        # the most recent cold-start prefill for that system-prompt prefix.
        # Using a prompt-derived key rather than session_id means that the
        # stored representation persists across different API calls that share
        # the same agent system prompt, even when the client generates a fresh
        # session_id for each request.
        self._prompt_hidden: Dict[tuple, "torch.Tensor"] = {}

        # Legacy session-keyed cache (kept as a secondary fallback for
        # multi-turn conversations that DO reuse session_id).
        self._session_hidden: Dict[str, "torch.Tensor"] = {}

        # tool_name → list of token IDs that form the tool's KV-cache prefix
        self._tool_tokens: Dict[str, List[int]] = {}

        logger.info(
            "[PrefetchController] Initialized | top_k=%d | max_history=%d | min_prob=%.2f",
            top_k,
            max_history_per_session,
            min_prob,
        )

    # ── Tool registration ─────────────────────────────────────────────────────

    def register_tool_tokens(self, tool_name: str, token_ids: List[int]) -> None:
        """
        Associate *tool_name* with the token-ID sequence that forms its
        KV-cache prefix in the Radix Tree.

        Parameters
        ----------
        tool_name:
            Name as it appears in the tool vocabulary (e.g. ``"Image Classification"``).
        token_ids:
            Full token-ID list representing the tool's definition text as it
            appears in the LLM prompt (must match exactly).
        """
        ids = list(token_ids)
        self._tool_tokens[tool_name] = ids
        # Show first-5 and last-5 token IDs to verify the chat template is applied.
        head = ids[:5]
        tail = ids[-5:] if len(ids) > 5 else []
        logger.info(
            "[Register] '%s': %d tokens | first5=%s last5=%s",
            tool_name,
            len(ids),
            head,
            tail,
        )

    def load_tool_defs(self, defs_path: str, tokenizer: Any) -> None:
        """
        Load tool definitions from a JSON file, tokenize each definition
        **with chat-template wrapping**, and register the resulting token
        sequences.

        This is called automatically at startup when ``--my-tool-defs`` is set.

        JSON format
        -----------
        ``{"Agent Name": "system prompt text", ...}``

        The values are the **raw system-prompt texts** as sent by the client
        (i.e. the ``content`` of the ``{"role":"system"}`` message).  This
        method applies the tokenizer's chat template to produce the exact token
        sequence that sglang caches in the Radix Tree:

        .. code-block::

            <|im_start|>system\\n{text}<|im_end|>\\n

        This ensures that the registered token IDs form a true prefix of every
        request that uses this agent, so ``match_prefix()`` returns a cache hit.

        Parameters
        ----------
        defs_path:
            Path to the JSON file.
        tokenizer:
            A HuggingFace ``PreTrainedTokenizer`` with ``apply_chat_template``
            and ``encode`` methods.
        """
        try:
            with open(defs_path, encoding="utf-8") as f:
                tool_defs: Dict[str, str] = json.load(f)
        except Exception as exc:
            logger.error(
                "[PrefetchController] Failed to load tool defs from '%s': %s",
                defs_path,
                exc,
            )
            return

        # Check whether the tokenizer supports apply_chat_template.
        has_chat_template = (
            hasattr(tokenizer, "apply_chat_template")
            and getattr(tokenizer, "chat_template", None) is not None
        )
        if not has_chat_template:
            logger.warning(
                "[PrefetchController] Tokenizer has no chat_template; "
                "falling back to raw-text tokenization. "
                "Token IDs may not match the cached prefix."
            )

        registered = 0
        skipped = 0
        for tool_name, definition_text in tool_defs.items():
            # When using an external predictor (self.predictor is None),
            # accept all tools from the defs file.
            if self.predictor is not None and not self.predictor.is_known_tool(tool_name):
                logger.warning(
                    "[PrefetchController] Tool '%s' from defs file is not in predictor "
                    "vocabulary — skipping token registration (prediction still works)",
                    tool_name,
                )
                skipped += 1
                continue
            try:
                if has_chat_template:
                    # Format as a system-only chat message so the token IDs
                    # match the prefix that sglang stores in the Radix Tree.
                    # apply_chat_template handles model-specific special tokens
                    # (e.g. Llama-3, Qwen, Mistral, etc.) automatically.
                    messages_sys = [{"role": "system", "content": definition_text}]
                    formatted = tokenizer.apply_chat_template(
                        messages_sys,
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                    token_ids = tokenizer.encode(formatted, add_special_tokens=False)
                else:
                    token_ids = tokenizer.encode(
                        definition_text, add_special_tokens=False
                    )
                self.register_tool_tokens(tool_name, token_ids)
                registered += 1
            except Exception as exc:
                logger.warning(
                    "[PrefetchController] Failed to tokenize tool '%s': %s",
                    tool_name,
                    exc,
                )
                skipped += 1

        logger.info(
            "[PrefetchController] Loaded tool defs from '%s': "
            "%d registered (chat_template=%s), %d skipped",
            defs_path,
            registered,
            has_chat_template,
            skipped,
        )

    # ── Session history ───────────────────────────────────────────────────────

    def get_history(self, session_id: str) -> List[str]:
        """Return the tool-call history for *session_id* (may be empty)."""
        return list(self._session_history[session_id])

    def on_request_complete(
        self,
        session_id: str,
        tool_name: Optional[str],
    ) -> None:
        """
        Record that *tool_name* was called in *session_id*.

        Call this after a request finishes and the called tool is known.
        Pass ``None`` if the request did not result in a tool call (no-op).
        """
        if tool_name is None:
            return
        hist = self._session_history[session_id]
        hist.append(tool_name)
        if len(hist) > self.max_history:
            self._session_history[session_id] = hist[-self.max_history :]

    def clear_session(self, session_id: str) -> None:
        """Remove history and session-keyed hidden state for a session."""
        self._session_history.pop(session_id, None)
        self._session_hidden.pop(session_id, None)
        # Note: _prompt_hidden is intentionally NOT cleared per session because
        # the prompt key is shared across sessions for the same agent.

    # ── Core: predict + prefetch ──────────────────────────────────────────────

    def on_new_request(
        self,
        session_id: str,
        tree_cache: Any,
        text_feat: Optional[torch.Tensor] = None,
        n_new_tokens: int = 0,
        prompt_key: Optional[tuple] = None,
        chain_id: Optional[str] = None,
        history: Optional[List[str]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Run the tool predictor and (if possible) prefetch the predicted tools'
        KV caches from CPU to GPU.

        Called by the scheduler inside ``_my_post_prefill_predict()`` after
        each EXTEND (prefill) batch forward.

        Parameters
        ----------
        session_id:
            Identifier for the current session / conversation.
        tree_cache:
            The scheduler's ``tree_cache`` instance.  CPU→GPU prefetching is
            only attempted when it is a ``HiRadixCache`` (has ``load_back``).
        text_feat:
            1-D float tensor of shape ``(d_text,)`` mean-pooled from the LLM's
            last hidden layer for this request's new tokens.  ``None`` falls
            back to a zero vector (history-only prediction).
        n_new_tokens:
            Number of *new* tokens processed in this prefill batch for this
            request (i.e. excluding cached tokens).  Used to distinguish a
            cold-start (large n_new_tokens, rich hidden state) from a warm-start
            cache-hit (small n_new_tokens, e.g. 1, dominated by boundary tokens).
        prompt_key:
            A tuple of the first N token IDs of the request's original input.
            Used as a stable, session-independent cache key for the hidden-state
            representation.  Two requests with the same system prompt will have
            the same ``prompt_key`` even if their ``session_id`` differs.
            Pass ``None`` to skip prompt-keyed caching.
        chain_id:
            Optional call-chain identifier.  When provided, it is forwarded to
            ``tree_cache.add_chain_predicted_ref`` so that prefetched nodes are
            marked as "predicted for this chain".
        history:
            Explicit tool-call history (list of tool names, oldest → newest)
            to feed into the predictor.  When provided, overrides the internal
            ``_session_history`` lookup.  This is the preferred source when the
            scheduler already tracks the full call-chain history via
            ``req.my_chain_history`` — the call chain IS the session for the
            prefetch use case, so the authoritative history lives on the
            scheduler side.  Pass ``None`` to fall back to the controller's
            local ``_session_history`` (still updated via
            :meth:`on_request_complete`).

        Returns
        -------
        List of ``(tool_name, probability)`` pairs sorted by probability desc.
        """
        # ── Resolve effective text feature ──────────────────────────────────
        # On a warm start (cache hit), sglang only runs the forward pass on
        # the uncached tokens.  If n_new_tokens < _MIN_COLD_TOKENS the captured
        # hidden state belongs to structural boundary tokens (e.g.
        # "<|im_start|>user\n"), NOT to the system-prompt content.  Feeding
        # that into the predictor yields garbage predictions (e.g. p=0.999 for
        # a random label).
        #
        # Lookup priority for a reliable hidden-state representation:
        #   1. prompt_key cache  (_prompt_hidden) — survives across sessions,
        #      best for multi-agent pipelines where each call has a fresh rid.
        #   2. session cache     (_session_hidden) — fallback for multi-turn
        #      conversations that explicitly reuse the same session_id.
        #   3. zero vector / None — last resort; history-only prediction.
        if n_new_tokens >= _MIN_COLD_TOKENS and text_feat is not None:
            # Cold-start: update BOTH caches so future warm calls can find it.
            if prompt_key is not None:
                self._prompt_hidden[prompt_key] = text_feat.detach()
            self._session_hidden[session_id] = text_feat.detach()
            effective_feat = text_feat
            feat_source = f"cold({n_new_tokens}tok)"
        elif prompt_key is not None and prompt_key in self._prompt_hidden:
            # Warm-start with known prompt prefix — use the prompt-keyed cache.
            effective_feat = self._prompt_hidden[prompt_key]
            feat_source = f"cached-prompt(warm,{n_new_tokens}tok)"
        elif session_id in self._session_hidden:
            # Warm-start, same session — use the session-keyed cache.
            effective_feat = self._session_hidden[session_id]
            feat_source = f"cached-session(warm,{n_new_tokens}tok)"
        else:
            # No cached representation available — fall back to zeros/None.
            effective_feat = text_feat  # may be None → predictor uses zero vector
            feat_source = f"zeros(no-cache,{n_new_tokens}tok)"

        # Prefer the explicit history passed in by the scheduler (sourced from
        # ``req.my_chain_history``).  Only fall back to the controller-local
        # ``_session_history`` when no explicit history is available.
        if history is None:
            history = self.get_history(session_id)
        # Defensive copy + truncate so downstream mutation / large histories
        # never break the predictor.
        history = list(history)[-self.max_history :]

        predictions = self.predictor.predict(
            history=history,
            text_feat=effective_feat,
            top_k=self.top_k,
        )

        # ── [Predict] INFO log ──────────────────────────────────────────────
        feat_desc = feat_source
        pred_str = ", ".join(f"{n}:{p:.3f}" for n, p in predictions)
        logger.info(
            "[Predict] session=%s | history=%s | feat=%s | top%d=[%s]",
            session_id,
            history if history else "[]",
            feat_desc,
            self.top_k,
            pred_str,
        )
        try:
            from sglang.srt.my_modules import kvcache_logger as _kvcl
            _kvcl.log_predict(session_id, feat_source, predictions, history)
        except Exception:
            pass

        triggered = []
        for tool_name, prob in predictions:
            # Filter special tokens — <END> means "no more tool calls"
            if tool_name in ("<END>", "<START>"):
                logger.debug(
                    "[Prefetch] '%s' p=%.3f → SKIP: special token",
                    tool_name, prob,
                )
                continue
            # Skip low-probability candidates to avoid wasteful prefetch
            if prob < self.min_prob:
                logger.info(
                    "[Prefetch] '%s' p=%.3f → SKIP: below min_prob=%.2f",
                    tool_name, prob, self.min_prob,
                )
                try:
                    from sglang.srt.my_modules import kvcache_logger as _kvcl
                    _kvcl.log_prefetch_controller(
                        tool_name, 0, False, prob,
                        f"below min_prob={self.min_prob:.2f}",
                    )
                except Exception:
                    pass
                continue
            result = self._try_prefetch(tool_name, prob, tree_cache, chain_id=chain_id)
            if result:
                triggered.append(f"{tool_name}:{prob:.3f}")

        # ── [Summary] per-request prefetch summary ──────────────────────────
        logger.info(
            "[Summary] session=%s | prefetch_triggered=%d/%d | tools=[%s]",
            session_id,
            len(triggered),
            len([p for p in predictions if p[0] not in ("<END>", "<START>")]),
            ", ".join(triggered) if triggered else "none",
        )

        return predictions

    def _try_prefetch(
        self,
        tool_name: str,
        prob: float,
        tree_cache: Any,
        chain_id: Optional[str] = None,
    ) -> bool:
        """
        Attempt a CPU→GPU prefetch for *tool_name*'s KV-cache prefix.

        Returns ``True`` if ``load_back`` was successfully called, ``False``
        in all skip / error cases.

        Uses MatchResult fields from HiRadixCache.match_prefix:
        - ``last_host_node``: deepest backed-up (CPU-resident) node in the match.
        - ``last_device_node``: deepest GPU-resident node in the match.
        - ``host_hit_length``: token count that is in CPU but NOT on GPU.
          A value > 0 means the cache was evicted from GPU → prefetch is useful.
        """
        token_ids = self._tool_tokens.get(tool_name)
        if token_ids is None:
            logger.debug(
                "[Prefetch] '%s' p=%.3f → SKIP: no token prefix registered",
                tool_name, prob,
            )
            return False

        if not hasattr(tree_cache, "load_back"):
            logger.debug(
                "[Prefetch] '%s' p=%.3f → SKIP: cache has no load_back "
                "(HiRadixCache not enabled)",
                tool_name, prob,
            )
            return False

        try:
            from sglang.srt.mem_cache.radix_cache import RadixKey
            from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams

            match_params = MatchPrefixParams(key=RadixKey(token_ids=token_ids))
            match_result = tree_cache.match_prefix(match_params)
        except Exception as exc:
            logger.warning(
                "[Prefetch] '%s' p=%.3f → SKIP: match_prefix error: %s",
                tool_name, prob, exc,
            )
            return False

        host_hit_length = match_result.host_hit_length
        last_host_node = match_result.last_host_node
        gpu_hit_tokens = len(match_result.device_indices)

        # Debug: raw match result (helps diagnose token-prefix mismatches)
        logger.debug(
            "[Prefetch] '%s' match: gpu_hit=%d host_hit=%d registered=%d",
            tool_name, gpu_hit_tokens, host_hit_length, len(token_ids),
        )

        if host_hit_length == 0:
            # No CPU-only cache: either fully on GPU or prefix not cached yet.
            if gpu_hit_tokens < max(1, len(token_ids) // 2):
                logger.info(
                    "[Prefetch] '%s' p=%.3f → SKIP: not in radix tree "
                    "(gpu_hit=%d / %d tokens)",
                    tool_name, prob, gpu_hit_tokens, len(token_ids),
                )
                try:
                    from sglang.srt.my_modules import kvcache_logger as _kvcl
                    _kvcl.log_prefetch_controller(
                        tool_name, len(token_ids), False, prob, "not in tree"
                    )
                except Exception:
                    pass
            else:
                logger.info(
                    "[Prefetch] '%s' p=%.3f → SKIP: already on GPU "
                    "(gpu_hit=%d tokens, host_hit=0)",
                    tool_name, prob, gpu_hit_tokens,
                )
                try:
                    from sglang.srt.my_modules import kvcache_logger as _kvcl
                    _kvcl.log_prefetch_controller(
                        tool_name, gpu_hit_tokens, False, prob, "already on GPU"
                    )
                except Exception:
                    pass
            return False

        # host_hit_length > 0: some prefix tokens were evicted from GPU to CPU.
        logger.info(
            "[Eviction-Obs] '%s': %d tokens evicted GPU→CPU "
            "(gpu_hit=%d, cpu_only=%d)",
            tool_name, host_hit_length, gpu_hit_tokens, host_hit_length,
        )
        try:
            tree_cache.load_back(last_host_node, is_prefetch=True)
            logger.info(
                "[Prefetch] '%s' p=%.3f → TRIGGERED "
                "(%d CPU tokens → GPU, load_back called)",
                tool_name, prob, host_hit_length,
            )
            # Record this chain as predicted to access the prefetched node
            if chain_id and hasattr(tree_cache, "add_chain_predicted_ref"):
                try:
                    tree_cache.add_chain_predicted_ref(last_host_node, chain_id)
                except Exception:
                    pass
            try:
                from sglang.srt.my_modules import kvcache_logger as _kvcl
                _kvcl.log_prefetch_controller(
                    tool_name, host_hit_length, True, prob
                )
            except Exception:
                pass
            return True
        except Exception as exc:
            logger.warning(
                "[Prefetch] '%s' p=%.3f → ERROR: load_back failed: %s",
                tool_name, prob, exc,
            )
            try:
                from sglang.srt.my_modules import kvcache_logger as _kvcl
                _kvcl.log_prefetch_controller(
                    tool_name, host_hit_length, False, prob,
                    f"load_back error: {exc}",
                )
            except Exception:
                pass
            return False

    # ── External-prediction entry point ───────────────────────────────────

    def prefetch_for_predictions(
        self,
        predictions: List[Tuple[str, float]],
        tree_cache: Any,
        chain_id: Optional[str] = None,
    ) -> List[str]:
        """
        Execute prefetch using externally-provided predictions (e.g. from
        the chain predictor).

        This bypasses the internal ``self.predictor`` entirely — the caller
        is responsible for running the prediction model and supplying the
        sorted ``(agent_name, probability)`` list.

        Parameters
        ----------
        predictions:
            Top-k ``(agent_name, probability)`` pairs, sorted by prob desc.
        tree_cache:
            The scheduler's ``tree_cache`` (must be ``HiRadixCache`` for
            ``load_back`` to work).
        chain_id:
            Optional call-chain identifier for ``add_chain_predicted_ref``.

        Returns
        -------
        List of ``"agent_name:prob"`` strings for which ``load_back`` was
        successfully triggered.
        """
        triggered: List[str] = []
        for agent_name, prob in predictions:
            if prob < self.min_prob:
                logger.info(
                    "[Prefetch] '%s' p=%.3f → SKIP: below min_prob=%.2f",
                    agent_name, prob, self.min_prob,
                )
                try:
                    from sglang.srt.my_modules import kvcache_logger as _kvcl
                    _kvcl.log_prefetch_controller(
                        agent_name, 0, False, prob,
                        f"below min_prob={self.min_prob:.2f}",
                    )
                except Exception:
                    pass
                continue
            result = self._try_prefetch(agent_name, prob, tree_cache, chain_id=chain_id)
            if result:
                triggered.append(f"{agent_name}:{prob:.3f}")

        logger.info(
            "[Prefetch] prefetch_for_predictions: triggered=%d/%d | tools=[%s]",
            len(triggered),
            len(predictions),
            ", ".join(triggered) if triggered else "none",
        )
        return triggered
