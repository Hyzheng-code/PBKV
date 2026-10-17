"""
Chain-aware next-agent predictor for ``--my-pred-chain-evict-enable``.

PBKV serves an existing GraphSAGE ``model.pt`` and returns all K probability
distributions. The historical CompressMLP wrapper is retained for standalone
callers; the PBKV scheduler uses GraphSAGEChainPredictor exclusively.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model definition (must match training code exactly)
# ---------------------------------------------------------------------------
class CompressMLP(nn.Module):
    """Two-input MLP: compress embedding then concat with history features."""

    def __init__(
        self,
        hist_dim: int,
        emb_dim: int,
        num_classes: int,
        compress_dim: int = 32,
        hidden_layers: List[int] = (128, 32),
        dropout: float = 0.3,
    ):
        super().__init__()
        self.emb_compress = nn.Sequential(
            nn.Linear(emb_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, compress_dim),
            nn.ReLU(),
        )
        mods: list = []
        in_d = hist_dim + compress_dim
        for h in hidden_layers:
            mods.append(nn.Linear(in_d, h))
            mods.append(nn.BatchNorm1d(h))
            mods.append(nn.ReLU())
            mods.append(nn.Dropout(dropout))
            in_d = h
        mods.append(nn.Linear(in_d, num_classes))
        self.classifier = nn.Sequential(*mods)

    def forward(self, xh: torch.Tensor, xe: torch.Tensor) -> torch.Tensor:
        z = self.emb_compress(xe)
        x = torch.cat([xh, z], dim=1)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Feature engineering (mirrors training notebook exactly)
# ---------------------------------------------------------------------------
def _build_history_feature(
    step_id: int,
    history_str: str,
    chain_history: List[str],
    current_agent: str,
    encoder: Dict,
) -> np.ndarray:
    """Build a single history feature vector (same as ``transform_history_features``
    in the training notebook, but for one record)."""
    pattern_to_idx = encoder["pattern_to_idx"]
    pattern_list = encoder["pattern_list"]
    agents_set = encoder["agents_set"]
    agent_to_idx = encoder["agent_to_idx"]

    row: list = []
    # 1) step_id
    row.append(step_id)

    # 2) history length
    hist_agents = history_str.split("\u2192") if history_str else []
    row.append(len(hist_agents))

    # 3) last agent one-hot
    last_oh = [0] * len(agents_set)
    if hist_agents:
        last_a = hist_agents[-1]
        if last_a in agent_to_idx:
            last_oh[agent_to_idx[last_a]] = 1
    row.extend(last_oh)

    # 4) chain_history length
    row.append(len(chain_history) if chain_history else 0)

    # 5) pattern one-hot
    pat_oh = [0] * len(pattern_list)
    if history_str in pattern_to_idx:
        pat_oh[pattern_to_idx[history_str]] = 1
    row.extend(pat_oh)

    # 6) agent counts
    agent_counts = [0] * len(agents_set)
    for a in hist_agents:
        if a in agent_to_idx:
            agent_counts[agent_to_idx[a]] += 1
    row.extend(agent_counts)

    # 7) current_agent one-hot
    cur_oh = [0] * len(agents_set)
    if current_agent in agent_to_idx:
        cur_oh[agent_to_idx[current_agent]] = 1
    row.extend(cur_oh)

    return np.array(row, dtype=np.float32)


# ---------------------------------------------------------------------------
# Predictor wrapper
# ---------------------------------------------------------------------------
class ChainPredictor:
    """Loads model artifacts and provides ``predict()`` for single requests."""

    def __init__(self, artifacts_dir: str, device: str = "cuda:1"):
        self.device = torch.device(device)

        # ── Load history encoder ──────────────────────────────────────────
        enc_path = os.path.join(artifacts_dir, "history_encoder.json")
        with open(enc_path, "r", encoding="utf-8") as f:
            self.encoder: Dict = json.load(f)

        # ── Load checkpoint ───────────────────────────────────────────────
        ckpt_path = os.path.join(artifacts_dir, "best_model.pt")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        self.agent_labels: List[str] = ckpt["agent_labels"]
        self.label_to_id: Dict[str, int] = ckpt["label_to_id"]
        self.id_to_label: Dict[int, str] = {v: k for k, v in self.label_to_id.items()}

        hp = ckpt["model_hparams"]
        self.model = CompressMLP(
            hist_dim=hp["hist_dim"],
            emb_dim=hp["emb_dim"],
            num_classes=hp["num_classes"],
            compress_dim=hp["compress_dim"],
            hidden_layers=hp["hidden_layers"],
            dropout=hp["dropout"],
        )
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device)
        self.model.eval()

        logger.info(
            "[ChainPredictor] Loaded from %s | labels=%s | device=%s | "
            "hist_dim=%d emb_dim=%d compress_dim=%d",
            artifacts_dir,
            self.agent_labels,
            self.device,
            hp["hist_dim"],
            hp["emb_dim"],
            hp["compress_dim"],
        )

    @torch.no_grad()
    def predict(
        self,
        embedding_last: torch.Tensor,
        step_id: int,
        history_str: str,
        chain_history: List[str],
        current_agent: str,
    ) -> Tuple[str, float, List[Tuple[str, float]]]:
        """
        Predict the next agent.

        Parameters
        ----------
        embedding_last : Tensor
            1-D tensor ``[emb_dim]`` (CPU float32) — last-token hidden state
            from ``model.norm``.
        step_id : int
            Current step index in the chain.
        history_str : str
            Arrow-separated history string, e.g. ``"Planner→Retriever→Planner"``.
        chain_history : list[str]
            List form of history, e.g. ``["Planner", "Retriever", "Planner"]``.
        current_agent : str
            Name of the agent that just finished, e.g. ``"Planner"``.

        Returns
        -------
        (predicted_label, confidence, all_probs)
            - predicted_label: str — e.g. ``"Retriever"``
            - confidence: float — softmax probability of the top prediction
            - all_probs: list of (label, prob) sorted descending
        """
        # Guard the CUDA device context: this predictor runs on a secondary
        # device (e.g. cuda:1) but the caller's main model is on cuda:0.
        # We must restore the original device after our CUDA operations.
        _prev_device = torch.cuda.current_device()
        try:
            return self._predict_impl(
                embedding_last, step_id, history_str, chain_history, current_agent
            )
        finally:
            torch.cuda.set_device(_prev_device)

    def _predict_impl(
        self,
        embedding_last: torch.Tensor,
        step_id: int,
        history_str: str,
        chain_history: List[str],
        current_agent: str,
    ) -> Tuple[str, float, List[Tuple[str, float]]]:
        # Build history features
        xh = _build_history_feature(
            step_id=step_id,
            history_str=history_str,
            chain_history=chain_history,
            current_agent=current_agent,
            encoder=self.encoder,
        )
        xh_t = torch.tensor(xh, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Build embedding features
        xe_t = embedding_last.unsqueeze(0).to(self.device)

        # Forward
        logits = self.model(xh_t, xe_t)  # [1, num_classes]
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        # Sort by probability descending
        sorted_indices = np.argsort(-probs)
        all_probs = [
            (self.id_to_label[int(idx)], float(probs[idx]))
            for idx in sorted_indices
        ]
        predicted_label = all_probs[0][0]
        confidence = all_probs[0][1]

        return predicted_label, confidence, all_probs


# ---------------------------------------------------------------------------
# GraphSAGE model definition (must match training code)
# ---------------------------------------------------------------------------
class GraphSAGEHistSmall(nn.Module):
    """GraphSAGE with history attention, matching train_graphsage_hist_small.py."""

    def __init__(
        self,
        n_agents: int,
        d_node: int,
        emb_dim: int,
        text_proj_dim: int,
        head_dims: List[int],
        num_classes: int,
        future_k: int,
        dropout: float,
        A_fwd: torch.Tensor,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.future_k = future_k
        self.num_classes = num_classes
        self.register_buffer("A_fwd", A_fwd)

        self.node_emb = nn.Parameter(torch.randn(n_agents, d_node) * 0.1)
        self.W1 = nn.Linear(2 * d_node, d_node)
        self.W2 = nn.Linear(2 * d_node, d_node)
        self.attn_proj = nn.Linear(d_node, d_node)
        self.attn_scale = d_node ** 0.5
        self.text_proj = nn.Linear(emb_dim, text_proj_dim)

        head_layers: list = []
        head_in = d_node + d_node + text_proj_dim
        for hidden_dim in head_dims:
            head_layers.append(nn.Linear(head_in, hidden_dim))
            head_layers.append(nn.ReLU())
            head_layers.append(nn.Dropout(dropout))
            head_in = hidden_dim
        head_layers.append(nn.Linear(head_in, future_k * num_classes))
        self.head = nn.Sequential(*head_layers)

    def propagate(self) -> torch.Tensor:
        h = self.node_emb
        agg1 = self.A_fwd @ h
        h1 = torch.relu(self.W1(torch.cat([h, agg1], dim=-1)))
        agg2 = self.A_fwd @ h1
        h2 = torch.relu(self.W2(torch.cat([h1, agg2], dim=-1)))
        return h2

    def forward_batch(self, cur_id, ch_ids, ch_len, text_emb):
        h_nodes = self.propagate()
        cur_h = h_nodes[cur_id]
        h_padded = torch.cat(
            [h_nodes, torch.zeros(1, h_nodes.size(1), device=h_nodes.device)], dim=0
        )
        hist_h = h_padded[ch_ids]
        positions = torch.arange(ch_ids.size(1), device=cur_id.device)[None, :]
        mask = positions < ch_len[:, None]
        q = self.attn_proj(cur_h)[:, None, :]
        scores = (q * hist_h).sum(-1) / self.attn_scale
        weights = torch.softmax(scores.masked_fill(~mask, -1e9), dim=-1)
        hist_agg = (weights[..., None] * hist_h).sum(1)
        hist_agg = hist_agg.masked_fill((ch_len == 0)[:, None], 0.0)
        text = torch.relu(self.text_proj(text_emb.float()))
        out = self.head(torch.cat([cur_h, hist_agg, text], dim=-1))
        return out.view(-1, self.future_k, self.num_classes)

    def forward_single(self, cur_id, ch_ids, ch_len, text_emb):
        return self.forward_batch(cur_id, ch_ids.unsqueeze(0), ch_len, text_emb)


# ---------------------------------------------------------------------------
# GraphSAGE chain predictor wrapper
# ---------------------------------------------------------------------------
class PredictionTicket:
    """Pinned host result whose CUDA completion can be polled without waiting."""

    def __init__(self, output, event=None, keepalive=()):
        self.output, self.event, self.keepalive = output, event, keepalive

    def ready(self):
        return self.event is None or self.event.query()

    def result(self):
        if not self.ready():
            raise RuntimeError("Prediction is not ready")
        return self.output.numpy()


class GraphSAGEChainPredictor:
    """Serve an existing GraphSAGE checkpoint; expose all K conditional heads.

    Histories contain prior invocations only (i < t in paper Eq. 4); the current
    agent has its own input. No training or weight update occurs in this class.
    """

    def __init__(self, artifacts_dir: str, device: str = "cuda:2"):
        self.device = torch.device(device)
        ckpt_path = (
            artifacts_dir
            if os.path.isfile(artifacts_dir)
            else os.path.join(artifacts_dir, "model.pt")
        )
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        meta, cfg = ckpt["meta"], ckpt["cfg"]
        if (
            not cfg.get("use_attn", True)
            or cfg.get("use_bidir", False)
            or len(cfg["head_dims"]) != 1
        ):
            raise ValueError(
                "PBKV requires history attention, forward GraphSAGE, and a two-layer MLP head"
            )
        agents = list(meta["agents"])
        self.pad_id = int(meta["pad_id"])
        self.max_hist = int(meta["max_hist"])
        self.future_k = int(meta.get("future_k", 3))
        n_classes = int(meta["n_classes"])
        end_names = {"Chain_End", "chain_end", "END", "<END>", "⟨END⟩"}
        self.end_id = meta.get(
            "end_id", next((i for i, a in enumerate(agents) if a in end_names), None)
        )
        if self.end_id is None or not 0 <= self.end_id < n_classes:
            raise ValueError("Checkpoint must identify the END output class")
        self.agent_to_id = {
            a: i for i, a in enumerate(agents) if i < self.pad_id and i != self.end_id
        }
        if len(agents) == n_classes - 1 and self.end_id == len(agents):
            agents.append("<END>")
        if len(agents) != n_classes or self.max_hist < 1:
            raise ValueError(
                "Checkpoint class vocabulary or history length is inconsistent"
            )
        self.agent_labels = agents
        self.id_to_label = dict(enumerate(agents))
        self.emb_dim = int(meta["emb_dim"])
        A_fwd = torch.as_tensor(ckpt["A_fwd"], dtype=torch.float32)
        self.model = GraphSAGEHistSmall(
            n_agents=self.pad_id,
            d_node=cfg["d_node"],
            emb_dim=self.emb_dim,
            text_proj_dim=cfg["text_proj_dim"],
            head_dims=cfg["head_dims"],
            num_classes=n_classes,
            future_k=self.future_k,
            dropout=cfg["dropout"],
            A_fwd=A_fwd,
        )
        state = dict(ckpt["state_dict"])
        # Both published parameter names denote exactly the same query layer.
        for suffix in ("weight", "bias"):
            old, new = f"q_proj.{suffix}", f"attn_proj.{suffix}"
            if old in state:
                if new in state and not torch.equal(state[old], state[new]):
                    raise ValueError(
                        "Conflicting query-projection weights in checkpoint"
                    )
                state[new] = state.pop(old)
        state["A_fwd"] = A_fwd
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()
        self.stream = (
            torch.cuda.Stream(device=self.device)
            if self.device.type == "cuda"
            else None
        )
        if self.stream is not None:
            self.stream.wait_stream(torch.cuda.current_stream(self.device))
        self.transfer_streams = {}
        logger.info(
            "Loaded GraphSAGE predictor: path=%s K=%d device=%s",
            ckpt_path,
            self.future_k,
            self.device,
        )

    def _inputs(self, current_agents, histories):
        unknown = set(current_agents) - self.agent_to_id.keys()
        unknown.update(
            a for hist in histories for a in hist if a not in self.agent_to_id
        )
        if unknown:
            raise ValueError(
                f"Agents absent from predictor vocabulary: {sorted(unknown)}"
            )
        cur = torch.tensor(
            [self.agent_to_id[a] for a in current_agents],
            dtype=torch.long,
            device=self.device,
        )
        ids = [
            [self.agent_to_id[a] for a in hist[-self.max_hist :]] for hist in histories
        ]
        lengths = torch.tensor(
            [len(hist) for hist in ids], dtype=torch.long, device=self.device
        )
        padded = torch.tensor(
            [hist + [self.pad_id] * (self.max_hist - len(hist)) for hist in ids],
            dtype=torch.long,
            device=self.device,
        )
        return cur, padded, lengths

    @torch.no_grad()
    def submit(self, embeddings, current_agents, histories, ready_event=None):
        if len(current_agents) != len(histories) or embeddings.shape != (
            len(histories),
            self.emb_dim,
        ):
            raise ValueError("Predictor batch dimensions do not match the checkpoint")
        if self.stream is None:
            if ready_event is not None:
                ready_event.synchronize()
            cur, ids, lengths = self._inputs(current_agents, histories)
            logits = self.model.forward_batch(
                cur, ids, lengths, embeddings.to(self.device)
            )
            return PredictionTicket(torch.softmax(logits, dim=-1).cpu())
        transfer_input = embeddings
        if embeddings.is_cuda and embeddings.device != self.device:
            # Transfer only the selected last-token vectors. A pinned bridge
            # also supports PCIe systems where peer access is unavailable or
            # unreliable, without synchronizing either GPU on the host.
            source = embeddings.device
            if source not in self.transfer_streams:
                self.transfer_streams[source] = torch.cuda.Stream(device=source)
            transfer_stream = self.transfer_streams[source]
            if ready_event is None:
                ready_event = torch.cuda.Event()
                ready_event.record(torch.cuda.current_stream(source))
            with torch.cuda.device(source), torch.cuda.stream(transfer_stream):
                transfer_stream.wait_event(ready_event)
                transfer_input = torch.empty_like(
                    embeddings, device="cpu", pin_memory=True
                )
                transfer_input.copy_(embeddings, non_blocking=True)
                ready_event = torch.cuda.Event()
                ready_event.record(transfer_stream)
        with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            if ready_event is not None:
                self.stream.wait_event(ready_event)
            cur, ids, lengths = self._inputs(current_agents, histories)
            text = transfer_input.to(
                self.device, dtype=torch.float32, non_blocking=True
            )
            probabilities = torch.softmax(
                self.model.forward_batch(cur, ids, lengths, text), dim=-1
            )
            output = torch.empty(
                probabilities.shape,
                dtype=probabilities.dtype,
                device="cpu",
                pin_memory=True,
            )
            output.copy_(probabilities, non_blocking=True)
            done = torch.cuda.Event()
            done.record(self.stream)
        return PredictionTicket(
            output, done, (embeddings, transfer_input, text, probabilities, ready_event)
        )

    @torch.no_grad()
    def predict_distributions(self, embedding_last, chain_history, current_agent):
        """Synchronous full-horizon API for validation and standalone callers."""
        ready = None
        if embedding_last.is_cuda:
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream(embedding_last.device))
        ticket = self.submit(
            embedding_last[None, :], [current_agent], [chain_history], ready
        )
        if ticket.event is not None:
            ticket.event.synchronize()
        return ticket.result()[0]

    def predict(
        self, embedding_last, step_id, history_str, chain_history, current_agent
    ):
        """Legacy next-agent interface; serving uses submit() and all K heads."""
        probabilities = self.predict_distributions(
            embedding_last, chain_history, current_agent
        )[0]
        order = np.argsort(-probabilities)
        all_probs = [(self.agent_labels[i], float(probabilities[i])) for i in order]
        return all_probs[0][0], all_probs[0][1], all_probs
