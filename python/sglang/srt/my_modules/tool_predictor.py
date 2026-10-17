"""
MLP-S1 Tool Call Predictor.

Reconstructs the ToolPredictor model from the checkpoint saved by the
training pipeline (4_train.py / train_utils.py).

Architecture (S1 / concatenation scheme):
  input = cat([text_feat (2048-dim), seq_feat (5 × 32 = 160-dim)])  →  2208-dim
  shared MLP: Linear(2208, 256) → ReLU → Dropout(0.2)
              → Linear(256, 256) → ReLU → Dropout(0.2)
  head1:  Linear(256, n_classes)   ← next tool
  head2:  Linear(256, n_classes)   ← tool at +2 steps (auxiliary)

Text feature:
  During training, this was the mean-pooled last hidden state from the LLM
  over the user-request tokens.  For the current serving implementation,
  callers may pass a real hidden-state tensor or None (in which case a
  zero vector is used as a fallback so that history-only prediction still
  works).

Usage::

    manager = ToolPredictorManager(
        checkpoint_path="/path/to/MLP-S1.pt",
        vocab_path="/path/to/tool_vocab.json",
        device="cuda:1",
    )
    preds = manager.predict(
        history=["Image Classification", "Audio Classification"],
        text_feat=None,   # or a 1-D torch.Tensor of shape (2048,)
        top_k=3,
    )
    # preds → [("Object Detection", 0.62), ("Image Segmentation", 0.21), ...]
"""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ── Constants matching training hyper-parameters ────────────────────────────
_D_TOOL: int = 32   # embedding dim per tool in seq encoder
_K_CONCAT: int = 5  # number of recent tools to concatenate (S1 scheme)


# ── Model classes (must mirror train_utils.py exactly) ──────────────────────

class SeqEncoderS1(nn.Module):
    """Concatenate the last *k* tool embeddings from the call history."""

    def __init__(self, n_seq_vocab: int, d_tool: int = _D_TOOL, k: int = _K_CONCAT):
        super().__init__()
        self.k = k
        self.d_out = k * d_tool
        # padding_idx=0 → <START> / empty slots are zero-vectors
        self.emb = nn.Embedding(n_seq_vocab, d_tool, padding_idx=0)

    def forward(self, histories: List[List[int]]) -> torch.Tensor:
        """
        Args:
            histories: list of B history index sequences (variable length each).
        Returns:
            Tensor of shape [B, k * d_tool].
        """
        B = len(histories)
        dev = self.emb.weight.device
        padded = torch.zeros(B, self.k, dtype=torch.long, device=dev)
        for i, hist in enumerate(histories):
            recent = hist[-self.k :]
            if recent:
                offset = self.k - len(recent)
                padded[i, offset:] = torch.tensor(recent, dtype=torch.long, device=dev)
        return self.emb(padded).view(B, -1)  # [B, k*d_tool]


class ToolPredictor(nn.Module):
    """Two-head MLP that predicts next and step-two tools."""

    def __init__(
        self,
        d_text: int,
        seq_encoder: SeqEncoderS1,
        n_classes: int,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        use_head2: bool = True,
    ):
        super().__init__()
        self.seq_encoder = seq_encoder
        self.use_head2 = use_head2
        d_in = d_text + seq_encoder.d_out

        self.shared = nn.Sequential(
            nn.Linear(d_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head1 = nn.Linear(hidden_dim, n_classes)
        if use_head2:
            self.head2 = nn.Linear(hidden_dim, n_classes)

    def forward(
        self,
        text_feat: torch.Tensor,
        histories: List[List[int]],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        seq_feat = self.seq_encoder(histories)
        combined = torch.cat([text_feat, seq_feat], dim=-1)
        shared = self.shared(combined)
        logits1 = self.head1(shared)
        logits2 = self.head2(shared) if self.use_head2 else None
        return logits1, logits2


# ── Public manager class ─────────────────────────────────────────────────────

class ToolPredictorManager:
    """
    High-level wrapper that loads MLP-S1 from a checkpoint and provides a
    :meth:`predict` interface for use in the scheduler.

    Parameters
    ----------
    checkpoint_path:
        Path to the ``.pt`` file saved by ``4_train.py``.
    vocab_path:
        Path to ``tool_vocab.json`` (contains ``tool_vocab``, ``seq_vocab``,
        ``n_classes``, ``n_seq_vocab``, ``d_text``).
    device:
        PyTorch device string, e.g. ``"cuda:1"`` or ``"cpu"``.
    """

    def __init__(self, checkpoint_path: str, vocab_path: str, device: str):
        self.device = torch.device(device)

        # ── Load vocabulary ──────────────────────────────────────────────────
        with open(vocab_path, encoding="utf-8") as f:
            vocab_data: dict = json.load(f)

        self.tool_vocab: Dict[str, int] = vocab_data["tool_vocab"]
        self.seq_vocab: Dict[str, int] = vocab_data["seq_vocab"]
        self.n_classes: int = vocab_data["n_classes"]
        self.n_seq_vocab: int = vocab_data["n_seq_vocab"]
        self.d_text: int = vocab_data["d_text"]

        # Reverse mapping: class index → tool name
        self.idx_to_tool: Dict[int, str] = {
            v: k for k, v in self.tool_vocab.items()
        }

        # ── Load checkpoint ──────────────────────────────────────────────────
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        scheme: str = ckpt["scheme"]
        use_head2: bool = ckpt["use_head2"]

        if scheme != "S1":
            raise ValueError(
                f"ToolPredictorManager only supports scheme 'S1', got '{scheme}'."
            )

        # ── Build model ──────────────────────────────────────────────────────
        seq_enc = SeqEncoderS1(n_seq_vocab=self.n_seq_vocab)
        self.model = ToolPredictor(
            d_text=self.d_text,
            seq_encoder=seq_enc,
            n_classes=self.n_classes,
            use_head2=use_head2,
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        val_metrics = ckpt.get("val_metrics", {})
        logger.info(
            "[ToolPredictor] Loaded MLP-S1 on %s | "
            "d_text=%d, n_classes=%d, epoch=%d | "
            "val top-1=%.4f, top-3=%.4f",
            device,
            self.d_text,
            self.n_classes,
            ckpt.get("epoch", -1),
            val_metrics.get("top1_acc", float("nan")),
            val_metrics.get("top3_hit", float("nan")),
        )

    # ── Helpers ──────────────────────────────────────────────────────────────

    def history_to_seq_indices(self, history: List[str]) -> List[int]:
        """Convert a list of tool names to seq_vocab indices (skip unknowns)."""
        return [
            self.seq_vocab[name]
            for name in history
            if name in self.seq_vocab
        ]

    def is_known_tool(self, tool_name: str) -> bool:
        """Return True if *tool_name* is in the predictor's vocabulary."""
        return tool_name in self.tool_vocab

    # ── Main inference entry-point ────────────────────────────────────────────

    @torch.no_grad()
    def predict(
        self,
        history: List[str],
        text_feat: Optional[torch.Tensor] = None,
        top_k: int = 3,
    ) -> List[Tuple[str, float]]:
        """
        Predict the most likely next tool(s).

        Parameters
        ----------
        history:
            Ordered list of tool names called so far in the current session
            (most-recent last).  May be empty.
        text_feat:
            Optional 1-D float tensor of shape ``(d_text,)`` extracted from
            the LLM's last hidden layer via the hook mechanism.  If ``None``,
            a zero vector is used as a fallback (history-only prediction).
        top_k:
            How many top predictions to return.

        Returns
        -------
        List of ``(tool_name, probability)`` tuples sorted by probability
        descending.  Probabilities sum to ≈ 1 across all classes.
        """
        import numpy as np

        hist_indices = self.history_to_seq_indices(history)

        if text_feat is None:
            feat = torch.zeros(1, self.d_text, device=self.device, dtype=torch.float32)
        else:
            # Accepts both 1-D (d_text,) and 2-D (1, d_text) tensors.
            feat = text_feat.to(device=self.device, dtype=torch.float32)
            if feat.dim() == 1:
                feat = feat.unsqueeze(0)  # [1, d_text]

        logits1, _ = self.model(feat, [hist_indices])
        probs = torch.softmax(logits1[0], dim=-1).cpu().numpy()

        k = min(top_k, self.n_classes)
        top_indices = np.argsort(probs)[::-1][:k]

        return [
            (self.idx_to_tool.get(int(idx), f"<tool_{idx}>"), float(probs[idx]))
            for idx in top_indices
        ]
