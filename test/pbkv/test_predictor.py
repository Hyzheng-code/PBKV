"""Checkpoint compatibility and asynchronous inference; never trains a model."""

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from sglang.srt.my_modules import embedding_buffer as eb
from sglang.srt.my_modules.chain_predictor import (
    GraphSAGEChainPredictor,
    GraphSAGEHistSmall,
)


def reference(checkpoint, embedding, current, history):
    """Single-sample equations evaluated directly from checkpoint tensors."""
    s = checkpoint["state_dict"]
    meta = checkpoint["meta"]
    cfg = checkpoint["cfg"]
    a = torch.as_tensor(checkpoint["A_fwd"])
    h = s["node_emb"]
    for layer in ["W1", "W2"]:
        h = F.relu(
            F.linear(
                torch.cat([h, a @ h], -1), s[layer + ".weight"], s[layer + ".bias"]
            )
        )
    cur = h[current]
    hist = torch.zeros_like(cur)
    if history:
        h_hist = h[history[-meta["max_hist"] :]]
        qkey = "attn_proj" if "attn_proj.weight" in s else "q_proj"
        q = F.linear(cur, s[qkey + ".weight"], s[qkey + ".bias"])
        weights = torch.softmax((q * h_hist).sum(-1) / cfg["d_node"] ** 0.5, dim=-1)
        hist = (weights[:, None] * h_hist).sum(0)
    text = F.relu(
        F.linear(embedding.float(), s["text_proj.weight"], s["text_proj.bias"])
    )
    hidden = F.relu(
        F.linear(torch.cat([cur, hist, text]), s["head.0.weight"], s["head.0.bias"])
    )
    logits = F.linear(hidden, s["head.3.weight"], s["head.3.bias"])
    return torch.softmax(logits.view(meta["future_k"], meta["n_classes"]), -1).numpy()


class PredictorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "model.pt"
        model = GraphSAGEHistSmall(3, 4, 8, 4, [8], 3, 3, 0.2, torch.eye(3))
        self.ckpt = {
            "meta": {
                "agents": ["<END>", "A", "B"],
                "n_classes": 3,
                "pad_id": 3,
                "end_id": 0,
                "max_hist": 4,
                "future_k": 3,
                "emb_dim": 8,
            },
            "cfg": {
                "d_node": 4,
                "text_proj_dim": 4,
                "head_dims": [8],
                "dropout": 0.2,
                "use_attn": True,
                "use_bidir": False,
            },
            "A_fwd": torch.eye(3),
            "state_dict": model.state_dict(),
        }
        torch.save(self.ckpt, self.path)

    def tearDown(self):
        self.temp.cleanup()
        eb._staging_layers.clear()
        eb._queue.clear()

    def test_all_heads_match_checkpoint_equations_and_batching(self):
        p = GraphSAGEChainPredictor(str(self.path), "cpu")
        x = torch.randn(3, 8)
        histories = [[], ["A"], ["B", "A"] * 5]
        currents = ["A", "B", "A"]
        actual = p.submit(x, currents, histories).result()
        for i in range(3):
            expected = reference(
                self.ckpt,
                x[i],
                p.agent_to_id[currents[i]],
                [p.agent_to_id[a] for a in histories[i]],
            )
            np.testing.assert_allclose(actual[i], expected, atol=1e-6)
        self.assertEqual(actual.shape, (3, 3, 3))
        np.testing.assert_allclose(actual.sum(-1), 1, atol=1e-6)

    def test_legacy_query_name_strict_load_without_changing_weights(self):
        expected = GraphSAGEChainPredictor(str(self.path), "cpu")
        for suffix in ["weight", "bias"]:
            self.ckpt["state_dict"]["q_proj." + suffix] = self.ckpt["state_dict"].pop(
                "attn_proj." + suffix
            )
        torch.save(self.ckpt, self.path)
        p = GraphSAGEChainPredictor(self.temp.name, "cpu")
        x = torch.randn(8)
        np.testing.assert_array_equal(
            p.predict_distributions(x, ["A"], "B"),
            expected.predict_distributions(x, ["A"], "B"),
        )

    def test_unknown_agent_and_horizon_errors_are_explicit(self):
        p = GraphSAGEChainPredictor(str(self.path), "cpu")
        with self.assertRaises(ValueError):
            p.submit(torch.zeros(1, 8), ["missing"], [[]])
        with self.assertRaises(ValueError):
            p.submit(torch.zeros(1, 7), ["A"], [[]])

    def test_embedding_buffer_selects_last_tokens_and_preserves_fifo(self):
        full = torch.arange(35).view(7, 5)
        eb.put_last(full)
        eb.commit([2, 5])
        eb.put_last(full + 100)
        eb.commit([7])
        a = eb.get()
        b = eb.get()
        torch.testing.assert_close(a.layers["last"], full[[1, 6]])
        torch.testing.assert_close(b.layers["last"], (full + 100)[[6]])
        self.assertEqual(a.layers["last"].numel(), 10)
        self.assertIsNone(eb.get())

    def test_empty_prefill_and_length_guard(self):
        eb.put_last(torch.empty(0, 8))
        eb.commit([0])
        self.assertEqual(eb.get().layers["last"].shape, (1, 8))
        eb.put_last(torch.empty(1, 8))
        with self.assertRaises(ValueError):
            eb.commit([2])

    @unittest.skipUnless(
        os.environ.get("PBKV_TEST_CHECKPOINT"),
        "set PBKV_TEST_CHECKPOINT to test existing weights",
    )
    def test_existing_checkpoint_cpu_matches_equations(self):
        path = os.environ["PBKV_TEST_CHECKPOINT"]
        p = GraphSAGEChainPredictor(path, "cpu")
        ckpt = torch.load(
            Path(path) / "model.pt" if Path(path).is_dir() else path,
            map_location="cpu",
            weights_only=False,
        )
        names = list(p.agent_to_id)
        x = torch.randn(p.emb_dim)
        history = names[:3]
        expected = reference(
            ckpt, x, p.agent_to_id[names[0]], [p.agent_to_id[a] for a in history]
        )
        np.testing.assert_allclose(
            p.predict_distributions(x, history, names[0]),
            expected,
            atol=2e-6,
            rtol=2e-5,
        )

    @unittest.skipUnless(
        os.environ.get("PBKV_TEST_CUDA") == "1" and torch.cuda.device_count() >= 2,
        "requires two CUDA devices and PBKV_TEST_CUDA=1",
    )
    def test_async_cross_device_existing_or_synthetic_checkpoint(self):
        path = os.environ.get("PBKV_TEST_CHECKPOINT", self.temp.name)
        cpu = GraphSAGEChainPredictor(path, "cpu")
        gpu = GraphSAGEChainPredictor(path, "cuda:1")
        names = list(cpu.agent_to_id)
        current = [names[0], names[-1]]
        histories = [[], names * 4]
        source = torch.cuda.Stream(device=0)
        for dtype in [torch.float32, torch.bfloat16]:
            x = torch.randn(2, cpu.emb_dim).to(dtype)
            expected = cpu.submit(x, current, histories).result()
            with torch.cuda.stream(source):
                embedded = x.to("cuda:0", non_blocking=True)
                ready = torch.cuda.Event()
                ready.record(source)
            ticket = gpu.submit(embedded, current, histories, ready)
            ticket.event.synchronize()
            np.testing.assert_allclose(ticket.result(), expected, atol=3e-6, rtol=3e-5)


if __name__ == "__main__":
    unittest.main()
