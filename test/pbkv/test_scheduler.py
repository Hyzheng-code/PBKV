"""Workflow state and stale-result tests using the actual scheduler methods."""

import unittest
from collections import deque
from types import SimpleNamespace as Obj

import numpy as np
import torch

from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.radix_cache import TreeNode
from sglang.srt.my_modules.chain_predictor import PredictionTicket
from sglang.srt.my_modules.pbkv_policy import PBKVPolicy
from sglang.srt.server_args import ServerArgs


class SchedulerTests(unittest.TestCase):
    def scheduler(self):
        p = PBKVPolicy(lambda n: 0)
        p.configure(["<END>", "A", "B"], 3, 0.7, 0)
        return Obj(
            _my_chain_next_step={},
            _my_chain_histories={},
            _my_chain_prediction_step={},
            _my_pending_predictions=deque(),
            server_args=Obj(
                my_base_chain_evict_enable=False,
                my_pred_chain_evict_enable=True,
            ),
            tp_rank=0,
            tp_size=1,
            _my_chain_predictor=Obj(agent_labels=p.labels),
            tree_cache=Obj(
                invalidate_chain_prediction=p.invalidate,
                update_chain_prediction=p.update,
                remove_chain_prediction=p.terminate,
                terminated_chain_ids=p.terminated,
            ),
            policy=p,
        )

    def req(self, agent, history=None, selector=False):
        return Obj(
            my_chain_id="w",
            my_current_agent=agent,
            my_chain_history=history,
            my_is_selector=selector,
            rid="r",
        )

    def test_history_is_prior_only_and_selector_is_excluded(self):
        s = self.scheduler()
        self.assertEqual(Scheduler._my_prepare_chain_context(s, self.req("A"))[2], [])
        self.assertEqual(
            Scheduler._my_prepare_chain_context(s, self.req(None, selector=True))[2],
            ["A"],
        )
        self.assertEqual(
            Scheduler._my_prepare_chain_context(s, self.req("B"))[2], ["A"]
        )
        self.assertEqual(s._my_chain_histories["w"], ["A", "B"])
        self.assertEqual(
            Scheduler._my_prepare_chain_context(s, self.req("A", history=[]))[2], []
        )

    def test_old_and_terminated_predictions_are_ignored(self):
        s = self.scheduler()
        _, step, _ = Scheduler._my_prepare_chain_context(s, self.req("A"))
        p = torch.tensor([[[0.1, 0.7, 0.2]] * 3])
        s._my_pending_predictions.append(([("w", step, "A", [])], PredictionTicket(p)))
        Scheduler._my_prepare_chain_context(s, self.req("B"))
        Scheduler._my_poll_predictions(s)
        self.assertNotIn("w", s.policy.forecasts)
        step = s._my_chain_prediction_step["w"]
        s._my_pending_predictions.append(
            ([("w", step, "B", ["A"])], PredictionTicket(p))
        )
        Scheduler._my_poll_predictions(s)
        np.testing.assert_allclose(s.policy.forecasts["w"], p.numpy()[0])
        s.policy.terminate("w")
        s._my_pending_predictions.append(
            ([("w", step, "B", ["A"])], PredictionTicket(p))
        )
        Scheduler._my_poll_predictions(s)
        self.assertNotIn("w", s.policy.forecasts)

    def test_pending_invocation_retains_cache_protection_until_new_forecast(self):
        s = self.scheduler()
        root, b, a = TreeNode(), TreeNode(), TreeNode()
        for node, agent in [(b, "B"), (a, "A")]:
            node.parent = root
            root.children[node.id] = node
            node.value = [node.id]
            node.chain_agent_pairs.add(("w", agent))
            node.visited_chain_ids.add("w")
            s.policy.sync(node)

        _, old_step, _ = Scheduler._my_prepare_chain_context(s, self.req("A"))
        previous = torch.tensor([[[0.0, 0.0, 1.0]] * 3])
        s._my_pending_predictions.append(
            ([("w", old_step, "A", [])], PredictionTicket(previous))
        )
        Scheduler._my_poll_predictions(s)
        self.assertIs(s.policy.gpu.peek()[1], a)

        # B is now queued, but its predicted prefix must remain protected.
        _, step, _ = Scheduler._my_prepare_chain_context(s, self.req("B"))
        self.assertIs(s.policy.gpu.peek()[1], a)
        np.testing.assert_allclose(s.policy.forecasts["w"], previous.numpy()[0])

        replacement = torch.tensor([[[0.0, 1.0, 0.0]] * 3])
        ready = Obj(done=False)
        ticket = PredictionTicket(replacement, event=Obj(query=lambda: ready.done))
        s._my_pending_predictions.extend(
            [
                ([("w", old_step, "A", [])], PredictionTicket(replacement)),
                ([("w", step, "B", ["A"])], ticket),
            ]
        )
        # An obsolete result cannot overwrite the retained forecast, and an
        # unfinished replacement must not interrupt its protection either.
        Scheduler._my_poll_predictions(s)
        self.assertEqual(len(s._my_pending_predictions), 1)
        self.assertIs(s.policy.gpu.peek()[1], a)
        np.testing.assert_allclose(s.policy.forecasts["w"], previous.numpy()[0])

        ready.done = True
        Scheduler._my_poll_predictions(s)
        self.assertFalse(s._my_pending_predictions)
        np.testing.assert_allclose(s.policy.forecasts["w"], replacement.numpy()[0])
        self.assertAlmostEqual(s.policy.score(a), 1 + 0.7 + 0.7**2)
        self.assertAlmostEqual(s.policy.score(b), 0)
        self.assertIs(s.policy.gpu.peek()[1], b)

    def test_termination_clears_forecast_retained_for_pending_invocation(self):
        s = self.scheduler()
        Scheduler._my_prepare_chain_context(s, self.req("A"))
        p = torch.tensor([[[0.1, 0.7, 0.2]] * 3])
        s.tree_cache.update_chain_prediction("w", p.numpy()[0])
        _, step, _ = Scheduler._my_prepare_chain_context(s, self.req("B"))
        self.assertIn("w", s.policy.forecasts)
        s._my_pending_predictions.append(
            ([("w", step, "B", ["A"])], PredictionTicket(p))
        )

        result = Scheduler._my_handle_chain_terminated_wrapped(s, Obj(chain_id="w"))
        self.assertTrue(result.success)
        self.assertIn("w", s.policy.terminated)
        self.assertNotIn("w", s.policy.forecasts)
        self.assertNotIn("w", s.policy.vectors)
        self.assertNotIn("w", s._my_chain_prediction_step)
        self.assertNotIn("w", s._my_chain_histories)
        self.assertNotIn("w", s._my_chain_next_step)
        Scheduler._my_poll_predictions(s)
        self.assertFalse(s._my_pending_predictions)
        self.assertNotIn("w", s.policy.forecasts)

    def test_prefetch_trigger_only_on_pure_decode(self):
        calls = []
        s = Obj(
            server_args=Obj(
                my_prefetch_enable=True,
                my_prefetch_evict_live=False,
                my_prefetch_active_fraction=0.2,
            ),
            _my_decode_events=deque(),
            tree_cache=Obj(
                try_batch_prefetch=lambda **kwargs: calls.append(kwargs)
                or {"skipped": True}
            ),
        )
        for decode in [False, False, True]:
            Scheduler._my_after_forward(
                s, Obj(forward_mode=Obj(is_decode=lambda: decode)), None
            )
        self.assertEqual(len(calls), 1)

    def test_cli_validation_rejects_truncated_scores_and_invalid_budget(self):
        args = Obj(
            **{
                name: getattr(ServerArgs, name)
                for name in [
                    "my_base_chain_evict_enable",
                    "my_pred_chain_evict_enable",
                    "my_prefetch_enable",
                    "my_pred_evict_K",
                    "my_pred_evict_gamma",
                    "my_prediction_noise",
                    "my_prefetch_active_fraction",
                    "my_pred_chain_evict_model_type",
                    "my_pred_chain_evict_model",
                    "enable_hierarchical_cache",
                    "my_prefetch_bw",
                    "my_prefetch_step_sec",
                    "my_prefetch_top_k",
                    "my_prefetch_min_prob",
                    "my_prefetch_value_quantile",
                    "disable_radix_cache",
                    "pp_size",
                    "dp_size",
                    "enable_dp_attention",
                ]
            }
        )
        args.my_pred_chain_evict_enable = True
        args.my_prefetch_enable = True
        args.enable_hierarchical_cache = True
        args.my_pred_chain_evict_model = "existing/model.pt"
        ServerArgs._validate_pbkv_args(args)
        for base, prediction, prefetch in (
            (True, False, False),
            (False, True, False),
            (False, False, True),
            (True, True, True),
        ):
            with self.subTest(base=base, prediction=prediction, prefetch=prefetch):
                incomplete = Obj(**vars(args))
                incomplete.my_base_chain_evict_enable = base
                incomplete.my_pred_chain_evict_enable = prediction
                incomplete.my_prefetch_enable = prefetch
                with self.assertRaisesRegex(ValueError, "only full mode"):
                    ServerArgs._validate_pbkv_args(incomplete)
        ordinary = Obj(**vars(args))
        ordinary.my_pred_chain_evict_enable = False
        ordinary.my_prefetch_enable = False
        ServerArgs._validate_pbkv_args(ordinary)
        args.my_prefetch_top_k = 1
        with self.assertRaises(ValueError):
            ServerArgs._validate_pbkv_args(args)
        args.my_prefetch_top_k = None
        args.my_prefetch_bw = float("nan")
        with self.assertRaises(ValueError):
            ServerArgs._validate_pbkv_args(args)


class MetricTests(unittest.TestCase):
    def test_gpu_metric_is_token_weighted_and_excludes_host(self):
        import io
        import logging
        from unittest.mock import patch

        from sglang.srt.my_modules import kvcache_logger as kv

        output = io.StringIO()
        logger = logging.Logger("pbkv_metric_test")
        logger.addHandler(logging.StreamHandler(output))
        state = {
            "_kv_logger": logger,
            "_req_count": 0,
            "_total_input_tokens": 0,
            "_total_device_tokens": 0,
            "_prefill_req_count": 0,
            "_total_prefill_ms": 0.0,
        }
        with patch.dict(vars(kv), state):
            for tokens, device, host in [(100, 10, 80), (400, 20, 100)]:
                kv.log_request_metrics(
                    rid="test",
                    input_tokens=tokens,
                    output_tokens=1,
                    cached_tokens=device + host,
                    cached_device=device,
                    cached_host=host,
                    prefill_ms=None,
                    decode_ms=None,
                    total_forward_ms=None,
                    hit_rate_pct=100 * device / tokens,
                )
        self.assertIn("avg_gpu_hit=6.00%", output.getvalue())


if __name__ == "__main__":
    unittest.main()
