"""PBKV request rejection, metadata forwarding, and full-mode startup checks."""

import json
import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as Obj
from unittest.mock import AsyncMock, Mock, patch

import httpx
from fastapi import FastAPI

from sglang.srt.entrypoints import http_server
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.managers.io_struct import EmbeddingReqInput, GenerateReqInput
from sglang.srt.managers.tokenizer_manager import ServerStatus, TokenizerManager


def workflow_fields():
    return dict(
        my_chain_id="workflow-1",
        my_current_agent="A",
        my_chain_history=[],
        my_is_selector=False,
    )


def manager():
    result = TokenizerManager.__new__(TokenizerManager)
    result.server_args = Obj(
        my_pred_chain_evict_enable=True,
        my_prefetch_enable=True,
        disaggregation_mode="null",
    )
    return result


class RequestValidationTests(unittest.TestCase):
    def test_every_field_is_required_even_for_selectors(self):
        for selector in (False, True):
            for field in workflow_fields():
                for omit in (True, False):
                    with self.subTest(selector=selector, field=field, omit=omit):
                        fields = workflow_fields()
                        fields["my_is_selector"] = selector
                        if omit:
                            fields.pop(field)
                        else:
                            fields[field] = None
                        with self.assertRaisesRegex(ValueError, field):
                            manager().validate_pbkv_request(
                                GenerateReqInput(text="hello", **fields)
                            )

    def test_empty_history_and_false_selector_are_valid(self):
        for history, selector in (([], False), (["A", "B"], False), ([], True)):
            fields = workflow_fields()
            fields.update(my_chain_history=history, my_is_selector=selector)
            manager().validate_pbkv_request(GenerateReqInput(text="hello", **fields))

    def test_blank_identifiers_and_malformed_metadata_are_rejected(self):
        invalid = {
            "my_chain_id": ("", " \t", 5, []),
            "my_current_agent": ("", "\n", 5, []),
            "my_chain_history": ("A", [None], [""], [" "], [42]),
            "my_is_selector": ("false", 0, 1, []),
        }
        for field, values in invalid.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    fields = workflow_fields()
                    fields[field] = value
                    with self.assertRaisesRegex(ValueError, field):
                        manager().validate_pbkv_request(
                            GenerateReqInput(text="hello", **fields)
                        )

    def test_non_pbkv_and_embedding_requests_keep_existing_contract(self):
        ordinary = manager()
        ordinary.server_args.my_pred_chain_evict_enable = False
        ordinary.validate_pbkv_request(GenerateReqInput(text="hello"))
        manager().validate_pbkv_request(EmbeddingReqInput(text="hello"))

    def test_batch_splitting_preserves_workflow_fields(self):
        for sampling_params in ({}, {"n": 2}):
            with self.subTest(sampling_params=sampling_params):
                fields = workflow_fields()
                fields["my_chain_history"] = ["B"]
                request = GenerateReqInput(
                    text=["hello", "world"],
                    sampling_params=sampling_params,
                    **fields,
                )
                request.normalize_batch_and_arguments()
                for index in range(len(request.rid)):
                    item = request[index]
                    manager().validate_pbkv_request(item)
                    for field, expected in fields.items():
                        self.assertEqual(getattr(item, field), expected)
                    self.assertIsNot(item.my_chain_history, request.my_chain_history)


class RequestEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.manager = manager()
        self.state = patch.object(
            http_server, "_global_state", Obj(tokenizer_manager=self.manager)
        )
        self.state.start()
        self.addCleanup(self.state.stop)
        self.app = FastAPI()
        self.app.post("/generate")(http_server.generate_request)

    async def test_native_requests_return_400_before_streaming_or_dispatch(self):
        self.manager.generate_request = Mock(side_effect=AssertionError("dispatched"))
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app), base_url="http://test"
        ) as client:
            for stream in (False, True):
                for text in ("hello", ["hello", "world"]):
                    for field in workflow_fields():
                        with self.subTest(stream=stream, text=text, field=field):
                            fields = workflow_fields()
                            fields.pop(field)
                            response = await client.post(
                                "/generate", json=dict(text=text, stream=stream, **fields)
                            )
                            self.assertEqual(response.status_code, 400)
                            self.assertIn(field, response.json()["error"]["message"])
        self.manager.generate_request.assert_not_called()

    async def test_valid_native_requests_reach_generation(self):
        received = []

        async def generate(obj, request):
            received.append(obj)
            yield {"text": "ok"}

        self.manager.generate_request = generate
        self.manager.create_abort_task = lambda obj: None
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app), base_url="http://test"
        ) as client:
            for stream in (False, True):
                response = await client.post(
                    "/generate",
                    json=dict(text="hello", stream=stream, **workflow_fields()),
                )
                self.assertEqual(response.status_code, 200)
                self.assertIn("ok", response.text)
        self.assertEqual(len(received), 2)

    async def test_openai_requests_return_400_before_streaming_or_dispatch(self):
        serving = OpenAIServingChat.__new__(OpenAIServingChat)
        serving.tokenizer_manager = self.manager
        serving._validate_request = lambda request: None
        serving._handle_streaming_request = AsyncMock()
        serving._handle_non_streaming_request = AsyncMock()
        for stream in (False, True):
            for field in workflow_fields():
                fields = workflow_fields()
                fields.pop(field)
                request = ChatCompletionRequest(
                    model="test",
                    messages=[{"role": "user", "content": "hello"}],
                    stream=stream,
                    **fields,
                )
                # Chat-template rendering is independent of metadata validation.
                serving._convert_to_internal_request = Mock(
                    return_value=(GenerateReqInput(text="hello", **fields), request)
                )
                response = await serving.handle_request(request, None)
                self.assertEqual(response.status_code, 400)
                self.assertIn(field, json.loads(response.body)["message"])
        serving._handle_streaming_request.assert_not_awaited()
        serving._handle_non_streaming_request.assert_not_awaited()

    async def test_direct_manager_calls_fail_before_starting_work(self):
        self.manager.auto_create_handle_loop = Mock()
        # A client-controlled health-check-like ID does not bypass validation.
        for rid in (None, "HEALTH_CHECK_client"):
            with self.assertRaisesRegex(ValueError, "my_chain_id"):
                await self.manager.generate_request(
                    GenerateReqInput(text="hello", rid=rid)
                ).__anext__()
        self.manager.auto_create_handle_loop.assert_not_called()

    async def test_internal_validation_exemption_is_explicit(self):
        self.manager.auto_create_handle_loop = Mock(side_effect=RuntimeError("accepted"))
        with self.assertRaisesRegex(RuntimeError, "accepted"):
            await self.manager.generate_request(
                GenerateReqInput(text="hello"), _skip_pbkv_validation=True
            ).__anext__()

    async def test_health_endpoint_uses_internal_request(self):
        self.manager.gracefully_exit = False
        self.manager.server_status = ServerStatus.Up
        self.manager.is_image_gen = False
        self.manager.is_generation = True
        self.manager.last_receive_tstamp = 0
        self.manager.rid_to_state = {}
        received = []

        async def generate(obj, request, *, _skip_pbkv_validation=False):
            received.append(_skip_pbkv_validation)
            self.manager.last_receive_tstamp = time.time()
            yield {}

        self.manager.generate_request = generate
        self.app.get("/health_generate")(http_server.health_generate)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app), base_url="http://test"
        ) as client:
            response = await client.get("/health_generate")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(received, [True])


class StartupTests(unittest.TestCase):
    def test_http_warmup_supplies_metadata_and_retires_its_workflow(self):
        for multimodal, disaggregation_mode in (
            (False, "null"),
            (True, "null"),
            (False, "prefill"),
        ):
            with self.subTest(multimodal=multimodal, mode=disaggregation_mode):
                args = Obj(
                    url=lambda: "http://test",
                    api_key=None,
                    skip_tokenizer_init=False,
                    disaggregation_mode=disaggregation_mode,
                    dp_size=1,
                    tp_size=2,
                    debug_tensor_dump_input_file=None,
                    my_pred_chain_evict_enable=True,
                )
                tm = manager()
                tm.served_model_name = "test"
                calls = []

                def post(url, *, json, **kwargs):
                    calls.append((url, json))
                    if url.endswith("/my/chain_terminated"):
                        self.assertEqual(json["chain_id"], calls[0][1]["my_chain_id"])
                    else:
                        fields = {key: json[key] for key in workflow_fields()}
                        tm.validate_pbkv_request(
                            GenerateReqInput(text="warmup", **fields)
                        )
                        self.assertIs(json["my_is_selector"], True)
                    return Mock(status_code=200, json=lambda: {"success": True})

                info = Mock(
                    status_code=200,
                    json=lambda: {
                        "is_generation": True,
                        "has_image_understanding": multimodal,
                    },
                )
                with (
                    patch.object(http_server, "_global_state", Obj(tokenizer_manager=tm)),
                    patch.object(http_server.time, "sleep"),
                    patch.object(http_server.requests, "get", return_value=info),
                    patch.object(http_server.requests, "post", side_effect=post),
                    patch.object(http_server, "kill_process_tree") as kill,
                ):
                    self.assertTrue(http_server._execute_server_warmup(args))
                    kill.assert_not_called()
                self.assertEqual(len(calls), 2)
                self.assertEqual(tm.server_status, ServerStatus.Up)

    def test_launcher_only_accepts_full_mode_and_enables_both_features(self):
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as tmp:
            fake_python = Path(tmp) / "python"
            fake_python.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
            fake_python.chmod(0o755)
            env = dict(
                os.environ,
                MODEL_PATH="/model",
                PREDICTOR_DIR="/predictor",
                PYTHON_BIN=str(fake_python),
            )
            env.pop("PBKV_MODE", None)
            for mode in (None, "full", "lru", "lae", "he", "unknown"):
                with self.subTest(mode=mode):
                    if mode is not None:
                        env["PBKV_MODE"] = mode
                    result = subprocess.run(
                        ["bash", str(root / "scripts/pbkv/launch.sh")],
                        env=env,
                        capture_output=True,
                        text=True,
                    )
                    if mode in (None, "full"):
                        self.assertEqual(result.returncode, 0, result.stderr)
                        self.assertIn("--my-pred-chain-evict-enable", result.stdout)
                        self.assertIn("--my-prefetch-enable", result.stdout)
                    else:
                        self.assertEqual(result.returncode, 2)
                        self.assertIn("only full mode", result.stderr)


if __name__ == "__main__":
    unittest.main()
