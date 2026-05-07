"""Tests for ``stream_text_deltas`` covering Fix #2 (stalled-stream
detection) and Fix #4 (resync on event divergence).

We mock ``websockets.connect`` to simulate Meta's WebSocket without going
out to the network. Fixtures emit raw bytes that ``extract_json_objects``
can recover, exactly as the real Meta gateway does.
"""
from __future__ import annotations

import asyncio
import json
import unittest
from typing import Any
from unittest.mock import patch

from muse_spark.client import stream_text_deltas
from muse_spark.errors import (
    ProviderEmptyResponseError,
    ProviderProtocolError,
    ProviderStallError,
)


def _frame(event: dict[str, Any]) -> bytes:
    """Wrap a JSON event in a small binary preamble like Meta's frames so
    ``extract_json_objects`` finds it via its byte scanner."""
    payload = json.dumps(event).encode("utf-8")
    # Random non-JSON header that the scanner skips past until it sees ``{``.
    return b"\x00\x01\x02ECTO" + payload + b"\x00"


class _FakeWebSocket:
    def __init__(self, recv_script: list[Any]) -> None:
        # Each item is either ``bytes`` (returned from recv) or the special
        # marker ``"STALL"`` (recv blocks forever to trigger timeout).
        self._script = list(recv_script)
        self.sent: list[bytes] = []

    async def send(self, data: bytes) -> None:
        self.sent.append(data)

    async def recv(self) -> bytes:
        if not self._script:
            await asyncio.sleep(60)  # block forever, will be cancelled
            raise AssertionError("unreachable")
        item = self._script.pop(0)
        if item == "STALL":
            await asyncio.sleep(60)
            raise AssertionError("unreachable")
        return item


class _FakeWebSocketContext:
    def __init__(self, ws: _FakeWebSocket) -> None:
        self._ws = ws

    async def __aenter__(self) -> _FakeWebSocket:
        return self._ws

    async def __aexit__(self, *exc_info: Any) -> None:
        return None


def _patched_connect(ws: _FakeWebSocket):
    def _connect(uri: str, additional_headers: Any = None) -> _FakeWebSocketContext:
        return _FakeWebSocketContext(ws)

    return _connect


async def _collect(prompt: str, ws: _FakeWebSocket, timeout: float = 0.05) -> list[str]:
    """Drive ``stream_text_deltas`` against ``ws`` and return the deltas."""
    chunks: list[str] = []
    with patch("websockets.connect", _patched_connect(ws)):
        async for chunk in stream_text_deltas(
            prompt=prompt,
            conversation_id="00000000-0000-0000-0000-000000000000",
            authorization="Bearer fake",
            cookie_header="cookie=fake",
            receive_timeout=timeout,
        ):
            chunks.append(chunk)
    return chunks


class StreamTextDeltasTests(unittest.TestCase):
    def test_stalled_after_yield_raises_stall_error(self):
        """Mid-response stall raises ``ProviderStallError`` (a subclass of
        ``ProviderProtocolError``). The dedicated subclass lets the SSE
        pipeline surface partial output with ``finish_reason="length"``
        instead of treating the request as a hard error.
        """
        ws = _FakeWebSocket(
            recv_script=[
                _frame({
                    "type": "patch",
                    "operations": [
                        {
                            "op": "delta",
                            "path": "/sections/0/view_model/primitive/text",
                            "value": "Hello",
                        }
                    ],
                }),
                "STALL",
            ]
        )
        with self.assertRaises(ProviderStallError) as ctx:
            asyncio.run(_collect("hi", ws, timeout=0.05))
        # Subclass relationship is part of the contract — existing handlers
        # that catch ``ProviderProtocolError`` keep working.
        self.assertIsInstance(ctx.exception, ProviderProtocolError)
        self.assertIn("stalled", str(ctx.exception).lower())

    def test_completion_signal_exits_loop_promptly(self):
        """When Meta sends an explicit completion signal we must stop right
        away — no waiting for the recv() idle timeout."""
        ws = _FakeWebSocket(
            recv_script=[
                _frame({
                    "type": "patch",
                    "operations": [
                        {
                            "op": "delta",
                            "path": "/sections/0/view_model/primitive/text",
                            "value": "done",
                        },
                        {
                            "op": "replace",
                            "path": "/sections/0/state",
                            "value": "COMPLETE",
                        },
                    ],
                }),
                "STALL",  # would trigger the stall-error if we didn't exit early
            ]
        )
        chunks = asyncio.run(_collect("hi", ws, timeout=0.05))
        self.assertEqual(chunks, ["done"])

    def test_first_byte_timeout_fires_before_receive_timeout_when_no_data(self):
        """When Meta is silent before the first patch — the typical
        stuck-conversation symptom — ``stream_text_deltas`` must give up
        after ``first_byte_timeout`` instead of waiting the full
        ``receive_timeout``. The SSE recovery path keys off the resulting
        ``ProviderEmptyResponseError`` to purge + retry, so faster detection
        directly translates into shorter user-facing latency.
        """
        ws = _FakeWebSocket(recv_script=["STALL"])

        async def _run() -> None:
            with patch("websockets.connect", _patched_connect(ws)):
                start = asyncio.get_event_loop().time()
                with self.assertRaises(ProviderEmptyResponseError):
                    async for _ in stream_text_deltas(
                        prompt="hi",
                        conversation_id="00000000-0000-0000-0000-000000000000",
                        authorization="Bearer fake",
                        cookie_header="cookie=fake",
                        receive_timeout=10.0,        # would dominate without the optimisation
                        first_byte_timeout=0.05,     # tight pre-first-byte ceiling
                    ):
                        pass
                elapsed = asyncio.get_event_loop().time() - start
                # Comfortably below ``receive_timeout`` (10s) — sanity check
                # that the smaller window actually drove the deadline.
                self.assertLess(elapsed, 1.0)

        asyncio.run(_run())

    def test_first_byte_timeout_does_not_apply_after_first_token(self):
        """Once Meta has streamed at least one token, between-tokens silence
        is *generation* slowness — not a stuck-conversation symptom — and the
        full ``receive_timeout`` window must apply so we surface a graceful
        ``ProviderStallError`` (→ ``finish_reason="length"``) instead of
        misclassifying it as an empty-response failure.
        """
        ws = _FakeWebSocket(
            recv_script=[
                _frame({
                    "type": "patch",
                    "operations": [
                        {
                            "op": "delta",
                            "path": "/sections/0/view_model/primitive/text",
                            "value": "Hi",
                        }
                    ],
                }),
                "STALL",  # mid-response stall — should raise stall error after
                          # ``receive_timeout`` (NOT after ``first_byte_timeout``).
            ]
        )

        async def _run() -> None:
            with patch("websockets.connect", _patched_connect(ws)):
                with self.assertRaises(ProviderStallError) as ctx:
                    async for _ in stream_text_deltas(
                        prompt="hi",
                        conversation_id="00000000-0000-0000-0000-000000000000",
                        authorization="Bearer fake",
                        cookie_header="cookie=fake",
                        # ``receive_timeout`` is what should fire here;
                        # ``first_byte_timeout`` is irrelevant once we yielded.
                        receive_timeout=0.05,
                        first_byte_timeout=0.01,
                    ):
                        pass
                # Error message must reflect the stall window, not the tighter
                # first-byte window — this guards against a regression where
                # the wrong timeout value bleeds into the error string.
                msg = str(ctx.exception).lower()
                self.assertIn("stalled mid-response", msg)
                self.assertNotIn("0.01s", msg)
                self.assertNotIn("0.0s", msg)

        asyncio.run(_run())

    def test_first_byte_timeout_is_disabled_when_zero_or_above_receive(self):
        """Caller can opt out by passing ``first_byte_timeout`` <= 0 or >=
        ``receive_timeout``. In both cases, the existing single-timeout
        behaviour must be preserved."""
        for opt_out_value in (0.0, 10.0):
            ws = _FakeWebSocket(recv_script=["STALL"])

            async def _run(value: float = opt_out_value) -> None:
                with patch("websockets.connect", _patched_connect(ws)):
                    start = asyncio.get_event_loop().time()
                    with self.assertRaises(ProviderEmptyResponseError):
                        async for _ in stream_text_deltas(
                            prompt="hi",
                            conversation_id="00000000-0000-0000-0000-000000000000",
                            authorization="Bearer fake",
                            cookie_header="cookie=fake",
                            receive_timeout=0.05,
                            first_byte_timeout=value,
                        ):
                            pass
                    # ``receive_timeout`` (0.05s) drove the deadline — neither
                    # 0 nor a value at/above receive_timeout shortened it.
                    elapsed = asyncio.get_event_loop().time() - start
                    self.assertGreaterEqual(elapsed, 0.04)

            asyncio.run(_run())

    def test_full_event_resyncs_divergent_text(self):
        """If Meta corrects an earlier token via a ``full`` event, we must
        emit the divergent tail rather than dropping it (which the previous
        heuristic did, leaving clients with stale text)."""
        ws = _FakeWebSocket(
            recv_script=[
                _frame({
                    "type": "patch",
                    "operations": [
                        {
                            "op": "delta",
                            "path": "/sections/0/view_model/primitive/text",
                            "value": "Hello, world!",
                        }
                    ],
                }),
                _frame({
                    "type": "full",
                    "response": {
                        "sections": [
                            {
                                "view_model": {
                                    "primitive": {"text": "Hello, World!"},
                                }
                            }
                        ],
                    },
                }),
                _frame({
                    "type": "patch",
                    "operations": [
                        {
                            "op": "replace",
                            "path": "/sections/0/state",
                            "value": "COMPLETE",
                        }
                    ],
                }),
            ]
        )
        chunks = asyncio.run(_collect("hi", ws, timeout=0.5))
        # First chunk is the streamed "Hello, world!". The full-event
        # diverges at the W/w; we expect the divergent tail to be emitted
        # so the joined stream ends with the corrected text.
        joined_after_resync = "Hello, world!" + "".join(chunks[1:])
        self.assertTrue(joined_after_resync.endswith("World!"))


if __name__ == "__main__":
    unittest.main()
