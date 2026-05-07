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
from muse_spark.errors import ProviderProtocolError, ProviderStallError


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
