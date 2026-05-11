"""Verifies that the bootstrap round-trip has been removed.

Previously the API made *two* sequential provider calls on the first turn of
a fresh conversation: a hidden ``bootstrap`` priming call followed by the
user's actual prompt. That round-trip is gone — any system/developer
messages are now folded into a single combined prompt and sent in one
provider call. These tests guard against accidental re-introduction.
"""

import tempfile
import unittest
from pathlib import Path
from fastapi.testclient import TestClient

from muse_spark.api import create_app
from muse_spark.provider import MuseProviderResponse


class NoBootstrapRoundtripTests(unittest.TestCase):
    def test_chat_completions_does_not_make_separate_bootstrap_call(self):
        """First turn of a new conversation must use exactly *one* provider
        call. The legacy bootstrap+main pair would have produced two
        invocations; the lean design folds system instructions into the
        single user-turn prompt.
        """
        seen = []

        async def fake_provider(request, state_path=None):
            seen.append(request)
            return MuseProviderResponse(
                text="hello",
                conversation_id=request.conversation_id,
                template_name=request.template_name,
            )

        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "state.json"
            app = create_app(provider_generate_fn=fake_provider, state_path=state_path)
            client = TestClient(app)

            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "meta/muse-spark",
                    "messages": [
                        {"role": "system", "content": "Be terse."},
                        {"role": "user", "content": "first turn"},
                    ],
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(seen), 1, f"Expected exactly 1 provider call, got {seen}")
        # Critically: system instructions still reach Meta — they're just
        # inlined into the same prompt rather than sent as a separate turn.
        self.assertIn("Be terse.", seen[0].prompt)
        self.assertIn("first turn", seen[0].prompt)
        # And the legacy bootstrap_response field never appears on responses.
        self.assertNotIn("bootstrap_response", response.json())

    def test_stream_chat_completions_does_not_emit_bootstrap_response_field(self):
        """Streaming counterpart: SSE chunks must not carry the legacy
        ``bootstrap_response`` field. Clients that previously opted into it
        now just get the same chunks as everyone else.
        """
        async def fake_provider_stream(request, state_path=None):
            for chunk in ["final", " answer"]:
                yield chunk

        async def fake_provider_generate(request, state_path=None):
            return MuseProviderResponse(
                text="unused",
                conversation_id=request.conversation_id,
                template_name=request.template_name,
            )

        app = create_app(provider_generate_fn=fake_provider_generate, provider_stream_fn=fake_provider_stream)
        client = TestClient(app)

        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "meta/muse-spark",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
        ) as response:
            body = b"".join(response.iter_bytes()).decode("utf-8")

        self.assertEqual(response.status_code, 200)
        self.assertNotIn("bootstrap_response", body)
        self.assertIn('"content":"final"', body)
        self.assertIn("data: [DONE]", body)
