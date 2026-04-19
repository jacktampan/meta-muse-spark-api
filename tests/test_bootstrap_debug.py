import tempfile
import unittest
from pathlib import Path
from fastapi.testclient import TestClient

from muse_spark.api import create_app
from muse_spark.provider import MuseProviderResponse


class BootstrapDebugTests(unittest.TestCase):
    def test_chat_completions_can_expose_bootstrap_response_when_requested(self):
        seen = []

        async def fake_provider(request, state_path=None):
            seen.append(request)
            if len(seen) == 1:
                return MuseProviderResponse(
                    text="bootstrap debug reply",
                    conversation_id=request.conversation_id,
                    template_name="home",
                )
            return MuseProviderResponse(
                text="final answer",
                conversation_id=request.conversation_id,
                template_name="chat",
            )

        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "state.json"
            app = create_app(provider_generate_fn=fake_provider, state_path=state_path)
            client = TestClient(app)

            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "meta/muse-spark",
                    "include_bootstrap_response": True,
                    "messages": [{"role": "user", "content": "first turn"}],
                },
            )

            payload = response.json()

            self.assertEqual(response.status_code, 200)
            self.assertEqual(payload["choices"][0]["message"]["content"], "final answer")
            self.assertEqual(payload["bootstrap_response"], "bootstrap debug reply")
            self.assertEqual(len(seen), 2)
            self.assertEqual(seen[0].template_name, "home")
            self.assertEqual(seen[1].template_name, "chat")

    def test_stream_chat_completions_can_expose_bootstrap_response_when_requested(self):
        async def fake_provider(request, state_path=None):
            return MuseProviderResponse(
                text="bootstrap debug reply" if request.template_name == "home" else "final answer",
                conversation_id=request.conversation_id,
                template_name=request.template_name,
            )

        async def fake_provider_stream(request, state_path=None):
            for chunk in ["final", " answer"]:
                yield chunk

        app = create_app(provider_generate_fn=fake_provider, provider_stream_fn=fake_provider_stream)
        client = TestClient(app)

        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "meta/muse-spark",
                "include_bootstrap_response": True,
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
        ) as response:
            body = b"".join(response.iter_bytes()).decode("utf-8")

        self.assertEqual(response.status_code, 200)
        self.assertIn('"bootstrap_response":"bootstrap debug reply"', body)
        self.assertIn('"content":"final"', body)
        self.assertIn('data: [DONE]', body)
