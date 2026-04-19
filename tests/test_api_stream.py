import unittest
from fastapi.testclient import TestClient

from muse_spark.api import create_app
from muse_spark.provider import MuseProviderResponse


class ApiStreamingTests(unittest.TestCase):
    def test_chat_completions_stream_returns_incremental_sse_chunks_and_done(self):
        async def fake_provider(request, state_path=None):
            return MuseProviderResponse(
                text="should not be used when streaming is incremental",
                conversation_id="conv-123",
                template_name="home",
            )

        async def fake_provider_stream(request, state_path=None):
            for chunk in ["hello", " ", "world", " from muse spark"]:
                yield chunk

        app = create_app(provider_generate_fn=fake_provider, provider_stream_fn=fake_provider_stream)
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
        self.assertIn('"object":"chat.completion.chunk"', body)
        self.assertIn('"role":"assistant"', body)
        self.assertIn('"content":"hello"', body)
        self.assertIn('"content":"world"', body)
        self.assertIn('"content":" from muse spark"', body)
        self.assertIn('data: [DONE]', body)
