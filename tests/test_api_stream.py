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

    def test_chat_completions_stream_handles_error_gracefully(self):
        from muse_spark.errors import ProviderProtocolError

        async def fake_provider_stream(request, state_path=None):
            yield "initial"
            raise ProviderProtocolError("Meta transport returned no usable text response.")

        def fake_load_auth(state_path=None):
            return {"cookie_header": "fake", "authorization": "fake", "mode": "fake", "user_agent": "fake"}

        import muse_spark.api
        from unittest.mock import MagicMock
        from muse_spark.provider import ResolvedConversation

        muse_spark.api.resolve_api_conversation = MagicMock(return_value=ResolvedConversation(
            client_conversation_id="fake-client-id",
            meta_conversation_id="fake-meta-id",
            template_name="home"
        ))

        async def fake_provider_generate(request, state_path=None):
            from muse_spark.provider import MuseProviderResponse
            return MuseProviderResponse(text="fake", conversation_id="fake-meta-id", template_name="home")

        app = create_app(
            provider_stream_fn=fake_provider_stream,
            load_auth_fn=fake_load_auth,
            provider_generate_fn=fake_provider_generate
        )
        client = TestClient(app, raise_server_exceptions=False)

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
        self.assertIn('"content":"initial"', body)
        # Should NOT contain DONE because it failed before reaching it
        self.assertNotIn('data: [DONE]', body)
