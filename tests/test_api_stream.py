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
        # On stream failure we must still emit a terminal chunk + [DONE] so
        # OpenAI SDK clients don't hang waiting for the stream to close.
        # The terminal chunk carries finish_reason="error" so callers can
        # detect that output is partial.
        self.assertIn('"finish_reason":"error"', body)
        self.assertIn('data: [DONE]', body)

    def test_chat_completions_stream_strips_scaffolding_tags_across_chunks(self):
        async def fake_provider_stream(request, state_path=None):
            # Tag split across chunks must still be removed.
            for chunk in ["Hello", " <user_mes", "sage>world</user_message> done"]:
                yield chunk

        async def fake_provider_generate(request, state_path=None):
            return MuseProviderResponse(text="fake", conversation_id="m", template_name="home")

        app = create_app(
            provider_generate_fn=fake_provider_generate,
            provider_stream_fn=fake_provider_stream,
        )
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
        # Concatenate all delta.content fragments and verify scaffolding gone.
        import json
        contents = []
        for line in body.split("\n"):
            line = line.strip()
            if not line.startswith("data: ") or line.endswith("[DONE]"):
                continue
            event = json.loads(line[len("data: "):])
            delta = event["choices"][0].get("delta", {})
            if "content" in delta:
                contents.append(delta["content"])
        joined = "".join(contents)
        self.assertNotIn("<user_message", joined)
        self.assertNotIn("</user_message", joined)
        self.assertIn("Hello", joined)
        self.assertIn("world", joined)
        self.assertIn("done", joined)
        self.assertIn('data: [DONE]', body)

    def test_chat_completions_stream_marks_stall_as_length_truncation(self):
        """A mid-response stall (``ProviderStallError``) must surface partial
        output with ``finish_reason="length"`` rather than ``"error"`` —
        OpenAI-compatible clients then treat it as graceful truncation
        instead of failing the request.
        """
        from muse_spark.errors import ProviderStallError

        async def fake_provider_stream(request, state_path=None):
            yield "Halo, ini jawaban yang"
            raise ProviderStallError(
                "Meta stream stalled mid-response: no data received within 60.0s."
            )

        async def fake_provider_generate(request, state_path=None):
            return MuseProviderResponse(text="fake", conversation_id="m", template_name="home")

        app = create_app(
            provider_generate_fn=fake_provider_generate,
            provider_stream_fn=fake_provider_stream,
        )
        client = TestClient(app, raise_server_exceptions=False)

        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "meta/muse-spark",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        ) as response:
            body = b"".join(response.iter_bytes()).decode("utf-8")

        self.assertEqual(response.status_code, 200)
        # Partial output must reach the client.
        self.assertIn('"content":"Halo, ini jawaban yang"', body)
        # Truncation marker, NOT error marker.
        self.assertIn('"finish_reason":"length"', body)
        self.assertNotIn('"finish_reason":"error"', body)
        self.assertIn('data: [DONE]', body)

    def test_chat_completions_stream_flushes_buffered_tail_on_stall(self):
        """The scaffolding stripper holds back content behind a potential
        ``<`` or ``{{`` marker. When a stall happens while content is held,
        that buffered text must still reach the client — otherwise we'd
        silently drop legitimate tokens while claiming graceful truncation.
        """
        from muse_spark.errors import ProviderStallError

        async def fake_provider_stream(request, state_path=None):
            # First chunk emits cleanly. Second chunk ends with an opener
            # ``<`` that the stripper holds (could be a partial scaffolding
            # tag). Third would close the tag — but the stall happens first.
            yield "Hasil kemarin: "
            yield "tim menang besar dengan skor <"
            raise ProviderStallError("stalled before tag closer arrived")

        async def fake_provider_generate(request, state_path=None):
            return MuseProviderResponse(text="fake", conversation_id="m", template_name="home")

        app = create_app(
            provider_generate_fn=fake_provider_generate,
            provider_stream_fn=fake_provider_stream,
        )
        client = TestClient(app, raise_server_exceptions=False)

        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "meta/muse-spark",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        ) as response:
            body = b"".join(response.iter_bytes()).decode("utf-8")

        self.assertEqual(response.status_code, 200)
        # The text up to the opener must stream as content chunks.
        self.assertIn("Hasil kemarin", body)
        self.assertIn("tim menang besar", body)
        # And — critically — the held tail (``... dengan skor <``) must also
        # be flushed; before this fix the stripper buffer was silently
        # dropped on stall, hiding the last few tokens from the client.
        self.assertIn("dengan skor", body)
        self.assertIn('"finish_reason":"length"', body)

    def test_chat_completions_stream_sets_sticky_conversation_cookie(self):
        async def fake_provider_stream(request, state_path=None):
            yield "ok"

        async def fake_provider_generate(request, state_path=None):
            return MuseProviderResponse(text="fake", conversation_id="m", template_name="home")

        app = create_app(
            provider_generate_fn=fake_provider_generate,
            provider_stream_fn=fake_provider_stream,
        )
        client = TestClient(app)

        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "meta/muse-spark",
                "messages": [{"role": "user", "content": "ping"}],
                "stream": True,
            },
        ) as response:
            # Drain so the response completes and cookies become available.
            b"".join(response.iter_bytes())
            self.assertEqual(response.status_code, 200)
            self.assertIn("muse_spark_conv", response.cookies)
