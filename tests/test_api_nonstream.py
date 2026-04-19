import tempfile
import unittest
from pathlib import Path
from fastapi.testclient import TestClient

from muse_spark.api import create_app
from muse_spark.errors import MissingAuthError, ReauthRequiredError
from muse_spark.provider import MuseProviderResponse


class ApiNonStreamingTests(unittest.TestCase):
    def test_chat_completions_returns_openai_compatible_shape(self):
        async def fake_provider(request, state_path=None):
            return MuseProviderResponse(
                text="refactored component code",
                conversation_id="meta-conv-123",
                template_name="home",
            )

        app = create_app(provider_generate_fn=fake_provider)
        client = TestClient(app)

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "meta/muse-spark",
                "messages": [{"role": "user", "content": "Refactor this React component."}],
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["object"], "chat.completion")
        self.assertEqual(payload["model"], "meta/muse-spark")
        self.assertEqual(payload["choices"][0]["message"]["role"], "assistant")
        self.assertEqual(payload["choices"][0]["message"]["content"], "refactored component code")
        self.assertEqual(payload["choices"][0]["finish_reason"], "stop")
        self.assertIn("conversation_id", payload)

    def test_chat_completions_reuses_conversation_id_mapping_across_turns(self):
        seen = []

        async def fake_provider(request, state_path=None):
            seen.append(request)
            return MuseProviderResponse(
                text="ok",
                conversation_id=request.conversation_id,
                template_name="chat" if len(seen) > 1 else "home",
            )

        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "state.json"
            app = create_app(provider_generate_fn=fake_provider, state_path=state_path)
            client = TestClient(app)

            first = client.post(
                "/v1/chat/completions",
                json={
                    "model": "meta/muse-spark",
                    "messages": [{"role": "user", "content": "first turn"}],
                },
            )
            first_payload = first.json()
            conversation_id = first_payload["conversation_id"]

            second = client.post(
                "/v1/chat/completions",
                json={
                    "model": "meta/muse-spark",
                    "conversation_id": conversation_id,
                    "messages": [{"role": "user", "content": "second turn"}],
                },
            )
            second_payload = second.json()

            self.assertEqual(first.status_code, 200)
            self.assertEqual(second.status_code, 200)
            self.assertEqual(second_payload["conversation_id"], conversation_id)
            self.assertEqual(seen[0].template_name, "home")
            self.assertEqual(seen[1].template_name, "chat")
            self.assertEqual(seen[0].conversation_id, seen[1].conversation_id)

    def test_chat_completions_rejects_unknown_model(self):
        async def fake_provider(request, state_path=None):
            return MuseProviderResponse(
                text="should not be used",
                conversation_id="conv-123",
                template_name="home",
            )

        app = create_app(provider_generate_fn=fake_provider)
        client = TestClient(app)

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-not-real",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

        self.assertEqual(response.status_code, 400)
        payload = response.json()
        self.assertEqual(payload["error"]["code"], "invalid_model")

    def test_chat_completions_maps_reauth_required(self):
        async def fake_provider(request, state_path=None):
            raise ReauthRequiredError("refresh auth")

        app = create_app(provider_generate_fn=fake_provider)
        client = TestClient(app)

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "meta/muse-spark",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

        self.assertEqual(response.status_code, 401)
        payload = response.json()
        self.assertEqual(payload["error"]["code"], "reauth_required")

    def test_readyz_returns_503_when_auth_missing(self):
        def fake_load_auth(state_path=None):
            raise MissingAuthError("missing auth")

        app = create_app(load_auth_fn=fake_load_auth)
        client = TestClient(app)

        response = client.get("/readyz")

        self.assertEqual(response.status_code, 503)
        payload = response.json()
        self.assertEqual(payload["error"]["code"], "missing_auth")

    def test_models_endpoint_lists_meta_muse_spark(self):
        app = create_app()
        client = TestClient(app)

        response = client.get("/v1/models")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["data"][0]["id"], "meta/muse-spark")
