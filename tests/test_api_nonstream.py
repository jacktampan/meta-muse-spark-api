import tempfile
import unittest
from pathlib import Path
from typing import Optional
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

    def test_chat_completions_uses_stateful_bootstrap_then_current_user_turn(self):
        seen = []

        async def fake_provider(request, state_path=None):
            seen.append(request)
            return MuseProviderResponse(
                text="ok",
                conversation_id=request.conversation_id,
                template_name="chat",
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
            # In single conversation mode or persistent state, we expect second turn to be chat.
            # But the mock should have recorded it.
            self.assertEqual(seen[1].template_name, "chat")
            self.assertEqual(seen[0].conversation_id, seen[1].conversation_id)

    def test_provider_request_inherits_configured_receive_timeout(self):
        """The single per-turn provider call must honour
        ``ApiSettings.receive_timeout`` — otherwise users who raise the env
        var to handle slow networks still get the dataclass default.
        """
        from muse_spark.config import ApiSettings

        seen_timeouts: list[float] = []

        async def fake_provider(request, state_path=None):
            seen_timeouts.append(request.receive_timeout)
            return MuseProviderResponse(
                text="ok",
                conversation_id=request.conversation_id,
                template_name="chat",
            )

        settings = ApiSettings(receive_timeout=120.0)
        app = create_app(provider_generate_fn=fake_provider, settings=settings)
        client = TestClient(app)

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "meta/muse-spark",
                "messages": [{"role": "user", "content": "first turn"}],
            },
        )

        self.assertEqual(response.status_code, 200)
        # Exactly one provider call per turn now (bootstrap was removed).
        self.assertEqual(len(seen_timeouts), 1)
        self.assertEqual(seen_timeouts[0], 120.0)

    def test_provider_request_inherits_configured_first_byte_timeout(self):
        """``ApiSettings.first_byte_timeout`` must propagate to the single
        per-turn provider call so a stuck conversation surfaces inside the
        first-byte window rather than the longer between-byte window.
        """
        from muse_spark.config import ApiSettings

        seen: list[tuple[Optional[float], Optional[float]]] = []

        async def fake_provider(request, state_path=None):
            seen.append((request.receive_timeout, request.first_byte_timeout))
            return MuseProviderResponse(
                text="ok",
                conversation_id=request.conversation_id,
                template_name="chat",
            )

        settings = ApiSettings(receive_timeout=120.0, first_byte_timeout=15.0)
        app = create_app(provider_generate_fn=fake_provider, settings=settings)
        client = TestClient(app)

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "meta/muse-spark",
                "messages": [{"role": "user", "content": "first turn"}],
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(seen), 1)
        receive_t, first_byte_t = seen[0]
        self.assertEqual(receive_t, 120.0)
        self.assertEqual(first_byte_t, 15.0)

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

    def test_chat_completions_uses_x_conversation_id_header_when_body_lacks_one(self):
        seen = []

        async def fake_provider(request, state_path=None):
            seen.append(request)
            return MuseProviderResponse(
                text="ok",
                conversation_id=request.conversation_id,
                template_name="chat",
            )

        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "state.json"
            app = create_app(provider_generate_fn=fake_provider, state_path=state_path)
            client = TestClient(app)

            first = client.post(
                "/v1/chat/completions",
                headers={"X-Conversation-Id": "agent-session-42"},
                json={
                    "model": "meta/muse-spark",
                    "messages": [{"role": "user", "content": "first"}],
                },
            )
            second = client.post(
                "/v1/chat/completions",
                headers={"X-Conversation-Id": "agent-session-42"},
                json={
                    "model": "meta/muse-spark",
                    "messages": [{"role": "user", "content": "second"}],
                },
            )

            self.assertEqual(first.status_code, 200)
            self.assertEqual(second.status_code, 200)
            self.assertEqual(first.json()["conversation_id"], "agent-session-42")
            self.assertEqual(second.json()["conversation_id"], "agent-session-42")
            # One provider call per turn now (bootstrap removed). First
            # turn uses HOME template (fresh conv on Meta), second uses
            # CHAT (Meta already knows the conversation).
            self.assertEqual([req.template_name for req in seen], ["home", "chat"])

    def test_chat_completions_skips_warmup_for_followup_conversation(self):
        seen = []

        async def fake_provider(request, state_path=None):
            seen.append(request)
            return MuseProviderResponse(
                text="ok",
                conversation_id=request.conversation_id,
                template_name="chat",
            )

        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "state.json"
            app = create_app(provider_generate_fn=fake_provider, state_path=state_path)
            client = TestClient(app)

            client.post(
                "/v1/chat/completions",
                json={
                    "model": "meta/muse-spark",
                    "conversation_id": "conv-warmup",
                    "messages": [{"role": "user", "content": "first"}],
                },
            )
            client.post(
                "/v1/chat/completions",
                json={
                    "model": "meta/muse-spark",
                    "conversation_id": "conv-warmup",
                    "messages": [{"role": "user", "content": "second"}],
                },
            )

            # First turn warms the conversation (one provider call). Follow-up
            # turn skips warmup — Meta already knows the conversation, so the
            # warmup + mode_switch GraphQL round-trips are unnecessary.
            self.assertEqual(len(seen), 2)
            self.assertTrue(seen[0].needs_warmup)
            self.assertFalse(seen[1].needs_warmup)

    def test_chat_completions_strips_scaffolding_tags_from_response(self):
        async def fake_provider(request, state_path=None):
            return MuseProviderResponse(
                text="<conversation_turn><user_message>Hello, world!</user_message></conversation_turn>",
                conversation_id="meta-1",
                template_name="chat",
            )

        app = create_app(provider_generate_fn=fake_provider)
        client = TestClient(app)

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "meta/muse-spark",
                "messages": [{"role": "user", "content": "ping"}],
            },
        )

        self.assertEqual(response.status_code, 200)
        content = response.json()["choices"][0]["message"]["content"]
        self.assertEqual(content, "Hello, world!")

    def test_chat_completions_rejects_empty_messages_list_with_400(self):
        async def fake_provider(request, state_path=None):
            return MuseProviderResponse(
                text="should not be used",
                conversation_id="meta-1",
                template_name="home",
            )

        app = create_app(provider_generate_fn=fake_provider)
        client = TestClient(app)

        response = client.post(
            "/v1/chat/completions",
            json={"model": "meta/muse-spark", "messages": []},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["code"], "invalid_request")

    def test_chat_completions_sets_sticky_conversation_cookie(self):
        async def fake_provider(request, state_path=None):
            return MuseProviderResponse(
                text="ok",
                conversation_id=request.conversation_id,
                template_name="chat",
            )

        app = create_app(provider_generate_fn=fake_provider)
        client = TestClient(app)

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "meta/muse-spark",
                "messages": [{"role": "user", "content": "ping"}],
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("muse_spark_conv", response.cookies)

    def test_chat_completions_recovers_stuck_conversation_on_empty_response(self):
        """When Meta returns an empty response on an *existing* conversation
        — the symptom of the conv being wedged on Meta's backend after a
        prior stall — the endpoint should silently purge the stuck mapping
        and retry once on a fresh meta_conversation_id. Bootstrap was
        removed, so the retry adds no extra round-trip beyond the second
        main call.
        """
        from muse_spark.client import save_state
        from muse_spark.errors import ProviderEmptyResponseError

        attempts: list[str] = []

        async def fake_provider(request, state_path=None):
            attempts.append(request.conversation_id)
            if request.conversation_id == "stuck-meta-id":
                raise ProviderEmptyResponseError(
                    "Meta transport returned no usable text response."
                )
            return MuseProviderResponse(
                text="real reply",
                conversation_id=request.conversation_id,
                template_name=request.template_name,
            )

        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "state.json"
            save_state(
                state_path,
                {
                    "auth": {
                        "cookie_header": "x",
                        "authorization": "Bearer x",
                        "mode": "fast",
                        "user_agent": "test",
                    },
                    "api_conversations": {
                        "tg-99": {
                            "meta_conversation_id": "stuck-meta-id",
                            "created_at": 1,
                            "last_used_at": 1,
                        }
                    },
                },
            )

            app = create_app(
                provider_generate_fn=fake_provider,
                state_path=state_path,
            )
            client = TestClient(app)

            response = client.post(
                "/v1/chat/completions",
                headers={"x-conversation-id": "tg-99"},
                json={
                    "model": "meta/muse-spark",
                    "messages": [{"role": "user", "content": "ping"}],
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["choices"][0]["message"]["content"], "real reply")
        # Exactly two attempts: stuck conv + retry on fresh meta id.
        self.assertEqual(len(attempts), 2, f"Expected 2 attempts, got {attempts}")
        self.assertEqual(attempts[0], "stuck-meta-id")
        self.assertNotEqual(
            attempts[1], "stuck-meta-id",
            "Retry must use a fresh meta conversation id after purge",
        )

    def test_chat_completions_returns_503_when_retry_also_fails(self):
        """If the silent 1-shot retry *also* returns an empty response, the
        endpoint surfaces 503 ``stuck_conversation`` so the client knows the
        problem is unrecoverable transparently and can call ``/v1/reset``
        (or back off).
        """
        from muse_spark.client import save_state
        from muse_spark.errors import ProviderEmptyResponseError

        attempts: list[str] = []

        async def fake_provider(request, state_path=None):
            attempts.append(request.conversation_id)
            # Both the original and the retry empty out.
            raise ProviderEmptyResponseError(
                "Meta transport returned no usable text response."
            )

        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "state.json"
            save_state(
                state_path,
                {
                    "auth": {
                        "cookie_header": "x",
                        "authorization": "Bearer x",
                        "mode": "fast",
                        "user_agent": "test",
                    },
                    "api_conversations": {
                        "tg-503": {
                            "meta_conversation_id": "stuck-meta-id",
                            "created_at": 1,
                            "last_used_at": 1,
                        }
                    },
                },
            )

            app = create_app(
                provider_generate_fn=fake_provider,
                state_path=state_path,
            )
            client = TestClient(app)

            response = client.post(
                "/v1/chat/completions",
                headers={"x-conversation-id": "tg-503"},
                json={
                    "model": "meta/muse-spark",
                    "messages": [{"role": "user", "content": "ping"}],
                },
            )

        self.assertEqual(response.status_code, 503)
        payload = response.json()
        self.assertEqual(payload["error"]["code"], "stuck_conversation")
        self.assertEqual(payload["error"]["type"], "service_unavailable")
        # Confirms both attempts were made before surfacing 503.
        self.assertEqual(len(attempts), 2, f"Expected 2 attempts before 503, got {attempts}")

    def test_models_endpoint_lists_meta_muse_spark(self):
        app = create_app()
        client = TestClient(app)

        response = client.get("/v1/models")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["data"][0]["id"], "meta/muse-spark")
