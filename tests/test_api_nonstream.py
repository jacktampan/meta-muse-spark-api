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

    def test_bootstrap_request_inherits_configured_receive_timeout(self):
        """Bootstrap call must honour ``ApiSettings.receive_timeout`` —
        otherwise users who raise the env var to handle slow networks still
        get the dataclass default during the bootstrap phase, causing
        spurious bootstrap failures while the main call uses the configured
        value.
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
        # Both bootstrap and main call must see the configured timeout, not
        # the dataclass default.
        self.assertEqual(len(seen_timeouts), 2)
        self.assertTrue(
            all(t == 120.0 for t in seen_timeouts),
            f"Expected all calls to honour configured receive_timeout=120.0, got {seen_timeouts}",
        )

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
            # First turn boots up: bootstrap (home) + main (chat). Second turn
            # is a follow-up: chat only, no bootstrap.
            self.assertEqual([req.template_name for req in seen], ["home", "chat", "chat"])

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

            # First turn: bootstrap warms the conversation, main call skips warmup.
            self.assertTrue(seen[0].needs_warmup)
            self.assertFalse(seen[1].needs_warmup)
            # Follow-up turn: no warmup at all (conversation already exists).
            self.assertFalse(seen[2].needs_warmup)

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
        """Non-streaming counterpart to the SSE recovery test: when Meta
        returns an empty response on an existing conversation, the endpoint
        should purge the stuck mapping and retry once with a fresh meta id.

        Bookkeeping: an existing conversation hits ``CHAT`` template (no
        bootstrap call), so the first call is the main request that fails.
        The retry purges and re-resolves to a brand-new conversation, which
        forces template ``HOME`` and therefore an additional bootstrap
        call before the second main call. Total expected provider calls:
        1. main on stuck conv → empty response → triggers recovery
        2. bootstrap on fresh conv (warmup)
        3. main on fresh conv → real reply
        """
        from muse_spark.client import save_state
        from muse_spark.errors import ProviderEmptyResponseError

        main_attempts: list[str] = []
        bootstrap_attempts: list[str] = []

        async def fake_provider(request, state_path=None):
            # Distinguish bootstrap from main by template_name. Bootstrap
            # always runs on the HOME template; the main user turn switches
            # to CHAT after the bootstrap call.
            if request.template_name == "home":
                bootstrap_attempts.append(request.conversation_id)
                return MuseProviderResponse(
                    text="bootstrap-ack",
                    conversation_id=request.conversation_id,
                    template_name="home",
                )
            main_attempts.append(request.conversation_id)
            if len(main_attempts) == 1:
                raise ProviderEmptyResponseError(
                    "Meta transport returned no usable text response."
                )
            return MuseProviderResponse(
                text="real reply",
                conversation_id=request.conversation_id,
                template_name="chat",
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
        # Two main calls: original (stuck) + retry on fresh conv.
        self.assertEqual(
            len(main_attempts), 2,
            f"Expected 2 main attempts, got {main_attempts}",
        )
        self.assertEqual(main_attempts[0], "stuck-meta-id")
        self.assertNotEqual(
            main_attempts[1], "stuck-meta-id",
            "Retry must use a fresh meta conversation id after purge",
        )
        # Bootstrap fires exactly once — for the recovery conversation.
        self.assertEqual(
            len(bootstrap_attempts), 1,
            f"Expected 1 bootstrap call, got {bootstrap_attempts}",
        )
        self.assertEqual(bootstrap_attempts[0], main_attempts[1])

    def test_chat_completions_recovery_propagates_bootstrap_response_text(self):
        """When recovery fires for a stuck conversation, the recovery path
        re-bootstraps from a fresh meta conversation (HOME template). If the
        client requested ``include_bootstrap_response=true``, that fresh
        bootstrap text must reach the response — discarding it silently
        breaks the ``bootstrap_response`` field for any caller that relies
        on it (regression for Devin Review BUG_pr-review-job-…_0001).
        """
        from muse_spark.client import save_state
        from muse_spark.errors import ProviderEmptyResponseError

        async def fake_provider(request, state_path=None):
            # Bootstrap call → return a recognisable greeting. Main user
            # call → first invocation fails empty (stuck conv); second
            # invocation (after recovery) succeeds.
            if request.template_name == "home":
                return MuseProviderResponse(
                    text="HELLO FROM RECOVERED BOOTSTRAP",
                    conversation_id=request.conversation_id,
                    template_name="home",
                )
            if request.conversation_id == "stuck-meta-id":
                raise ProviderEmptyResponseError(
                    "Meta transport returned no usable text response."
                )
            return MuseProviderResponse(
                text="real reply",
                conversation_id=request.conversation_id,
                template_name="chat",
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
                        "tg-100": {
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
                headers={"x-conversation-id": "tg-100"},
                json={
                    "model": "meta/muse-spark",
                    "messages": [{"role": "user", "content": "ping"}],
                    "include_bootstrap_response": True,
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["choices"][0]["message"]["content"], "real reply")
        # The recovery's fresh bootstrap text must appear in the response.
        # Without the fix, ``bootstrap_response`` would be ``None``/missing
        # because the original attempt was on an existing conv with no
        # bootstrap, and the retry's bootstrap text was being discarded.
        self.assertEqual(payload.get("bootstrap_response"), "HELLO FROM RECOVERED BOOTSTRAP")

    def test_models_endpoint_lists_meta_muse_spark(self):
        app = create_app()
        client = TestClient(app)

        response = client.get("/v1/models")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["data"][0]["id"], "meta/muse-spark")
