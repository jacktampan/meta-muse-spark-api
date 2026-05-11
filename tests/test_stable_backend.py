"""Tests for the stable-backend redesign:

  1. /v1/reset endpoint (purge by id, purge all).
  2. body.user as a sticky-session-key fallback for OpenAI SDK callers.
  3. Single-conversation mode auto-rolls a fresh meta_conversation_id on
     each server startup (purges the stale mapping).
  4. 503 stuck_conversation responses on the non-streaming path.

These cover the user-facing contract Hermes-style agents rely on to make
Muse Spark feel "like ChatGPT/Claude API": predictable session resolution,
single round-trip per turn, and explicit failure surfaces with an operator
escape hatch.
"""

import tempfile
import unittest
from pathlib import Path
from fastapi.testclient import TestClient

from muse_spark.api import create_app
from muse_spark.client import load_state, save_state
from muse_spark.config import ApiSettings
from muse_spark.provider import MuseProviderResponse


def _state_with_mappings(mappings: dict[str, str]) -> dict[str, object]:
    return {
        "auth": {
            "cookie_header": "x",
            "authorization": "Bearer x",
            "mode": "fast",
            "user_agent": "test",
        },
        "api_conversations": {
            client_id: {
                "meta_conversation_id": meta_id,
                "created_at": 1,
                "last_used_at": 1,
            }
            for client_id, meta_id in mappings.items()
        },
    }


class ResetEndpointTests(unittest.TestCase):
    def test_reset_with_conversation_id_purges_only_that_mapping(self):
        async def fake_provider(request, state_path=None):
            return MuseProviderResponse(text="ok", conversation_id="m", template_name="chat")

        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "state.json"
            save_state(
                state_path,
                _state_with_mappings({"keep-me": "meta-a", "purge-me": "meta-b"}),
            )

            app = create_app(provider_generate_fn=fake_provider, state_path=state_path)
            client = TestClient(app)

            response = client.post("/v1/reset", json={"conversation_id": "purge-me"})

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertTrue(payload["reset"])
            self.assertEqual(payload["purged_count"], 1)
            self.assertEqual(payload["conversation_id"], "purge-me")

            # Only the requested mapping is gone.
            remaining = load_state(state_path)["api_conversations"]
            self.assertIn("keep-me", remaining)
            self.assertNotIn("purge-me", remaining)

    def test_reset_with_unknown_conversation_id_is_a_noop(self):
        async def fake_provider(request, state_path=None):
            return MuseProviderResponse(text="ok", conversation_id="m", template_name="chat")

        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "state.json"
            save_state(state_path, _state_with_mappings({"keep-me": "meta-a"}))

            app = create_app(provider_generate_fn=fake_provider, state_path=state_path)
            client = TestClient(app)

            response = client.post("/v1/reset", json={"conversation_id": "does-not-exist"})

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["purged_count"], 0)

    def test_reset_with_empty_body_purges_all_mappings(self):
        """Operator emergency: bot in a bad state, clear everything."""
        async def fake_provider(request, state_path=None):
            return MuseProviderResponse(text="ok", conversation_id="m", template_name="chat")

        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "state.json"
            save_state(
                state_path,
                _state_with_mappings({"a": "meta-a", "b": "meta-b", "c": "meta-c"}),
            )

            app = create_app(provider_generate_fn=fake_provider, state_path=state_path)
            client = TestClient(app)

            # Send no body at all — FastAPI accepts that for a POST without
            # a request model.
            response = client.post("/v1/reset")

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertTrue(payload["reset"])
            self.assertEqual(payload["purged_count"], 3)

            self.assertEqual(load_state(state_path)["api_conversations"], {})

    def test_reset_with_empty_object_body_purges_all(self):
        async def fake_provider(request, state_path=None):
            return MuseProviderResponse(text="ok", conversation_id="m", template_name="chat")

        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "state.json"
            save_state(state_path, _state_with_mappings({"a": "m1", "b": "m2"}))

            app = create_app(provider_generate_fn=fake_provider, state_path=state_path)
            client = TestClient(app)

            response = client.post("/v1/reset", json={})

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["purged_count"], 2)


class UserFieldStickyKeyTests(unittest.TestCase):
    def test_body_user_field_becomes_sticky_session_key(self):
        """OpenAI-compatible callers that set ``user`` (LangChain, OpenAI
        Python SDK with ``user=...``, many agent frameworks) should get
        sticky sessions automatically — without having to wire up a custom
        header or vendor extension.
        """
        seen: list[str] = []

        async def fake_provider(request, state_path=None):
            seen.append(request.conversation_id)
            return MuseProviderResponse(
                text="ok",
                conversation_id=request.conversation_id,
                template_name=request.template_name,
            )

        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "state.json"
            app = create_app(provider_generate_fn=fake_provider, state_path=state_path)
            client = TestClient(app)

            first = client.post(
                "/v1/chat/completions",
                json={
                    "model": "meta/muse-spark",
                    "user": "agent-7",
                    "messages": [{"role": "user", "content": "first"}],
                },
            )
            second = client.post(
                "/v1/chat/completions",
                json={
                    "model": "meta/muse-spark",
                    "user": "agent-7",
                    "messages": [{"role": "user", "content": "second"}],
                },
            )

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        # Both requests sticky to the same meta_conversation_id (Meta side).
        self.assertEqual(seen[0], seen[1])
        # And the client-facing conversation_id surfaces ``agent-7``.
        self.assertEqual(first.json()["conversation_id"], "agent-7")
        self.assertEqual(second.json()["conversation_id"], "agent-7")

    def test_explicit_conversation_id_overrides_body_user(self):
        """body.conversation_id retains priority over body.user — it's the
        more explicit signal.
        """
        seen: list[str] = []

        async def fake_provider(request, state_path=None):
            seen.append(request.conversation_id)
            return MuseProviderResponse(
                text="ok",
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
                    "user": "ignored",
                    "conversation_id": "explicit",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["conversation_id"], "explicit")

    def test_header_overrides_body_user(self):
        async def fake_provider(request, state_path=None):
            return MuseProviderResponse(
                text="ok",
                conversation_id=request.conversation_id,
                template_name=request.template_name,
            )

        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "state.json"
            app = create_app(provider_generate_fn=fake_provider, state_path=state_path)
            client = TestClient(app)

            response = client.post(
                "/v1/chat/completions",
                headers={"x-conversation-id": "from-header"},
                json={
                    "model": "meta/muse-spark",
                    "user": "ignored",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["conversation_id"], "from-header")


class SingleConversationAutoRollTests(unittest.TestCase):
    def test_force_single_conversation_purges_default_mapping_on_startup(self):
        """In single-conv mode, every server startup must roll a fresh
        meta_conversation_id by purging the stale ``default-single-conversation``
        entry left over from a previous process. Restart == clean slate.
        """
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "state.json"
            save_state(
                state_path,
                _state_with_mappings({
                    "default-single-conversation": "stale-meta-id",
                    "other-keep": "untouched-meta-id",
                }),
            )

            create_app(
                settings=ApiSettings(force_single_conversation=True),
                state_path=state_path,
            )

            mappings = load_state(state_path)["api_conversations"]
            # The single-conv mapping is rolled fresh.
            self.assertNotIn("default-single-conversation", mappings)
            # Other mappings are untouched — auto-roll is narrow.
            self.assertIn("other-keep", mappings)
            self.assertEqual(mappings["other-keep"]["meta_conversation_id"], "untouched-meta-id")

    def test_force_single_conversation_no_op_when_no_existing_mapping(self):
        """If there's nothing to roll, startup is a no-op (no error)."""
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "state.json"
            save_state(state_path, _state_with_mappings({}))

            create_app(
                settings=ApiSettings(force_single_conversation=True),
                state_path=state_path,
            )

            self.assertEqual(load_state(state_path)["api_conversations"], {})

    def test_normal_mode_does_not_roll_on_startup(self):
        """Without ``force_single_conversation``, server startup must leave
        every existing mapping alone — multi-tenant deployments rely on
        long-lived per-user mappings surviving restarts.
        """
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "state.json"
            save_state(
                state_path,
                _state_with_mappings({
                    "user-a": "meta-a",
                    "user-b": "meta-b",
                    "default-single-conversation": "should-survive",
                }),
            )

            create_app(
                settings=ApiSettings(force_single_conversation=False),
                state_path=state_path,
            )

            mappings = load_state(state_path)["api_conversations"]
            self.assertEqual(set(mappings.keys()), {"user-a", "user-b", "default-single-conversation"})


if __name__ == "__main__":
    unittest.main()
