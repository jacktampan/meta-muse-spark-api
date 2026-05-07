import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.error import HTTPError

from muse_spark.client import build_mode_switch_body, configure_auth, _graphql_request
from muse_spark.errors import MissingAuthError, ProviderProtocolError, ProviderTransportError, ReauthRequiredError
from muse_spark.provider import (
    MuseProviderRequest,
    generate_from_state,
    generate_from_state_async,
    load_provider_auth,
    purge_api_conversation,
    resolve_api_conversation,
    stream_from_state_async,
)


class MuseSparkProviderTests(unittest.TestCase):
    def test_generate_from_state_uses_ephemeral_conversation_and_returns_response(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "state.json"
            configure_auth(state_path, cookie_header="cookie=value", authorization="ecto1:abc")
            fake_generate = Mock(return_value="hello from muse")
            fake_warmup = Mock()
            fake_mode_switch = Mock()
            id_source = Mock(return_value="11111111-2222-3333-4444-555555555555")

            response = generate_from_state(
                MuseProviderRequest(prompt="hello"),
                state_path=state_path,
                generate_fn=fake_generate,
                warmup_fn=fake_warmup,
                mode_switch_fn=fake_mode_switch,
                conversation_id_factory=id_source,
            )

            self.assertEqual(response.text, "hello from muse")
            self.assertEqual(response.conversation_id, "11111111-2222-3333-4444-555555555555")
            fake_warmup.assert_called_once_with(
                "11111111-2222-3333-4444-555555555555",
                "cookie=value",
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36",
            )
            fake_mode_switch.assert_called_once()
            fake_generate.assert_called_once()

    def test_load_provider_auth_accepts_muse_spark_cookie_env_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "state.json"
            with patch.dict(
                os.environ,
                {"MUSE_SPARK_COOKIE": "cookie=value", "MUSE_SPARK_AUTHORIZATION": "ecto1:abc"},
                clear=False,
            ):
                auth = load_provider_auth(state_path)

            self.assertEqual(auth["cookie_header"], "cookie=value")
            self.assertEqual(auth["authorization"], "ecto1:abc")


    def test_generate_from_state_raises_missing_auth_error_without_auth(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "state.json"

            with self.assertRaises(MissingAuthError):
                generate_from_state(MuseProviderRequest(prompt="hello"), state_path=state_path)

    def test_generate_from_state_raises_protocol_error_on_empty_reply(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "state.json"
            configure_auth(state_path, cookie_header="cookie=value", authorization="ecto1:abc")

            with self.assertRaises(ProviderProtocolError):
                generate_from_state(
                    MuseProviderRequest(prompt="hello"),
                    state_path=state_path,
                    warmup_fn=Mock(),
                    mode_switch_fn=Mock(),
                    generate_fn=Mock(return_value=""),
                    conversation_id_factory=Mock(return_value="11111111-2222-3333-4444-555555555555"),
                )

    def test_graphql_request_maps_401_to_reauth_required(self):
        body = build_mode_switch_body("conversation-id", mode="think_fast")
        error = HTTPError(
            url="https://meta.ai/api/graphql",
            code=401,
            msg="Unauthorized",
            hdrs=None,
            fp=None,
        )

        with patch("urllib.request.urlopen", side_effect=error):
            with self.assertRaises(ReauthRequiredError):
                _graphql_request(body, cookie_header="cookie=value")

    def test_graphql_request_maps_non_auth_http_error_to_transport_error(self):
        body = build_mode_switch_body("conversation-id", mode="think_fast")
        error = HTTPError(
            url="https://meta.ai/api/graphql",
            code=500,
            msg="Server Error",
            hdrs=None,
            fp=None,
        )

        with patch("urllib.request.urlopen", side_effect=error):
            with self.assertRaises(ProviderTransportError):
                _graphql_request(body, cookie_header="cookie=value")

    def test_graphql_request_maps_invalid_json_success_to_protocol_error(self):
        body = build_mode_switch_body("conversation-id", mode="think_fast")
        fake_response = Mock()
        fake_response.read.return_value = b"not-json"
        context_manager = Mock()
        context_manager.__enter__ = Mock(return_value=fake_response)
        context_manager.__exit__ = Mock(return_value=False)

        with patch("urllib.request.urlopen", return_value=context_manager):
            with self.assertRaises(ProviderProtocolError):
                _graphql_request(body, cookie_header="cookie=value")

    def test_graphql_request_maps_graphql_errors_payload_to_protocol_error(self):
        body = build_mode_switch_body("conversation-id", mode="think_fast")
        fake_response = Mock()
        fake_response.read.return_value = b'{"errors":[{"message":"bad auth"}]}'
        context_manager = Mock()
        context_manager.__enter__ = Mock(return_value=fake_response)
        context_manager.__exit__ = Mock(return_value=False)

        with patch("urllib.request.urlopen", return_value=context_manager):
            with self.assertRaises(ProviderProtocolError):
                _graphql_request(body, cookie_header="cookie=value")


class MuseSparkAsyncProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_generate_from_state_async_returns_response(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "state.json"
            configure_auth(state_path, cookie_header="cookie=value", authorization="ecto1:abc")
            fake_warmup = Mock()
            fake_mode_switch = Mock()
            id_source = Mock(return_value="11111111-2222-3333-4444-555555555555")

            async def fake_generate(**kwargs):
                return "hello from muse"

            response = await generate_from_state_async(
                MuseProviderRequest(prompt="hello"),
                state_path=state_path,
                generate_fn=fake_generate,
                warmup_fn=fake_warmup,
                mode_switch_fn=fake_mode_switch,
                conversation_id_factory=id_source,
            )

            self.assertEqual(response.text, "hello from muse")
            self.assertEqual(response.conversation_id, "11111111-2222-3333-4444-555555555555")
            fake_warmup.assert_called_once_with(
                "11111111-2222-3333-4444-555555555555",
                "cookie=value",
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36",
            )
            fake_mode_switch.assert_called_once()

    async def test_stream_from_state_async_yields_incremental_chunks(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "state.json"
            configure_auth(state_path, cookie_header="cookie=value", authorization="ecto1:abc")
            fake_warmup = Mock()
            fake_mode_switch = Mock()
            id_source = Mock(return_value="11111111-2222-3333-4444-555555555555")

            async def fake_stream(**kwargs):
                for chunk in ["hello", " ", "world"]:
                    yield chunk

            chunks = []
            async for chunk in stream_from_state_async(
                MuseProviderRequest(prompt="hello"),
                state_path=state_path,
                stream_fn=fake_stream,
                warmup_fn=fake_warmup,
                mode_switch_fn=fake_mode_switch,
                conversation_id_factory=id_source,
            ):
                chunks.append(chunk)

            self.assertEqual(chunks, ["hello", " ", "world"])
            fake_warmup.assert_called_once_with(
                "11111111-2222-3333-4444-555555555555",
                "cookie=value",
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36",
            )
            fake_mode_switch.assert_called_once()

    def test_purge_api_conversation_removes_existing_mapping_and_subsequent_resolve_creates_fresh_meta_id(self):
        """``purge_api_conversation`` is the recovery primitive for stuck
        conversations: after a purge, ``resolve_api_conversation`` must
        re-create the mapping with a *new* meta conversation id rather than
        reusing the broken one. This is the contract the SSE recovery path
        relies on, so it gets a focused unit test in addition to the
        end-to-end api tests.
        """
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "state.json"
            configure_auth(state_path, cookie_header="cookie=v", authorization="x:y")

            initial = resolve_api_conversation(
                state_path=state_path,
                client_conversation_id="tg-42",
                meta_conversation_id_factory=lambda: "old-meta",
            )
            self.assertEqual(initial.meta_conversation_id, "old-meta")
            self.assertTrue(initial.is_new)

            # Resolving again before purge should reuse the cached mapping.
            cached = resolve_api_conversation(
                state_path=state_path,
                client_conversation_id="tg-42",
                meta_conversation_id_factory=lambda: "should-not-be-used",
            )
            self.assertEqual(cached.meta_conversation_id, "old-meta")
            self.assertFalse(cached.is_new)

            self.assertTrue(
                purge_api_conversation(state_path, "tg-42"),
                "Purge must report a removal when the mapping existed",
            )

            after_purge = resolve_api_conversation(
                state_path=state_path,
                client_conversation_id="tg-42",
                meta_conversation_id_factory=lambda: "new-meta",
            )
            self.assertEqual(after_purge.meta_conversation_id, "new-meta")
            self.assertTrue(after_purge.is_new)

            # Purging again should be a no-op and report False.
            self.assertTrue(purge_api_conversation(state_path, "tg-42"))
            self.assertFalse(
                purge_api_conversation(state_path, "tg-does-not-exist"),
                "Purge must report False when there's nothing to remove",
            )

    def test_purge_api_conversation_is_noop_for_falsy_id(self):
        """Defensive: passing ``None`` / empty string must not blow up or
        wipe unrelated state — the caller may pass through a missing
        client conversation id without checking it themselves.
        """
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "state.json"
            configure_auth(state_path, cookie_header="cookie=v", authorization="x:y")

            self.assertFalse(purge_api_conversation(state_path, None))
            self.assertFalse(purge_api_conversation(state_path, ""))