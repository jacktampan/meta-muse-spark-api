import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from muse_spark.client import (
    INTRO_FRAME_TYPE,
    PROMPT_FRAME_TYPE,
    PROMPT_FRAME_FLAG,
    build_intro_frame,
    build_mode_switch_body,
    build_prompt_frame,
    configure_auth,
    current_conversation,
    decode_prompt_frame,
    extract_json_objects,
    list_conversations,
    main,
    merge_response_events,
    new_conversation,
    send_chat_message,
    use_conversation,
)


class MuseSparkClientTests(unittest.TestCase):
    def test_build_intro_frame_injects_conversation_id(self):
        conversation_id = "11111111-2222-3333-4444-555555555555"

        frame = build_intro_frame(conversation_id)

        self.assertEqual(frame[0], INTRO_FRAME_TYPE)
        # bytes 1-2: sub-session idx (LE), bytes 3-5: payload length (LE u24)
        sub_session = int.from_bytes(frame[1:3], "little")
        body_len = int.from_bytes(frame[3:6], "little")
        self.assertEqual(sub_session, 0)
        self.assertEqual(body_len, len(frame) - 6)
        payload = json.loads(frame[6:].decode("utf-8"))
        self.assertEqual(payload["x-dgw-app-x-ecto-conversation-id"], conversation_id)
        self.assertEqual(payload["x-dgw-app-client-payload-type"], "PROTO_INSIDE_JSON")

    def test_build_prompt_frame_updates_prompt_and_ids(self):
        prompt = "Muse Spark, show me what you've got"
        conversation_id = "11111111-2222-3333-4444-555555555555"
        request_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        user_message_id = "ffffffff-1111-2222-3333-444444444444"
        submitted_ms = 1777000000123
        unique_message_id = 7450273077659807999

        frame = build_prompt_frame(
            prompt=prompt,
            conversation_id=conversation_id,
            request_id=request_id,
            user_message_id=user_message_id,
            submitted_ms=submitted_ms,
            unique_message_id=unique_message_id,
        )
        decoded = decode_prompt_frame(frame)

        self.assertEqual(decoded["outer_request_id"], request_id)
        self.assertEqual(decoded["request_id"], request_id)
        self.assertEqual(decoded["user_message_id"], user_message_id)
        self.assertEqual(decoded["conversation_id"], conversation_id)
        self.assertEqual(decoded["prompt"], prompt)
        self.assertEqual(decoded["submitted_ms"], submitted_ms)
        self.assertEqual(decoded["unique_message_id"], unique_message_id)

    def test_build_chat_prompt_frame_updates_prompt_and_ids(self):
        frame = build_prompt_frame(
            prompt="follow up probe",
            conversation_id="11111111-2222-3333-4444-555555555555",
            request_id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
            template_name="chat",
        )

        decoded = decode_prompt_frame(frame)

        self.assertEqual(decoded["prompt"], "follow up probe")
        self.assertEqual(decoded["conversation_id"], "11111111-2222-3333-4444-555555555555")

    def test_build_prompt_frame_has_dgw_message_framing(self):
        frame = build_prompt_frame(
            prompt="hi",
            conversation_id="11111111-2222-3333-4444-555555555555",
            request_id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
            sub_session_idx=1,
            message_seq=2,
        )

        self.assertEqual(frame[0], PROMPT_FRAME_TYPE)
        sub_session = int.from_bytes(frame[1:3], "little")
        body_len = int.from_bytes(frame[3:6], "little")
        msg_seq = frame[6]
        flag = frame[7]
        self.assertEqual(sub_session, 1)
        self.assertEqual(body_len, len(frame) - 6)
        self.assertEqual(msg_seq, 2)
        self.assertEqual(flag, PROMPT_FRAME_FLAG)
        # Ensure inner JSON is well-formed
        inner = json.loads(frame[8:].decode("utf-8"))
        self.assertEqual(inner["req-id"], "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
        self.assertIn("payload", inner)

    def test_build_mode_switch_body_uses_think_fast_doc_id(self):
        body = json.loads(build_mode_switch_body("1234", mode="think_fast").decode("utf-8"))

        self.assertEqual(body["doc_id"], "c32bbe999c48e64e855dc63177d5153f")
        self.assertEqual(body["variables"]["input"]["conversationId"], "1234")
        self.assertEqual(body["variables"]["input"]["mode"], "think_fast")

    def test_extract_json_objects_finds_embedded_json(self):
        payload = (
            b"\x0f\x00\x00\x7f\x00\x00"
            b'{"seq":0,"type":"full","response":{"sections":[{"view_model":{"primitive":{"text":"Hi"}}}]}}'
            b"tail"
        )

        objects = extract_json_objects(payload)

        self.assertEqual(len(objects), 1)
        self.assertEqual(objects[0]["type"], "full")
        self.assertEqual(
            objects[0]["response"]["sections"][0]["view_model"]["primitive"]["text"],
            "Hi",
        )

    def test_merge_response_events_applies_patch_deltas(self):
        events = [
            {
                "type": "full",
                "response": {
                    "sections": [
                        {"view_model": {"primitive": {"text": "Hey"}}},
                    ]
                },
            },
            {
                "type": "patch",
                "operations": [
                    {
                        "op": "delta",
                        "path": "/sections/0/view_model/primitive/text",
                        "value": " Kamell",
                    }
                ],
            },
            {
                "type": "patch",
                "operations": [
                    {
                        "op": "delta",
                        "path": "/sections/0/view_model/primitive/text",
                        "value": " let's cook.",
                    }
                ],
            },
        ]

        text = merge_response_events(events)

        self.assertEqual(text, "Hey Kamell let's cook.")

    def test_new_conversation_creates_state_and_sets_current(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "state.json"
            configure_auth(state_path, cookie_header="cookie=value", authorization="ecto1:abc")
            fake_generate = Mock(return_value="hello from muse")
            fake_warmup = Mock()
            fake_mode_switch = Mock()
            id_source = Mock(return_value="11111111-2222-3333-4444-555555555555")

            conversation = new_conversation(
                state_path,
                prompt="new convo probe 1",
                generate_fn=fake_generate,
                warmup_fn=fake_warmup,
                mode_switch_fn=fake_mode_switch,
                conversation_id_factory=id_source,
            )

            self.assertEqual(conversation["id"], "11111111-2222-3333-4444-555555555555")
            self.assertEqual(current_conversation(state_path)["id"], conversation["id"])
            self.assertEqual(list_conversations(state_path)[0]["id"], conversation["id"])
            self.assertEqual(list_conversations(state_path)[0]["template_name"], "chat")
            fake_warmup.assert_called_once_with(
                "11111111-2222-3333-4444-555555555555",
                "cookie=value",
                user_agent=unittest.mock.ANY,
            )
            fake_mode_switch.assert_called_once_with(
                conversation_id="11111111-2222-3333-4444-555555555555",
                cookie_header="cookie=value",
                mode="think_fast",
                user_agent=unittest.mock.ANY,
            )
            fake_generate.assert_called_once()
            self.assertEqual(fake_generate.call_args.kwargs["template_name"], "home")

    def test_chat_uses_current_conversation_without_manual_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "state.json"
            configure_auth(state_path, cookie_header="cookie=value", authorization="ecto1:abc")
            new_conversation(
                state_path,
                prompt="first prompt",
                generate_fn=Mock(return_value="first reply"),
                conversation_id_factory=Mock(return_value="11111111-2222-3333-4444-555555555555"),
            )
            fake_generate = Mock(return_value="second reply")
            fake_warmup = Mock()
            fake_mode_switch = Mock()

            reply = send_chat_message(
                state_path,
                prompt="follow up probe 2",
                generate_fn=fake_generate,
                warmup_fn=fake_warmup,
                mode_switch_fn=fake_mode_switch,
            )

            self.assertEqual(reply, "second reply")
            fake_warmup.assert_called_once_with(
                "11111111-2222-3333-4444-555555555555",
                "cookie=value",
                user_agent=unittest.mock.ANY,
            )
            fake_mode_switch.assert_called_once_with(
                conversation_id="11111111-2222-3333-4444-555555555555",
                cookie_header="cookie=value",
                mode="think_fast",
                user_agent=unittest.mock.ANY,
            )
            self.assertEqual(fake_generate.call_args.kwargs["template_name"], "chat")

    def test_use_conversation_switches_current_conversation(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "state.json"
            configure_auth(state_path, cookie_header="cookie=value", authorization="ecto1:abc")
            new_conversation(
                state_path,
                prompt="first prompt",
                generate_fn=Mock(return_value="first reply"),
                conversation_id_factory=Mock(return_value="11111111-2222-3333-4444-555555555555"),
            )
            new_conversation(
                state_path,
                prompt="second prompt",
                generate_fn=Mock(return_value="second reply"),
                conversation_id_factory=Mock(return_value="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"),
            )

            selected = use_conversation(state_path, "11111111-2222-3333-4444-555555555555")

            self.assertEqual(selected["id"], "11111111-2222-3333-4444-555555555555")
            self.assertEqual(current_conversation(state_path)["id"], "11111111-2222-3333-4444-555555555555")

    def test_main_serve_runs_api_server_with_host_port_and_state_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "state.json"
            with patch("muse_spark.api.run_api_server") as fake_run:
                exit_code = main(
                    [
                        "--state-path",
                        str(state_path),
                        "serve",
                        "--host",
                        "0.0.0.0",
                        "--port",
                        "9001",
                    ]
                )

            self.assertEqual(exit_code, 0)
            # When tuning flags are omitted on the CLI, we forward None so
            # ApiSettings.from_env() / MUSE_SPARK_* env vars stay authoritative.
            fake_run.assert_called_once_with(
                host="0.0.0.0",
                port=9001,
                state_path=state_path,
                force_single_conversation=None,
                stream_chunk_size=None,
                receive_timeout=None,
                first_byte_timeout=None,
            )

    def test_main_serve_forwards_explicit_tuning_flags_when_provided(self):
        from muse_spark.client import main

        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "state.json"
            with patch("muse_spark.api.run_api_server") as fake_run:
                exit_code = main(
                    [
                        "--state-path",
                        str(state_path),
                        "serve",
                        "--host",
                        "127.0.0.1",
                        "--port",
                        "8000",
                        "--single-conversation",
                        "--chunk-size",
                        "120",
                        "--timeout",
                        "45.0",
                    ]
                )

            self.assertEqual(exit_code, 0)
            fake_run.assert_called_once_with(
                host="127.0.0.1",
                port=8000,
                state_path=state_path,
                force_single_conversation=True,
                stream_chunk_size=120,
                receive_timeout=45.0,
                first_byte_timeout=None,
            )

    def test_main_serve_forwards_first_byte_timeout_flag(self):
        """``--first-byte-timeout`` must be plumbed through to
        ``run_api_server`` so operators can override the default 20s
        ceiling without touching env vars (useful for production hot-fix
        when the optimisation needs tightening or disabling)."""
        from muse_spark.client import main

        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "state.json"
            with patch("muse_spark.api.run_api_server") as fake_run:
                exit_code = main(
                    [
                        "--state-path",
                        str(state_path),
                        "serve",
                        "--first-byte-timeout",
                        "10.0",
                    ]
                )

            self.assertEqual(exit_code, 0)
            kwargs = fake_run.call_args.kwargs
            self.assertEqual(kwargs.get("first_byte_timeout"), 10.0)

    def test_run_api_server_prints_startup_banner(self):
        from muse_spark.api import run_api_server

        with patch("uvicorn.run") as fake_uvicorn, patch("builtins.print") as fake_print:
            run_api_server(host="127.0.0.1", port=8123, state_path=Path("/tmp/test-state.json"))

        printed = "\n".join(" ".join(str(arg) for arg in call.args) for call in fake_print.call_args_list)
        self.assertIn("Muse Spark API", printed)
        self.assertIn("http://127.0.0.1:8123", printed)
        self.assertIn("/tmp/test-state.json", printed)


if __name__ == "__main__":
    unittest.main()
