from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import struct
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Iterable, Optional, Union

from .errors import (
    MissingAuthError,
    ProviderProtocolError,
    ProviderStallError,
    ProviderTransportError,
    ReauthRequiredError,
)
from .logging_utils import get_logger

logger = get_logger(__name__)

INTRO_FRAME_TYPE = 0x0f
PROMPT_FRAME_TYPE = 0x0d
PROMPT_FRAME_FLAG = 0x80
PROTO_INSIDE_JSON = "PROTO_INSIDE_JSON"
MODE_SWITCH_DOC_ID = "c32bbe999c48e64e855dc63177d5153f"
WARMUP_CONVERSATION_DOC_ID = "e7f802582dbfed8e181b012e010993eb"
META_APP_ID = "1522763855472543"
META_APP_VERSION = "1.0.0"
META_AUTHTYPE = "15:0"
META_DGW_VERSION = "5"
META_DGW_UUID = "0"
META_TIER = "prod"
DEFAULT_MODE = "think_fast"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36"
)
DEFAULT_STATE_PATH = Path.home() / ".muse_spark" / "state.json"
HOME_TEMPLATE_NAME = "home"
CHAT_TEMPLATE_NAME = "chat"
HOME_WS_TEMPLATE = {
    "req-id": "ee7a35eb-df8c-4793-a1c0-10ae414f5e6e",
    "payload": "CrYGCsQDCiBLQURBQlJBX19IT01FX19VTklGSUVEX0lOUFVUX0JBUhIQMTUyMjc2Mzg1NTQ3MjU0MyInNWE1Yi04ZDRlLWYwNTQtOTllZi1iMmRlLWRiMDItMGQwNS01MmM3KigqJgokOGYxMjliMjUtYzNlMC00NzNiLWFlNzktNWViM2YyNGU1NjRjMAU6C0hVTUFOX0FHRU5UQiIKDzg2NzA1MTMxNDc2NzY5NhIPODY3MDUxMzE0NzY3Njk2UgVFQ1RPMVoRQWJyYSBXZWIgTWFpbiBLZXliCRoDCOgHIgIIAWoITWFjIE9TIFhyCnVzZXJfaW5wdXR6dU1vemlsbGEvNS4wIChNYWNpbnRvc2g7IEludGVsIE1hYyBPUyBYIDEwXzE1XzcpIEFwcGxlV2ViS2l0LzUzNy4zNiAoS0hUTUwsIGxpa2UgR2Vja28pIENocm9tZS8xNDYuMC4wLjAgU2FmYXJpLzUzNy4zNoIBC2Rlc2t0b3Bfd2VimgFHCkBlMmI4OGY5ODQ2Mzc5Y2JjMjY5NjBmYTNhZTFkMjIyMDFkZmIxOWRmNzg5MGFlNmEzYWM4YTI4ODcwYmFjNjgyFQAAAEASFAi4w6XTk4/yARC4w6XTk4/yARgCGgIgASIAKg4Ix6D+ldkzGJ6g/pXZMzIkZWU3YTM1ZWItZGY4Yy00NzkzLWExYzAtMTBhZTQxNGY1ZTZlOgBKBxIFZW4tVVNScgokNTYwN2Y0YzAtYjljZi00ZjZlLWJlYTYtZTc2N2E1OGJhMjhlGiRlMDliN2FhMC1jYzYwLTQyYTktYjk2OS00YzY1YjViZGZlNGIiJDhmMTI5YjI1LWMzZTAtNDczYi1hZTc5LTVlYjNmMjRlNTY0Y3oRIg9BbWVyaWNhL0NoaWNhZ2+CAQOwAQGSAQwKBnN0b2NrcxICCAGSAQ0KB3dlYXRoZXISAggBkgEkCh5tZXRhX2tub3dsZWRnZV9zZWFyY2hfY2Fyb3VzZWwSAggBkgEiChxtZXRhX2NhdGFsb2dfc2VhcmNoX2Nhcm91c2VsEgIIAZIBEwoNbWVkaWFfZ2FsbGVyeRICCAGiAQEDEpIBCmEKJGFiOWRkNzg5LWRlOGQtNDc5MS05ODE1LWI5YjBmMTU1MDdiNBI3CiQ4ZjEyOWIyNS1jM2UwLTQ3M2ItYWU3OS01ZWIzZjI0ZTU2NGMQyKD+ldkzGKbcxozB/KuyZygBEihIZWxsbyB0aGlzIGlzIGFub3RoZXIgdGVzdCBvZiB5b3VyIHBvd2VyIgMKATA=",
}
CHAT_WS_TEMPLATE = {
    "req-id": "c6b5d261-6624-49af-90c7-09b45c0a6bef",
    "payload": "CrIGCsADCiBLQURBQlJBX19DSEFUX19VTklGSUVEX0lOUFVUX0JBUhIQMTUyMjc2Mzg1NTQ3MjU0MyInNWE1Yi04ZDRlLWYwNTQtOTllZi1iMmRlLWRiMDItMGQwNS01MmM3KigqJgokYjA4Mzg1YTYtNWE1My00ZjE0LTk2NmUtMzQ3ZjI4MDg4NDU0MAU6C0hVTUFOX0FHRU5UQiIKDzg2NzA1MTMxNDc2NzY5NhIPODY3MDUxMzE0NzY3Njk2UgVFQ1RPMVoRQWJyYSBXZWIgTWFpbiBLZXliBRoDCOgHaghNYWMgT1MgWHIKdXNlcl9pbnB1dHp1TW96aWxsYS81LjAgKE1hY2ludG9zaDsgSW50ZWwgTWFjIE9TIFggMTBfMTVfNykgQXBwbGVXZWJLaXQvNTM3LjM2IChLSFRNTCwgbGlrZSBHZWNrbykgQ2hyb21lLzE0Ni4wLjAuMCBTYWZhcmkvNTM3LjM2ggELZGVza3RvcF93ZWKaAUcKQGUyYjg4Zjk4NDYzNzljYmMyNjk2MGZhM2FlMWQyMjIwMWRmYjE5ZGY3ODkwYWU2YTNhYzhhMjg4NzBiYWM2ODIVAAAAQBIUCLjDpdOTj/IBELjDpdOTj/IBGAIaAiABIgAqDgikgvuW2TMYoYL7ltkzMiRjNmI1ZDI2MS02NjI0LTQ5YWYtOTBjNy0wOWI0NWMwYTZiZWY6AEoHEgVlbi1VU1JyCiQxZDNjZGQzYy1jYTFhLTRlMDItODk1My1kZTBiYTM0NzI5ODkaJDcxODNhMzM0LTFiNWEtNGQyNi1iMjcxLWJjY2Y1NDY2NmJiZiIkYjA4Mzg1YTYtNWE1My00ZjE0LTk2NmUtMzQ3ZjI4MDg4NDU0ehEiD0FtZXJpY2EvQ2hpY2Fnb4IBA7ABAZIBDAoGc3RvY2tzEgIIAZIBDQoHd2VhdGhlchICCAGSASQKHm1ldGFfa25vd2xlZGdlX3NlYXJjaF9jYXJvdXNlbBICCAGSASIKHG1ldGFfY2F0YWxvZ19zZWFyY2hfY2Fyb3VzZWwSAggBkgETCg1tZWRpYV9nYWxsZXJ5EgIIAaIBAQMSlgEKfAokMTc4MDVmYjEtOTY3Zi00YmYyLTlmMjctOWRhYmRhMzYyMTJkEjcKJGIwODM4NWE2LTVhNTMtNGYxNC05NjZlLTM0N2YyODA4ODQ1NBCkgvuW2TMYxN23xoT2rbJnIhtlLjAwcHlKMUtxa3BHTmg5Sk9oWElNdnJRWlYSEWZvbGxvdyB1cCBwcm9iZSAyIgMKATI=",
}
WS_TEMPLATES = {
    HOME_TEMPLATE_NAME: HOME_WS_TEMPLATE,
    CHAT_TEMPLATE_NAME: CHAT_WS_TEMPLATE,
}


@dataclass
class ProtoField:
    number: int
    wire_type: int
    value: Any



def _empty_state() -> dict[str, Any]:
    return {
        "auth": {
            "cookie_header": None,
            "authorization": None,
            "mode": DEFAULT_MODE,
            "user_agent": DEFAULT_USER_AGENT,
        },
        "current_conversation_id": None,
        "conversations": {},
    }



def load_state(state_path: Union[Path, str] = DEFAULT_STATE_PATH) -> dict[str, Any]:
    path = Path(state_path)
    if not path.exists():
        return _empty_state()
    data = json.loads(path.read_text())
    state = _empty_state()
    state.update(data)
    state["auth"].update(data.get("auth", {}))
    state["conversations"].update(data.get("conversations", {}))
    return state



def save_state(state_path: Union[Path, str], state: dict[str, Any]) -> None:
    path = Path(state_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True))



def configure_auth(
    state_path: Union[Path, str],
    cookie_header: str,
    authorization: str,
    mode: str = DEFAULT_MODE,
    user_agent: str = DEFAULT_USER_AGENT,
) -> dict[str, Any]:
    state = load_state(state_path)
    state["auth"] = {
        "cookie_header": cookie_header,
        "authorization": authorization,
        "mode": mode,
        "user_agent": user_agent,
    }
    save_state(state_path, state)
    return state["auth"]



def _require_auth(state: dict[str, Any]) -> dict[str, str]:
    auth = state.get("auth", {})
    has_cookie = bool(auth.get("cookie_header") or os.getenv("MUSE_SPARK_COOKIE_HEADER") or os.getenv("MUSE_SPARK_COOKIE"))
    has_authorization = bool(auth.get("authorization") or os.getenv("MUSE_SPARK_AUTHORIZATION"))
    if not has_cookie or not has_authorization:
        raise MissingAuthError(
            "missing stored auth. Run `muse-spark auth set --cookie ... --authorization ...` first."
        )
    return {
        "cookie_header": auth.get("cookie_header") or os.getenv("MUSE_SPARK_COOKIE_HEADER") or os.getenv("MUSE_SPARK_COOKIE") or "",
        "authorization": auth.get("authorization") or os.getenv("MUSE_SPARK_AUTHORIZATION") or "",
        "mode": auth.get("mode") or os.getenv("MUSE_SPARK_MODE", DEFAULT_MODE),
        "user_agent": auth.get("user_agent") or DEFAULT_USER_AGENT,
    }



def current_conversation(state_path: Union[Path, str] = DEFAULT_STATE_PATH) -> Optional[dict[str, Any]]:
    state = load_state(state_path)
    current_id = state.get("current_conversation_id")
    if not current_id:
        return None
    conversation = state.get("conversations", {}).get(current_id)
    if not conversation:
        return None
    return {"id": current_id, **conversation}



def list_conversations(state_path: Union[Path, str] = DEFAULT_STATE_PATH) -> list[dict[str, Any]]:
    state = load_state(state_path)
    current_id = state.get("current_conversation_id")
    items = []
    for conversation_id, conversation in state.get("conversations", {}).items():
        items.append({"id": conversation_id, "is_current": conversation_id == current_id, **conversation})
    items.sort(key=lambda item: (item.get("last_used_at", 0), item["id"]), reverse=True)
    return items



def _conversation_title(prompt: str) -> str:
    return prompt.strip()[:60] or "Untitled conversation"



def _upsert_conversation(
    state: dict[str, Any],
    conversation_id: str,
    *,
    template_name: str = CHAT_TEMPLATE_NAME,
    title: Optional[str] = None,
    last_prompt: Optional[str] = None,
    last_response: Optional[str] = None,
) -> dict[str, Any]:
    now = int(time.time())
    conversations = state.setdefault("conversations", {})
    conversation = conversations.get(conversation_id, {})
    if not conversation:
        conversation["created_at"] = now
    conversation["template_name"] = template_name
    conversation["title"] = title or conversation.get("title") or conversation_id
    conversation["last_prompt"] = last_prompt or conversation.get("last_prompt")
    conversation["last_response"] = last_response or conversation.get("last_response")
    conversation["last_used_at"] = now
    conversations[conversation_id] = conversation
    state["current_conversation_id"] = conversation_id
    return {"id": conversation_id, **conversation}



def use_conversation(state_path: Union[Path, str], conversation_id: str) -> dict[str, Any]:
    state = load_state(state_path)
    conversation = _upsert_conversation(state, conversation_id, title=state.get("conversations", {}).get(conversation_id, {}).get("title") or conversation_id)
    save_state(state_path, state)
    return conversation



def encode_varint(value: int) -> bytes:
    if value < 0:
        raise ValueError("varints must be non-negative")
    out = bytearray()
    while True:
        to_write = value & 0x7F
        value >>= 7
        if value:
            out.append(to_write | 0x80)
        else:
            out.append(to_write)
            return bytes(out)



def decode_varint(data: bytes, offset: int = 0) -> tuple[int, int]:
    shift = 0
    value = 0
    while True:
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            return value, offset
        shift += 7



def parse_message(data: bytes) -> list[ProtoField]:
    fields: list[ProtoField] = []
    offset = 0
    while offset < len(data):
        tag, offset = decode_varint(data, offset)
        number = tag >> 3
        wire_type = tag & 0x07
        if wire_type == 0:
            value, offset = decode_varint(data, offset)
        elif wire_type == 1:
            value = struct.unpack("<Q", data[offset : offset + 8])[0]
            offset += 8
        elif wire_type == 2:
            length, offset = decode_varint(data, offset)
            value = data[offset : offset + length]
            offset += length
        elif wire_type == 5:
            value = struct.unpack("<I", data[offset : offset + 4])[0]
            offset += 4
        else:
            raise ValueError(f"unsupported wire type: {wire_type}")
        fields.append(ProtoField(number=number, wire_type=wire_type, value=value))
    return fields



def serialize_message(fields: Iterable[ProtoField]) -> bytes:
    out = bytearray()
    for field in fields:
        tag = (field.number << 3) | field.wire_type
        out.extend(encode_varint(tag))
        if field.wire_type == 0:
            out.extend(encode_varint(int(field.value)))
        elif field.wire_type == 1:
            out.extend(struct.pack("<Q", int(field.value)))
        elif field.wire_type == 2:
            raw = bytes(field.value)
            out.extend(encode_varint(len(raw)))
            out.extend(raw)
        elif field.wire_type == 5:
            out.extend(struct.pack("<I", int(field.value)))
        else:
            raise ValueError(f"unsupported wire type: {field.wire_type}")
    return bytes(out)



def _get_field(fields: list[ProtoField], number: int, occurrence: int = 0) -> ProtoField:
    matches = [field for field in fields if field.number == number]
    if occurrence >= len(matches):
        raise KeyError(f"field {number} occurrence {occurrence} not found")
    return matches[occurrence]



def _get_nested_message(fields: list[ProtoField], path: list[int]) -> list[ProtoField]:
    current_fields = fields
    for number in path:
        field = _get_field(current_fields, number)
        if field.wire_type != 2:
            raise ValueError(f"field {number} is not length-delimited")
        current_fields = parse_message(field.value)
    return current_fields



def _replace_text(field: ProtoField, text: str) -> None:
    if field.wire_type != 2:
        raise ValueError("field is not text-capable")
    field.value = text.encode("utf-8")



def _replace_varint(field: ProtoField, value: int) -> None:
    if field.wire_type != 0:
        raise ValueError("field is not varint")
    field.value = value



def _replace_trailing_uuid(field: ProtoField, new_uuid: str) -> None:
    raw = bytes(field.value)
    if len(raw) < 36:
        raise ValueError("field too short to hold a UUID")
    field.value = raw[:-36] + new_uuid.encode("utf-8")



def _mutate_message(fields: list[ProtoField], path: list[int], mutator: Callable[[list[ProtoField]], None]) -> None:
    if not path:
        mutator(fields)
        return
    field = _get_field(fields, path[0])
    if field.wire_type != 2:
        raise ValueError(f"field {path[0]} is not length-delimited")
    nested = parse_message(field.value)
    _mutate_message(nested, path[1:], mutator)
    field.value = serialize_message(nested)



def _u24_le(value: int) -> bytes:
    if value < 0 or value > 0xFFFFFF:
        raise ValueError(f"u24 length out of range: {value}")
    return value.to_bytes(3, "little")


def build_intro_frame(conversation_id: str, sub_session_idx: int = 0) -> bytes:
    payload = json.dumps(
        {
            "x-dgw-app-x-ecto-conversation-id": conversation_id,
            "x-dgw-app-client-payload-type": PROTO_INSIDE_JSON,
        },
        separators=(",", ":"),
    ).encode("utf-8")
    header = bytes([INTRO_FRAME_TYPE]) + sub_session_idx.to_bytes(2, "little") + _u24_le(len(payload))
    return header + payload



def build_prompt_frame(
    prompt: str,
    conversation_id: str,
    request_id: Optional[str] = None,
    user_message_id: Optional[str] = None,
    submitted_ms: Optional[int] = None,
    unique_message_id: Optional[int] = None,
    template_name: str = HOME_TEMPLATE_NAME,
    sub_session_idx: int = 0,
    message_seq: int = 0,
) -> bytes:
    if template_name not in WS_TEMPLATES:
        raise ValueError(f"unknown template: {template_name}")
    template = dict(WS_TEMPLATES[template_name])
    request_id = request_id or str(uuid.uuid4())
    user_message_id = user_message_id or str(uuid.uuid4())
    submitted_ms = submitted_ms or int(time.time() * 1000)
    unique_message_id = unique_message_id or int(f"{submitted_ms}{str(uuid.uuid4().int)[:4]}")

    proto_fields = parse_message(base64.b64decode(template["payload"]))

    _mutate_message(
        proto_fields,
        [1, 1],
        lambda fields: _replace_trailing_uuid(_get_field(fields, 5), conversation_id),
    )
    _mutate_message(
        proto_fields,
        [2, 1],
        lambda fields: _replace_text(_get_field(fields, 1), user_message_id),
    )

    def mutate_root2_1_2(fields: list[ProtoField]) -> None:
        _replace_text(_get_field(fields, 1), conversation_id)
        _replace_varint(_get_field(fields, 2), submitted_ms)
        _replace_varint(_get_field(fields, 3), unique_message_id)

    _mutate_message(proto_fields, [2, 1, 2], mutate_root2_1_2)
    _mutate_message(
        proto_fields,
        [2],
        lambda fields: _replace_text(_get_field(fields, 2), prompt),
    )

    def mutate_root1_5(fields: list[ProtoField]) -> None:
        _replace_varint(_get_field(fields, 1), submitted_ms + 1)
        _replace_varint(_get_field(fields, 3), submitted_ms)

    _mutate_message(proto_fields, [1, 5], mutate_root1_5)
    _mutate_message(
        proto_fields,
        [1],
        lambda fields: _replace_text(_get_field(fields, 6), request_id),
    )
    _mutate_message(
        proto_fields,
        [1, 10],
        lambda fields: _replace_text(_get_field(fields, 4), conversation_id),
    )

    updated_payload = base64.b64encode(serialize_message(proto_fields)).decode("ascii")
    outer = {"req-id": request_id, "payload": updated_payload}
    inner = json.dumps(outer, separators=(",", ":")).encode("utf-8")
    msg_body = bytes([message_seq, PROMPT_FRAME_FLAG]) + inner
    header = (
        bytes([PROMPT_FRAME_TYPE])
        + sub_session_idx.to_bytes(2, "little")
        + _u24_le(len(msg_body))
    )
    return header + msg_body



def decode_prompt_frame(frame: bytes) -> dict[str, Any]:
    if not frame:
        raise ValueError("empty frame")
    if frame[0] == PROMPT_FRAME_TYPE and len(frame) >= 8:
        # Strip 6-byte outer header + 2-byte msg header
        inner = frame[8:]
    else:
        inner = frame
    outer = json.loads(inner.decode("utf-8"))
    proto_fields = parse_message(base64.b64decode(outer["payload"]))
    root2 = _get_nested_message(proto_fields, [2])
    root2_1 = _get_nested_message(proto_fields, [2, 1])
    root2_1_2 = _get_nested_message(proto_fields, [2, 1, 2])
    root1 = _get_nested_message(proto_fields, [1])
    root1_5 = _get_nested_message(proto_fields, [1, 5])
    root1_10 = _get_nested_message(proto_fields, [1, 10])
    return {
        "outer_request_id": outer["req-id"],
        "request_id": _get_field(root1, 6).value.decode("utf-8"),
        "user_message_id": _get_field(root2_1, 1).value.decode("utf-8"),
        "conversation_id": _get_field(root1_10, 4).value.decode("utf-8"),
        "prompt": _get_field(root2, 2).value.decode("utf-8"),
        "submitted_ms": _get_field(root1_5, 3).value,
        "unique_message_id": _get_field(root2_1_2, 3).value,
    }



def build_mode_switch_body(conversation_id: str, mode: str = DEFAULT_MODE) -> bytes:
    body = {
        "doc_id": MODE_SWITCH_DOC_ID,
        "variables": {"input": {"conversationId": conversation_id, "mode": mode}},
    }
    return json.dumps(body, separators=(",", ":")).encode("utf-8")



def build_warmup_body(conversation_id: str) -> bytes:
    body = {
        "doc_id": WARMUP_CONVERSATION_DOC_ID,
        "variables": {"conversationId": conversation_id},
    }
    return json.dumps(body, separators=(",", ":")).encode("utf-8")



def extract_json_objects(payload: bytes) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    start: Optional[int] = None
    depth = 0
    in_string = False
    escape = False
    for index, byte in enumerate(payload):
        char = chr(byte)
        if start is None:
            if char == "{":
                start = index
                depth = 1
                in_string = False
                escape = False
            continue
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start is not None:
                candidate = payload[start : index + 1]
                try:
                    decoded = json.loads(candidate.decode("utf-8"))
                except Exception:
                    pass
                else:
                    if isinstance(decoded, dict):
                        results.append(decoded)
                start = None
    return results



def merge_response_events(events: Iterable[dict[str, Any]]) -> str:
    text = ""
    for event in events:
        event_type = event.get("type")
        if event_type == "full":
            sections = event.get("response", {}).get("sections", [])
            for section in sections:
                primitive = section.get("view_model", {}).get("primitive", {})
                if isinstance(primitive.get("text"), str):
                    text = primitive["text"]
                    break
        elif event_type == "patch":
            for operation in event.get("operations", []):
                if (
                    operation.get("op") == "delta"
                    and operation.get("path") == "/sections/0/view_model/primitive/text"
                    and isinstance(operation.get("value"), str)
                ):
                    text += operation["value"]
    return text


def _event_full_text(event: dict[str, Any]) -> Optional[str]:
    sections = event.get("response", {}).get("sections", [])
    for section in sections:
        primitive = section.get("view_model", {}).get("primitive", {})
        if isinstance(primitive.get("text"), str):
            return primitive["text"]
    return None


async def stream_text_deltas(
    prompt: str,
    conversation_id: str,
    authorization: str,
    cookie_header: str,
    mode: str = DEFAULT_MODE,
    user_agent: str = DEFAULT_USER_AGENT,
    switch_mode_first: bool = False,
    receive_timeout: float = 30.0,
    template_name: str = HOME_TEMPLATE_NAME,
) -> AsyncIterator[str]:
    try:
        import websockets
    except ImportError as exc:
        raise RuntimeError("websockets is not installed. Create a venv and `pip install websockets`.") from exc
    from websockets.exceptions import ConnectionClosed

    connection_request_id = str(uuid.uuid4())
    prompt_request_id = str(uuid.uuid4())
    if switch_mode_first:
        graphql_mode_switch(
            conversation_id=conversation_id,
            cookie_header=cookie_header,
            mode=mode,
            user_agent=user_agent,
        )

    uri = websocket_url(authorization=authorization, request_id=connection_request_id)
    headers = {
        "Cookie": cookie_header,
        "User-Agent": user_agent,
        "Origin": "https://meta.ai",
    }

    yielded = False
    current_text = ""
    seen_completion_signal = False

    async with websockets.connect(uri, additional_headers=headers) as websocket:
        await websocket.send(build_intro_frame(conversation_id))
        await websocket.send(
            build_prompt_frame(
                prompt=prompt,
                conversation_id=conversation_id,
                request_id=prompt_request_id,
                template_name=template_name,
            )
        )
        while True:
            try:
                payload = await asyncio.wait_for(websocket.recv(), timeout=receive_timeout)
            except asyncio.TimeoutError:
                # Idle timeout. If we never yielded anything, the request
                # didn't produce text — let the post-loop check raise. If we
                # already saw a completion signal, this is the natural end of
                # stream. Otherwise the response stalled mid-generation: raise
                # ``ProviderStallError`` so the SSE pipeline can surface the
                # partial output we already streamed with finish_reason
                # ``"length"`` (graceful truncation) instead of ``"error"``.
                if yielded and not seen_completion_signal:
                    raise ProviderStallError(
                        "Meta stream stalled mid-response: no data received within "
                        f"{receive_timeout:.1f}s. Output may be truncated."
                    )
                break
            except ConnectionClosed:
                # Server closed the WebSocket — treat as natural end.
                break
            payload_bytes = payload.encode("utf-8") if isinstance(payload, str) else payload
            objects = extract_json_objects(payload_bytes)
            for event in objects:
                event_type = event.get("type")
                if event_type == "patch":
                    for operation in event.get("operations", []):
                        op_kind = operation.get("op")
                        op_path = operation.get("path")
                        if (
                            op_kind == "delta"
                            and op_path == "/sections/0/view_model/primitive/text"
                            and isinstance(operation.get("value"), str)
                        ):
                            delta = operation["value"]
                            current_text += delta
                            yielded = True
                            yield delta
                        elif (
                            op_kind == "replace"
                            and op_path == "/sections/0/view_model/primitive/text"
                            and isinstance(operation.get("value"), str)
                        ):
                            new_text = operation["value"]
                            for resync_chunk in _resync_text(current_text, new_text):
                                yielded = True
                                yield resync_chunk
                            current_text = new_text
                        elif _is_completion_op(op_kind, op_path, operation.get("value")):
                            seen_completion_signal = True
                elif event_type == "full":
                    full_text = _event_full_text(event)
                    if isinstance(full_text, str) and full_text != current_text:
                        for resync_chunk in _resync_text(current_text, full_text):
                            yielded = True
                            yield resync_chunk
                        current_text = full_text
                    if _event_is_complete(event):
                        seen_completion_signal = True
                elif event_type in {"complete", "completion", "done"}:
                    seen_completion_signal = True
            if seen_completion_signal:
                # Meta signalled end-of-stream for this turn. Exit immediately
                # instead of waiting on the recv() idle timeout — anything
                # arriving after this would belong to the next turn.
                break

    if not yielded:
        raise ProviderProtocolError("Meta transport returned no usable text response.")


def _is_completion_op(op_kind: Optional[str], op_path: Optional[str], op_value: Any) -> bool:
    """Best-effort detection of an end-of-stream signal in a patch operation."""
    if op_kind != "replace" or not isinstance(op_path, str):
        return False
    if op_path.endswith("/state") and isinstance(op_value, str) and op_value.upper() in {
        "COMPLETE",
        "DONE",
        "FINISHED",
    }:
        return True
    if op_path.endswith("/is_complete") and op_value is True:
        return True
    return False


def _event_is_complete(event: dict[str, Any]) -> bool:
    response = event.get("response")
    if not isinstance(response, dict):
        return False
    state = response.get("state") or response.get("status")
    if isinstance(state, str) and state.upper() in {"COMPLETE", "DONE", "FINISHED"}:
        return True
    if response.get("is_complete") is True:
        return True
    return False


# When Meta corrects an already-streamed token (typo fix at the tail of
# the response), emitting the corrected suffix is acceptable — callers see
# a small amount of duplication but the final text is correct. When the
# divergence is much earlier (Meta rewrote a long span we already streamed),
# emitting the divergent tail produces visibly garbled output because SSE
# has no way to retract previously streamed bytes. In that case we drop the
# correction; subsequent deltas usually reconverge as the response stabilises.
_RESYNC_MAX_BACKTRACK = 32


def _resync_text(
    current: str,
    new_full: str,
    *,
    max_backtrack: int = _RESYNC_MAX_BACKTRACK,
) -> list[str]:
    """Return the chunks needed to bring streamed text in line with ``new_full``.

    - If ``new_full`` extends ``current`` cleanly, return only the suffix.
    - If they diverge in the last ``max_backtrack`` characters of
      ``current`` (typo-style late correction), return the divergent tail
      so the user-visible text catches up to Meta's final intent. This
      duplicates a short overlap on the wire — the trade-off is a few
      duplicated chars vs. permanently stale text.
    - If they diverge earlier (Meta rewrote a long span we already
      streamed), return ``[]``: the SSE protocol can't retract already
      sent bytes, so emitting the tail would just stack garbled
      duplication on top of the original. Subsequent deltas tend to
      reconverge as the response stabilises.
    - If ``new_full`` is shorter or identical, return [].
    """
    if not isinstance(new_full, str):
        return []
    if new_full == current:
        return []
    if new_full.startswith(current):
        suffix = new_full[len(current):]
        return [suffix] if suffix else []
    if not current:
        return [new_full]
    # Diverged. Find the longest common prefix.
    common_len = 0
    for ch_a, ch_b in zip(current, new_full):
        if ch_a != ch_b:
            break
        common_len += 1
    backtrack = len(current) - common_len
    if backtrack > max_backtrack:
        logger.debug(
            "stream resync: dropping divergent correction (backtrack=%d > %d, current_len=%d, new_len=%d)",
            backtrack,
            max_backtrack,
            len(current),
            len(new_full),
        )
        return []
    tail = new_full[common_len:]
    if not tail:
        return []
    logger.debug(
        "stream resync: applying late correction (backtrack=%d, current_len=%d, new_len=%d)",
        backtrack,
        len(current),
        len(new_full),
    )
    return [tail]


def _ensure_text_reply(text: Any) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ProviderProtocolError("Meta transport returned no usable text response.")
    return text



def _graphql_request(
    body: bytes,
    cookie_header: str,
    user_agent: str = DEFAULT_USER_AGENT,
    endpoint: str = "https://meta.ai/api/graphql",
) -> dict[str, Any]:
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={
            "content-type": "application/json",
            "accept": "multipart/mixed, application/json",
            "origin": "https://meta.ai",
            "cookie": cookie_header,
            "user-agent": user_agent,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
            if not isinstance(payload, dict):
                raise ProviderProtocolError("Meta GraphQL returned a non-object JSON response.")
            if payload.get("errors"):
                raise ProviderProtocolError("Meta GraphQL returned GraphQL errors in a successful HTTP response.")
            return payload
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise ReauthRequiredError("Meta auth expired or was rejected. Refresh local auth state.") from exc
        raise ProviderTransportError(f"Meta GraphQL request failed with HTTP {exc.code}.") from exc
    except urllib.error.URLError as exc:
        raise ProviderTransportError("Meta GraphQL request failed before a response was received.") from exc
    except json.JSONDecodeError as exc:
        raise ProviderProtocolError("Meta GraphQL returned a successful response with invalid JSON.") from exc



def graphql_mode_switch(
    conversation_id: str,
    cookie_header: str,
    mode: str = DEFAULT_MODE,
    user_agent: str = DEFAULT_USER_AGENT,
    endpoint: str = "https://meta.ai/api/graphql",
) -> dict[str, Any]:
    return _graphql_request(
        build_mode_switch_body(conversation_id, mode=mode),
        cookie_header=cookie_header,
        user_agent=user_agent,
        endpoint=endpoint,
    )



def graphql_warmup_conversation(
    conversation_id: str,
    cookie_header: str,
    user_agent: str = DEFAULT_USER_AGENT,
    endpoint: str = "https://meta.ai/api/graphql",
) -> dict[str, Any]:
    return _graphql_request(
        build_warmup_body(conversation_id),
        cookie_header=cookie_header,
        user_agent=user_agent,
        endpoint=endpoint,
    )



def websocket_url(authorization: str, request_id: str) -> str:
    query = urllib.parse.urlencode(
        {
            "x-dgw-appid": META_APP_ID,
            "x-dgw-appversion": META_APP_VERSION,
            "x-dgw-authtype": META_AUTHTYPE,
            "x-dgw-version": META_DGW_VERSION,
            "x-dgw-uuid": META_DGW_UUID,
            "x-dgw-tier": META_TIER,
            "Authorization": authorization,
            "x-dgw-app-origin": "meta.ai",
            "x-dgw-app-clippy-request-id": request_id,
            "x-dgw-app-clippy-async": "true",
        }
    )
    return f"wss://gateway.meta.ai/ws/clippy?{query}"



async def generate_text(
    prompt: str,
    conversation_id: str,
    authorization: str,
    cookie_header: str,
    mode: str = DEFAULT_MODE,
    user_agent: str = DEFAULT_USER_AGENT,
    switch_mode_first: bool = False,
    receive_timeout: float = 30.0,
    template_name: str = HOME_TEMPLATE_NAME,
) -> str:
    parts: list[str] = []
    async for chunk in stream_text_deltas(
        prompt=prompt,
        conversation_id=conversation_id,
        authorization=authorization,
        cookie_header=cookie_header,
        mode=mode,
        user_agent=user_agent,
        switch_mode_first=switch_mode_first,
        receive_timeout=receive_timeout,
        template_name=template_name,
    ):
        parts.append(chunk)
    return _ensure_text_reply("".join(parts))



def new_conversation(
    state_path: Union[Path, str],
    prompt: str,
    generate_fn: Callable[..., str] = lambda **kwargs: asyncio.run(generate_text(**kwargs)),
    warmup_fn: Callable[..., Any] = graphql_warmup_conversation,
    mode_switch_fn: Callable[..., Any] = graphql_mode_switch,
    conversation_id_factory: Callable[[], str] = lambda: str(uuid.uuid4()),
) -> dict[str, Any]:
    state = load_state(state_path)
    auth = _require_auth(state)
    conversation_id = str(conversation_id_factory())
    warmup_fn(
        conversation_id,
        auth["cookie_header"],
        user_agent=auth["user_agent"],
    )
    mode_switch_fn(
        conversation_id=conversation_id,
        cookie_header=auth["cookie_header"],
        mode=auth["mode"],
        user_agent=auth["user_agent"],
    )
    reply = generate_fn(
        prompt=prompt,
        conversation_id=conversation_id,
        authorization=auth["authorization"],
        cookie_header=auth["cookie_header"],
        mode=auth["mode"],
        user_agent=auth["user_agent"],
        switch_mode_first=False,
        template_name=HOME_TEMPLATE_NAME,
    )
    conversation = _upsert_conversation(
        state,
        conversation_id,
        template_name=CHAT_TEMPLATE_NAME,
        title=_conversation_title(prompt),
        last_prompt=prompt,
        last_response=reply,
    )
    save_state(state_path, state)
    return {**conversation, "response": reply}



def send_chat_message(
    state_path: Union[Path, str],
    prompt: str,
    generate_fn: Callable[..., str] = lambda **kwargs: asyncio.run(generate_text(**kwargs)),
    warmup_fn: Callable[..., Any] = graphql_warmup_conversation,
    mode_switch_fn: Callable[..., Any] = graphql_mode_switch,
) -> str:
    state = load_state(state_path)
    auth = _require_auth(state)
    current = current_conversation(state_path)
    if not current:
        raise ValueError("no current conversation. Run `muse-spark new \"prompt\"` or `muse-spark use <id>` first.")
    conversation_id = current["id"]
    warmup_fn(
        conversation_id,
        auth["cookie_header"],
        user_agent=auth["user_agent"],
    )
    mode_switch_fn(
        conversation_id=conversation_id,
        cookie_header=auth["cookie_header"],
        mode=auth["mode"],
        user_agent=auth["user_agent"],
    )
    reply = generate_fn(
        prompt=prompt,
        conversation_id=conversation_id,
        authorization=auth["authorization"],
        cookie_header=auth["cookie_header"],
        mode=auth["mode"],
        user_agent=auth["user_agent"],
        switch_mode_first=False,
        template_name=CHAT_TEMPLATE_NAME,
    )
    _upsert_conversation(
        state,
        conversation_id,
        template_name=CHAT_TEMPLATE_NAME,
        title=current.get("title") or _conversation_title(prompt),
        last_prompt=prompt,
        last_response=reply,
    )
    save_state(state_path, state)
    return reply



def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Meta AI Muse Spark CLI")
    parser.add_argument("--state-path", default=str(DEFAULT_STATE_PATH))
    subparsers = parser.add_subparsers(dest="command", required=True)

    auth_parser = subparsers.add_parser("auth", help="Store auth locally")
    auth_subparsers = auth_parser.add_subparsers(dest="auth_command", required=True)
    auth_set = auth_subparsers.add_parser("set", help="Store cookie and authorization")
    auth_set.add_argument("--cookie", required=True)
    auth_set.add_argument("--authorization", required=True)
    auth_set.add_argument("--mode", default=DEFAULT_MODE)

    new_parser = subparsers.add_parser("new", help="Start a new conversation")
    new_parser.add_argument("prompt")

    chat_parser = subparsers.add_parser("chat", help="Send a message to the current conversation")
    chat_parser.add_argument("prompt")

    use_parser = subparsers.add_parser("use", help="Switch current conversation")
    use_parser.add_argument("conversation_id")

    subparsers.add_parser("current", help="Show current conversation")
    subparsers.add_parser("list", help="List known conversations")

    debug_parser = subparsers.add_parser("debug-frame", help="Print a generated frame payload")
    debug_parser.add_argument("prompt")
    debug_parser.add_argument("--conversation-id", required=True)
    debug_parser.add_argument("--template", choices=sorted(WS_TEMPLATES), default=HOME_TEMPLATE_NAME)

    serve_parser = subparsers.add_parser("serve", help="Run the local OpenAI-compatible API server")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument(
        "--single-conversation",
        action="store_true",
        default=None,
        help="Force a single persistent conversation. When omitted, MUSE_SPARK_FORCE_SINGLE_CONVERSATION env var is honored.",
    )
    serve_parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="SSE chunk size (0 to disable). When omitted, MUSE_SPARK_STREAM_CHUNK_SIZE env var is honored.",
    )
    serve_parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Receive timeout in seconds. When omitted, MUSE_SPARK_RECEIVE_TIMEOUT env var is honored.",
    )

    return parser



def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    state_path = Path(args.state_path)

    if args.command == "auth" and args.auth_command == "set":
        configure_auth(
            state_path,
            cookie_header=args.cookie,
            authorization=args.authorization,
            mode=args.mode,
        )
        print(f"Stored auth in {state_path}")
        return 0

    if args.command == "new":
        conversation = new_conversation(state_path, args.prompt)
        print(f"Conversation: {conversation['id']}")
        print(conversation["response"])
        return 0

    if args.command == "chat":
        print(send_chat_message(state_path, args.prompt))
        return 0

    if args.command == "use":
        conversation = use_conversation(state_path, args.conversation_id)
        print(f"Current conversation: {conversation['id']}")
        return 0

    if args.command == "current":
        conversation = current_conversation(state_path)
        if not conversation:
            print("No current conversation")
            return 0
        print(json.dumps(conversation, indent=2, sort_keys=True))
        return 0

    if args.command == "list":
        conversations = list_conversations(state_path)
        if not conversations:
            print("No known conversations")
            return 0
        for conversation in conversations:
            marker = "*" if conversation.get("is_current") else " "
            print(f"{marker} {conversation['id']}  {conversation.get('title', conversation['id'])}")
        return 0

    if args.command == "debug-frame":
        frame = build_prompt_frame(
            prompt=args.prompt,
            conversation_id=args.conversation_id,
            template_name=args.template,
        )
        print(frame.decode("utf-8"))
        return 0

    if args.command == "serve":
        from .api import run_api_server

        run_api_server(
            host=args.host,
            port=args.port,
            state_path=state_path,
            force_single_conversation=args.single_conversation,
            stream_chunk_size=args.chunk_size,
            receive_timeout=args.timeout,
        )
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
