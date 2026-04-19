from __future__ import annotations

import json
import time
import uuid
from typing import Any, Optional


CHAT_COMPLETION_OBJECT = "chat.completion"
CHAT_COMPLETION_CHUNK_OBJECT = "chat.completion.chunk"
MODELS_OBJECT = "list"



def build_chat_completion_response(
    *,
    content: str,
    model: str,
    finish_reason: str = "stop",
    response_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
) -> dict[str, Any]:
    payload = {
        "id": response_id or f"chatcmpl-{uuid.uuid4()}",
        "object": CHAT_COMPLETION_OBJECT,
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
    }
    if conversation_id is not None:
        payload["conversation_id"] = conversation_id
    return payload



def build_chat_completion_chunk(
    *,
    model: str,
    response_id: str,
    delta: dict[str, Any],
    finish_reason: Optional[str] = None,
    index: int = 0,
    conversation_id: Optional[str] = None,
) -> dict[str, Any]:
    payload = {
        "id": response_id,
        "object": CHAT_COMPLETION_CHUNK_OBJECT,
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": index,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }
    if conversation_id is not None:
        payload["conversation_id"] = conversation_id
    return payload



def encode_sse_data(payload: dict[str, Any]) -> bytes:
    rendered = json.dumps(payload, separators=(",", ":"))
    return f"data: {rendered}\n\n".encode("utf-8")



def encode_sse_done() -> bytes:
    return b"data: [DONE]\n\n"



def build_error_response(*, code: str, message: str, error_type: str = "invalid_request_error") -> dict[str, Any]:
    return {
        "error": {
            "message": message,
            "type": error_type,
            "code": code,
        }
    }



def build_models_response(models: list[str]) -> dict[str, Any]:
    return {
        "object": MODELS_OBJECT,
        "data": [{"id": model, "object": "model", "owned_by": "meta"} for model in models],
    }
