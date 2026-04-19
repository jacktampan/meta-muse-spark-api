from __future__ import annotations

from typing import Any, Optional, Union

from pydantic import BaseModel, ConfigDict


class ResponseFormat(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: Any


class ChatCompletionsRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[ChatMessage]
    conversation_id: Optional[str] = None
    stream: bool = False
    max_tokens: Optional[int] = None
    stop: Optional[Union[str, list[str]]] = None
    response_format: Optional[ResponseFormat] = None
    user: Optional[str] = None
    temperature: Optional[float] = None
