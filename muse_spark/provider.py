from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable, Optional, Union

from .client import (
    CHAT_TEMPLATE_NAME,
    DEFAULT_MODE,
    DEFAULT_STATE_PATH,
    DEFAULT_USER_AGENT,
    HOME_TEMPLATE_NAME,
    _ensure_text_reply,
    _require_auth,
    generate_text,
    graphql_mode_switch,
    graphql_warmup_conversation,
    load_state,
    save_state,
    stream_text_deltas,
)



@dataclass
class MuseProviderRequest:
    prompt: str
    conversation_id: Optional[str] = None
    template_name: str = HOME_TEMPLATE_NAME
    # Kept in sync with ``ApiSettings.receive_timeout`` so callers that build a
    # request without an explicit value still honour the configured idle
    # timeout. Update both together when adjusting the platform default.
    receive_timeout: float = 60.0
    # Optional pre-first-byte idle ceiling — see ``ApiSettings.first_byte_timeout``.
    # ``None`` (or any value >= ``receive_timeout``) disables the optimisation
    # and falls back to the existing single-timeout behaviour.
    first_byte_timeout: Optional[float] = 20.0
    user_prompt: Optional[str] = None
    # When False, callers signal the conversation is already warm on Meta's
    # side and the per-request warmup + mode_switch GraphQL round-trips can
    # be skipped, halving follow-up latency.
    needs_warmup: bool = True


@dataclass
class MuseProviderResponse:
    text: str
    conversation_id: str
    template_name: str
    mode: str = DEFAULT_MODE
    user_agent: str = DEFAULT_USER_AGENT


@dataclass
class ResolvedConversation:
    client_conversation_id: str
    meta_conversation_id: str
    template_name: str
    is_new: bool = True


WarmupFn = Callable[..., Any]
ModeSwitchFn = Callable[..., Any]
ConversationIdFactory = Callable[[], str]
AsyncGeneratedTextFn = Callable[..., Awaitable[str]]
AsyncStreamTextFn = Callable[..., AsyncIterator[str]]
API_CONVERSATIONS_KEY = "api_conversations"



def load_provider_auth(state_path: Union[Path, str] = DEFAULT_STATE_PATH) -> dict[str, str]:
    state = load_state(state_path)
    return _require_auth(state)



def purge_api_conversation(
    state_path: Union[Path, str] = DEFAULT_STATE_PATH,
    client_conversation_id: Optional[str] = None,
) -> bool:
    """Drop the cached client → meta conversation mapping.

    Used when Meta starts returning empty responses for an existing mapping —
    a symptom of the conversation entering a stuck state on Meta's backend
    (typically observed after a mid-response stall). After a purge, the next
    call to :func:`resolve_api_conversation` for the same client id rolls a
    fresh meta conversation id and re-bootstraps from scratch.

    Returns ``True`` if a mapping was removed, ``False`` if there was
    nothing to purge (no-op).
    """
    if not client_conversation_id:
        return False
    state = load_state(state_path)
    mappings = state.get(API_CONVERSATIONS_KEY, {})
    if client_conversation_id not in mappings:
        return False
    del mappings[client_conversation_id]
    save_state(state_path, state)
    return True


def resolve_api_conversation(
    state_path: Union[Path, str] = DEFAULT_STATE_PATH,
    client_conversation_id: Optional[str] = None,
    meta_conversation_id_factory: ConversationIdFactory = lambda: str(uuid.uuid4()),
    force_single_conversation: bool = False,
) -> ResolvedConversation:
    state = load_state(state_path)
    mappings = state.setdefault(API_CONVERSATIONS_KEY, {})
    now = int(time.time())

    if force_single_conversation:
        client_conversation_id = "default-single-conversation"

    if client_conversation_id and client_conversation_id in mappings:
        mapping = mappings[client_conversation_id]
        mapping["last_used_at"] = now
        save_state(state_path, state)
        return ResolvedConversation(
            client_conversation_id=client_conversation_id,
            meta_conversation_id=mapping["meta_conversation_id"],
            template_name=CHAT_TEMPLATE_NAME,
            is_new=False,
        )

    resolved_client_id = client_conversation_id or str(uuid.uuid4())
    resolved_meta_id = str(meta_conversation_id_factory())
    mappings[resolved_client_id] = {
        "meta_conversation_id": resolved_meta_id,
        "created_at": now,
        "last_used_at": now,
    }
    save_state(state_path, state)
    return ResolvedConversation(
        client_conversation_id=resolved_client_id,
        meta_conversation_id=resolved_meta_id,
        template_name=HOME_TEMPLATE_NAME,
        is_new=True,
    )


async def generate_from_state_async(
    request: MuseProviderRequest,
    state_path: Union[Path, str] = DEFAULT_STATE_PATH,
    generate_fn: AsyncGeneratedTextFn = generate_text,
    warmup_fn: WarmupFn = graphql_warmup_conversation,
    mode_switch_fn: ModeSwitchFn = graphql_mode_switch,
    conversation_id_factory: ConversationIdFactory = lambda: str(uuid.uuid4()),
) -> MuseProviderResponse:
    auth = await asyncio.to_thread(load_provider_auth, state_path)
    conversation_id = request.conversation_id or str(conversation_id_factory())

    if request.needs_warmup:
        await asyncio.to_thread(
            warmup_fn,
            conversation_id,
            auth["cookie_header"],
            user_agent=auth["user_agent"],
        )
        await asyncio.to_thread(
            mode_switch_fn,
            conversation_id=conversation_id,
            cookie_header=auth["cookie_header"],
            mode=auth["mode"],
            user_agent=auth["user_agent"],
        )
    user_prompt = request.user_prompt or request.prompt
    text = await generate_fn(
        prompt=user_prompt,
        conversation_id=conversation_id,
        authorization=auth["authorization"],
        cookie_header=auth["cookie_header"],
        mode=auth["mode"],
        user_agent=auth["user_agent"],
        switch_mode_first=False,
        receive_timeout=request.receive_timeout,
        first_byte_timeout=request.first_byte_timeout,
        template_name=request.template_name,
    )

    return MuseProviderResponse(
        text=_ensure_text_reply(text),
        conversation_id=conversation_id,
        template_name=request.template_name,
        mode=auth["mode"],
        user_agent=auth["user_agent"],
    )


async def stream_from_state_async(
    request: MuseProviderRequest,
    state_path: Union[Path, str] = DEFAULT_STATE_PATH,
    stream_fn: AsyncStreamTextFn = stream_text_deltas,
    warmup_fn: WarmupFn = graphql_warmup_conversation,
    mode_switch_fn: ModeSwitchFn = graphql_mode_switch,
    conversation_id_factory: ConversationIdFactory = lambda: str(uuid.uuid4()),
) -> AsyncIterator[str]:
    auth = await asyncio.to_thread(load_provider_auth, state_path)
    conversation_id = request.conversation_id or str(conversation_id_factory())

    if request.needs_warmup:
        await asyncio.to_thread(
            warmup_fn,
            conversation_id,
            auth["cookie_header"],
            user_agent=auth["user_agent"],
        )
        await asyncio.to_thread(
            mode_switch_fn,
            conversation_id=conversation_id,
            cookie_header=auth["cookie_header"],
            mode=auth["mode"],
            user_agent=auth["user_agent"],
        )
    user_prompt = request.user_prompt or request.prompt
    async for chunk in stream_fn(
        prompt=user_prompt,
        conversation_id=conversation_id,
        authorization=auth["authorization"],
        cookie_header=auth["cookie_header"],
        mode=auth["mode"],
        user_agent=auth["user_agent"],
        switch_mode_first=False,
        receive_timeout=request.receive_timeout,
        first_byte_timeout=request.first_byte_timeout,
        template_name=request.template_name,
    ):
        yield chunk


def generate_from_state(
    request: MuseProviderRequest,
    state_path: Union[Path, str] = DEFAULT_STATE_PATH,
    generate_fn: Optional[Callable[..., str]] = None,
    warmup_fn: WarmupFn = graphql_warmup_conversation,
    mode_switch_fn: ModeSwitchFn = graphql_mode_switch,
    conversation_id_factory: ConversationIdFactory = lambda: str(uuid.uuid4()),
) -> MuseProviderResponse:
    if generate_fn is None:
        return asyncio.run(
            generate_from_state_async(
                request,
                state_path=state_path,
                warmup_fn=warmup_fn,
                mode_switch_fn=mode_switch_fn,
                conversation_id_factory=conversation_id_factory,
            )
        )

    auth = load_provider_auth(state_path)
    conversation_id = request.conversation_id or str(conversation_id_factory())

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
    text = generate_fn(
        prompt=request.prompt,
        conversation_id=conversation_id,
        authorization=auth["authorization"],
        cookie_header=auth["cookie_header"],
        mode=auth["mode"],
        user_agent=auth["user_agent"],
        switch_mode_first=False,
        receive_timeout=request.receive_timeout,
        template_name=request.template_name,
    )

    return MuseProviderResponse(
        text=_ensure_text_reply(text),
        conversation_id=conversation_id,
        template_name=request.template_name,
        mode=auth["mode"],
        user_agent=auth["user_agent"],
    )
