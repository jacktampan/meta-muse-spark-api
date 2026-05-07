from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable, Optional, Union

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .client import CHAT_TEMPLATE_NAME, DEFAULT_STATE_PATH, HOME_TEMPLATE_NAME
from .config import ApiSettings
from .errors import (
    MissingAuthError,
    ProviderProtocolError,
    ProviderStallError,
    ProviderTransportError,
    ReauthRequiredError,
)
from .logging_utils import get_logger
from .openai_compat import (
    build_chat_completion_chunk,
    build_chat_completion_response,
    build_error_response,
    build_models_response,
    encode_sse_data,
    encode_sse_done,
)
from .prompt_compiler import build_stateful_turn_plan
from .provider import (
    MuseProviderRequest,
    generate_from_state_async,
    load_provider_auth,
    resolve_api_conversation,
    stream_from_state_async,
)
from .schemas import ChatCompletionsRequest

ProviderGenerateFn = Callable[..., Awaitable[Any]]
ProviderStreamFn = Callable[..., AsyncIterator[str]]
CompilerFn = Callable[..., Any]
LoadAuthFn = Callable[..., dict[str, str]]



def _normalize_stop(stop: Optional[Union[str, list[str]]]) -> Optional[list[str]]:
    if isinstance(stop, list):
        return stop
    if isinstance(stop, str):
        return [stop]
    return None



def _chunk_text(text: str, chunk_size: int) -> list[str]:
    if chunk_size <= 0:
        return [text] if text else [""]
    return [text[index : index + chunk_size] for index in range(0, len(text), chunk_size)] or [""]


# Tags emitted by build_stateful_turn_plan that may leak back from Meta's
# stateful conversation context into user-visible output. We strip them from
# both streaming and non-streaming responses.
_SCAFFOLDING_TAG_PATTERN = re.compile(
    r"</?(?:conversation_setup|conversation_turn|user_message|system_instructions|"
    r"conversation_preamble|instruction|acknowledgement)\b[^>]*>",
    re.IGNORECASE,
)
_SCAFFOLDING_BARE_READY = re.compile(r"^\s*READY\.?\s*$", re.IGNORECASE)

# Meta's stateful conversation responses occasionally leak internal "Inline
# Entity" markup of the form ``{{IE_<id>}}content{{/IE_<id>}}``. Meta's web UI
# renders these into hyperlinks, citations, and entity highlights — but they
# surface as raw text through the OpenAI-compatible API. Examples observed in
# production:
#   {{IE_30000}}Paris Saint-Germain{{/IE_30000}}   (entity name — unwrap)
#   {{IE_1}}d5cd{{/IE_1}}                          (citation hash — drop)
#   {{IE_3}}post-862616848552664650850{{/IE_3}}    (citation post-id — drop)
#   {{IE_2}}c3ce{{/IE_2}}0b57{{/IE_2}}             (orphan close — drop)
_INLINE_ENTITY_PAIR = re.compile(
    r"\{\{IE_([\w-]+)\}\}(.*?)\{\{/IE_\1\}\}",
    re.DOTALL,
)
_INLINE_ENTITY_OPENER = re.compile(r"\{\{IE_([\w-]+)\}\}")
_INLINE_ENTITY_ORPHAN = re.compile(r"\{\{/?IE_[\w-]+\}\}")
# Pure lowercase hex (≥4 chars) or ``post-…`` IDs are citation references and
# carry no user-facing meaning — drop the entire pair. Other content (e.g.
# proper-noun entity names like ``Paris Saint-Germain``) is unwrapped and kept.
_INLINE_ENTITY_CITATION_LIKE = re.compile(r"^(?:[a-f0-9]{4,}|post-[\w-]+)$")


def _replace_ie_pair(match: re.Match[str]) -> str:
    content = match.group(2)
    if not content or _INLINE_ENTITY_CITATION_LIKE.match(content):
        return ""
    return content


def _strip_inline_entities(text: str) -> str:
    """Remove Meta's internal inline-entity markup from assistant output.

    See ``_INLINE_ENTITY_PAIR`` above for examples. Pairs whose content looks
    like a citation reference are dropped entirely; pairs with human-readable
    content are unwrapped (content kept). Orphan tags (open without close, or
    close without open) are stripped.
    """
    if "{{" not in text:
        return text
    # Apply pair replacement until stable to handle the rare case where Meta
    # emits nested entity markers (inner pair drops first, outer pair drops
    # on the next iteration once it becomes a flat run of text).
    while True:
        new_text = _INLINE_ENTITY_PAIR.sub(_replace_ie_pair, text)
        if new_text == text:
            break
        text = new_text
    return _INLINE_ENTITY_ORPHAN.sub("", text)


def _clean_assistant_text(text: Optional[str]) -> str:
    """Strip stateful-turn scaffolding tags that occasionally leak into output."""
    if not text:
        return ""
    cleaned = _SCAFFOLDING_TAG_PATTERN.sub("", text)
    cleaned = _strip_inline_entities(cleaned)
    if _SCAFFOLDING_BARE_READY.match(cleaned):
        return ""
    return cleaned


class _ScaffoldingStripper:
    """Streaming-safe stripper for scaffolding tags and inline-entity markup.

    Holds back only when a chunk ends with what could be a partial marker —
    ``<`` without ``>``, ``{{`` without ``}}``, or a complete ``{{IE_X}}``
    opener whose matching ``{{/IE_X}}`` close hasn't arrived yet. All other
    content is emitted immediately, preserving streaming UX.
    """

    def __init__(self) -> None:
        self._buffer = ""

    @staticmethod
    def _clean(text: str) -> str:
        text = _SCAFFOLDING_TAG_PATTERN.sub("", text)
        return _strip_inline_entities(text)

    @staticmethod
    def _hold_position(text: str) -> int:
        """Earliest index from which content might be incomplete markup.

        Returns ``-1`` when the entire buffer is safe to emit.
        """
        candidates: list[int] = []

        # Partial scaffolding tag at the end (``<`` with no ``>`` after it).
        last_lt = text.rfind("<")
        if last_lt != -1 and ">" not in text[last_lt:]:
            candidates.append(last_lt)

        # Partial brace pair at the end (``{{`` with no ``}}`` after it).
        last_brace = text.rfind("{{")
        if last_brace != -1 and "}}" not in text[last_brace:]:
            candidates.append(last_brace)

        # Complete ``{{IE_X}}`` opener whose matching close hasn't arrived.
        # Walk earliest-first; first unmatched opener is enough since holding
        # there suspends every later opener too.
        for match in _INLINE_ENTITY_OPENER.finditer(text):
            needle = "{{/IE_" + match.group(1) + "}}"
            if needle not in text[match.end():]:
                candidates.append(match.start())
                break

        return min(candidates) if candidates else -1

    def feed(self, chunk: str) -> str:
        if not chunk:
            return ""
        self._buffer += chunk
        hold = self._hold_position(self._buffer)
        if hold == -1:
            emit = self._clean(self._buffer)
            self._buffer = ""
            return emit
        emit = self._clean(self._buffer[:hold])
        self._buffer = self._buffer[hold:]
        return emit

    def flush(self) -> str:
        cleaned = self._clean(self._buffer)
        self._buffer = ""
        if _SCAFFOLDING_BARE_READY.match(cleaned):
            return ""
        return cleaned



def run_api_server(
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    state_path: Union[Path, str] = DEFAULT_STATE_PATH,
    force_single_conversation: Optional[bool] = None,
    stream_chunk_size: Optional[int] = None,
    receive_timeout: Optional[float] = None,
) -> None:
    import uvicorn

    print("Muse Spark API")
    print(f"  URL: http://{host}:{port}")
    print(f"  State: {state_path}")
    print("  Endpoints: /healthz /readyz /v1/models /v1/chat/completions")

    settings = ApiSettings.from_env()
    if force_single_conversation is not None:
        settings.force_single_conversation = force_single_conversation
    if stream_chunk_size is not None:
        settings.stream_chunk_size = stream_chunk_size
    if receive_timeout is not None:
        settings.receive_timeout = receive_timeout

    app = create_app(state_path=state_path, settings=settings)
    uvicorn.run(app, host=host, port=port)


async def _stream_chat_completion(
    *,
    chunk_iter: AsyncIterator[str],
    model: str,
    response_id: str,
    conversation_id: str,
    chunk_size: int,
    logger: Any,
    bootstrap_response: Optional[str] = None,
) -> AsyncIterator[bytes]:
    # Send role chunk first so clients (including OpenAI SDK) see the connection
    # is alive immediately, even if the upstream provider is still warming up.
    yield encode_sse_data(
        build_chat_completion_chunk(
            model=model,
            response_id=response_id,
            delta={"role": "assistant"},
            conversation_id=conversation_id,
            bootstrap_response=bootstrap_response,
        )
    )

    stripper = _ScaffoldingStripper()
    finish_reason = "stop"
    try:
        async for provider_chunk in chunk_iter:
            cleaned = stripper.feed(provider_chunk)
            if not cleaned:
                continue
            for chunk in _chunk_text(cleaned, chunk_size):
                if chunk:
                    yield encode_sse_data(
                        build_chat_completion_chunk(
                            model=model,
                            response_id=response_id,
                            delta={"content": chunk},
                            conversation_id=conversation_id,
                            # Only include bootstrap in the very first content chunk
                            # to keep the payload small.
                            bootstrap_response=bootstrap_response,
                        )
                    )
                bootstrap_response = None
    except ProviderStallError as exc:
        # Mid-response stall: we already streamed real content to the client.
        # Surface it as a graceful truncation (``finish_reason="length"``)
        # rather than an error, so OpenAI-compatible clients keep the partial
        # output and don't raise on the caller side.
        logger.warning("streaming_stalled: %s", exc)
        finish_reason = "length"
    except Exception:
        logger.exception("streaming_failed")
        finish_reason = "error"

    # Always flush any buffered tail (e.g. content the stripper was holding
    # back behind a potential ``<`` or ``{{`` marker). The held content was
    # generated before the loop ended and should reach the client even on
    # stall or error — otherwise we'd silently drop legitimate tokens while
    # claiming graceful truncation. ``flush()`` sanitises orphan markers, so
    # incomplete entity scaffolding cannot leak.
    tail = stripper.flush()
    if tail:
        for chunk in _chunk_text(tail, chunk_size):
            if chunk:
                yield encode_sse_data(
                    build_chat_completion_chunk(
                        model=model,
                        response_id=response_id,
                        delta={"content": chunk},
                        conversation_id=conversation_id,
                        bootstrap_response=bootstrap_response,
                    )
                )
            bootstrap_response = None

    # Always emit a terminal chunk + [DONE] so clients don't hang. Use
    # finish_reason="error" on failure so callers can detect partial output.
    yield encode_sse_data(
        build_chat_completion_chunk(
            model=model,
            response_id=response_id,
            delta={},
            finish_reason=finish_reason,
            conversation_id=conversation_id,
            bootstrap_response=bootstrap_response,
        )
    )
    yield encode_sse_done()



def create_app(
    *,
    provider_generate_fn: ProviderGenerateFn = generate_from_state_async,
    provider_stream_fn: ProviderStreamFn = stream_from_state_async,
    compiler_fn: CompilerFn = build_stateful_turn_plan,
    load_auth_fn: LoadAuthFn = load_provider_auth,
    state_path: Union[Path, str] = DEFAULT_STATE_PATH,
    settings: Optional[ApiSettings] = None,
) -> FastAPI:
    settings = settings or ApiSettings.from_env()
    logger = get_logger("muse_spark.api", settings.log_level)
    app = FastAPI(title="Muse Spark OpenAI-Compatible API")
    app.state.settings = settings

    def error_json(status_code: int, code: str, message: str, error_type: str = "invalid_request_error") -> JSONResponse:
        return JSONResponse(
            status_code=status_code,
            content=build_error_response(code=code, message=message, error_type=error_type),
        )

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("request_failed method=%s path=%s", request.method, request.url.path)
            raise
        logger.info("request method=%s path=%s status=%s", request.method, request.url.path, response.status_code)
        return response

    @app.exception_handler(MissingAuthError)
    async def handle_missing_auth(request, exc: MissingAuthError):
        return error_json(503, "missing_auth", str(exc), error_type="auth_error")

    @app.exception_handler(ReauthRequiredError)
    async def handle_reauth_required(request, exc: ReauthRequiredError):
        return error_json(401, "reauth_required", str(exc), error_type="auth_error")

    @app.exception_handler(ProviderTransportError)
    async def handle_transport_error(request, exc: ProviderTransportError):
        return error_json(502, "provider_transport_error", str(exc), error_type="provider_error")

    @app.exception_handler(ProviderProtocolError)
    async def handle_protocol_error(request, exc: ProviderProtocolError):
        return error_json(502, "provider_protocol_error", str(exc), error_type="provider_error")

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {"ok": True}

    @app.get("/readyz")
    async def readyz() -> dict[str, Any]:
        load_auth_fn(state_path)
        return {"ok": True, "model": settings.model_name}

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        return build_models_response([settings.model_name])

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request, body: ChatCompletionsRequest):
        if body.model != settings.model_name:
            return error_json(400, "invalid_model", f"Unsupported model: {body.model}")

        if not body.messages:
            return error_json(400, "invalid_request", "messages must not be empty")

        try:
            compiled = compiler_fn([message.model_dump() for message in body.messages])
        except ValueError as exc:
            return error_json(400, "invalid_request", str(exc))

        # Sticky conversation resolution priority:
        #   1. body.conversation_id (explicit, OpenAI vendor extension)
        #   2. X-Conversation-Id header (agent frameworks that expose custom headers)
        #   3. muse_spark_conv cookie (browser-style clients)
        #   4. settings.force_single_conversation (server-wide single conversation mode)
        sticky_id = (
            body.conversation_id
            or request.headers.get("x-conversation-id")
            or request.cookies.get("muse_spark_conv")
        )
        resolved = resolve_api_conversation(
            state_path=state_path,
            client_conversation_id=sticky_id,
            force_single_conversation=settings.force_single_conversation,
        )

        main_template = resolved.template_name
        bootstrap_response_text: Optional[str] = None
        # Bootstrap warms the conversation on Meta side; subsequent calls in
        # the same request don't need their own warmup/mode_switch round-trip.
        bootstrap_already_warmed = False
        if resolved.template_name == HOME_TEMPLATE_NAME and compiled.bootstrap_prompt:
            bootstrap_response = await provider_generate_fn(
                MuseProviderRequest(
                    prompt=compiled.bootstrap_prompt,
                    conversation_id=resolved.meta_conversation_id,
                    template_name=HOME_TEMPLATE_NAME,
                    # Honour the configured timeout for the bootstrap call too —
                    # otherwise users who raise MUSE_SPARK_RECEIVE_TIMEOUT for
                    # slow networks still get the dataclass default here.
                    receive_timeout=settings.receive_timeout,
                    needs_warmup=resolved.is_new,
                ),
                state_path=state_path,
            )
            main_template = CHAT_TEMPLATE_NAME
            bootstrap_already_warmed = True
            if body.include_bootstrap_response:
                bootstrap_response_text = bootstrap_response.text

        provider_request = MuseProviderRequest(
            prompt=compiled.user_prompt,
            conversation_id=resolved.meta_conversation_id,
            template_name=main_template,
            user_prompt=compiled.user_prompt,
            receive_timeout=settings.receive_timeout,
            # Skip warmup for follow-ups (conversation already exists on Meta)
            # and for the second call in a bootstrap round (just warmed).
            needs_warmup=resolved.is_new and not bootstrap_already_warmed,
        )

        if body.stream:
            response_id = f"chatcmpl-{uuid.uuid4()}"
            chunk_iter = provider_stream_fn(provider_request, state_path=state_path)
            response = StreamingResponse(
                _stream_chat_completion(
                    chunk_iter=chunk_iter,
                    model=settings.model_name,
                    response_id=response_id,
                    conversation_id=resolved.client_conversation_id,
                    chunk_size=settings.stream_chunk_size,
                    logger=logger,
                    bootstrap_response=bootstrap_response_text,
                ),
                media_type="text/event-stream",
            )
            response.set_cookie(
                "muse_spark_conv",
                resolved.client_conversation_id,
                max_age=60 * 60 * 24 * 30,
                httponly=True,
                samesite="lax",
            )
            return response

        provider_response = await provider_generate_fn(provider_request, state_path=state_path)
        response = JSONResponse(
            content=build_chat_completion_response(
                content=_clean_assistant_text(provider_response.text),
                model=settings.model_name,
                conversation_id=resolved.client_conversation_id,
                bootstrap_response=bootstrap_response_text,
            )
        )
        response.set_cookie(
            "muse_spark_conv",
            resolved.client_conversation_id,
            max_age=60 * 60 * 24 * 30,
            httponly=True,
            samesite="lax",
        )
        return response

    return app
