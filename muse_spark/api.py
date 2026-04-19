from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable, Optional, Union

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .client import DEFAULT_STATE_PATH
from .config import ApiSettings
from .errors import MissingAuthError, ProviderProtocolError, ProviderTransportError, ReauthRequiredError
from .logging_utils import get_logger
from .openai_compat import (
    build_chat_completion_chunk,
    build_chat_completion_response,
    build_error_response,
    build_models_response,
    encode_sse_data,
    encode_sse_done,
)
from .prompt_compiler import compile_chat_messages
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
        chunk_size = 120
    return [text[index : index + chunk_size] for index in range(0, len(text), chunk_size)] or [""]



def run_api_server(*, host: str = "127.0.0.1", port: int = 8000, state_path: Union[Path, str] = DEFAULT_STATE_PATH) -> None:
    import uvicorn

    print("Muse Spark API")
    print(f"  URL: http://{host}:{port}")
    print(f"  State: {state_path}")
    print("  Endpoints: /healthz /readyz /v1/models /v1/chat/completions")
    app = create_app(state_path=state_path)
    uvicorn.run(app, host=host, port=port)


async def _stream_chat_completion(
    *,
    chunk_iter: AsyncIterator[str],
    model: str,
    response_id: str,
    conversation_id: str,
    chunk_size: int,
) -> AsyncIterator[bytes]:
    yield encode_sse_data(
        build_chat_completion_chunk(
            model=model,
            response_id=response_id,
            delta={"role": "assistant"},
            conversation_id=conversation_id,
        )
    )
    async for provider_chunk in chunk_iter:
        for chunk in _chunk_text(provider_chunk, chunk_size):
            if chunk:
                yield encode_sse_data(
                    build_chat_completion_chunk(
                        model=model,
                        response_id=response_id,
                        delta={"content": chunk},
                        conversation_id=conversation_id,
                    )
                )
    yield encode_sse_data(
        build_chat_completion_chunk(
            model=model,
            response_id=response_id,
            delta={},
            finish_reason="stop",
            conversation_id=conversation_id,
        )
    )
    yield encode_sse_done()



def create_app(
    *,
    provider_generate_fn: ProviderGenerateFn = generate_from_state_async,
    provider_stream_fn: ProviderStreamFn = stream_from_state_async,
    compiler_fn: CompilerFn = compile_chat_messages,
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
    async def chat_completions(body: ChatCompletionsRequest):
        if body.model != settings.model_name:
            return error_json(400, "invalid_model", f"Unsupported model: {body.model}")

        compiled = compiler_fn(
            [message.model_dump() for message in body.messages],
            response_format=body.response_format.model_dump() if body.response_format else None,
            max_tokens=body.max_tokens,
            stop=_normalize_stop(body.stop),
        )
        resolved = resolve_api_conversation(state_path=state_path, client_conversation_id=body.conversation_id)
        provider_request = MuseProviderRequest(
            prompt=compiled.prompt,
            conversation_id=resolved.meta_conversation_id,
            template_name=resolved.template_name,
        )

        if body.stream:
            response_id = f"chatcmpl-{uuid.uuid4()}"
            chunk_iter = provider_stream_fn(provider_request, state_path=state_path)
            return StreamingResponse(
                _stream_chat_completion(
                    chunk_iter=chunk_iter,
                    model=settings.model_name,
                    response_id=response_id,
                    conversation_id=resolved.client_conversation_id,
                    chunk_size=settings.stream_chunk_size,
                ),
                media_type="text/event-stream",
            )

        provider_response = await provider_generate_fn(provider_request, state_path=state_path)
        return build_chat_completion_response(
            content=provider_response.text,
            model=settings.model_name,
            conversation_id=resolved.client_conversation_id,
        )

    return app
