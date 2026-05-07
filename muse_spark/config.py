from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class ApiSettings:
    model_name: str = "meta/muse-spark"
    log_level: str = "INFO"
    stream_chunk_size: int = 0
    debug_frame_dumps: bool = False
    force_single_conversation: bool = False
    # Idle timeout per WebSocket recv. Bumped to 60s because Meta can pause
    # for tens of seconds mid-generation under load — a 30s ceiling caused
    # legitimate slow responses to fail the stall-detection guard. Override
    # via ``MUSE_SPARK_RECEIVE_TIMEOUT`` if your network/load tolerates less.
    receive_timeout: float = 60.0
    # Tighter idle window applied *before* any token has streamed. A healthy
    # Meta turn produces the first patch within a couple of seconds, so a
    # multi-second silence pre-first-byte is almost always a stuck-conversation
    # symptom (post-stall, throttling, or backend wedge). Failing fast here
    # lets the SSE recovery path purge + retry sooner instead of holding the
    # client connection open for the full ``receive_timeout`` (which is sized
    # for between-tokens slowness, a different failure mode). Override via
    # ``MUSE_SPARK_FIRST_BYTE_TIMEOUT``; set to the same value as
    # ``receive_timeout`` (or higher) to disable the optimisation.
    first_byte_timeout: Optional[float] = 20.0

    @classmethod
    def from_env(cls) -> "ApiSettings":
        first_byte_raw = os.getenv("MUSE_SPARK_FIRST_BYTE_TIMEOUT", "20.0")
        first_byte = float(first_byte_raw) if first_byte_raw else None
        return cls(
            model_name=os.getenv("MUSE_SPARK_MODEL_NAME", "meta/muse-spark"),
            log_level=os.getenv("MUSE_SPARK_LOG_LEVEL", "INFO").upper(),
            stream_chunk_size=int(os.getenv("MUSE_SPARK_STREAM_CHUNK_SIZE", "0")),
            debug_frame_dumps=os.getenv("MUSE_SPARK_DEBUG_FRAME_DUMPS", "0").lower() in {"1", "true", "yes", "on"},
            force_single_conversation=os.getenv("MUSE_SPARK_FORCE_SINGLE_CONVERSATION", "0").lower() in {"1", "true", "yes", "on"},
            receive_timeout=float(os.getenv("MUSE_SPARK_RECEIVE_TIMEOUT", "60.0")),
            first_byte_timeout=first_byte,
        )
