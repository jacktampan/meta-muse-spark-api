from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class ApiSettings:
    model_name: str = "meta/muse-spark"
    log_level: str = "INFO"
    stream_chunk_size: int = 0
    debug_frame_dumps: bool = False
    force_single_conversation: bool = False
    # Idle timeout per WebSocket recv. Bumped from 10s to 30s because Meta can
    # pause mid-generation and a too-short timeout silently truncates output.
    receive_timeout: float = 30.0

    @classmethod
    def from_env(cls) -> "ApiSettings":
        return cls(
            model_name=os.getenv("MUSE_SPARK_MODEL_NAME", "meta/muse-spark"),
            log_level=os.getenv("MUSE_SPARK_LOG_LEVEL", "INFO").upper(),
            stream_chunk_size=int(os.getenv("MUSE_SPARK_STREAM_CHUNK_SIZE", "0")),
            debug_frame_dumps=os.getenv("MUSE_SPARK_DEBUG_FRAME_DUMPS", "0").lower() in {"1", "true", "yes", "on"},
            force_single_conversation=os.getenv("MUSE_SPARK_FORCE_SINGLE_CONVERSATION", "0").lower() in {"1", "true", "yes", "on"},
            receive_timeout=float(os.getenv("MUSE_SPARK_RECEIVE_TIMEOUT", "30.0")),
        )
