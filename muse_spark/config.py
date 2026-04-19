from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class ApiSettings:
    model_name: str = "meta/muse-spark"
    log_level: str = "INFO"
    stream_chunk_size: int = 120
    debug_frame_dumps: bool = False

    @classmethod
    def from_env(cls) -> "ApiSettings":
        return cls(
            model_name=os.getenv("MUSE_SPARK_MODEL_NAME", "meta/muse-spark"),
            log_level=os.getenv("MUSE_SPARK_LOG_LEVEL", "INFO").upper(),
            stream_chunk_size=int(os.getenv("MUSE_SPARK_STREAM_CHUNK_SIZE", "120")),
            debug_frame_dumps=os.getenv("MUSE_SPARK_DEBUG_FRAME_DUMPS", "0").lower() in {"1", "true", "yes", "on"},
        )
