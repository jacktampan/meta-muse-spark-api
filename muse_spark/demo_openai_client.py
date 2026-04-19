from __future__ import annotations

import argparse
import os
from typing import Iterable, Optional

DEFAULT_BASE_URL = os.getenv("MUSE_SPARK_BASE_URL", "http://127.0.0.1:8000/v1")
DEFAULT_API_KEY = os.getenv("MUSE_SPARK_API_KEY", "x")
DEFAULT_MODEL = os.getenv("MUSE_SPARK_MODEL_NAME", "meta/muse-spark")
DEFAULT_PROMPT = "Reply with exactly: muse spark is live"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tiny OpenAI SDK demo for a local Muse Spark endpoint."
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--stream", action="store_true")
    return parser


def create_client(*, base_url: str, api_key: str):
    from openai import OpenAI

    return OpenAI(base_url=base_url, api_key=api_key)


def _extract_delta_text(chunk) -> str:
    choices = getattr(chunk, "choices", None) or []
    if not choices:
        return ""
    delta = getattr(choices[0], "delta", None)
    if delta is None:
        return ""
    return getattr(delta, "content", None) or ""


def run_completion(*, client, model: str, prompt: str, stream: bool) -> str:
    if not stream:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""

    chunks: list[str] = []
    stream_response: Iterable[object] = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    for chunk in stream_response:
        text = _extract_delta_text(chunk)
        if text:
            chunks.append(text)
    return "".join(chunks)


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    client = create_client(base_url=args.base_url, api_key=args.api_key)
    text = run_completion(
        client=client,
        model=args.model,
        prompt=args.prompt,
        stream=args.stream,
    )

    print(f"base_url: {args.base_url}")
    print(f"model: {args.model}")
    print(f"prompt: {args.prompt}")
    if args.stream:
        print("mode: stream")
    print()
    print("response:")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
