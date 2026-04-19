from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class CompiledPrompt:
    prompt: str
    truncated: bool
    dropped_messages: int
    kept_messages: int


@dataclass
class _Turn:
    role: str
    content: str



def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
                parts.append(item["text"].strip())
            elif isinstance(item, dict):
                parts.append(json.dumps(item, sort_keys=True))
            elif isinstance(item, str):
                parts.append(item.strip())
            elif item is not None:
                parts.append(str(item).strip())
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict):
        return json.dumps(content, sort_keys=True)
    if content is None:
        return ""
    return str(content).strip()



def _format_turn(turn: _Turn) -> str:
    return f"[{turn.role}]\n{turn.content.strip()}"



def _build_system_block(system_messages: list[str]) -> str:
    if not system_messages:
        return ""
    lines = ["System instructions:"]
    lines.extend(f"- {message}" for message in system_messages)
    return "\n".join(lines)



def _build_requirements_block(
    response_format: Optional[dict[str, Any]] = None,
    max_tokens: Optional[int] = None,
    stop: Optional[list[str]] = None,
) -> str:
    lines = [
        "Response requirements:",
        "- Answer the latest user request directly.",
        "- Preserve markdown code fences when returning code.",
    ]
    if response_format and response_format.get("type") == "json_object":
        lines.append("- Return only valid JSON.")
        lines.append("- Do not include markdown fences or explanatory text.")
    if max_tokens is not None:
        lines.append(f"- Keep the answer within about {max_tokens} tokens if possible.")
    if stop:
        rendered = ", ".join(str(item) for item in stop)
        lines.append(f"- Stop when you reach one of these sequences if possible: {rendered}")
    return "\n".join(lines)



def _build_prompt(system_block: str, transcript_entries: list[str], requirements_block: str) -> str:
    parts: list[str] = []
    if system_block:
        parts.append(system_block)
    if transcript_entries:
        parts.append("Conversation:\n" + "\n\n".join(transcript_entries))
    if requirements_block:
        parts.append(requirements_block)
    return "\n\n".join(part for part in parts if part).strip()



def _truncate_turn_to_fit(turn: _Turn, system_block: str, requirements_block: str, max_chars: int) -> _Turn:
    prefix = f"[{turn.role}]\n"
    skeleton_len = len(_build_prompt(system_block, [prefix], requirements_block))
    available = max(0, max_chars - skeleton_len)
    if len(turn.content) <= available:
        return turn
    if available <= 3:
        trimmed = turn.content[:available]
    else:
        trimmed = turn.content[: available - 3].rstrip() + "..."
    return _Turn(role=turn.role, content=trimmed)



def compile_chat_messages(
    messages: list[dict[str, Any]],
    *,
    max_chars: int = 12000,
    response_format: Optional[dict[str, Any]] = None,
    max_tokens: Optional[int] = None,
    stop: Optional[list[str]] = None,
) -> CompiledPrompt:
    if not messages:
        raise ValueError("messages must not be empty")

    system_messages: list[str] = []
    transcript_turns: list[_Turn] = []
    for message in messages:
        role = str(message.get("role") or "user").strip() or "user"
        content = _content_to_text(message.get("content"))
        if not content:
            continue
        if role in {"system", "developer"}:
            system_messages.append(content)
        else:
            transcript_turns.append(_Turn(role=role, content=content))

    if not system_messages and not transcript_turns:
        raise ValueError("messages must contain at least one non-empty content block")

    system_block = _build_system_block(system_messages)
    requirements_block = _build_requirements_block(
        response_format=response_format,
        max_tokens=max_tokens,
        stop=stop,
    )

    if not transcript_turns:
        full_prompt = _build_prompt(system_block, [], requirements_block)
        was_truncated = len(full_prompt) > max_chars
        prompt = full_prompt[:max_chars].rstrip() if was_truncated else full_prompt
        return CompiledPrompt(prompt=prompt, truncated=was_truncated, dropped_messages=0, kept_messages=0)

    kept_turns: list[_Turn] = []
    truncated = False
    full_prompt = _build_prompt(
        system_block,
        [_format_turn(item) for item in transcript_turns],
        requirements_block,
    )
    if len(full_prompt) <= max_chars:
        kept_turns = transcript_turns
    else:
        truncated = True
        for start in range(1, len(transcript_turns)):
            candidate = transcript_turns[start:]
            candidate_prompt = _build_prompt(
                system_block,
                [_format_turn(item) for item in candidate],
                requirements_block,
            )
            if len(candidate_prompt) <= max_chars:
                kept_turns = candidate
                break

    if not kept_turns:
        kept_turns = [_truncate_turn_to_fit(transcript_turns[-1], system_block, requirements_block, max_chars)]
        truncated = True

    prompt = _build_prompt(
        system_block,
        [_format_turn(item) for item in kept_turns],
        requirements_block,
    )

    if len(prompt) > max_chars:
        latest_only = _truncate_turn_to_fit(transcript_turns[-1], system_block, requirements_block, max_chars)
        latest_prompt = _build_prompt(system_block, [_format_turn(latest_only)], requirements_block)
        if len(latest_prompt) <= max_chars:
            kept_turns = [latest_only]
            prompt = latest_prompt
        else:
            prompt = latest_prompt[:max_chars].rstrip()
        truncated = True

    kept_messages = len(kept_turns)
    dropped_messages = max(0, len(transcript_turns) - kept_messages)
    return CompiledPrompt(
        prompt=prompt,
        truncated=truncated,
        dropped_messages=dropped_messages,
        kept_messages=kept_messages,
    )
