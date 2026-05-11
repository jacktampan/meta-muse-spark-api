from __future__ import annotations

import html
from dataclasses import dataclass
from typing import Any


@dataclass
class StatefulTurnPlan:
    """Compiled prompt for a single Meta turn.

    ``system_preamble`` carries any user-provided ``system``/``developer``
    instructions wrapped in XML so they can be prepended to the very first
    user turn. ``user_prompt`` carries the most recent user message wrapped
    in ``<conversation_turn><user_message>...`` scaffolding. For follow-up
    turns the caller should send ``user_prompt`` alone — Meta retains its
    own conversation state and re-sending the preamble each turn would
    just waste tokens.
    """

    user_prompt: str
    system_preamble: str = ""
    truncated: bool = False
    dropped_messages: int = 0
    kept_messages: int = 0


@dataclass
class _Message:
    role: str
    content: str


SYSTEM_ROLES = {"system", "developer"}



def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
                parts.append(item["text"].strip())
            elif isinstance(item, dict):
                parts.append(str(item))
            elif isinstance(item, str):
                parts.append(item.strip())
            elif item is not None:
                parts.append(str(item).strip())
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict):
        return str(content)
    if content is None:
        return ""
    return str(content).strip()



def _escape_xml(text: str) -> str:
    return html.escape(text, quote=False)



def _extract_messages(messages: list[dict[str, Any]]) -> tuple[list[str], list[_Message]]:
    system_messages: list[str] = []
    user_messages: list[_Message] = []
    for message in messages:
        role = str(message.get("role") or "user").strip() or "user"
        content = _content_to_text(message.get("content"))
        if not content:
            continue
        if role in SYSTEM_ROLES:
            system_messages.append(content)
        elif role == "user":
            user_messages.append(_Message(role=role, content=content))
    return system_messages, user_messages



def _build_system_preamble(system_messages: list[str], max_chars: int) -> tuple[str, bool]:
    """Render user-provided system/developer instructions as an XML preamble.

    Returns an empty string when no system messages were supplied — the
    caller treats that as "nothing to prepend". Unlike the previous
    bootstrap-prompt scheme, this preamble has *no* fixed scaffolding text
    (no "Reply with exactly READY", no fixed instruction list). Only the
    caller's own system messages are emitted, so a request without any
    system messages results in an empty preamble.
    """
    if not system_messages:
        return "", False
    setup_bits = [
        "<conversation_setup>",
        "  <system_instructions>",
    ]
    for index, message in enumerate(system_messages, start=1):
        setup_bits.append(
            f"    <instruction index=\"{index}\">{_escape_xml(message)}</instruction>"
        )
    setup_bits.extend(
        [
            "  </system_instructions>",
            "</conversation_setup>",
        ]
    )
    prompt = "\n".join(setup_bits).strip()
    if len(prompt) <= max_chars:
        return prompt, False
    return prompt[:max_chars].rstrip(), True



def _build_user_prompt(message: _Message, max_chars: int) -> tuple[str, bool]:
    prompt = "\n".join(
        [
            "<conversation_turn>",
            f"  <user_message>{_escape_xml(message.content)}</user_message>",
            "</conversation_turn>",
        ]
    )
    if len(prompt) <= max_chars:
        return prompt, False
    if max_chars <= 3:
        return prompt[:max_chars], True
    return prompt[: max_chars - 3].rstrip() + "...", True



def build_stateful_turn_plan(
    messages: list[dict[str, Any]],
    *,
    max_chars: int = 12000,
) -> StatefulTurnPlan:
    if not messages:
        raise ValueError("messages must not be empty")

    system_messages, user_messages = _extract_messages(messages)
    if not user_messages:
        raise ValueError("messages must include at least one user message")

    latest_user = user_messages[-1]
    system_preamble, preamble_truncated = _build_system_preamble(system_messages, max_chars)
    user_prompt, user_truncated = _build_user_prompt(latest_user, max_chars)

    return StatefulTurnPlan(
        user_prompt=user_prompt,
        system_preamble=system_preamble,
        truncated=preamble_truncated or user_truncated,
        dropped_messages=max(0, len(user_messages) - 1),
        kept_messages=1,
    )
