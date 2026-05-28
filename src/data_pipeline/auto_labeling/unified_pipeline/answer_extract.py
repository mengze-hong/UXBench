"""Best-effort plain text from raw `answer` field (JSON string or nested dict)."""

from __future__ import annotations

import json
from typing import Any


def split_answer_reasoning_and_body(raw: Any, *, max_chars: int = 120_000) -> tuple[str, str]:
    """
    AI Assistant deep_search 等卡片：把「思考/工具/引用卡片」与最终对用户回复的正文拆开。

    典型结构：content: [ {type: deepSearch, contents: [...]}, {type: searchGuid, ...}, {type: text, msg: 最终答复} ]

    Returns:
        (reasoning, body) 均为 strip 后的纯文本；无法识别时 reasoning==""，body 走旧版全文兜底。
    """
    half = max(8, max_chars // 2)

    def clip(s: str, n: int) -> str:
        s = (s or "").strip()
        if len(s) <= n:
            return s
        return s[:n] + "…"

    if raw is None:
        return "", ""
    if isinstance(raw, str):
        s = raw.strip()
        if s.startswith("{") and len(s) > 2:
            try:
                return split_answer_reasoning_and_body(json.loads(s), max_chars=max_chars)
            except json.JSONDecodeError:
                return "", clip(s, max_chars)
        return "", clip(s, max_chars)

    if isinstance(raw, (int, float, bool)):
        return "", clip(str(raw), max_chars)

    if isinstance(raw, dict):
        inner = raw.get("message")
        if isinstance(inner, str) and inner.strip().startswith("{"):
            try:
                r0, b0 = split_answer_reasoning_and_body(json.loads(inner), max_chars=max_chars)
                if b0 or r0:
                    return clip(r0, half), clip(b0, half)
            except json.JSONDecodeError:
                pass

        blocks = raw.get("content")
        if isinstance(blocks, list) and blocks:
            reasoning_chunks: list[str] = []
            reply_chunks: list[str] = []

            for block in blocks:
                if not isinstance(block, dict):
                    continue
                bt = block.get("type")

                if bt == "deepSearch":
                    for inner in block.get("contents") or []:
                        if not isinstance(inner, dict):
                            continue
                        it = inner.get("type")
                        if it == "text" and isinstance(inner.get("msg"), str) and inner["msg"].strip():
                            reasoning_chunks.append(inner["msg"].strip())
                        elif it == "toolCall":
                            title = inner.get("title") or ""
                            tcn = inner.get("toolCallName") or ""
                            bits = [x for x in (title, tcn) if x]
                            if bits:
                                reasoning_chunks.append("[" + " | ".join(bits) + "]")

                elif bt == "searchGuid":
                    title = block.get("title") or block.get("subTitle") or ""
                    if isinstance(title, str) and title.strip():
                        reasoning_chunks.append(f"[{title.strip()}]")

                elif bt == "think":
                    inner_text = block.get("content")
                    if isinstance(inner_text, str) and inner_text.strip():
                        reasoning_chunks.append(inner_text.strip())

                elif bt == "text" and isinstance(block.get("msg"), str) and block["msg"].strip():
                    reply_chunks.append(block["msg"].strip())

                elif isinstance(block.get("msg"), str) and block["msg"].strip() and bt not in (
                    "deepSearch",
                    "searchGuid",
                    "think",
                ):
                    reply_chunks.append(block["msg"].strip())

            reasoning = clip("\n\n".join(reasoning_chunks), half)
            body = clip("\n\n".join(reply_chunks), half)
            if body:
                return reasoning, body
            if reasoning:
                full = assistant_text_from_answer_field(raw, max_chars=max_chars)
                if len(full) > len(reasoning) + 80:
                    tail = full[len(reasoning) :].strip()
                    if len(tail) > 40:
                        return reasoning, clip(tail, half)
                return reasoning, clip(full, half)

    body = assistant_text_from_answer_field(raw, max_chars=max_chars)
    return "", body


def assistant_text_from_answer_field(raw: Any, *, max_chars: int = 120_000) -> str:
    """
    Flatten AI Assistant-style answer payloads (deep_search / Draw / text) into UTF-8 text.
    Preserves rough reading order; may include titles/URLs from search cards.
    """
    if raw is None:
        return ""
    if isinstance(raw, str):
        s = raw.strip()
        if s.startswith("{") and len(s) > 2:
            try:
                return assistant_text_from_answer_field(json.loads(s), max_chars=max_chars)
            except json.JSONDecodeError:
                return s[:max_chars]
        return s[:max_chars]
    if isinstance(raw, (int, float, bool)):
        return str(raw)
    if isinstance(raw, list):
        parts: list[str] = []
        n = 0
        for x in raw:
            t = assistant_text_from_answer_field(x, max_chars=max_chars - n).strip()
            if t:
                parts.append(t)
                n += len(t) + 1
                if n >= max_chars:
                    break
        return "\n".join(parts)[:max_chars]
    if isinstance(raw, dict):
        if isinstance(raw.get("msg"), str) and raw.get("msg", "").strip():
            return (raw.get("msg") or "")[:max_chars]
        priority_keys = ("text", "content", "contents", "answer", "message", "data", "steps", "docs")
        parts: list[str] = []
        n = 0
        for k in priority_keys:
            if k not in raw:
                continue
            t = assistant_text_from_answer_field(raw[k], max_chars=max_chars - n).strip()
            if t:
                parts.append(t)
                n += len(t) + 1
                if n >= max_chars:
                    break
        if parts:
            return "\n".join(parts)[:max_chars]
        for v in raw.values():
            t = assistant_text_from_answer_field(v, max_chars=max_chars - n).strip()
            if t:
                parts.append(t)
                n += len(t) + 1
                if n >= max_chars:
                    break
        return "\n".join(parts)[:max_chars]
    return str(raw)[:max_chars]
