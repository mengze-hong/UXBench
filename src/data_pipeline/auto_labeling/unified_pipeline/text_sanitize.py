"""Strip decorative Markdown / emoji from dialogue text (rule-based, no LLM)."""

from __future__ import annotations

import re


def _strip_emojis(text: str) -> str:
    out: list[str] = []
    for ch in text:
        o = ord(ch)
        if ch in ("\u200d", "\ufe0f", "\uFE0F"):
            continue
        if 0xFE00 <= o <= 0xFE0F:
            continue
        if (
            (0x1F300 <= o <= 0x1FAFF)
            or (0x2600 <= o <= 0x27BF)
            or (0x1F600 <= o <= 0x1F64F)
            or (0x1F680 <= o <= 0x1F6FF)
            or (0x1F900 <= o <= 0x1F9FF)
        ):
            continue
        out.append(ch)
    return "".join(out)


def strip_markdown_noise(text: str) -> str:
    if not text:
        return ""
    t = text.replace("\r\n", "\n")
    t = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", t)
    for _ in range(6):
        nt = re.sub(r"\*\*([^*]+)\*\*", r"\1", t)
        nt = re.sub(r"\*([^*]+)\*", r"\1", nt)
        nt = re.sub(r"__([^_]+)__", r"\1", nt)
        if nt == t:
            break
        t = nt
    t = re.sub(r"`([^`]+)`", r"\1", t)
    t = re.sub(r"^#{1,6}\s+", "", t, flags=re.MULTILINE)
    t = re.sub(r"^```[^\n]*\n.*?^```", "", t, flags=re.MULTILINE | re.DOTALL)
    t = re.sub(r"\n{5,}", "\n\n\n\n", t)
    return t.strip()


def strip_decorative_text(text: str) -> str:
    t = strip_markdown_noise(text or "")
    t = _strip_emojis(t)
    t = re.sub(r"[ \t]+\n", "\n", t)
    t = re.sub(r"\n{4,}", "\n\n\n", t)
    return t.strip()


def sanitize_user_profile(obj: object) -> object:
    if isinstance(obj, dict):
        return {str(k): sanitize_user_profile(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_user_profile(x) for x in obj]
    if isinstance(obj, str):
        return strip_decorative_text(obj)
    return obj
