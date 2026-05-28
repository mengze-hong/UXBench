"""Rule-based cleaning before LLM judge (aligned with legacy prefilter spirit)."""

from __future__ import annotations

import re

UNSAFE_PATTERNS = [
    r"(色情|裸聊|约炮|黄片|成人视频|情色|卖淫)",
    r"(操你|草你|fuck\s*you|傻逼|弱智|智障)",
    r"(怎么(自杀|上吊|割腕)|自杀方法)",
    r"(制作(毒品|冰毒|炸弹|爆炸物))",
]
_UNSAFE_RE = re.compile("|".join(UNSAFE_PATTERNS), re.IGNORECASE)

INTERRUPTED_MARKERS = (
    "回答中断",
    "已暂停生成",
    "视频生成失败",
    "未找到相关图片",
    "图片下载失败",
    "（无输出）",
    "(无输出)",
    "内容审核未通过",
)


def _has_unsafe(text: str) -> bool:
    return bool(text and _UNSAFE_RE.search(text))


def _is_interrupted(text: str) -> bool:
    t = (text or "").strip()
    return any(m in t for m in INTERRUPTED_MARKERS)


def clean_lite_case(case: dict, *, mode: str) -> tuple[bool, str]:
    """
    Returns (keep, reason).
    mode: 'bad' | 'good'
    """
    sq = case.get("source_query") or {}
    q = (sq.get("message") or "").strip()
    if len(q) < 2:
        return False, "query_too_short"

    if mode == "bad":
        body = (case.get("agent_response_full") or "").strip()
    else:
        body = (case.get("liked_response_full") or "").strip()

    if len(body) < 15:
        return False, "response_too_short"

    if _is_interrupted(body):
        return False, "interrupted_or_empty"

    if _has_unsafe(q) or _has_unsafe(body):
        return False, "unsafe"

    hist = case.get("selected_history") or []
    combined = q + "\n" + body + "\n" + "\n".join((h.get("message") or "") for h in hist)
    if _has_unsafe(combined):
        return False, "unsafe_in_history"

    return True, "pass"
