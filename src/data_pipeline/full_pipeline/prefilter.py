"""Rule-based prefilter before LLM mining."""

from __future__ import annotations

import re

UNSAFE_PATTERNS = [r"(色情|裸聊|约炮|黄片|卖淫)", r"(操你|fuck\\s*you|傻逼|智障)", r"(怎么(自杀|上吊|割腕)|自杀方法)", r"(制作(毒品|冰毒|炸弹|爆炸物))"]
_UNSAFE_RE = re.compile("|".join(UNSAFE_PATTERNS), re.IGNORECASE)


def check_unsafe(history: list) -> tuple[bool, str]:
    for msg in history:
        text = msg.get("message", "") or ""
        m = _UNSAFE_RE.search(text)
        if m:
            return True, f"unsafe_match:{m.group(0)[:30]}"
    return False, ""


def prefilter(record: dict, enrichment: dict) -> tuple[bool, str]:
    history = record.get("history", [])
    if not enrichment.get("unliked_turns"):
        return False, "no_unliked_turns"
    if enrichment.get("turn_count", 0) < 2:
        return False, "too_short"
    asst_turns = [t for t in enrichment.get("turns", []) if t.get("role") == "assistant"]
    if asst_turns and all(t.get("interrupted") for t in asst_turns):
        return False, "all_interrupted"
    unsafe, reason = check_unsafe(history)
    if unsafe:
        return False, reason
    return True, "pass"
