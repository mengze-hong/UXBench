"""
Stage 0: Rule-based pre-filter.

Fast pure-Python filters to remove dialogues that:
- Contain unsafe content (sexual / abuse / sensitive)
- Have no meaningful unliked signal
- Are entirely composed of system/interrupted messages
- Are trivial (too short to be meaningful bad cases)

Output: (keep: bool, reject_reason: str)
"""

import re

# ⚠ Block-list for unsafe content. Keep conservative: only clearly dangerous patterns.
# We do NOT block general sensitive topics (politics, medical, legal) — those are legit user questions.
UNSAFE_PATTERNS = [
    # Sexual / porn
    r"(色情|裸聊|约炮|黄片|成人视频|情色|卖淫)",
    # Explicit abuse / slurs (very conservative — only blatant ones)
    r"(操你|草你|fuck\s*you|傻逼|弱智|智障)",
    # Self-harm instructions
    r"(怎么(自杀|上吊|割腕)|自杀方法)",
    # Illegal weapons/drugs manufacturing
    r"(制作(毒品|冰毒|炸弹|爆炸物))",
]
_UNSAFE_RE = re.compile("|".join(UNSAFE_PATTERNS), re.IGNORECASE)


def check_unsafe(history: list) -> tuple[bool, str]:
    """Check for unsafe content in user or assistant messages."""
    for msg in history:
        text = msg.get("message", "") or ""
        m = _UNSAFE_RE.search(text)
        if m:
            return True, f"unsafe_match:{m.group(0)[:30]}"
    return False, ""


def prefilter(record: dict, enrichment: dict) -> tuple[bool, str]:
    """
    Decide whether this dialogue passes pre-filter.
    Returns (keep: bool, reason: str).
    """
    history = record.get("history", [])

    # Rule 1: Must have at least one assistant turn with unlike signal
    if not enrichment.get("unliked_turns"):
        return False, "no_unliked_turns"

    # Rule 2: Must have enough substance
    if enrichment["turn_count"] < 2:
        return False, "too_short"

    # Rule 3: All assistant messages are system-interrupted → no real content to judge
    asst_turns = [t for t in enrichment["turns"] if t["role"] == "assistant"]
    if asst_turns and all(t["interrupted"] for t in asst_turns):
        return False, "all_interrupted"

    # Rule 4: Unsafe content check
    unsafe, reason = check_unsafe(history)
    if unsafe:
        return False, reason

    # Rule 5: Image-related check (already filtered in valid.jsonl, but double-check)
    # We trust is_image_related=false in the deepseek label if present

    return True, "pass"
