"""Signal enrichment for full bad-case pipeline."""

from __future__ import annotations

import re
from datetime import datetime

COMPLAINT_PATTERNS = [r"不对", r"错了?", r"答非所问", r"废话", r"没用", r"不是这个", r"我问的是", r"无语"]
_COMPLAINT_RE = re.compile("|".join(f"({p})" for p in COMPLAINT_PATTERNS), re.IGNORECASE)

INTERRUPTED_MARKERS = {"回答中断", "已暂停生成", "视频生成失败", "未找到相关图片", "内容审核未通过"}


def parse_time(s: str):
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def detect_complaint(text: str) -> tuple[bool, str]:
    if not text or len(text) > 300:
        return False, ""
    m = _COMPLAINT_RE.search(text)
    return (True, m.group(0)) if m else (False, "")


def is_interrupted(text: str) -> bool:
    t = (text or "").strip()
    return t in INTERRUPTED_MARKERS or len(t) < 2


def compute_turn_signals(history: list) -> list[dict]:
    out = []
    prev_time = None
    prev_role = None
    prev_len = 0
    for msg in history:
        t = parse_time(msg.get("create_time", ""))
        gap_s = None
        gap_ratio = None
        skip_read = False
        if t is not None and prev_time is not None:
            gap_s = (t - prev_time).total_seconds()
            if prev_role == "assistant" and msg.get("role") == "user" and prev_len > 50:
                gap_ratio = gap_s / prev_len
                skip_read = gap_ratio < 0.15 and gap_s < 30
        out.append(
            {
                "turn_index": msg.get("turn_index"),
                "role": msg.get("role"),
                "is_unliked": int(msg.get("is_unliked", 0)) == 1,
                "length": msg.get("length") or len(msg.get("message", "") or ""),
                "gap_s": round(gap_s, 2) if gap_s is not None else None,
                "gap_ratio": round(gap_ratio, 4) if gap_ratio is not None else None,
                "skip_read": skip_read,
                "interrupted": msg.get("role") == "assistant" and is_interrupted(msg.get("message", "")),
            }
        )
        if t is not None:
            prev_time = t
        prev_role = msg.get("role")
        prev_len = msg.get("length") or len(msg.get("message", "") or "")
    return out


def enrich_dialogue(history: list) -> dict:
    turns = compute_turn_signals(history)
    unliked_turns = [t["turn_index"] for t in turns if t["role"] == "assistant" and t["is_unliked"]]
    complaints = []
    for i, msg in enumerate(history):
        if msg.get("role") != "user":
            continue
        ok, snippet = detect_complaint(msg.get("message", "") or "")
        if not ok:
            continue
        trigger = None
        for j in range(i - 1, -1, -1):
            if history[j].get("role") == "assistant":
                trigger = history[j].get("turn_index")
                break
        complaints.append(
            {"turn_index": msg.get("turn_index"), "snippet": snippet, "full_text": (msg.get("message", "") or "")[:200], "triggered_asst_turn_id": trigger}
        )
    return {
        "turns": turns,
        "unliked_turns": unliked_turns,
        "explicit_complaints": complaints,
        "has_interrupted": any(t["interrupted"] for t in turns),
        "turn_count": len(history),
        "total_dissatisfied": len(unliked_turns) + len(complaints),
    }


def format_dialogue_for_llm(history: list, enrichment: dict, max_chars_per_turn: int = 1500) -> str:
    signal_by_turn = {t["turn_index"]: t for t in enrichment.get("turns", [])}
    complaint_turns = {c["turn_index"] for c in enrichment.get("explicit_complaints", [])}
    lines: list[str] = []
    for msg in history:
        tid = msg.get("turn_index")
        role = msg.get("role", "?")
        text = (msg.get("message", "") or "").strip()
        text = text[:max_chars_per_turn] + ("...[truncated]" if len(text) > max_chars_per_turn else "")
        sig = signal_by_turn.get(tid, {})
        markers = []
        if sig.get("is_unliked"):
            markers.append("DISLIKED")
        if tid in complaint_turns:
            markers.append("EXPLICIT_COMPLAINT")
        if sig.get("interrupted"):
            markers.append("INTERRUPTED")
        marker_str = f" [{' '.join(markers)}]" if markers else ""
        lines.append(f"[turn {tid} | {role}{marker_str}]")
        lines.append(text if text else "(empty)")
        lines.append("")
    return "\n".join(lines)
