"""
Pure-Python signal enrichment for dialogue bad case detection.

Computes per-dialogue signals:
- Rolling gap (user reply time - asst response time) / asst length → skip-read detection
- Explicit complaint detection (regex patterns on user messages)
- Unliked turn identification
- System message / interruption detection
- [NEW] Repeat query detection (user re-asks nearly same question)
- [NEW] Rephrase query detection (user rephrases same intent)
- [NEW] Short dismiss detection ("哦""算了""行吧")
- [NEW] Topic abandon detection (user abruptly switches topic after AI reply)
- [NEW] User reaction classification per assistant turn
"""

import re
from datetime import datetime
from difflib import SequenceMatcher

# ═══════════════════════════════════════════════════════════════
# Explicit complaint patterns (ordered by specificity)
# ═══════════════════════════════════════════════════════════════
COMPLAINT_PATTERNS = [
    # Direct denial
    r"不对", r"错了?", r"不是这个", r"不是.*我.*(要|说|问)",
    # Re-stating demand
    r"我(要|说|问)的是", r"我的意思是", r"我(刚才|之前)?(说|问|要)的",
    # Frustration
    r"没用", r"答非所问", r"废话", r"不要(再)?说.*这(类|种)", r"不需要(这些)?",
    r"牛头不对马嘴", r"跑题", r"文不对题", r"完全不对",
    # Emotional
    r"无语", r"服了", r"真的假的", r"看不懂", r"醉了", r"晕",
    r"你(是|在)?逗我", r"搞什么", r"什么鬼", r"离谱",
    # Meta-complaint about AI behavior
    r"打广告", r"消费(消费者)?", r"别(再)?(废话|啰嗦)",
    r"你(这)?是(ai|机器人|人工智能)", r"你能不能(好好|认真)",
    r"你(到底)?(会不会|能不能|懂不懂)", r"说(了|的).*和.*一样",
    r"重复.*回答", r"换(一个|个)(答案|回答|说法)",
]
_COMPLAINT_RE = re.compile("|".join(f"({p})" for p in COMPLAINT_PATTERNS), re.IGNORECASE)

# ═══════════════════════════════════════════════════════════════
# Short dismiss patterns — user gives up / politely rejects
# ═══════════════════════════════════════════════════════════════
SHORT_DISMISS_EXACT = {
    "哦", "噢", "嗯", "好吧", "算了", "行吧", "知道了", "好的吧",
    "不用了", "没事了", "不需要了", "罢了", "得了", "随便吧",
    "好", "ok", "嗯嗯", "哦哦", "行", "可以了", "就这样吧",
    "不用", "没事", "不需要", "pass", "跳过", "下一个",
    "不了", "谢谢不用了", "不用谢谢", "好的谢谢",
}

# System / interrupted messages (not real answers)
INTERRUPTED_MARKERS = {
    "回答中断", "已暂停生成。", "已暂停生成", "已暂停生成.",
    "视频生成失败。", "视频生成失败",
    "未找到相关图片。", "未找到相关图片",
    "图片下载失败", "（无输出）", "(无输出)",
    "内容审核未通过", "内容审核未通过。",
}


# ═══════════════════════════════════════════════════════════════
# Utility functions
# ═══════════════════════════════════════════════════════════════
def parse_time(s: str):
    """Parse '2026-04-16 11:06:44' → datetime or None."""
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def detect_complaint(text: str) -> tuple[bool, str]:
    """Return (is_complaint, matched_snippet)."""
    if not text:
        return False, ""
    if len(text) > 300:
        return False, ""
    m = _COMPLAINT_RE.search(text)
    if m:
        return True, m.group(0)
    return False, ""


def is_interrupted(text: str) -> bool:
    t = (text or "").strip()
    return t in INTERRUPTED_MARKERS or len(t) < 4


def _clean_for_compare(text: str) -> str:
    """Normalize text for similarity comparison: lowercase, strip punct/spaces."""
    t = (text or "").strip().lower()
    t = re.sub(r'[，。！？、；：\u201c\u201d\u2018\u2019（）()\[\]{}\s\n\r]+', '', t)
    return t


def text_similarity(a: str, b: str) -> float:
    """Fast char-level similarity using SequenceMatcher. 0-1."""
    ca, cb = _clean_for_compare(a), _clean_for_compare(b)
    if not ca or not cb:
        return 0.0
    if ca == cb:
        return 1.0
    return SequenceMatcher(None, ca, cb).ratio()


def jaccard_char_sim(a: str, b: str) -> float:
    """Char-level Jaccard similarity (faster for long strings)."""
    ca, cb = _clean_for_compare(a), _clean_for_compare(b)
    if not ca or not cb:
        return 0.0
    sa, sb = set(ca), set(cb)
    intersection = sa & sb
    union = sa | sb
    return len(intersection) / len(union) if union else 0.0


def is_short_dismiss(text: str) -> bool:
    """Check if user message is a short dismissive reply."""
    t = (text or "").strip()
    if len(t) > 20:
        return False
    t_lower = t.lower().rstrip("。.！!~～")
    return t_lower in SHORT_DISMISS_EXACT


def detect_topic_change(prev_user_msg: str, curr_user_msg: str, threshold: float = 0.15) -> bool:
    """Detect if user completely changed topic (very low similarity)."""
    if not prev_user_msg or not curr_user_msg:
        return False
    # Both must be real queries (not too short)
    if len(prev_user_msg.strip()) < 5 or len(curr_user_msg.strip()) < 5:
        return False
    sim = max(text_similarity(prev_user_msg, curr_user_msg),
              jaccard_char_sim(prev_user_msg, curr_user_msg))
    return sim < threshold


# ═══════════════════════════════════════════════════════════════
# Core: per-turn signal computation
# ═══════════════════════════════════════════════════════════════
def compute_rolling_gaps(history: list) -> list:
    """
    For each turn compute signals. Returns list of dict with:
      - turn_index, role, is_unliked
      - gap_s (sec from previous message's create_time)
      - gap_ratio (gap_s / prev_message_length), only for user turns after asst
      - skip_read: True if user replied too fast after long asst msg
    """
    out = []
    prev_time = None
    prev_len = 0
    prev_role = None
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
        out.append({
            "turn_index": msg.get("turn_index"),
            "role": msg.get("role"),
            "is_unliked": int(msg.get("is_unliked", 0)) == 1,
            "length": msg.get("length") or len(msg.get("message", "") or ""),
            "gap_s": round(gap_s, 2) if gap_s is not None else None,
            "gap_ratio": round(gap_ratio, 4) if gap_ratio is not None else None,
            "skip_read": skip_read,
            "interrupted": msg.get("role") == "assistant" and is_interrupted(msg.get("message", "")),
        })
        if t is not None:
            prev_time = t
        prev_len = msg.get("length") or len(msg.get("message", "") or "")
        prev_role = msg.get("role")
    return out


# ═══════════════════════════════════════════════════════════════
# NEW: User reaction classification for each assistant turn
# ═══════════════════════════════════════════════════════════════
REACTION_TYPES = [
    "satisfied",        # Normal follow-up / new topic (no dissatisfaction signal)
    "repeat",           # User re-asks nearly same question
    "rephrase",         # User rephrases same intent
    "complain",         # Explicit complaint
    "short_dismiss",    # "哦""算了""行吧"
    "abandon",          # Abruptly switches to unrelated topic
    "skip_read",        # Replied too fast to have read
    "no_followup",      # Conversation ends (user stops responding)
]


def classify_user_reactions(history: list, turn_signals: list) -> list:
    """
    For each assistant turn, classify the NEXT user's reaction.
    Returns list of dicts: {asst_turn_id, next_user_turn_id, reaction, details}
    """
    reactions = []
    signal_by_turn = {t["turn_index"]: t for t in turn_signals}

    # Build ordered list of (role, turn_index, message)
    msgs = [(m.get("role"), m.get("turn_index"), m.get("message", "") or "") for m in history]

    # Collect all user messages for repeat/rephrase detection
    user_msgs = [(tid, text) for role, tid, text in msgs if role == "user"]

    for i, (role, tid, text) in enumerate(msgs):
        if role != "assistant":
            continue

        # Find next user message
        next_user = None
        for j in range(i + 1, len(msgs)):
            if msgs[j][0] == "user":
                next_user = msgs[j]
                break

        if next_user is None:
            reactions.append({
                "asst_turn_id": tid,
                "next_user_turn_id": None,
                "reaction": "no_followup",
                "details": "conversation ends after this assistant turn",
                "dissatisfied": False,
            })
            continue

        nu_role, nu_tid, nu_text = next_user
        sig = signal_by_turn.get(nu_tid, {})

        # Priority-ordered detection
        is_comp, comp_snippet = detect_complaint(nu_text)

        if is_comp:
            reaction = "complain"
            details = f"complaint: '{comp_snippet}'"
            dissatisfied = True
        elif sig.get("skip_read"):
            reaction = "skip_read"
            details = f"gap_ratio={sig.get('gap_ratio')}"
            dissatisfied = True  # likely didn't read
        elif is_short_dismiss(nu_text):
            reaction = "short_dismiss"
            details = f"'{nu_text.strip()[:30]}'"
            dissatisfied = True
        else:
            # Check repeat/rephrase: compare with the user query that triggered this assistant turn
            # Find the user msg right before this assistant turn
            prev_user_text = ""
            for k in range(i - 1, -1, -1):
                if msgs[k][0] == "user":
                    prev_user_text = msgs[k][2]
                    break

            sim = text_similarity(prev_user_text, nu_text) if prev_user_text else 0
            jsim = jaccard_char_sim(prev_user_text, nu_text) if prev_user_text else 0
            best_sim = max(sim, jsim)

            if best_sim > 0.80 and len(nu_text.strip()) > 4:
                reaction = "repeat"
                details = f"similarity={best_sim:.2f} with previous query"
                dissatisfied = True
            elif best_sim > 0.45 and len(nu_text.strip()) > 4:
                # Rephrase: moderate similarity + user re-asks
                reaction = "rephrase"
                details = f"similarity={best_sim:.2f} with previous query"
                dissatisfied = True
            elif detect_topic_change(prev_user_text, nu_text):
                reaction = "abandon"
                details = "topic completely changed"
                dissatisfied = True  # likely gave up
            else:
                reaction = "satisfied"
                details = ""
                dissatisfied = False

        reactions.append({
            "asst_turn_id": tid,
            "next_user_turn_id": nu_tid,
            "reaction": reaction,
            "details": details,
            "dissatisfied": dissatisfied,
        })

    return reactions


# ═══════════════════════════════════════════════════════════════
# NEW: Recovery chain extraction
# ═══════════════════════════════════════════════════════════════
def extract_recovery_chains(history: list, reactions: list) -> list:
    """
    From reactions, find failure → recovery chains.
    A chain starts at a dissatisfied reaction and follows until topic changes or conversation ends.
    """
    chains = []
    reaction_by_asst = {r["asst_turn_id"]: r for r in reactions}
    msgs_by_tid = {m.get("turn_index"): m for m in history}

    # Find all failed assistant turns
    failed_asst_tids = [r["asst_turn_id"] for r in reactions if r["dissatisfied"]]

    for fail_tid in failed_asst_tids:
        fail_reaction = reaction_by_asst[fail_tid]
        fail_msg = msgs_by_tid.get(fail_tid)
        if not fail_msg:
            continue

        # Collect the chain: subsequent turns until topic change or end
        chain_turns = []
        started = False
        for msg in history:
            tid = msg.get("turn_index")
            if tid == fail_tid:
                started = True
                continue
            if not started:
                continue

            role = msg.get("role")
            text = (msg.get("message", "") or "")[:300]

            if role == "user":
                is_comp, snippet = detect_complaint(text)
                if is_comp:
                    action = "complain"
                elif is_short_dismiss(text):
                    action = "dismiss"
                else:
                    action = "followup"
            else:
                action = "retry"

            chain_turns.append({
                "turn_id": tid,
                "role": role,
                "action": action,
                "text_preview": text[:150],
            })

            # Stop chain after 6 turns or if user abandons
            if len(chain_turns) >= 6:
                break
            if role == "user" and action == "dismiss":
                break

        # Determine outcome
        if not chain_turns:
            outcome = "no_recovery"
        elif chain_turns[-1]["role"] == "user" and chain_turns[-1]["action"] == "dismiss":
            outcome = "user_gave_up"
        elif any(t["action"] == "retry" for t in chain_turns):
            # Check if any subsequent user reaction is satisfied
            last_asst_in_chain = None
            for t in reversed(chain_turns):
                if t["role"] == "assistant":
                    last_asst_in_chain = t["turn_id"]
                    break
            if last_asst_in_chain and last_asst_in_chain in reaction_by_asst:
                if not reaction_by_asst[last_asst_in_chain]["dissatisfied"]:
                    outcome = "recovered"
                else:
                    outcome = "failed_recovery"
            else:
                outcome = "partial_recovery"
        else:
            outcome = "no_recovery"

        chains.append({
            "failure_turn_id": fail_tid,
            "failure_reaction": fail_reaction["reaction"],
            "recovery_chain": chain_turns,
            "recovery_outcome": outcome,
            "chain_length": len(chain_turns),
        })

    return chains


# ═══════════════════════════════════════════════════════════════
# Main enrichment function (upgraded)
# ═══════════════════════════════════════════════════════════════
def enrich_dialogue(history: list) -> dict:
    """
    Given raw history, produce comprehensive enrichment dict with:
    - turns: per-turn signals (gap, skip_read, interrupted, unliked)
    - unliked_turns: list of assistant turn_ids with is_unliked=1
    - explicit_complaints: list of {turn_index, snippet, triggered_asst_turn_id}
    - user_reactions: per-assistant-turn reaction classification
    - dissatisfied_turns: all assistant turns where user showed any dissatisfaction
    - recovery_chains: failure → recovery → outcome sequences
    - has_interrupted: any asst msg is interrupted/empty
    """
    turns = compute_rolling_gaps(history)

    unliked_turns = [t["turn_index"] for t in turns if t["role"] == "assistant" and t["is_unliked"]]

    # Detect explicit complaints
    explicit_complaints = []
    for i, msg in enumerate(history):
        if msg.get("role") != "user":
            continue
        text = msg.get("message", "") or ""
        is_comp, snippet = detect_complaint(text)
        if is_comp:
            trigger = None
            for j in range(i - 1, -1, -1):
                if history[j].get("role") == "assistant":
                    trigger = history[j].get("turn_index")
                    break
            explicit_complaints.append({
                "turn_index": msg.get("turn_index"),
                "snippet": snippet,
                "full_text": text[:200],
                "triggered_asst_turn_id": trigger,
            })

    has_interrupted = any(t["interrupted"] for t in turns)

    # NEW: User reactions & recovery chains
    user_reactions = classify_user_reactions(history, turns)
    dissatisfied_turns = [r["asst_turn_id"] for r in user_reactions if r["dissatisfied"]]
    recovery_chains = extract_recovery_chains(history, user_reactions)

    # Summary stats
    reaction_counts = {}
    for r in user_reactions:
        reaction_counts[r["reaction"]] = reaction_counts.get(r["reaction"], 0) + 1

    return {
        "turns": turns,
        "unliked_turns": unliked_turns,
        "explicit_complaints": explicit_complaints,
        "user_reactions": user_reactions,
        "dissatisfied_turns": dissatisfied_turns,
        "recovery_chains": recovery_chains,
        "reaction_summary": reaction_counts,
        "has_interrupted": has_interrupted,
        "turn_count": len(history),
        "total_dissatisfied": len(dissatisfied_turns),
        "total_unliked": len(unliked_turns),
    }


# ═══════════════════════════════════════════════════════════════
# Format for LLM (upgraded with new signals)
# ═══════════════════════════════════════════════════════════════
def format_dialogue_for_llm(history: list, enrichment: dict, max_chars_per_turn: int = 1500) -> str:
    """
    Render the full dialogue in a structured, LLM-friendly format with ALL signals.
    """
    signal_by_turn = {t["turn_index"]: t for t in enrichment["turns"]}
    complaint_turns = {c["turn_index"] for c in enrichment["explicit_complaints"]}
    reaction_by_asst = {r["asst_turn_id"]: r for r in enrichment.get("user_reactions", [])}

    lines = []
    for msg in history:
        tid = msg.get("turn_index")
        role = msg.get("role", "?")
        text = (msg.get("message", "") or "").strip()
        if len(text) > max_chars_per_turn:
            text = text[:max_chars_per_turn] + f"...[truncated, total {len(msg.get('message',''))} chars]"

        sig = signal_by_turn.get(tid, {})
        markers = []
        if sig.get("is_unliked"):
            markers.append("👎DISLIKED")
        if sig.get("interrupted"):
            markers.append("⚠INTERRUPTED")
        # NOTE: skip_read is intentionally NOT marked on the user turn here.
        # skip_read semantically describes the PRECEDING agent response that was skipped,
        # so it is displayed on the assistant turn via USER_REACTION below.
        if tid in complaint_turns:
            markers.append("💢EXPLICIT_COMPLAINT")

        # Add user reaction markers on assistant turns (correct placement)
        # skip_read is shown here — on the agent response that was skipped — not on the user reply
        if role == "assistant" and tid in reaction_by_asst:
            r = reaction_by_asst[tid]
            if r["dissatisfied"]:
                reaction_emoji = {
                    "complain": "💢", "repeat": "🔄", "rephrase": "🔀",
                    "short_dismiss": "😐", "abandon": "🏃", "skip_read": "⏩",
                }.get(r["reaction"], "⚠")
                markers.append(f"{reaction_emoji}USER_REACTION={r['reaction']}")

        marker_str = f" [{' '.join(markers)}]" if markers else ""
        gap_str = f" (+{sig.get('gap_s', 0)}s)" if sig.get("gap_s") else ""

        lines.append(f"[turn {tid} | {role}{marker_str}{gap_str}] ({len(msg.get('message','') or '')}字)")
        lines.append(text if text else "(empty)")
        lines.append("")
    return "\n".join(lines)
