"""
Build full chat sessions from raw AI Assistant rows (turn-level exports).

Adapted from the internal COS job snippet: group by cid, sort turns,
expand user/assistant messages, then convert to `auto_labeling.core.signals`
history shape (`message`, `turn_index`, `is_unliked`, `is_liked`, …).
"""

from __future__ import annotations

import json
from typing import Any

from .answer_extract import assistant_text_from_answer_field, split_answer_reasoning_and_body


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except Exception:
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def normalize_answer(answers: Any) -> str:
    if answers is None:
        return ""
    if isinstance(answers, list):
        return "\n".join([safe_str(a) for a in answers if safe_str(a).strip()])
    return safe_str(answers).strip()


def get_ability_label(atom_ability: str) -> tuple[str, int]:
    ability = safe_str(atom_ability).strip()
    if ability in ["闲聊", "AI搜索", "AI写作"]:
        return "text", 0
    if ability in ["AI生图", "教育讲解题"]:
        return "image", 1
    return "unknown", 0


def get_record_id(item: dict) -> str:
    for key in ["据ID", "记录ID", "record_id", "id", "_id"]:
        if key in item and item.get(key) not in [None, ""]:
            return safe_str(item.get(key))
    return ""


def assistant_reasoning_and_plain_text(item: dict, *, answer_max_chars: int) -> tuple[str, str]:
    """(reasoning, final_reply)；优先解析 answer 里的 deep_search 卡片，再退回 answer_msg / 模型回复。"""
    r, b = split_answer_reasoning_and_body(item.get("answer"), max_chars=answer_max_chars)
    if b or r:
        return r, b
    ans = item.get("answer_msg")
    if ans is not None and (not isinstance(ans, str) or ans.strip()):
        t = normalize_answer(ans)
        if t.strip():
            return "", t[:answer_max_chars]
    ans2 = item.get("模型回复")
    if ans2:
        t = normalize_answer(ans2)
        if t.strip():
            return "", t[:answer_max_chars]
    return "", assistant_text_from_answer_field(item.get("answer"), max_chars=answer_max_chars)


def assistant_plain_text(item: dict, *, answer_max_chars: int) -> str:
    """仅正文（不含 deep_search 思考段），兼容旧调用。"""
    return assistant_reasoning_and_plain_text(item, answer_max_chars=answer_max_chars)[1]


def raw_row_to_turn(item: dict, *, answer_max_chars: int) -> dict:
    """One logical turn (user + assistant pair metadata) from a raw export row."""
    if not isinstance(item, dict):
        return {}

    cid = safe_str(item.get("cid")).strip()
    if not cid:
        return {}

    atom_ability = safe_str(item.get("atom_ability"))
    ability_type, involves_image_processing = get_ability_label(atom_ability)

    user_content = safe_str(item.get("modelprompt")).strip()
    reasoning, assistant_content = assistant_reasoning_and_plain_text(item, answer_max_chars=answer_max_chars)

    return {
        "record_id": get_record_id(item),
        "cid": cid,
        "convidx": safe_int(item.get("convidx")),
        "chainid": safe_str(item.get("chainid")),
        "regeneateidx": safe_int(item.get("regeneateidx")),
        "regenerateidx": safe_int(item.get("regenerateidx")),
        "ftime": safe_str(item.get("ftime")),
        "user_content": user_content,
        "assistant_content": assistant_content,
        "assistant_reasoning": reasoning,
        "copy_cnt": safe_int(item.get("copy_cnt")),
        "like_cnt": safe_int(item.get("like_cnt")),
        "unlike_cnt": safe_int(item.get("unlike_cnt")),
        "share_cnt": safe_int(item.get("share_cnt")),
        "click_regen_cnt": safe_int(item.get("click_regen_cnt")),
        "click_picture_cnt": safe_int(item.get("click_picture_cnt")),
        "click_picture_save_cnt": safe_int(item.get("click_picture_save_cnt")),
        "stopgenerate": safe_int(item.get("stopgenerate")),
        "timecosttotal": safe_float(item.get("timecosttotal")),
        "mainbody_len": safe_int(item.get("mainbody_len")),
        "has_attachment": safe_int(item.get("has_attachment")),
        "promptcreatetime": safe_str(item.get("promptcreatetime")),
        "answercreatetime": safe_str(item.get("answercreatetime")),
        "atom_ability": atom_ability,
        "real_ability": safe_str(item.get("real_ability")),
        "output_category1": safe_str(item.get("output_category1")),
        "mas_intent_v2_1st": safe_str(item.get("mas_intent_v2_1st")),
        "badcase_type": safe_str(item.get("badcase_type")),
        "ability_type": ability_type,
        "involves_image_processing": involves_image_processing,
        "mas_intent": safe_str(item.get("mas_intent")),
        "mas_1nd_intent": safe_str(item.get("mas_1nd_intent")),
        "mas_1nd_sub_intent": safe_str(item.get("mas_1nd_sub_intent")),
        "mas_intent_v2_2nd": safe_str(item.get("mas_intent_v2_2nd")),
        "is_online_search": safe_str(item.get("is_online_search")),
        "search_type": safe_str(item.get("search_type")),
        "is_search": safe_str(item.get("is_search")),
        "is_has_citation": safe_str(item.get("is_has_citation")),
        "realpluginid": safe_str(item.get("realpluginid")),
        "is_deep_seek": safe_str(item.get("is_deep_seek")),
        "output_category2": safe_str(item.get("output_category2")),
        "output_category3": safe_str(item.get("output_category3")),
        "chatmodel_type": safe_str(item.get("chatmodel_type")),
        "product_type": safe_str(item.get("product_type")),
        "terminal": safe_str(item.get("terminal")),
        "systemprompt": safe_str(item.get("systemprompt")),
        "has_feedback": 1
        if (safe_int(item.get("like_cnt")) > 0 or safe_int(item.get("unlike_cnt")) > 0)
        else 0,
        "is_liked": 1 if safe_int(item.get("like_cnt")) > 0 else 0,
        "is_unliked": 1 if safe_int(item.get("unlike_cnt")) > 0 else 0,
    }


def _collect_user_profiles(items: list[dict]) -> dict[str, dict]:
    profiles: dict[str, dict] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        cid = safe_str(item.get("cid")).strip()
        if not cid or cid in profiles:
            continue
        profiles[cid] = {
            "cid": cid,
            "age": safe_str(item.get("age")),
            "gender": safe_str(item.get("gender")),
            "occupation": safe_str(item.get("occupation")),
            "industry": safe_str(item.get("industry")),
            "education": safe_str(item.get("education")),
            "nine_groups": safe_str(item.get("nine_groups")),
            "interest_tags": item.get("interest_tags", ""),
            "openid": safe_str(item.get("openid")),
            "register_day": safe_str(item.get("register_day")),
            "ftime": safe_str(item.get("ftime")),
        }
    return profiles


def build_sessions_from_items(items: list[dict], *, answer_max_chars: int = 120_000) -> list[dict]:
    """
    Returns sessions shaped like the legacy job output:
      {cid, history, user_profile, session_involves_image_processing, session_ability_types, session_stats}
    but each history entry uses `content` + `turn_stats` (rich). Use `session_to_core_record` for LLM core.
    """
    conversations: dict[str, list[dict]] = {}
    for item in items:
        turn = raw_row_to_turn(item, answer_max_chars=answer_max_chars)
        if not turn:
            continue
        conversations.setdefault(turn["cid"], []).append(turn)

    user_profiles = _collect_user_profiles(items)
    result: list[dict] = []

    for cid, turns in conversations.items():
        if not turns:
            continue
        turns_sorted = sorted(
            turns,
            key=lambda x: (
                safe_int(x.get("convidx", 0)),
                safe_int(x.get("regenerateidx", 0)),
                safe_int(x.get("regeneateidx", 0)),
                safe_str(x.get("ftime", "")),
                safe_str(x.get("promptcreatetime", "")),
            ),
        )

        history: list[dict] = []

        for turn in turns_sorted:
            common_stats = {k: v for k, v in turn.items() if k not in ("user_content", "assistant_content", "assistant_reasoning")}

            if safe_str(turn["user_content"]).strip():
                history.append(
                    {
                        "role": "user",
                        "content": turn["user_content"],
                        "content_len": len(turn["user_content"]),
                        "turn_stats": dict(common_stats),
                    }
                )

            if safe_str(turn["assistant_content"]).strip():
                entry = {
                    "role": "assistant",
                    "content": turn["assistant_content"],
                    "content_len": len(turn["assistant_content"]),
                    "turn_stats": dict(common_stats),
                }
                rs = safe_str(turn.get("assistant_reasoning", "")).strip()
                if rs:
                    entry["reasoning"] = rs
                    entry["reasoning_len"] = len(rs)
                history.append(entry)

        if not history:
            continue

        alternation_issues = 0
        for i in range(1, len(history)):
            if history[i]["role"] == history[i - 1]["role"]:
                alternation_issues += 1

        session_involves_image_processing = (
            1
            if any(
                safe_int(msg.get("turn_stats", {}).get("involves_image_processing", 0)) == 1
                for msg in history
            )
            else 0
        )

        session_ability_types = sorted(
            {safe_str(msg.get("turn_stats", {}).get("ability_type", "unknown")) for msg in history}
        )

        session_stats = {
            "turn_count": len(turns_sorted),
            "message_count": len(history),
            "user_message_count": sum(1 for h in history if h["role"] == "user"),
            "assistant_message_count": sum(1 for h in history if h["role"] == "assistant"),
            "total_like_cnt": sum(safe_int(t.get("like_cnt", 0)) for t in turns_sorted),
            "total_unlike_cnt": sum(safe_int(t.get("unlike_cnt", 0)) for t in turns_sorted),
            "total_copy_cnt": sum(safe_int(t.get("copy_cnt", 0)) for t in turns_sorted),
            "total_share_cnt": sum(safe_int(t.get("share_cnt", 0)) for t in turns_sorted),
            "total_click_regen_cnt": sum(safe_int(t.get("click_regen_cnt", 0)) for t in turns_sorted),
            "total_click_picture_cnt": sum(safe_int(t.get("click_picture_cnt", 0)) for t in turns_sorted),
            "total_click_picture_save_cnt": sum(
                safe_int(t.get("click_picture_save_cnt", 0)) for t in turns_sorted
            ),
            "total_stopgenerate_cnt": sum(safe_int(t.get("stopgenerate", 0)) for t in turns_sorted),
            "total_timecosttotal": sum(safe_float(t.get("timecosttotal", 0)) for t in turns_sorted),
            "total_mainbody_len": sum(safe_int(t.get("mainbody_len", 0)) for t in turns_sorted),
            "total_content_len": sum(safe_int(h.get("content_len", 0)) for h in history),
            "has_feedback": 1 if any(safe_int(t.get("has_feedback", 0)) == 1 for t in turns_sorted) else 0,
            "alternation_issues": alternation_issues,
            "session_start_time": turns_sorted[0].get("promptcreatetime", ""),
            "session_end_time": turns_sorted[-1].get("answercreatetime", ""),
        }

        result.append(
            {
                "cid": cid,
                "history": history,
                "user_profile": user_profiles.get(cid, {}),
                "session_involves_image_processing": session_involves_image_processing,
                "session_ability_types": session_ability_types,
                "session_stats": session_stats,
            }
        )

    return result


def session_to_core_record(session: dict) -> dict:
    """
    Convert rich session `history` (content + turn_stats) into the schema expected by
    `auto_labeling.core.signals` / `点赞对话数据...signals_like` (message + flags per turn).
    """
    hist_in = session.get("history") or []
    history_out: list[dict] = []
    tid = 0
    for msg in hist_in:
        role = safe_str(msg.get("role"))
        text = safe_str(msg.get("content"))
        ts = msg.get("turn_stats") or {}
        create_time = (
            safe_str(ts.get("promptcreatetime"))
            if role == "user"
            else safe_str(ts.get("answercreatetime"))
        )
        row = {
            "role": role,
            "message": text,
            "turn_index": tid,
            "create_time": create_time,
            "length": len(text),
            "is_unliked": safe_int(ts.get("is_unliked", 0)),
            "is_liked": safe_int(ts.get("is_liked", 0)),
        }
        if role == "assistant":
            rsn = safe_str(msg.get("reasoning")).strip()
            if rsn:
                row["reasoning"] = rsn
                row["reasoning_len"] = len(rsn)
        history_out.append(row)
        tid += 1

    return {
        "cid": session.get("cid", ""),
        "history": history_out,
        "user_profile": session.get("user_profile", {}),
        "_session_meta": {
            "session_stats": session.get("session_stats"),
            "session_ability_types": session.get("session_ability_types"),
            "session_involves_image_processing": session.get("session_involves_image_processing"),
        },
    }


def build_sessions_from_payload(file_data: bytes | str, *, answer_max_chars: int = 120_000) -> list[dict]:
    """COS-style: full JSON in memory -> items list -> sessions."""
    if isinstance(file_data, bytes):
        file_data = file_data.decode("utf-8", errors="replace")
    data = json.loads(file_data)
    from .raw_loader import parse_top_level_payload

    items = parse_top_level_payload(data)
    return build_sessions_from_items(items, answer_max_chars=answer_max_chars)
