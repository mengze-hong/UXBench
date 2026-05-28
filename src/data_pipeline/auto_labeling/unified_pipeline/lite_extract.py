"""
Lightweight case extraction: only turns with explicit like / unlike feedback.

No signal mining, no Miner, no multi-candidate — one export row per feedback turn
(unless --one-per-session collapses to the first such turn per dialogue).
"""

from __future__ import annotations

from typing import Any, Literal

Mode = Literal["bad", "good"]


def _last_user_turn_before(history: list[dict], asst_list_index: int) -> dict | None:
    j = asst_list_index - 1
    while j >= 0:
        if history[j].get("role") == "user":
            return history[j]
        j -= 1
    return None


def _selected_before(history: list[dict], before_turn_index: int) -> tuple[list[int], list[dict]]:
    selected: list[dict] = []
    indices: list[int] = []
    for m in history:
        tid = m.get("turn_index", -1)
        if tid < before_turn_index:
            indices.append(tid)
            msg = m.get("message", "") or ""
            entry: dict[str, Any] = {
                "role": m.get("role"),
                "message": msg,
                "turn_index": tid,
                "create_time": m.get("create_time", "") or "",
                "length": m.get("length", len(msg)),
            }
            if m.get("role") == "assistant":
                rsn = (m.get("reasoning") or "").strip() if isinstance(m.get("reasoning"), str) else str(m.get("reasoning") or "").strip()
                if rsn:
                    entry["reasoning"] = rsn
                    entry["reasoning_len"] = int(m.get("reasoning_len", len(rsn)))
            selected.append(entry)
    return indices, selected


def extract_lite_bad_cases(
    record: dict,
    *,
    one_per_session: bool = False,
    include_full_history: bool = False,
) -> list[dict]:
    """
    One row per assistant turn with is_unliked == 1.
    """
    cid = record.get("cid") or ""
    history = list(record.get("history") or [])
    user_profile = record.get("user_profile") or {}
    out: list[dict] = []

    for i, m in enumerate(history):
        if m.get("role") != "assistant":
            continue
        if int(m.get("is_unliked", 0) or 0) != 1:
            continue

        user_turn = _last_user_turn_before(history, i)
        if not user_turn:
            continue

        src_tid = int(user_turn.get("turn_index", -1))
        dislike_tid = int(m.get("turn_index", -1))
        indices, selected_history = _selected_before(history, src_tid)

        rsn = (m.get("reasoning") or "").strip() if isinstance(m.get("reasoning"), str) else str(m.get("reasoning") or "").strip()
        row: dict[str, Any] = {
            "case_id": f"{cid}:{dislike_tid}",
            "cid": cid,
            "ground_truth": -1,
            "signal": "dislike",
            "dislike_turn_index": dislike_tid,
            "source_query": {
                "turn_index": src_tid,
                "message": user_turn.get("message", "") or "",
            },
            "selected_turn_indices": indices,
            "selected_history": selected_history,
            "agent_response_full": m.get("message", "") or "",
            "agent_response_reasoning": rsn,
            "user_profile": user_profile,
        }
        if include_full_history:
            row["full_history"] = history
        out.append(row)
        if one_per_session:
            break

    return out


def extract_lite_good_cases(
    record: dict,
    *,
    one_per_session: bool = False,
    include_full_history: bool = False,
) -> list[dict]:
    """
    One row per assistant turn with is_liked == 1.
    """
    cid = record.get("cid") or ""
    history = list(record.get("history") or [])
    user_profile = record.get("user_profile") or {}
    out: list[dict] = []

    for i, m in enumerate(history):
        if m.get("role") != "assistant":
            continue
        if int(m.get("is_liked", 0) or 0) != 1:
            continue

        user_turn = _last_user_turn_before(history, i)
        if not user_turn:
            continue

        src_tid = int(user_turn.get("turn_index", -1))
        liked_tid = int(m.get("turn_index", -1))
        _, selected_history = _selected_before(history, src_tid)

        rsn = (m.get("reasoning") or "").strip() if isinstance(m.get("reasoning"), str) else str(m.get("reasoning") or "").strip()
        row: dict[str, Any] = {
            "case_id": f"{cid}:{liked_tid}",
            "cid": cid,
            "ground_truth": 1,
            "signal": "like",
            "liked_turn_index": liked_tid,
            "source_query": {
                "turn_index": src_tid,
                "message": user_turn.get("message", "") or "",
            },
            "selected_history": selected_history,
            "liked_response_full": m.get("message", "") or "",
            "agent_response_reasoning": rsn,
            "user_profile": user_profile,
        }
        if include_full_history:
            row["full_history"] = history
        out.append(row)
        if one_per_session:
            break

    return out


def extract_lite_cases(
    record: dict,
    mode: Mode,
    *,
    one_per_session: bool = False,
    include_full_history: bool = False,
) -> list[dict]:
    if mode == "bad":
        return extract_lite_bad_cases(
            record,
            one_per_session=one_per_session,
            include_full_history=include_full_history,
        )
    return extract_lite_good_cases(
        record,
        one_per_session=one_per_session,
        include_full_history=include_full_history,
    )


def _reasoning_str(m: dict) -> str:
    r = m.get("reasoning")
    if isinstance(r, str):
        return r.strip()
    return str(r or "").strip()


def extract_full_session_train_case(record: dict, mode: Mode) -> list[dict]:
    """
    每个 core session 产出 0 或 1 条与 lite 同结构的 case。

    - selected_history：与 lite 一致，仅为**信号轮对应 user 的 turn_index 之前**的上下文
      （``_selected_before(history, src_tid)``），不含本轮 user/assistant
    - 目标 assistant：**仅**最后一个带 ``is_unliked==1``（bad）或 ``is_liked==1``（good）的轮；
      若该 session 无任何对应信号，**不写行**
    - 有信号但找不到其前序 user 时：不写行
    """
    cid = record.get("cid") or ""
    history = list(record.get("history") or [])
    user_profile = record.get("user_profile") or {}
    if not history:
        return []

    def last_assistant_index_with_flag(flag: str) -> int:
        idxs: list[int] = []
        for i, m in enumerate(history):
            if m.get("role") != "assistant":
                continue
            if flag == "unliked":
                if int(m.get("is_unliked", 0) or 0) == 1:
                    idxs.append(i)
            else:
                if int(m.get("is_liked", 0) or 0) == 1:
                    idxs.append(i)
        return idxs[-1] if idxs else -1

    tgt_i = last_assistant_index_with_flag("unliked" if mode == "bad" else "liked")
    if tgt_i < 0:
        return []

    target = history[tgt_i]
    user_turn = _last_user_turn_before(history, tgt_i)
    if not user_turn:
        user_turn = next((h for h in history if h.get("role") == "user"), None)
    if not user_turn:
        return []

    src_tid = int(user_turn.get("turn_index", -1))
    tgt_tid = int(target.get("turn_index", -1))
    body = (target.get("message", "") or "").strip()
    rsn = _reasoning_str(target)

    indices, selected = _selected_before(history, src_tid)

    if mode == "bad":
        row: dict[str, Any] = {
            "case_id": f"{cid}:{tgt_tid}",
            "cid": cid,
            "ground_truth": -1,
            "signal": "dislike",
            "dislike_turn_index": tgt_tid,
            "source_query": {"turn_index": src_tid, "message": user_turn.get("message", "") or ""},
            "selected_turn_indices": indices,
            "selected_history": selected,
            "agent_response_full": body,
            "agent_response_reasoning": rsn,
            "user_profile": user_profile,
        }
    else:
        row = {
            "case_id": f"{cid}:{tgt_tid}",
            "cid": cid,
            "ground_truth": 1,
            "signal": "like",
            "liked_turn_index": tgt_tid,
            "source_query": {"turn_index": src_tid, "message": user_turn.get("message", "") or ""},
            "selected_history": selected,
            "liked_response_full": body,
            "agent_response_reasoning": rsn,
            "user_profile": user_profile,
        }
    return [row]


def extract_all_core_records(
    records: list[dict],
    mode: Mode,
    *,
    one_per_session: bool = False,
    include_full_history: bool = False,
) -> tuple[list[dict], dict[str, int]]:
    rows: list[dict] = []
    sessions_with_signal = 0
    sessions_empty = 0
    for rec in records:
        part = extract_lite_cases(
            rec,
            mode,
            one_per_session=one_per_session,
            include_full_history=include_full_history,
        )
        if part:
            sessions_with_signal += 1
            rows.extend(part)
        else:
            sessions_empty += 1
    stats = {
        "input_sessions": len(records),
        "sessions_with_extracted_cases": sessions_with_signal,
        "sessions_without_target_signal": sessions_empty,
        "output_rows": len(rows),
        "mode": mode,
        "one_per_session": int(one_per_session),
    }
    return rows, stats
