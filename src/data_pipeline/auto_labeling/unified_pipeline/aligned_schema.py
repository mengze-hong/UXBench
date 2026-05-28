"""Build train-jsonl rows aligned with UXBENCH-DATASET/uxbench-internal/*_train_15k.jsonl."""

from __future__ import annotations

from datetime import datetime
from typing import Any


def _preview(text: str, limit: int = 420) -> str:
    t = (text or "").replace("\r\n", "\n").strip()
    if len(t) <= limit:
        return t
    return t[:limit] + "…"


def good_style_selected_history(items: list[dict]) -> list[dict]:
    out: list[dict] = []
    for i, m in enumerate(items or []):
        msg = m.get("message", "") or ""
        tid_raw = m.get("turn_index")
        try:
            tid_use = int(tid_raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            tid_use = -1
        if tid_use < 0:
            tid_use = (i // 2) * 2
        d: dict[str, Any] = {
            "turn_index": tid_use,
            "role": m.get("role"),
            "message": msg,
            "create_time": m.get("create_time", "") or "",
            "length": int(m.get("length", len(msg))),
        }
        if m.get("role") == "assistant":
            rsn = (m.get("reasoning") or "").strip() if isinstance(m.get("reasoning"), str) else str(m.get("reasoning") or "").strip()
            if rsn:
                d["reasoning"] = rsn
                d["reasoning_len"] = int(m.get("reasoning_len", len(rsn)))
        out.append(d)
    return out


def bad_style_selected_history(items: list[dict]) -> list[dict]:
    out: list[dict] = []
    for m in items or []:
        msg = m.get("message", "") or ""
        tid = int(m.get("turn_index", -1))
        d: dict[str, Any] = {
            "turn_index": tid,
            "role": m.get("role"),
            "message": msg,
            "create_time": m.get("create_time", "") or "",
            "length": int(m.get("length", len(msg))),
        }
        if m.get("role") == "assistant":
            rsn = (m.get("reasoning") or "").strip() if isinstance(m.get("reasoning"), str) else str(m.get("reasoning") or "").strip()
            if rsn:
                d["reasoning"] = rsn
                d["reasoning_len"] = int(m.get("reasoning_len", len(rsn)))
        out.append(d)
    return out


def judge_average_from_scores(scores: dict[str, Any]) -> float | None:
    vals = []
    for k in ("query_completeness", "signal_credibility", "representativeness", "severity", "annotation_clarity"):
        v = scores.get(k)
        if isinstance(v, (int, float)):
            vals.append(float(v))
    if not vals:
        return None
    return round(sum(vals) / len(vals), 2)


def _defaults_bad_judge() -> dict[str, Any]:
    return {
        "failure_dimension": "实用性差",
        "failure_dimension_raw": "未分类",
        "scenario": "其他",
        "severity_tier": None,
        "explanation": "模型未返回有效 judge JSON。",
        "signal_confidence": "medium",
        "representativeness": "medium",
        "needs_context": False,
        "needs_image": False,
        "is_duplicate": False,
        "judge_scores": {
            "query_completeness": 3,
            "signal_credibility": 3,
            "representativeness": 3,
            "severity": 3,
            "annotation_clarity": 3,
        },
        "judge_audit": "judge_fallback",
        "overall_quality": "low",
        "confidence": 0.5,
        "qa_verdict": "delete",
        "qa_issues": ["judge_parse_fail"],
        "qa_notes": "自动降级删除",
    }


def _defaults_good_judge(model: str) -> dict[str, Any]:
    return {
        "success_dimension": "准确回答",
        "scenario": "其他",
        "sentiment": "implicit",
        "explanation": "模型未返回有效 judge JSON。",
        "tags": ["success_testcase"],
        "signal_confidence": "medium",
        "representativeness": "medium",
        "praise_snippet": None,
        "judge_average": 3.0,
        "overall_quality": "low",
        "qa_verdict": {
            "verdict": "delete",
            "quality": "low",
            "issues": ["judge_parse_fail"],
            "notes": "自动降级删除",
            "corrected_dimension": None,
            "model": model,
        },
    }


def build_bad_train_row(
    case: dict,
    judge: dict[str, Any] | None,
    *,
    system_prompt: str,
    save_time: str | None = None,
    use_judge_fallback: bool = False,
) -> dict[str, Any]:
    if use_judge_fallback:
        j = _defaults_bad_judge()
    else:
        j = dict(judge or {})

    scores = j.get("judge_scores")
    if not isinstance(scores, dict):
        scores = _defaults_bad_judge()["judge_scores"]
    ja = judge_average_from_scores(scores)

    sel_hist = bad_style_selected_history(case.get("selected_history") or [])
    indices = [int(x["turn_index"]) for x in sel_hist]

    rsn_sig = (case.get("agent_response_reasoning") or "").strip()
    row: dict[str, Any] = {
        "cid": case.get("cid", ""),
        "save_time": save_time or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ground_truth": -1,
        "source_query": {
            "turn_index": int((case.get("source_query") or {}).get("turn_index", -1)),
            "message": (case.get("source_query") or {}).get("message", "") or "",
        },
        "selected_turn_indices": indices,
        "selected_history": sel_hist,
        "agent_response_full": case.get("agent_response_full") or "",
        "agent_response_reasoning": rsn_sig,
        "agent_response_preview": _preview(case.get("agent_response_full") or "", 520),
        "dislike_turn_id": int(case.get("dislike_turn_index", -1)),
        "failure_dimension": str(j.get("failure_dimension") or "实用性差"),
        "failure_dimension_raw": str(j.get("failure_dimension_raw") or j.get("failure_dimension") or ""),
        "failure_dimension_reclassified": False,
        "reclassify_confidence": None,
        "scenario": str(j.get("scenario") or "其他"),
        "severity_tier": j.get("severity_tier"),
        "signal_type": "dislike",
        "signal_confidence": str(j.get("signal_confidence") or "medium"),
        "representativeness": str(j.get("representativeness") or "medium"),
        "needs_context": bool(j.get("needs_context", False)),
        "needs_image": bool(j.get("needs_image", False)),
        "is_duplicate": bool(j.get("is_duplicate", False)),
        "explanation": str(j.get("explanation") or ""),
        "judge_average": ja if ja is not None else 3.0,
        "judge_scores": scores,
        "judge_audit": str(j.get("judge_audit") or ""),
        "overall_quality": str(j.get("overall_quality") or "medium"),
        "confidence": float(j.get("confidence") or 0.75),
        "qa_verdict": str(j.get("qa_verdict") or "keep"),
        "qa_issues": j.get("qa_issues") if isinstance(j.get("qa_issues"), list) else [],
        "qa_notes": str(j.get("qa_notes") or ""),
        "user_profile": case.get("user_profile") or {},
        "system_prompt": system_prompt or "",
    }
    return row


def build_good_train_row(
    case: dict,
    judge: dict[str, Any] | None,
    *,
    system_prompt: str,
    judge_model: str,
    save_time: str | None = None,
    use_judge_fallback: bool = False,
) -> dict[str, Any]:
    if use_judge_fallback:
        j = _defaults_good_judge(judge_model)
    else:
        j = dict(judge or {})

    src = case.get("source_query") or {}
    src_lin = int(src.get("turn_index", -1))

    sel_hist = good_style_selected_history(case.get("selected_history") or [])

    qv = j.get("qa_verdict")
    if not isinstance(qv, dict):
        qv = {
            "verdict": str(j.get("qa_verdict") or "keep"),
            "quality": str(j.get("overall_quality") or "medium"),
            "issues": j.get("qa_issues") if isinstance(j.get("qa_issues"), list) else [],
            "notes": str(j.get("qa_notes") or ""),
            "corrected_dimension": j.get("corrected_dimension"),
            "model": judge_model,
        }
    else:
        qv = dict(qv)
        qv.setdefault("model", judge_model)

    tags = j.get("tags")
    if not isinstance(tags, list) or not tags:
        tags = ["success_testcase"]

    rsn_sig = (case.get("agent_response_reasoning") or "").strip()

    row: dict[str, Any] = {
        "cid": case.get("cid", ""),
        "save_time": save_time or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ground_truth": 1,
        "source_query": {
            # 与 _selected_before(history, src_tid) 使用同一线性 turn_index，避免奇数 user 轮与 history 不一致
            "turn_index": src_lin,
            "message": src.get("message", "") or "",
        },
        "selected_history": sel_hist,
        "agent_response_full": case.get("liked_response_full") or "",
        "agent_response_reasoning": rsn_sig,
        "praise_snippet": j.get("praise_snippet"),
        "success_dimension": str(j.get("success_dimension") or "准确回答"),
        "scenario": str(j.get("scenario") or "其他"),
        "signal_type": "like",
        "signal_confidence": str(j.get("signal_confidence") or "medium"),
        "sentiment": str(j.get("sentiment") or "implicit"),
        "representativeness": str(j.get("representativeness") or "medium"),
        "tags": tags,
        "explanation": str(j.get("explanation") or ""),
        "judge_average": float(j.get("judge_average") or 3.5),
        "overall_quality": str(j.get("overall_quality") or "medium"),
        "qa_verdict": qv,
        "user_profile": case.get("user_profile") or {},
        "system_prompt": system_prompt or "",
    }
    return row


def keep_bad_row(row: dict[str, Any], *, min_judge_average: float, min_confidence: float) -> bool:
    v = str(row.get("qa_verdict") or "").strip().lower()
    if v not in ("keep", "yes", "true", "1"):
        return False
    if float(row.get("judge_average", 0)) < min_judge_average:
        return False
    if float(row.get("confidence", 0)) < min_confidence:
        return False
    return True


def keep_good_row(row: dict[str, Any], *, min_judge_average: float) -> bool:
    qv = row.get("qa_verdict")
    if not isinstance(qv, dict):
        return False
    v = str(qv.get("verdict") or "").strip().lower()
    if v not in ("keep", "yes", "true", "1"):
        return False
    if float(row.get("judge_average", 0)) < min_judge_average:
        return False
    return True
