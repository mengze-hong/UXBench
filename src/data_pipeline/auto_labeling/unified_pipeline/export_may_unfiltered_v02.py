#!/usr/bin/env python3
"""
May 点赞/点踩 → 训练 JSONL，**不调用 Judge**、不做质量过滤。

仅做：
  - 与 lite_extract 一致：只保留点踩/点赞那一轮 + 之前 history；
  - 去掉装饰性 Markdown / Emoji（规则）；
  - 去掉明显 Outlier：单条过长、history 条数过多、全文总长过长。

写出字段顺序与键集合分别对齐：
  - bad  → UXBENCH-DATASET/uxbench-internal/bad_train_15k.jsonl
  - good → UXBENCH-DATASET/uxbench-internal/good_train_15k.jsonl

Judge 相关维度用中性占位填充（与参考文件同键），便于下游直接 concat。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from auto_labeling.unified_pipeline.aligned_schema import build_bad_train_row, build_good_train_row
from auto_labeling.unified_pipeline.may_data_paths import DEFAULT_DISLIKE_JSONL, DEFAULT_LIKE_JSONL
from auto_labeling.unified_pipeline.dedupe import dedupe_merge_from_iter, resolve_dedupe_key_field
from auto_labeling.unified_pipeline.lite_extract import extract_lite_cases
from auto_labeling.unified_pipeline.raw_loader import iter_raw_items_from_path
from auto_labeling.unified_pipeline.session_builder import build_sessions_from_items, session_to_core_record
from auto_labeling.unified_pipeline.text_sanitize import sanitize_user_profile, strip_decorative_text

BAD_KEY_ORDER = [
    "cid",
    "save_time",
    "ground_truth",
    "source_query",
    "selected_turn_indices",
    "selected_history",
    "agent_response_full",
    "agent_response_reasoning",
    "agent_response_preview",
    "dislike_turn_id",
    "failure_dimension",
    "failure_dimension_raw",
    "failure_dimension_reclassified",
    "reclassify_confidence",
    "scenario",
    "severity_tier",
    "signal_type",
    "signal_confidence",
    "representativeness",
    "needs_context",
    "needs_image",
    "is_duplicate",
    "explanation",
    "judge_average",
    "judge_scores",
    "judge_audit",
    "overall_quality",
    "confidence",
    "qa_verdict",
    "qa_issues",
    "qa_notes",
    "user_profile",
    "system_prompt",
]

GOOD_KEY_ORDER = [
    "cid",
    "save_time",
    "ground_truth",
    "source_query",
    "selected_history",
    "agent_response_full",
    "agent_response_reasoning",
    "praise_snippet",
    "success_dimension",
    "scenario",
    "signal_type",
    "signal_confidence",
    "sentiment",
    "representativeness",
    "tags",
    "explanation",
    "judge_average",
    "overall_quality",
    "qa_verdict",
    "user_profile",
    "system_prompt",
]

NEUTRAL_BAD_JUDGE: dict[str, Any] = {
    "failure_dimension": "未分类",
    "failure_dimension_raw": "未分类",
    "scenario": "其他",
    "severity_tier": None,
    "explanation": "",
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
    "judge_audit": "no_llm_judge_export",
    "overall_quality": "medium",
    "confidence": 0.5,
    "qa_verdict": "keep",
    "qa_issues": [],
    "qa_notes": "",
}

NEUTRAL_GOOD_JUDGE: dict[str, Any] = {
    "success_dimension": "未分类",
    "scenario": "其他",
    "sentiment": "implicit",
    "explanation": "",
    "tags": ["success_testcase"],
    "signal_confidence": "medium",
    "representativeness": "medium",
    "praise_snippet": None,
    "judge_average": 3.5,
    "overall_quality": "medium",
    "qa_verdict": {
        "verdict": "keep",
        "quality": "medium",
        "issues": [],
        "notes": "",
        "corrected_dimension": None,
        "model": "no_llm_judge",
    },
}


def system_prompt_from_rich(session: dict | None) -> str:
    if not session:
        return ""
    for h in session.get("history") or []:
        ts = h.get("turn_stats") or {}
        sp = (ts.get("systemprompt") or "").strip()
        if sp:
            return sp
    return ""


def _ingest_path(
    path: Path,
    *,
    dedupe_key: str,
    dedupe_keep: str,
    max_input_rows: int,
    answer_max_chars: int,
):
    key_field = resolve_dedupe_key_field(dedupe_key)
    raw_iter = iter_raw_items_from_path(path, max_rows=max_input_rows or 0)
    rows, dstats = dedupe_merge_from_iter(raw_iter, key_field=key_field, keep=dedupe_keep)
    sessions = build_sessions_from_items(rows, answer_max_chars=answer_max_chars)
    rich_by_cid = {s["cid"]: s for s in sessions}
    core = [session_to_core_record(s) for s in sessions]
    return core, rich_by_cid, dstats


def _enrich_system_prompt(cases: list[dict], rich_by_cid: dict[str, dict]) -> None:
    for c in cases:
        c["_system_prompt"] = strip_decorative_text(system_prompt_from_rich(rich_by_cid.get(c.get("cid") or "")))


def _cases_from_core(core: list[dict], mode: str, *, one_per_session: bool) -> list[dict]:
    out: list[dict] = []
    for rec in core:
        out.extend(
            extract_lite_cases(rec, mode, one_per_session=one_per_session, include_full_history=False)
        )
    return out


def _sanitize_lite_case(case: dict, *, mode: str) -> None:
    sq = case.setdefault("source_query", {})
    msg = strip_decorative_text(sq.get("message") or "")
    sq["message"] = msg
    for h in case.get("selected_history") or []:
        m = strip_decorative_text(h.get("message") or "")
        h["message"] = m
        h["length"] = len(m)
        if h.get("role") == "assistant" and h.get("reasoning"):
            rs = strip_decorative_text(h.get("reasoning") or "")
            h["reasoning"] = rs
            h["reasoning_len"] = len(rs)
    if mode == "bad":
        case["agent_response_full"] = strip_decorative_text(case.get("agent_response_full") or "")
    else:
        case["liked_response_full"] = strip_decorative_text(case.get("liked_response_full") or "")
    rsn0 = case.get("agent_response_reasoning") or ""
    if rsn0:
        case["agent_response_reasoning"] = strip_decorative_text(rsn0)
    up = case.get("user_profile")
    if isinstance(up, dict) and up:
        case["user_profile"] = sanitize_user_profile(up)


def _outlier_check(
    case: dict,
    *,
    mode: str,
    max_history_messages: int,
    max_agent_chars: int,
    max_any_message_chars: int,
    max_total_chars: int,
) -> tuple[bool, str]:
    hist = case.get("selected_history") or []
    if len(hist) > max_history_messages:
        return False, "history_messages_too_many"

    body = (case.get("agent_response_full") if mode == "bad" else case.get("liked_response_full")) or ""
    rsn = (case.get("agent_response_reasoning") or "").strip()
    if len(body) > max_agent_chars:
        return False, "agent_response_too_long"

    chunks: list[str] = [(case.get("source_query") or {}).get("message") or "", body, rsn]
    for h in hist:
        chunks.append(h.get("message") or "")
        if h.get("role") == "assistant" and h.get("reasoning"):
            chunks.append(h.get("reasoning") or "")
    for ch in chunks:
        if len(ch) > max_any_message_chars:
            return False, "single_message_too_long"
    total = sum(len(x) for x in chunks)
    if total > max_total_chars:
        return False, "total_chars_too_large"
    return True, "ok"


def _ordered_bad(row: dict[str, Any]) -> dict[str, Any]:
    return {k: row[k] for k in BAD_KEY_ORDER}


def _ordered_good(row: dict[str, Any]) -> dict[str, Any]:
    return {k: row[k] for k in GOOD_KEY_ORDER}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dislike-file", type=Path, default=DEFAULT_DISLIKE_JSONL)
    p.add_argument("--like-file", type=Path, default=DEFAULT_LIKE_JSONL)
    p.add_argument(
        "--output-bad",
        type=Path,
        default=PROJECT_ROOT / "UXBENCH-DATASET" / "uxbench-internal" / "may_bad_train_unfiltered.jsonl",
    )
    p.add_argument(
        "--output-good",
        type=Path,
        default=PROJECT_ROOT / "UXBENCH-DATASET" / "uxbench-internal" / "may_good_train_unfiltered.jsonl",
    )
    p.add_argument("--dedupe-key", default="auto")
    p.add_argument("--dedupe-keep", choices=("last", "first"), default="last")
    p.add_argument("--max-input-rows", type=int, default=0)
    p.add_argument("--answer-max-chars", type=int, default=120_000)
    p.add_argument("--one-per-session", action="store_true")
    p.add_argument("--max-history-messages", type=int, default=36, help="selected_history 最大条数（user/assistant 各算一条）")
    p.add_argument("--max-agent-chars", type=int, default=14_000, help="点踩/点赞对应 assistant 全文最大长度")
    p.add_argument("--max-any-message-chars", type=int, default=14_000, help="任意单条 message 最大长度")
    p.add_argument("--max-total-chars", type=int, default=80_000, help="source_query + agent + history 拼接总长上限")
    args = p.parse_args()

    save_tag = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report: dict[str, Any] = {"started": save_tag, "bad": {}, "good": {}}

    def run_side(mode: str, path: Path, out_path: Path) -> None:
        core, rich, dstats = _ingest_path(
            path,
            dedupe_key=args.dedupe_key,
            dedupe_keep=args.dedupe_keep,
            max_input_rows=args.max_input_rows,
            answer_max_chars=args.answer_max_chars,
        )
        cases = _cases_from_core(core, mode, one_per_session=args.one_per_session)
        _enrich_system_prompt(cases, rich)

        drop_reasons: Counter[str] = Counter()
        rows_out: list[dict[str, Any]] = []

        for c in cases:
            _sanitize_lite_case(c, mode=mode)
            ok, reason = _outlier_check(
                c,
                mode=mode,
                max_history_messages=args.max_history_messages,
                max_agent_chars=args.max_agent_chars,
                max_any_message_chars=args.max_any_message_chars,
                max_total_chars=args.max_total_chars,
            )
            if not ok:
                drop_reasons[reason] += 1
                continue

            sp = c.get("_system_prompt") or ""
            if mode == "bad":
                row = build_bad_train_row(
                    c,
                    dict(NEUTRAL_BAD_JUDGE),
                    system_prompt=sp,
                    save_time=save_tag,
                    use_judge_fallback=False,
                )
                rows_out.append(_ordered_bad(row))
            else:
                row = build_good_train_row(
                    c,
                    dict(NEUTRAL_GOOD_JUDGE),
                    system_prompt=sp,
                    judge_model="no_llm_judge",
                    save_time=save_tag,
                    use_judge_fallback=False,
                )
                rows_out.append(_ordered_good(row))

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for r in rows_out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        report["bad" if mode == "bad" else "good"] = {
            "input_file": str(path),
            "dedupe": dstats,
            "lite_cases": len(cases),
            "written": len(rows_out),
            "dropped_outliers": int(sum(drop_reasons.values())),
            "drop_reasons": dict(drop_reasons),
            "output": str(out_path),
        }

    run_side("bad", Path(args.dislike_file), Path(args.output_bad))
    run_side("good", Path(args.like_file), Path(args.output_good))

    rep_path = Path(args.output_bad).with_name("may_unfiltered_export_report.json")
    rep_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
