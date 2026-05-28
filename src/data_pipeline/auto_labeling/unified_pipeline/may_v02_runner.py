#!/usr/bin/env python3
"""May 点赞/点踩 → UXBench 对齐的 good_case.jsonl / bad_case.jsonl（清洗 + judge）。"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from auto_labeling.unified_pipeline.aligned_schema import (
    build_bad_train_row,
    build_good_train_row,
    keep_bad_row,
    keep_good_row,
)
from auto_labeling.unified_pipeline.cleaning import clean_lite_case
from auto_labeling.unified_pipeline.dedupe import dedupe_merge_from_iter, resolve_dedupe_key_field
from auto_labeling.unified_pipeline.judge_v02 import judge_bad_case, judge_good_case
from auto_labeling.unified_pipeline.lite_extract import extract_lite_cases
from auto_labeling.unified_pipeline.may_data_paths import (
    DEFAULT_DISLIKE_JSONL,
    DEFAULT_LIKE_JSONL,
    MAY_ROOT,
)
from auto_labeling.unified_pipeline.raw_loader import iter_raw_items_from_path
from auto_labeling.unified_pipeline.session_builder import build_sessions_from_items, session_to_core_record


def system_prompt_from_rich(session: dict | None) -> str:
    if not session:
        return ""
    for h in session.get("history") or []:
        ts = h.get("turn_stats") or {}
        sp = (ts.get("systemprompt") or "").strip()
        if sp:
            return sp
    return ""


PROGRESS_FILENAME = "v02_progress.json"


def _write_progress(out_dir: Path, payload: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / PROGRESS_FILENAME
    body = {"updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), **payload}
    path.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")


def _ingest_path(
    path: Path,
    *,
    dedupe_key: str,
    dedupe_keep: str,
    max_input_rows: int,
    answer_max_chars: int,
    out_dir: Path | None = None,
    progress_label: str = "",
):
    key_field = resolve_dedupe_key_field(dedupe_key)
    raw_iter = iter_raw_items_from_path(path, max_rows=max_input_rows or 0)

    def counted_iter():
        n = 0
        for row in raw_iter:
            n += 1
            if out_dir and n % 50_000 == 0:
                _write_progress(
                    out_dir,
                    {
                        "phase": f"read_jsonl_{progress_label}",
                        "file": str(path),
                        "raw_lines_read": n,
                    },
                )
            yield row

    rows, dstats = dedupe_merge_from_iter(counted_iter(), key_field=key_field, keep=dedupe_keep)
    if out_dir:
        _write_progress(
            out_dir,
            {
                "phase": f"after_dedupe_{progress_label}",
                "file": str(path),
                **dstats,
            },
        )
    if out_dir:
        _write_progress(
            out_dir,
            {
                "phase": f"building_sessions_{progress_label}",
                "file": str(path),
                "rows_for_sessions": len(rows),
            },
        )
    sessions = build_sessions_from_items(rows, answer_max_chars=answer_max_chars)
    rich_by_cid = {s["cid"]: s for s in sessions}
    core = [session_to_core_record(s) for s in sessions]
    if out_dir:
        _write_progress(
            out_dir,
            {
                "phase": f"after_core_{progress_label}",
                "file": str(path),
                "sessions": len(sessions),
                "core_records": len(core),
            },
        )
    return core, rich_by_cid, dstats


def _cases_from_core(core: list[dict], mode: str, *, one_per_session: bool) -> list[dict]:
    out: list[dict] = []
    for rec in core:
        out.extend(
            extract_lite_cases(rec, mode, one_per_session=one_per_session, include_full_history=False)
        )
    return out


def _enrich_system_prompt(cases: list[dict], rich_by_cid: dict[str, dict]) -> None:
    for c in cases:
        c["_system_prompt"] = system_prompt_from_rich(rich_by_cid.get(c.get("cid") or ""))


def _strip_judge_meta(j: dict[str, Any] | None) -> dict[str, Any]:
    if not j:
        return {}
    return {k: v for k, v in j.items() if not str(k).startswith("_")}


def _run_judges(
    cases: list[dict],
    mode: str,
    *,
    model: str,
    workers: int,
    out_dir: Path | None = None,
    progress_label: str = "",
):
    results: list[tuple[dict, dict[str, Any] | None, str]] = []
    total = len(cases)
    done = 0

    def job(c: dict):
        if mode == "bad":
            j, err = judge_bad_case(c, model=model)
        else:
            j, err = judge_good_case(c, model=model)
        return c, _strip_judge_meta(j), err

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(job, c) for c in cases]
        for fut in as_completed(futs):
            results.append(fut.result())
            done += 1
            if out_dir and total and (done % 40 == 0 or done == total):
                _write_progress(
                    out_dir,
                    {
                        "phase": f"judge_{progress_label}",
                        "mode": mode,
                        "judge_done": done,
                        "judge_total": total,
                        "pct": round(100.0 * done / total, 2),
                    },
                )
    return results


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dislike-file", type=Path, default=DEFAULT_DISLIKE_JSONL)
    p.add_argument("--like-file", type=Path, default=DEFAULT_LIKE_JSONL)
    p.add_argument("--output-dir", type=Path, default=MAY_ROOT / "v02_lite_aligned")
    p.add_argument("--dedupe-key", default="auto")
    p.add_argument("--dedupe-keep", choices=("last", "first"), default="last")
    p.add_argument("--max-input-rows", type=int, default=0)
    p.add_argument("--answer-max-chars", type=int, default=120_000)
    p.add_argument("--one-per-session", action="store_true")
    p.add_argument("--judge-model", default="gpt-5.1")
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--min-judge-average-bad", type=float, default=3.2)
    p.add_argument("--min-confidence-bad", type=float, default=0.42)
    p.add_argument("--min-judge-average-good", type=float, default=3.4)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_tag = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report: dict[str, Any] = {"started": save_tag}
    _write_progress(out_dir, {"phase": "started", "dislike_file": str(args.dislike_file), "like_file": str(args.like_file)})

    bad_core, bad_rich, bad_dedupe = _ingest_path(
        Path(args.dislike_file),
        dedupe_key=args.dedupe_key,
        dedupe_keep=args.dedupe_keep,
        max_input_rows=args.max_input_rows,
        answer_max_chars=args.answer_max_chars,
        out_dir=out_dir,
        progress_label="dislike",
    )
    bad_cases = _cases_from_core(bad_core, "bad", one_per_session=args.one_per_session)
    _enrich_system_prompt(bad_cases, bad_rich)

    bad_kept: list[dict] = []
    bad_clean_stats = {"input": len(bad_cases), "pass": 0, "reject": 0, "reasons": {}}
    for c in bad_cases:
        ok, reason = clean_lite_case(c, mode="bad")
        if ok:
            bad_clean_stats["pass"] += 1
            bad_kept.append(c)
        else:
            bad_clean_stats["reject"] += 1
            bad_clean_stats["reasons"][reason] = bad_clean_stats["reasons"].get(reason, 0) + 1

    _write_progress(
        out_dir,
        {
            "phase": "after_clean_dislike",
            "lite_cases": len(bad_cases),
            "after_clean": len(bad_kept),
            "clean_stats": bad_clean_stats,
        },
    )

    bad_judged = _run_judges(
        bad_kept,
        "bad",
        model=args.judge_model,
        workers=args.workers,
        out_dir=out_dir,
        progress_label="dislike",
    )
    bad_rows: list[dict] = []
    bad_judge_fail = 0
    for case, judge, err in bad_judged:
        use_fb = bool(err)
        if err:
            bad_judge_fail += 1
        row = build_bad_train_row(
            case,
            judge if not use_fb else None,
            system_prompt=case.get("_system_prompt") or "",
            save_time=save_tag,
            use_judge_fallback=use_fb,
        )
        if keep_bad_row(row, min_judge_average=args.min_judge_average_bad, min_confidence=args.min_confidence_bad):
            bad_rows.append(row)

    bad_path = out_dir / "bad_case.jsonl"
    with bad_path.open("w", encoding="utf-8") as f:
        for r in bad_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    _write_progress(out_dir, {"phase": "bad_case_written", "rows": len(bad_rows), "path": str(bad_path)})

    good_core, good_rich, good_dedupe = _ingest_path(
        Path(args.like_file),
        dedupe_key=args.dedupe_key,
        dedupe_keep=args.dedupe_keep,
        max_input_rows=args.max_input_rows,
        answer_max_chars=args.answer_max_chars,
        out_dir=out_dir,
        progress_label="like",
    )
    good_cases = _cases_from_core(good_core, "good", one_per_session=args.one_per_session)
    _enrich_system_prompt(good_cases, good_rich)

    good_kept: list[dict] = []
    good_clean_stats = {"input": len(good_cases), "pass": 0, "reject": 0, "reasons": {}}
    for c in good_cases:
        ok, reason = clean_lite_case(c, mode="good")
        if ok:
            good_clean_stats["pass"] += 1
            good_kept.append(c)
        else:
            good_clean_stats["reject"] += 1
            good_clean_stats["reasons"][reason] = good_clean_stats["reasons"].get(reason, 0) + 1

    _write_progress(
        out_dir,
        {
            "phase": "after_clean_like",
            "lite_cases": len(good_cases),
            "after_clean": len(good_kept),
            "clean_stats": good_clean_stats,
        },
    )

    good_judged = _run_judges(
        good_kept,
        "good",
        model=args.judge_model,
        workers=args.workers,
        out_dir=out_dir,
        progress_label="like",
    )
    good_rows: list[dict] = []
    good_judge_fail = 0
    for case, judge, err in good_judged:
        use_fb = bool(err)
        if err:
            good_judge_fail += 1
        row = build_good_train_row(
            case,
            judge if not use_fb else None,
            system_prompt=case.get("_system_prompt") or "",
            judge_model=args.judge_model,
            save_time=save_tag,
            use_judge_fallback=use_fb,
        )
        if keep_good_row(row, min_judge_average=args.min_judge_average_good):
            good_rows.append(row)

    good_path = out_dir / "good_case.jsonl"
    with good_path.open("w", encoding="utf-8") as f:
        for r in good_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    report["bad"] = {
        "dedupe": bad_dedupe,
        "lite_cases": len(bad_cases),
        "after_clean": len(bad_kept),
        "clean": bad_clean_stats,
        "judge_failures": bad_judge_fail,
        "final_rows": len(bad_rows),
        "output": str(bad_path),
    }
    report["good"] = {
        "dedupe": good_dedupe,
        "lite_cases": len(good_cases),
        "after_clean": len(good_kept),
        "clean": good_clean_stats,
        "judge_failures": good_judge_fail,
        "final_rows": len(good_rows),
        "output": str(good_path),
    }
    (out_dir / "run_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_progress(out_dir, {"phase": "done", "report": str(out_dir / "run_report.json")})
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
