#!/usr/bin/env python3
"""
May 原始 JSONL：仅按 message id 去重 → 拼完整 session → 各写一行 JSON。

不做：train/test 划分、清洗、截断（除 answer 字段已有的 max_chars 上限）、Judge、lite case。

输出：
  - 点踩源 → bad_sessions_may.jsonl（每行一个 session，结构见 session_builder.build_sessions_from_items）
  - 点赞源 → good_sessions_may.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from auto_labeling.unified_pipeline.dedupe import dedupe_merge_from_iter, resolve_dedupe_key_field
from auto_labeling.unified_pipeline.may_data_paths import (
    DEFAULT_BAD_SESSIONS_JSONL,
    DEFAULT_DISLIKE_JSONL,
    DEFAULT_GOOD_SESSIONS_JSONL,
    DEFAULT_LIKE_JSONL,
)
from auto_labeling.unified_pipeline.raw_loader import iter_raw_items_from_path
from auto_labeling.unified_pipeline.session_builder import build_sessions_from_items


def load_sessions(
    path: Path,
    *,
    dedupe_key: str,
    dedupe_keep: str,
    max_input_rows: int,
    answer_max_chars: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    key_field = resolve_dedupe_key_field(dedupe_key)
    raw_iter = iter_raw_items_from_path(path, max_rows=max_input_rows or 0)
    rows, dstats = dedupe_merge_from_iter(raw_iter, key_field=key_field, keep=dedupe_keep)
    sessions = build_sessions_from_items(rows, answer_max_chars=answer_max_chars)
    return sessions, dstats


def write_sessions_jsonl(sessions: list[dict[str, Any]], out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for s in sessions:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dislike-file", type=Path, default=DEFAULT_DISLIKE_JSONL)
    p.add_argument("--like-file", type=Path, default=DEFAULT_LIKE_JSONL)
    p.add_argument(
        "--output-bad",
        type=Path,
        default=DEFAULT_BAD_SESSIONS_JSONL,
    )
    p.add_argument(
        "--output-good",
        type=Path,
        default=DEFAULT_GOOD_SESSIONS_JSONL,
    )
    p.add_argument("--dedupe-key", default="auto")
    p.add_argument("--dedupe-keep", choices=("last", "first"), default="last")
    p.add_argument("--max-input-rows", type=int, default=0)
    p.add_argument("--answer-max-chars", type=int, default=120_000)
    args = p.parse_args()

    tag = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report: dict[str, Any] = {"started": tag}

    bad_sessions, bad_dedupe = load_sessions(
        Path(args.dislike_file),
        dedupe_key=args.dedupe_key,
        dedupe_keep=args.dedupe_keep,
        max_input_rows=args.max_input_rows,
        answer_max_chars=args.answer_max_chars,
    )
    good_sessions, good_dedupe = load_sessions(
        Path(args.like_file),
        dedupe_key=args.dedupe_key,
        dedupe_keep=args.dedupe_keep,
        max_input_rows=args.max_input_rows,
        answer_max_chars=args.answer_max_chars,
    )

    nb = write_sessions_jsonl(bad_sessions, Path(args.output_bad))
    ng = write_sessions_jsonl(good_sessions, Path(args.output_good))

    report["bad"] = {
        "input": str(args.dislike_file),
        "dedupe": bad_dedupe,
        "sessions": nb,
        "output": str(args.output_bad),
    }
    report["good"] = {
        "input": str(args.like_file),
        "dedupe": good_dedupe,
        "sessions": ng,
        "output": str(args.output_good),
    }

    rep_path = Path(args.output_bad).with_name("sessions_dedupe_report.json")
    rep_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
