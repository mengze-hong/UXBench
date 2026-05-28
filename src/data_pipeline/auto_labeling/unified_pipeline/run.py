#!/usr/bin/env python3
"""
Unified lightweight pipeline (local only).

1) Raw JSONL → message-id dedupe → full sessions
2) Export **only** turns with 点踩 (is_unliked) / 点赞 (is_liked) — no signal mining,
   no Miner, no multi-candidate Judge.

Examples
--------
  # One-shot: prepare + lite export
  python -m auto_labeling.unified_pipeline.run run \\
    --input \"may data/raw/点踩-五月1-10.jsonl\" \\
    --output-dir auto_labeling/unified_pipeline/out_may \\
    --mode bad

  python -m auto_labeling.unified_pipeline.run prepare \\
    --input \"may data/raw/点踩-五月1-10.jsonl\" --output-dir out/tmp --mode good

  python -m auto_labeling.unified_pipeline.run export \\
    --output-dir out/tmp --mode bad --one-per-session
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from auto_labeling.unified_pipeline.dedupe import dedupe_merge_from_iter, resolve_dedupe_key_field
from auto_labeling.unified_pipeline.lite_extract import extract_all_core_records
from auto_labeling.unified_pipeline.raw_loader import iter_raw_items_from_path
from auto_labeling.unified_pipeline.session_builder import build_sessions_from_items, session_to_core_record


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def cmd_prepare(args: argparse.Namespace) -> dict:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    key_field = resolve_dedupe_key_field(args.dedupe_key)

    raw_iter = iter_raw_items_from_path(Path(args.input), max_rows=args.max_input_rows or 0)
    rows, dstats = dedupe_merge_from_iter(raw_iter, key_field=key_field, keep=args.dedupe_keep)

    sessions = build_sessions_from_items(rows, answer_max_chars=args.answer_max_chars)
    core_records = [session_to_core_record(s) for s in sessions]

    (out / "01_raw_dedupe_stats.json").write_text(
        json.dumps({"type": "dedupe_stats", **dstats}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_jsonl(out / "02_sessions_rich.jsonl", sessions)
    _write_jsonl(out / "03_sessions_core.jsonl", core_records)

    summary = {"dedupe": dstats, "sessions": len(sessions), "core_records": len(core_records)}
    print(json.dumps(summary, ensure_ascii=False))
    return summary


def cmd_export(args: argparse.Namespace) -> dict:
    out = Path(args.output_dir)
    core_path = Path(args.core) if getattr(args, "core", None) else out / "03_sessions_core.jsonl"
    if not core_path.exists():
        raise SystemExit(f"Missing core file: {core_path} (run `prepare` first or pass --core)")

    records: list[dict] = []
    with core_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    mode = args.mode
    if mode not in ("bad", "good"):
        raise SystemExit("--mode must be bad or good")

    lite_rows, estats = extract_all_core_records(
        records,
        mode,
        one_per_session=args.one_per_session,
        include_full_history=args.include_full_history,
    )

    out_name = "04_lite_dislike_cases.jsonl" if mode == "bad" else "04_lite_like_cases.jsonl"
    _write_jsonl(out / out_name, lite_rows)
    (out / "05_lite_extract_stats.json").write_text(
        json.dumps({"type": "lite_extract_stats", **estats}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(estats, ensure_ascii=False))
    return estats


def cmd_run(args: argparse.Namespace) -> None:
    cmd_prepare(args)
    cmd_export(args)


def main() -> None:
    p = argparse.ArgumentParser(description="Unified lite pipeline (local, feedback-turn only)")
    sub = p.add_subparsers(dest="command", required=True)

    def add_prepare_flags(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--input", required=True, help="Raw JSONL (one JSON object per line)")
        sp.add_argument("--output-dir", required=True, help="Output directory")
        sp.add_argument(
            "--mode",
            choices=("bad", "good"),
            default="bad",
            help="good/bad only affects export filenames and which flag is scanned",
        )
        sp.add_argument("--dedupe-key", default="auto", help="message id field, or 'auto'")
        sp.add_argument("--dedupe-keep", choices=("last", "first"), default="last")
        sp.add_argument("--max-input-rows", type=int, default=0, help="0 = read entire file")
        sp.add_argument("--answer-max-chars", type=int, default=120_000)

    sp_prepare = sub.add_parser("prepare", help="Dedupe + build sessions (steps 01–03)")
    add_prepare_flags(sp_prepare)

    sp_export = sub.add_parser("export", help="Lite cases from 03_sessions_core.jsonl (step 04–05)")
    sp_export.add_argument("--output-dir", required=True)
    sp_export.add_argument("--mode", choices=("bad", "good"), default="bad")
    sp_export.add_argument(
        "--core",
        default="",
        help="Path to 03_sessions_core.jsonl (default: <output-dir>/03_sessions_core.jsonl)",
    )
    sp_export.add_argument(
        "--one-per-session",
        action="store_true",
        help="Keep only the first dislike (or like) turn per dialogue",
    )
    sp_export.add_argument(
        "--include-full-history",
        action="store_true",
        help="Attach full_history to each lite row (larger files)",
    )

    sp_run = sub.add_parser("run", help="prepare + export in one shot")
    add_prepare_flags(sp_run)
    sp_run.add_argument("--one-per-session", action="store_true")
    sp_run.add_argument("--include-full-history", action="store_true")

    sp_export.set_defaults(func=lambda a: cmd_export(a))
    sp_prepare.set_defaults(func=lambda a: cmd_prepare(a))
    sp_run.set_defaults(func=lambda a: cmd_run(a))

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
