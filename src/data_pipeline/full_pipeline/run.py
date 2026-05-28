#!/usr/bin/env python3
"""Full bad-case pipeline (raw JSONL -> saved/deleted/rejected)."""

from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from full_pipeline.dim_normalize import normalize_dimension
from full_pipeline.judge import DEFAULT_JUDGE_MODEL, judge_badcases_batch
from full_pipeline.miner import DEFAULT_MINER_MODEL, mine_badcases
from full_pipeline.prefilter import prefilter
from full_pipeline.signals import enrich_dialogue
from simple_pipeline.dedupe import dedupe_merge_from_iter, resolve_dedupe_key_field
from simple_pipeline.raw_loader import iter_raw_items_from_path
from simple_pipeline.session_builder import build_sessions_from_items, session_to_core_record


def _write_jsonl(fp: Path, obj: dict, lock: threading.Lock) -> None:
    with lock:
        with fp.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _prepare_core_records(input_path: Path, dedupe_key: str, dedupe_keep: str, answer_max_chars: int, max_input_rows: int = 0) -> list[dict]:
    key_field = resolve_dedupe_key_field(dedupe_key)
    raw_iter = iter_raw_items_from_path(input_path, max_rows=max_input_rows)
    rows, _ = dedupe_merge_from_iter(raw_iter, key_field=key_field, keep=dedupe_keep)
    sessions = build_sessions_from_items(rows, answer_max_chars=answer_max_chars)
    return [session_to_core_record(s) for s in sessions]


def build_saved_entry(record: dict, enrichment: dict, miner_output: dict, judge_output: dict, bc: dict) -> dict:
    history = record["history"]
    source_tid = bc.get("source_query_turn_id")
    dislike_tid = bc.get("dislike_turn_id")
    source_msg = ""
    for m in history:
        if m.get("turn_index") == source_tid:
            source_msg = m.get("message", "") or ""
            break
    selected = [m.get("turn_index") for m in history if m.get("turn_index", -1) < (source_tid or 0)]
    selected_history = [m for m in history if m.get("turn_index") in selected]

    sentiment = bc.get("sentiment", "implicit")
    tags = ["failure_testcase"] if sentiment == "explicit" else ["query_testcase"]
    raw_dim = bc.get("failure_dimension", "")
    norm_dim = normalize_dimension(raw_dim)
    parsed = judge_output.get("parsed") or {}

    return {
        "cid": record["cid"],
        "save_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tags": tags,
        "source_query": {"turn_index": source_tid, "message": source_msg},
        "selected_turn_indices": selected,
        "selected_history": selected_history,
        "full_history": history,
        "enrichment": {
            "unliked_turns": enrichment.get("unliked_turns", []),
            "explicit_complaints": enrichment.get("explicit_complaints", []),
            "turn_count": enrichment.get("turn_count", 0),
        },
        "auto_label": {
            "dislike_turn_id": dislike_tid,
            "failure_dimension": norm_dim,
            "failure_dimension_raw": raw_dim,
            "scenario": bc.get("scenario"),
            "sentiment": sentiment,
            "complaint_snippet": bc.get("complaint_snippet"),
            "explanation": bc.get("explanation"),
            "signal_type": bc.get("signal_type", "dislike"),
            "signal_confidence": bc.get("signal_confidence", "high"),
            "agent_response_preview": bc.get("agent_response_preview", ""),
            "user_reaction_after_failure": bc.get("user_reaction_after_failure", ""),
            "recovery_attempted": bc.get("recovery_attempted"),
            "recovery_successful": bc.get("recovery_successful"),
            "representativeness": bc.get("representativeness"),
            "confidence": bc.get("confidence"),
            "judge_scores": parsed.get("scores"),
            "judge_average": parsed.get("average"),
            "overall_quality": parsed.get("overall_quality"),
            "judge_audit": parsed.get("audit_notes"),
            "miner_model": miner_output.get("model"),
            "judge_model": judge_output.get("model"),
            "miner_latency_s": miner_output.get("latency_s", 0),
            "judge_latency_s": judge_output.get("latency_s", 0),
            "total_tokens": miner_output.get("tokens", 0) + judge_output.get("tokens", 0),
        },
    }


def load_processed_cids(saved_file: Path, deleted_file: Path, rejected_file: Path, processed_file: Path) -> set[str]:
    cids: set[str] = set()
    if processed_file.exists():
        try:
            data = json.loads(processed_file.read_text(encoding="utf-8"))
            cids.update(data.get("processed_cids", []))
        except Exception:
            pass
    for fp in (saved_file, deleted_file, rejected_file):
        if not fp.exists():
            continue
        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    cid = json.loads(line).get("cid", "")
                    if cid:
                        cids.add(cid)
                except Exception:
                    continue
    return cids


def save_processed_cids(cids: set[str], processed_file: Path) -> None:
    data = {
        "total_processed": len(cids),
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "processed_cids": sorted(cids),
    }
    processed_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def process_one(record: dict, miner_model: str, judge_model: str, out_files: dict[str, Path], log_lock: threading.Lock, stats: dict) -> None:
    cid = record.get("cid", "")
    history = record.get("history", [])
    trace = {"cid": cid, "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "stages": {}}

    t0 = time.time()
    enrichment = enrich_dialogue(history)
    trace["stages"]["enrich"] = {"duration_s": round(time.time() - t0, 3), "turn_count": enrichment.get("turn_count"), "unliked_turns": enrichment.get("unliked_turns", [])}

    keep, reason = prefilter(record, enrichment)
    trace["stages"]["prefilter"] = {"keep": keep, "reason": reason}
    if not keep:
        _write_jsonl(out_files["deleted"], {"cid": cid, "reason": "prefilter:" + reason}, log_lock)
        _write_jsonl(out_files["log"], trace, log_lock)
        stats["prefilter_rejected"] += 1
        return

    t0 = time.time()
    miner_out = mine_badcases(record, enrichment, model=miner_model)
    trace["stages"]["miner"] = {
        "duration_s": round(time.time() - t0, 3),
        "ok": miner_out["llm_ok"],
        "parse_ok": miner_out["parse_error"] == "",
        "error": miner_out["llm_error"] or miner_out["parse_error"],
        "tokens": miner_out["tokens"],
    }
    if not miner_out["llm_ok"] or miner_out["parsed"] is None:
        _write_jsonl(out_files["deleted"], {"cid": cid, "reason": "miner_failed", "miner_error": miner_out["llm_error"] or miner_out["parse_error"]}, log_lock)
        _write_jsonl(out_files["log"], trace, log_lock)
        stats["miner_failed"] += 1
        return

    parsed = miner_out["parsed"]
    usable = parsed.get("usable", False)
    badcases = parsed.get("badcases", []) or []
    if not usable or not badcases:
        _write_jsonl(out_files["deleted"], {"cid": cid, "reason": "miner_rejected:" + (parsed.get("reject_reason", "") or "no_badcases")}, log_lock)
        _write_jsonl(out_files["log"], trace, log_lock)
        stats["miner_rejected"] += 1
        return

    t0 = time.time()
    judge_batch = judge_badcases_batch(record, enrichment, badcases, model=judge_model)
    judge_duration = round(time.time() - t0, 3)
    trace["stages"]["judge"] = {"duration_s": judge_duration, "ok": judge_batch["llm_ok"], "tokens": judge_batch["tokens"], "n_candidates": len(badcases)}

    n_kept = 0
    n_rejected = 0
    for idx, bc in enumerate(badcases):
        result = judge_batch["results"][idx] if idx < len(judge_batch["results"]) else None
        if result is None or not result.get("parse_ok"):
            _write_jsonl(out_files["rejected"], {"cid": cid, "reason": "judge_parse_failed", "candidate": bc}, log_lock)
            n_rejected += 1
            continue
        j = result["parsed"]
        judge_single = {
            "model": judge_batch["model"],
            "latency_s": judge_duration / max(len(badcases), 1),
            "tokens": judge_batch["tokens"] // max(len(badcases), 1),
            "parsed": j,
        }
        if j.get("should_keep"):
            entry = build_saved_entry(record, enrichment, miner_out, judge_single, bc)
            _write_jsonl(out_files["saved"], entry, log_lock)
            n_kept += 1
            qk = "quality_" + (j.get("overall_quality") or "unknown")
            stats[qk] = stats.get(qk, 0) + 1
        else:
            _write_jsonl(
                out_files["rejected"],
                {"cid": cid, "reason": "judge_rejected:" + (j.get("overall_quality") or "low"), "judge_quality": j.get("overall_quality"), "judge_audit": j.get("audit_notes", ""), "candidate": bc},
                log_lock,
            )
            n_rejected += 1

    trace["n_kept"] = n_kept
    trace["n_rejected"] = n_rejected
    _write_jsonl(out_files["log"], trace, log_lock)
    stats["processed"] += 1
    stats["badcases_saved"] += n_kept
    stats["badcases_rejected"] += n_rejected


def main() -> None:
    parser = argparse.ArgumentParser(description="Full bad-case pipeline")
    parser.add_argument("--input", required=True, help="Raw JSONL input path")
    parser.add_argument("--output-dir", default="outputs/full", help="Output directory")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--dedupe-key", default="auto")
    parser.add_argument("--dedupe-keep", choices=("last", "first"), default="last")
    parser.add_argument("--max-input-rows", type=int, default=0)
    parser.add_argument("--answer-max-chars", type=int, default=120_000)
    parser.add_argument("--miner-model", default=DEFAULT_MINER_MODEL)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_file = out_dir / "saved_auto.jsonl"
    deleted_file = out_dir / "deleted_auto.jsonl"
    rejected_file = out_dir / "rejected_auto.jsonl"
    log_file = out_dir / "run_log.jsonl"
    processed_file = out_dir / "processed_cids.json"
    out_files = {"saved": saved_file, "deleted": deleted_file, "rejected": rejected_file, "log": log_file}

    if args.fresh:
        for fp in (saved_file, deleted_file, rejected_file, log_file, processed_file):
            if fp.exists():
                fp.unlink()

    print("Preparing core records from raw input...")
    records = _prepare_core_records(
        Path(args.input),
        dedupe_key=args.dedupe_key,
        dedupe_keep=args.dedupe_keep,
        answer_max_chars=args.answer_max_chars,
        max_input_rows=args.max_input_rows,
    )
    if args.offset:
        records = records[args.offset :]
    if args.limit:
        records = records[: args.limit]

    if args.resume:
        processed = load_processed_cids(saved_file, deleted_file, rejected_file, processed_file)
        before = len(records)
        records = [r for r in records if r.get("cid") not in processed]
        print(f"Resume: skipped {before - len(records)} already processed, remaining {len(records)}")

    print(f"Processing {len(records)} records | workers={args.workers} | miner={args.miner_model} | judge={args.judge_model}")
    stats = {
        "total": len(records),
        "processed": 0,
        "prefilter_rejected": 0,
        "miner_failed": 0,
        "miner_rejected": 0,
        "badcases_saved": 0,
        "badcases_rejected": 0,
    }
    log_lock = threading.Lock()
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_one, rec, args.miner_model, args.judge_model, out_files, log_lock, stats): rec.get("cid", "") for rec in records}
        n_done = 0
        for fut in as_completed(futures):
            n_done += 1
            try:
                fut.result()
            except Exception as e:
                cid = futures[fut]
                _write_jsonl(deleted_file, {"cid": cid, "reason": f"pipeline_exception:{str(e)[:200]}"}, log_lock)
            if n_done % 10 == 0 or n_done == len(records):
                elapsed = time.time() - t_start
                rate = n_done / elapsed if elapsed > 0 else 0.0
                print(
                    f"[{n_done}/{len(records)}] saved={stats['badcases_saved']} rejected={stats['badcases_rejected']} "
                    f"prefilter={stats['prefilter_rejected']} miner_fail={stats['miner_failed']} miner_rej={stats['miner_rejected']} | {rate:.2f} rec/s"
                )

    total_elapsed = time.time() - t_start
    all_processed = load_processed_cids(saved_file, deleted_file, rejected_file, processed_file)
    save_processed_cids(all_processed, processed_file)

    print("=" * 80)
    print(f"DONE in {total_elapsed/60:.1f} min")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"  saved: {saved_file}")
    print(f"  deleted: {deleted_file}")
    print(f"  rejected: {rejected_file}")
    print(f"  log: {log_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
