"""
Main pipeline orchestrator.

Input:  dialogues_valid.jsonl (validated user-feedback dialogues)
Output:
  - output/saved_auto.jsonl      — high/medium quality bad cases (matches saved_badcases.jsonl shape)
  - output/deleted_auto.jsonl    — records where Miner said usable=false or no bad case found
  - output/rejected_auto.jsonl   — records where Judge said should_keep=false
  - output/run_log.jsonl         — per-record trace (for debugging)

Usage:
  python pipeline.py --limit 20              # small test
  python pipeline.py --workers 5             # full run with 5 concurrent workers
  python pipeline.py --resume                # resume from checkpoint (skip cids already done)
"""

import argparse
import json
import sys
import io
import time
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from signals import enrich_dialogue
from prefilter import prefilter
from miner import mine_badcases, DEFAULT_MINER_MODEL
from judge import judge_badcases_batch, DEFAULT_JUDGE_MODEL
from dim_normalize import normalize_dimension
from constants import DislikeConstants, SharedConstants

DATA_FILE = ROOT.parent / DislikeConstants.INPUT_FILENAME
OUT_DIR = ROOT / SharedConstants.OUTPUTS_DIR
LOG_DIR = ROOT / SharedConstants.LOGS_DIR
OUT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

SAVED_FILE = OUT_DIR / DislikeConstants.SAVED_FILENAME
DELETED_FILE = OUT_DIR / DislikeConstants.DELETED_FILENAME
REJECTED_FILE = OUT_DIR / DislikeConstants.REJECTED_FILENAME
LOG_FILE = LOG_DIR / DislikeConstants.RUN_LOG_FILENAME




PROCESSED_CIDS_FILE = OUT_DIR / DislikeConstants.PROCESSED_CIDS_FILENAME


def load_processed_cids():
    """Load ALL processed cids from outputs + cached processed_cids.json."""
    cids = set()

    # 1. Read cached file (fast path — covers all historical runs)
    if PROCESSED_CIDS_FILE.exists():
        try:
            data = json.loads(PROCESSED_CIDS_FILE.read_text(encoding="utf-8"))
            cids.update(data.get("processed_cids", []))
        except Exception:
            pass

    # 2. Also scan current output files (catches anything written since last cache update)
    for fp in [SAVED_FILE, DELETED_FILE, REJECTED_FILE]:
        if fp.exists():
            for line in fp.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        cids.add(json.loads(line).get("cid", ""))
                    except Exception:
                        pass



    cids.discard("")
    return cids


def save_processed_cids(cids: set):
    """Update the processed_cids.json cache after a pipeline run."""
    # Load raw total for stats
    raw_total = 0
    if DATA_FILE.exists():
        raw_total = sum(1 for line in open(DATA_FILE, "r", encoding="utf-8") if line.strip())

    data = {
        "total_raw": raw_total,
        "total_processed": len(cids),
        "total_remaining": raw_total - len(cids),
        "last_updated": __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "processed_cids": sorted(cids),
    }
    with open(PROCESSED_CIDS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"\n  Updated processed_cids.json: {len(cids):,} cids")


def load_valid_records(limit=None):
    """Load & parse valid dataset."""
    records = []
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line: continue
            row = json.loads(line)
            history = row.get("history", "[]")
            if isinstance(history, str):
                history = json.loads(history)
            records.append({
                "cid": row.get("cid", ""),
                "history": history,
                "_raw": row,
            })
            if limit and len(records) >= limit:
                break
    return records


def _write_jsonl(fp, obj, lock):
    with lock:
        with open(fp, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            f.flush()


def build_saved_entry(record, enrichment, miner_output, judge_output, bc: dict):
    """
    Build a saved_auto.jsonl entry that matches saved_badcases.jsonl schema PLUS extra inspection fields.
    """
    history = record["history"]
    source_tid = bc.get("source_query_turn_id")
    dislike_tid = bc.get("dislike_turn_id")

    # source_query object (matches human annotation schema)
    source_msg = ""
    for m in history:
        if m.get("turn_index") == source_tid:
            source_msg = m.get("message", "") or ""
            break

    # selected_turn_indices: all turns strictly BEFORE source_query_turn_id
    selected = [m.get("turn_index") for m in history if m.get("turn_index", -1) < (source_tid or 0)]
    selected_history = [m for m in history if m.get("turn_index") in selected]

    # Infer auto-tag: explicit complaint → failure_testcase, else query_testcase
    sentiment = bc.get("sentiment", "implicit")
    tags = ["failure_testcase"] if sentiment == "explicit" else ["query_testcase"]

    # Normalize dimension (preserve raw)
    raw_dim = bc.get("failure_dimension", "")
    norm_dim = normalize_dimension(raw_dim)

    return {
        # ── matches human annotation ──
        "cid": record["cid"],
        "save_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tags": tags,
        "source_query": {
            "turn_index": source_tid,
            "message": source_msg,
        },
        "selected_turn_indices": selected,
        "selected_history": selected_history,
        "full_history": history,
        # ── enrichment for postprocess ──
        "enrichment": {
            "user_reactions": enrichment.get("user_reactions", []),
            "recovery_chains": enrichment.get("recovery_chains", []),
            "reaction_summary": enrichment.get("reaction_summary", {}),
            "total_dissatisfied": enrichment.get("total_dissatisfied", 0),
        },
        # ── additional auto-labeling metadata ──
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
            # Judge scores
            "judge_scores": (judge_output.get("parsed") or {}).get("scores"),
            "judge_average": (judge_output.get("parsed") or {}).get("average"),
            "overall_quality": (judge_output.get("parsed") or {}).get("overall_quality"),
            "judge_audit": (judge_output.get("parsed") or {}).get("audit_notes"),
            # Cross-verification
            "miner_model": miner_output["model"],
            "judge_model": judge_output["model"],
            "miner_latency_s": miner_output["latency_s"],
            "judge_latency_s": judge_output["latency_s"],
            "total_tokens": miner_output["tokens"] + judge_output["tokens"],
        },
    }


def process_one(record, miner_model, judge_model, log_lock, stats):
    """Process a single dialogue through the full pipeline."""
    cid = record["cid"]
    history = record["history"]

    trace = {
        "cid": cid,
        "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "stages": {},
    }

    # Stage 0: Enrich signals
    t0 = time.time()
    enrichment = enrich_dialogue(history)
    trace["stages"]["enrich"] = {
        "duration_s": round(time.time() - t0, 3),
        "turn_count": enrichment["turn_count"],
        "unliked_turns": enrichment["unliked_turns"],
        "explicit_complaints": len(enrichment["explicit_complaints"]),
        "total_dissatisfied": enrichment.get("total_dissatisfied", 0),
        "reaction_summary": enrichment.get("reaction_summary", {}),
        "recovery_chains": len(enrichment.get("recovery_chains", [])),
        "has_interrupted": enrichment["has_interrupted"],
    }

    # Stage 1: Pre-filter
    keep, reason = prefilter(record, enrichment)
    trace["stages"]["prefilter"] = {"keep": keep, "reason": reason}
    if not keep:
        _write_jsonl(DELETED_FILE, {
            "cid": cid,
            "reason": "prefilter:" + reason,
            "enrichment_summary": trace["stages"]["enrich"],
            "turn_count": len(history),
            "unliked_turns": enrichment.get("unliked_turns", []),
            "total_dissatisfied": enrichment.get("total_dissatisfied", 0),
        }, log_lock)
        _write_jsonl(LOG_FILE, trace, log_lock)
        stats["prefilter_rejected"] += 1
        return

    # Stage 2: Miner
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
        _write_jsonl(DELETED_FILE, {
            "cid": cid,
            "reason": "miner_failed",
            "miner_error": miner_out["llm_error"] or miner_out["parse_error"],
            "raw_output_head": miner_out["raw_output"][:300],
            "turn_count": len(history),
            "unliked_turns": enrichment.get("unliked_turns", []),
            "total_dissatisfied": enrichment.get("total_dissatisfied", 0),
        }, log_lock)
        _write_jsonl(LOG_FILE, trace, log_lock)
        stats["miner_failed"] += 1
        return

    parsed = miner_out["parsed"]
    usable = parsed.get("usable", False)
    badcases = parsed.get("badcases", []) or []
    trace["stages"]["miner"]["usable"] = usable
    trace["stages"]["miner"]["n_candidates"] = len(badcases)
    trace["stages"]["miner"]["reject_reason"] = parsed.get("reject_reason", "")

    if not usable or not badcases:
        _write_jsonl(DELETED_FILE, {
            "cid": cid,
            "reason": "miner_rejected:" + (parsed.get("reject_reason", "") or "no_badcases"),
            "miner_parsed": parsed,
            "turn_count": len(history),
            "unliked_turns": enrichment.get("unliked_turns", []),
        }, log_lock)
        _write_jsonl(LOG_FILE, trace, log_lock)
        stats["miner_rejected"] += 1
        return

    # Stage 3: Batch-judge ALL candidates in one LLM call
    t0 = time.time()
    judge_batch = judge_badcases_batch(record, enrichment, badcases, model=judge_model)
    judge_duration = round(time.time() - t0, 3)
    trace["stages"]["judge"] = {
        "duration_s": judge_duration,
        "ok": judge_batch["llm_ok"],
        "tokens": judge_batch["tokens"],
        "n_candidates": len(badcases),
        "candidates": [],
    }

    n_kept = 0
    n_rejected = 0
    for idx, bc in enumerate(badcases):
        result = judge_batch["results"][idx] if idx < len(judge_batch["results"]) else None
        jt = {"idx": idx}

        if result is None or not result.get("parse_ok"):
            jt["error"] = "batch_parse_failed"
            # Fallback: trust miner's confidence > 0.6 → keep as medium
            conf = bc.get("confidence", 0)
            fake_judge = {"model": judge_batch["model"], "latency_s": judge_duration / max(len(badcases), 1),
                          "tokens": judge_batch["tokens"] // max(len(badcases), 1),
                          "parsed": None, "llm_ok": False, "llm_error": "batch_miss"}
            if conf >= 0.6:
                entry = build_saved_entry(record, enrichment, miner_out, fake_judge, bc)
                entry["auto_label"]["overall_quality"] = "medium"
                entry["auto_label"]["judge_audit"] = "judge_batch_missed, fallback to miner confidence"
                _write_jsonl(SAVED_FILE, entry, log_lock)
                n_kept += 1
            else:
                _write_jsonl(REJECTED_FILE, {
                    "cid": cid, "reason": "judge_batch_missed_low_conf",
                    "candidate": bc,
                }, log_lock)
                n_rejected += 1
        else:
            j = result["parsed"]
            jt["quality"] = j.get("overall_quality")
            jt["should_keep"] = j.get("should_keep")

            # Apply Judge's suggested_corrections if present
            corrections = j.get("suggested_corrections") or {}
            for k in ("dislike_turn_id", "failure_dimension", "source_query_turn_id"):
                if corrections.get(k) is not None:
                    bc[k] = corrections[k]

            judge_single = {
                "model": judge_batch["model"],
                "latency_s": judge_duration / max(len(badcases), 1),
                "tokens": judge_batch["tokens"] // max(len(badcases), 1),
                "parsed": j, "llm_ok": True, "llm_error": "",
            }

            if j.get("should_keep"):
                entry = build_saved_entry(record, enrichment, miner_out, judge_single, bc)
                _write_jsonl(SAVED_FILE, entry, log_lock)
                n_kept += 1
                stats.setdefault("quality_" + (j.get("overall_quality") or "unknown"), 0)
                stats["quality_" + (j.get("overall_quality") or "unknown")] += 1
            else:
                _write_jsonl(REJECTED_FILE, {
                    "cid": cid,
                    "reason": "judge_rejected:" + (j.get("overall_quality") or "low"),
                    "judge_quality": j.get("overall_quality"),
                    "judge_audit": j.get("audit_notes", ""),
                    "candidate": bc,
                    "judge": j,
                }, log_lock)
                n_rejected += 1

        trace["stages"]["judge"]["candidates"].append(jt)

    trace["n_kept"] = n_kept
    trace["n_rejected"] = n_rejected
    _write_jsonl(LOG_FILE, trace, log_lock)
    stats["processed"] += 1
    stats["badcases_saved"] += n_kept
    stats["badcases_rejected"] += n_rejected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Max records to process (0=all)")
    parser.add_argument("--offset", type=int, default=0, help="Skip first N records")
    parser.add_argument("--workers", type=int, default=SharedConstants.DEFAULT_WORKERS, help="Concurrent workers")
    parser.add_argument("--miner-model", default=DEFAULT_MINER_MODEL)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--resume", action="store_true", help="Skip cids already in output")
    parser.add_argument("--fresh", action="store_true", help="Clear output files first")
    parser.add_argument("--tag", default="", help="Tag for progress file (e.g. progress_TAG.json)")
    args = parser.parse_args()

    if args.fresh:
        for fp in [SAVED_FILE, DELETED_FILE, REJECTED_FILE, LOG_FILE]:
            if fp.exists():
                fp.unlink()
        print(f"Cleared output files")

    print(f"Loading data from {DATA_FILE.name}...")
    records = load_valid_records(limit=None)
    print(f"Loaded {len(records)} records")

    if args.offset:
        records = records[args.offset:]
    if args.limit:
        records = records[:args.limit]

    if args.resume:
        processed = load_processed_cids()
        before = len(records)
        records = [r for r in records if r["cid"] not in processed]
        print(f"Resume: skipped {before - len(records)} already-processed, {len(records)} remaining (tracked {len(processed):,} cids)")

    print(f"\n{'='*80}")
    print(f"Processing {len(records)} records")
    print(f"  Miner: {args.miner_model}")
    print(f"  Judge: {args.judge_model}")
    print(f"  Workers: {args.workers}")
    print(f"  Output: {OUT_DIR}")
    print(f"{'='*80}\n")

    stats = {
        "total": len(records),
        "processed": 0,
        "prefilter_rejected": 0,
        "miner_failed": 0,
        "miner_rejected": 0,
        "badcases_saved": 0,
        "badcases_rejected": 0,
    }

    PROGRESS_FILE = LOG_DIR / (f"progress_{args.tag}.json" if args.tag else SharedConstants.PROGRESS_FILENAME)

    import threading
    log_lock = threading.Lock()

    def _write_progress(n_done, elapsed):
        rate = n_done / elapsed if elapsed > 0 else 0
        remaining = len(records) - n_done
        eta_s = remaining / rate if rate > 0 else 0
        progress = {
            "status": "running",
            "done": n_done,
            "total": len(records),
            "percent": round(n_done / len(records) * 100, 1) if records else 0,
            "rate_per_s": round(rate, 3),
            "elapsed_min": round(elapsed / 60, 1),
            "eta_min": round(eta_s / 60, 1),
            "miner_model": args.miner_model,
            "judge_model": args.judge_model,
            "workers": args.workers,
            **{k: v for k, v in stats.items() if k != "total"},
        }
        with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
            json.dump(progress, f, ensure_ascii=False)

    t_start = time.time()
    _write_progress(0, 0)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_one, rec, args.miner_model, args.judge_model, log_lock, stats): rec["cid"]
            for rec in records
        }
        n_done = 0
        for fut in as_completed(futures):
            n_done += 1
            try:
                fut.result()
            except Exception as e:
                cid = futures[fut]
                print(f"  ❌ [{n_done}/{len(records)}] {cid[:12]}... ERROR: {e}")
                _write_jsonl(DELETED_FILE, {"cid": cid, "reason": f"pipeline_exception: {str(e)[:200]}"}, log_lock)

            elapsed = time.time() - t_start
            _write_progress(n_done, elapsed)

            if n_done % 5 == 0 or n_done == len(records):
                rate = n_done / elapsed if elapsed > 0 else 0
                eta = (len(records) - n_done) / rate if rate > 0 else 0
                print(
                    f"  [{n_done:4d}/{len(records)}] "
                    f"saved={stats['badcases_saved']} rej={stats['badcases_rejected']} "
                    f"prefilter_rej={stats['prefilter_rejected']} miner_fail={stats['miner_failed']} "
                    f"miner_rej={stats['miner_rejected']} | "
                    f"{rate:.2f} rec/s, ETA {eta/60:.1f} min"
                )

    total_elapsed = time.time() - t_start
    # Write final progress
    final_progress = {
        "status": "done",
        "done": len(records),
        "total": len(records),
        "percent": 100.0,
        "elapsed_min": round(total_elapsed / 60, 1),
        "eta_min": 0,
        **{k: v for k, v in stats.items() if k != "total"},
    }
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(final_progress, f, ensure_ascii=False)

    print(f"\n{'='*80}")
    print(f"DONE in {total_elapsed/60:.1f} min")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"{'='*80}")
    print(f"  saved_auto.jsonl:    {SAVED_FILE}")
    print(f"  deleted_auto.jsonl:  {DELETED_FILE}")
    print(f"  rejected_auto.jsonl: {REJECTED_FILE}")
    print(f"  run_log.jsonl:       {LOG_FILE}")

    # Update processed cids cache
    all_processed = load_processed_cids()
    save_processed_cids(all_processed)


if __name__ == "__main__":
    main()
