"""
QA Layer: Multi-agent quality assurance for saved_auto.jsonl.

Strategy (optimized for speed):
  - Single fast model (gemini-2.5-flash) for initial scan — batch of 20 records
  - Second model (gpt-5.1) ONLY for items flagged as "reject" — confirmation
  - High concurrency: 20 workers for batch-level parallelism
  - Progress tracking via progress.json

Output: updates saved_auto.jsonl in place (adds qa_verdict field),
        moves rejected ones to rejected_auto.jsonl.
"""

import json, sys, io, time, argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from llm_client import call_llm, parse_json_output
from constants import DislikeConstants, SharedConstants

PROMPT_DIR = HERE.parent / SharedConstants.PROMPTS_DIR
QA_SYSTEM = (PROMPT_DIR / "qa_system.txt").read_text(encoding="utf-8")

OUTPUTS = HERE.parent / SharedConstants.OUTPUTS_DIR
SAVED_FILE = OUTPUTS / DislikeConstants.SAVED_FILENAME
REJECTED_FILE = OUTPUTS / DislikeConstants.REJECTED_FILENAME
LOG_DIR = HERE.parent / SharedConstants.LOGS_DIR
QA_LOG_FILE = LOG_DIR / "qa_log.jsonl"
PROGRESS_FILE = LOG_DIR / SharedConstants.PROGRESS_FILENAME

QA_MODEL_1 = "gemini-2.5-flash"   # fastest: ~2s, 100 tok/s
QA_MODEL_2 = "gpt-5.1"            # fast confirmation: ~1.1s

BATCH_SIZE = 20  # records per LLM call


def _summarize_for_qa(rec: dict) -> dict:
    """Extract minimal key fields for QA review (keep token count low)."""
    al = rec.get("auto_label", {})
    sq = rec.get("source_query", {})
    history = rec.get("full_history", [])

    dt_id = al.get("dislike_turn_id")
    agent_resp = ""
    for m in history:
        if isinstance(m, dict) and m.get("turn_index") == dt_id and m.get("role") == "assistant":
            agent_resp = (m.get("message", "") or "")[:300]
            break

    return {
        "source_query": (sq.get("message", "") or "")[:150],
        "agent_response_preview": agent_resp[:250],
        "failure_dimension": al.get("failure_dimension", ""),
        "explanation": (al.get("explanation", "") or "")[:150],
        "signal_type": al.get("signal_type", ""),
        "judge_average": al.get("judge_average", 0),
        "overall_quality": al.get("overall_quality", ""),
    }


def qa_batch(batch: list, model: str) -> list:
    """Run QA on a batch of records with one model. Returns list of verdicts."""
    summaries = [{"index": i, **_summarize_for_qa(rec)} for i, rec in enumerate(batch)]

    user_prompt = (
        f"# 待审核 Bad Cases（{len(summaries)} 条）\n"
        f"```json\n{json.dumps(summaries, ensure_ascii=False, indent=1)}\n```\n\n"
        f"请对每条进行质量验证，输出 JSON 数组。"
    )

    llm_res = call_llm(
        messages=[
            {"role": "system", "content": QA_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        model=model,
        max_tokens=1500 + 200 * len(batch),
        temperature=0.1,
    )

    if not llm_res.ok:
        return [None] * len(batch)

    parsed, _ = parse_json_output(llm_res.content)
    if parsed is None:
        return [None] * len(batch)

    results = [None] * len(batch)
    if isinstance(parsed, list):
        for item in parsed:
            idx = item.get("index", -1)
            if 0 <= idx < len(results):
                results[idx] = item
    elif isinstance(parsed, dict):
        results[0] = parsed

    return results


def merge_verdicts(v1, v2) -> dict:
    """Merge two QA verdicts. Conservative: if BOTH reject → reject; if only one → downgrade."""
    if v1 is None and v2 is None:
        return {"verdict": "keep", "quality": "medium", "notes": "both QA agents failed"}
    if v1 is None:
        return v2
    if v2 is None:
        return v1

    vd1 = v1.get("verdict", "keep")
    vd2 = v2.get("verdict", "keep")

    if vd1 == "reject" and vd2 == "reject":
        issues = list(set((v1.get("issues") or []) + (v2.get("issues") or [])))
        return {"verdict": "reject", "quality": "low", "issues": issues,
                "notes": f"Both agents reject",
                "corrected_dimension": v1.get("corrected_dimension") or v2.get("corrected_dimension")}

    if vd1 == "reject" or vd2 == "reject":
        # Only one rejects → downgrade instead (conservative)
        return {"verdict": "downgrade", "quality": "medium",
                "issues": list(set((v1.get("issues") or []) + (v2.get("issues") or []))),
                "notes": f"One agent rejects, downgraded",
                "corrected_dimension": v1.get("corrected_dimension") or v2.get("corrected_dimension")}

    if vd1 == "downgrade" or vd2 == "downgrade":
        return {"verdict": "downgrade", "quality": "medium",
                "notes": "downgraded",
                "corrected_dimension": v1.get("corrected_dimension") or v2.get("corrected_dimension")}

    return {"verdict": "keep", "quality": v1.get("quality", "high"),
            "notes": "pass",
            "corrected_dimension": v1.get("corrected_dimension") or v2.get("corrected_dimension")}


def process_batch(batch_idx, batch, lock, stats):
    """Process a single batch: fast scan + optional confirmation for rejects."""
    # Fast single-agent scan
    results1 = qa_batch(batch, QA_MODEL_1)

    # Only call second model for items agent1 wants to reject
    need_confirm = [i for i, v in enumerate(results1) if v and v.get("verdict") == "reject"]

    results2 = [None] * len(batch)
    if need_confirm:
        confirm_batch = [batch[i] for i in need_confirm]
        confirm_results = qa_batch(confirm_batch, QA_MODEL_2)
        for j, orig_i in enumerate(need_confirm):
            results2[orig_i] = confirm_results[j] if j < len(confirm_results) else None

    for i in range(len(batch)):
        if i in need_confirm and results2[i] is not None:
            v = merge_verdicts(results1[i], results2[i])
        else:
            v = results1[i] or {"verdict": "keep", "quality": "medium", "notes": "qa_failed_fallback"}

        batch[i]["qa_verdict"] = v

        if v.get("corrected_dimension"):
            batch[i]["auto_label"]["failure_dimension"] = v["corrected_dimension"]

        with lock:
            if v["verdict"] == "reject":
                stats["rejected"] += 1
            elif v["verdict"] == "downgrade":
                batch[i]["auto_label"]["overall_quality"] = "medium"
                stats["downgraded"] += 1
            else:
                stats["kept"] += 1


def main():
    parser = argparse.ArgumentParser(description="QA layer for saved_auto.jsonl")
    parser.add_argument("--workers", type=int, default=15, help="Concurrent workers (default: 15)")
    parser.add_argument("--limit", type=int, default=0, help="Max records to QA (0=all)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Records per LLM call")
    args = parser.parse_args()

    if not SAVED_FILE.exists():
        print("No saved_auto.jsonl found")
        return

    records = []
    with open(SAVED_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    # Skip already QA'd records
    todo = [r for r in records if "qa_verdict" not in r]
    if args.limit:
        todo = todo[:args.limit]

    print(f"Loaded {len(records):,} records, {len(todo):,} need QA")
    if not todo:
        print("All records already QA'd")
        return

    bs = args.batch_size
    batches = [todo[i:i+bs] for i in range(0, len(todo), bs)]
    print(f"Processing {len(batches):,} batches of up to {bs}, workers={args.workers}")

    lock = threading.Lock()
    stats = {"kept": 0, "downgraded": 0, "rejected": 0, "failed": 0}
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_batch, bi, batch, lock, stats): bi
            for bi, batch in enumerate(batches)
        }

        n_done = 0
        for fut in as_completed(futures):
            bi = futures[fut]
            n_done += 1
            try:
                fut.result()
            except Exception as e:
                print(f"  Batch {bi} ERROR: {e}")
                with lock:
                    stats["failed"] += len(batches[bi])

            elapsed = time.time() - t_start
            done_recs = min(n_done * bs, len(todo))
            pct = done_recs / len(todo) * 100
            rate = done_recs / elapsed if elapsed > 0 else 0
            eta = (len(todo) - done_recs) / rate / 60 if rate > 0 else 0

            # Update progress
            progress = {
                "status": "qa_running",
                "done": done_recs, "total": len(todo),
                "percent": round(pct, 1),
                "rate_per_s": round(rate, 2),
                "eta_min": round(eta, 1),
                "elapsed_min": round(elapsed / 60, 1),
                **stats,
            }
            with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
                json.dump(progress, f, ensure_ascii=False)

            if n_done % 10 == 0 or n_done == len(batches):
                print(
                    f"  [{done_recs:,}/{len(todo):,}] ({pct:.0f}%) "
                    f"kept={stats['kept']:,} down={stats['downgraded']:,} rej={stats['rejected']:,} "
                    f"| {rate:.1f} rec/s, ETA {eta:.1f} min"
                )

    # Separate kept vs rejected
    kept_records = []
    rejected_records = []
    todo_set = set(id(r) for r in todo)

    for rec in records:
        if id(rec) in todo_set:
            if rec.get("qa_verdict", {}).get("verdict") == "reject":
                rejected_records.append(rec)
            else:
                kept_records.append(rec)
        else:
            kept_records.append(rec)

    # Write back
    with open(SAVED_FILE, "w", encoding="utf-8") as f:
        for r in kept_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if rejected_records:
        with open(REJECTED_FILE, "a", encoding="utf-8") as f:
            for r in rejected_records:
                r["rejection_reason"] = "qa_rejected"
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Log
    with open(QA_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "input_count": len(todo),
            "stats": stats,
            "models": [QA_MODEL_1, QA_MODEL_2],
            "workers": args.workers,
            "batch_size": bs,
            "elapsed_min": round((time.time() - t_start) / 60, 1),
        }, ensure_ascii=False) + "\n")

    # Final progress
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "status": "qa_done", "done": len(todo), "total": len(todo),
            "percent": 100, **stats
        }, f, ensure_ascii=False)

    total_elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"QA DONE in {total_elapsed/60:.1f} min")
    print(f"  Kept:       {stats['kept']:,}")
    print(f"  Downgraded: {stats['downgraded']:,}")
    print(f"  Rejected:   {stats['rejected']:,}")
    print(f"  Failed:     {stats['failed']:,}")
    print(f"  saved_auto.jsonl: {len(kept_records):,} records")
    print(f"  rejected_auto.jsonl: +{len(rejected_records):,} records")
    print(f"  Speed: {len(todo)/total_elapsed:.1f} rec/s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
