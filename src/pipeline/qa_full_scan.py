"""
Full QA Scan: Re-validate ALL pipeline_saved_badcases.jsonl records with a strong model.

Strategy:
  - Uses claude-sonnet-4.5 (strong model) for primary validation
  - Async parallel processing with asyncio + aiohttp for maximum throughput
  - Batch size 20 records per LLM call
  - 15 concurrent workers
  - Saves qa_verdict_strong field (preserves old qa_verdict)
  - Rejected items moved to pipeline_rejected_cases.jsonl

Usage:
  python qa_full_scan.py --workers 15 --batch-size 20
"""

import json, sys, io, time, argparse, asyncio, re, os
from pathlib import Path
from datetime import datetime
from constants import DislikeConstants, SharedConstants

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

HERE = Path(__file__).parent
PROMPT_DIR = HERE.parent / SharedConstants.PROMPTS_DIR
OUTPUTS = HERE.parent / SharedConstants.OUTPUTS_DIR
LOG_DIR = HERE.parent / SharedConstants.LOGS_DIR
SAVED_FILE = OUTPUTS / DislikeConstants.LEGACY_SAVED_FILENAME
REJECTED_FILE = OUTPUTS / "pipeline_rejected_cases.jsonl"
QA_LOG_FILE = LOG_DIR / "qa_log.jsonl"
PROGRESS_FILE = LOG_DIR / SharedConstants.PROGRESS_FILENAME

# Strong model for full validation
QA_MODEL = "claude-sonnet-4.5"

API_URL = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
API_KEY = os.environ.get("OPENAI_API_KEY", "")

QA_SYSTEM_PROMPT = (PROMPT_DIR / "qa_system.txt").read_text(encoding="utf-8")


def _summarize_for_qa(rec: dict) -> dict:
    """Extract key fields for QA review, including prior turn for context."""
    al = rec.get("auto_label", {})
    sq = rec.get("source_query", {})
    history = rec.get("full_history", [])

    dt_id = al.get("dislike_turn_id")
    sq_turn_index = sq.get("turn_index", 0)

    agent_resp = ""
    prior_turn = ""

    for m in history:
        if not isinstance(m, dict):
            continue
        tidx = m.get("turn_index", -1)
        role = m.get("role", "")
        msg = (m.get("message", "") or "")

        # The disliked AI response
        if tidx == dt_id and role == "assistant":
            agent_resp = msg[:500]

        # The AI response immediately before the source_query (prior context)
        if tidx == sq_turn_index - 1 and role == "assistant":
            prior_turn = msg[:300]

    context_rounds = sq_turn_index // 2  # number of complete rounds before this turn

    return {
        "source_query": (sq.get("message", "") or "")[:300],
        "prior_turn": prior_turn,  # AI response before this query (context)
        "agent_response_preview": agent_resp,
        "failure_dimension": al.get("failure_dimension", ""),
        "explanation": (al.get("explanation", "") or "")[:300],
        "signal_type": al.get("signal_type", ""),
        "signal_confidence": al.get("signal_confidence", ""),
        "judge_average": al.get("judge_average", 0),
        "context_rounds": context_rounds,
    }


def parse_json_from_text(text: str):
    """Robust JSON extraction from LLM output."""
    if not text:
        return None
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try code block
    m = re.search(r'```(?:json)?\s*\n?([\s\S]*?)\n?```', text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Try finding array
    m = re.search(r'(\[[\s\S]*\])', text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    return None


async def call_llm_async(session, messages, model, max_tokens=4000, temperature=0.1, retries=2):
    """Async LLM call with retry."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    for attempt in range(retries + 1):
        try:
            async with session.post(API_URL, json=payload, headers=headers, timeout=120) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    tokens = data.get("usage", {}).get("total_tokens", 0)
                    return {"ok": True, "content": content, "tokens": tokens, "model": model}
                elif resp.status == 429:
                    await asyncio.sleep(2 ** attempt)
                    continue
                else:
                    text = await resp.text()
                    if attempt < retries:
                        await asyncio.sleep(1)
                        continue
                    return {"ok": False, "content": "", "tokens": 0, "model": model, "error": f"HTTP {resp.status}: {text[:200]}"}
        except asyncio.TimeoutError:
            if attempt < retries:
                await asyncio.sleep(2)
                continue
            return {"ok": False, "content": "", "tokens": 0, "model": model, "error": "timeout"}
        except Exception as e:
            if attempt < retries:
                await asyncio.sleep(1)
                continue
            return {"ok": False, "content": "", "tokens": 0, "model": model, "error": str(e)[:200]}

    return {"ok": False, "content": "", "tokens": 0, "model": model, "error": "max_retries"}


async def qa_batch_async(session, batch: list, batch_idx: int, semaphore, stats, progress_state):
    """Process a single batch of records through QA."""
    async with semaphore:
        summaries = [{"index": i, **_summarize_for_qa(rec)} for i, rec in enumerate(batch)]

        user_prompt = (
            f"# 待审核 Bad Cases（{len(summaries)} 条）\n"
            f"```json\n{json.dumps(summaries, ensure_ascii=False, indent=1)}\n```\n\n"
            f"请对每条进行质量验证，输出 JSON 数组。"
        )

        result = await call_llm_async(
            session,
            messages=[
                {"role": "system", "content": QA_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            model=QA_MODEL,
            max_tokens=2000 + 250 * len(batch),
            temperature=0.1,
        )

        if not result["ok"]:
            # LLM failed — mark all as keep with fallback note
            for rec in batch:
                rec["qa_verdict_strong"] = {
                    "verdict": "keep",
                    "quality": "medium",
                    "notes": f"qa_llm_failed: {result.get('error', '?')[:100]}",
                    "model": QA_MODEL,
                }
            stats["failed"] += len(batch)
            progress_state["done"] += len(batch)
            return

        parsed = parse_json_from_text(result["content"])

        # Map results by index
        result_map = {}
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict):
                    idx = item.get("index", -1)
                    if 0 <= idx < len(batch):
                        result_map[idx] = item
        elif isinstance(parsed, dict) and len(batch) == 1:
            result_map[0] = parsed

        for i, rec in enumerate(batch):
            v = result_map.get(i)
            if v is None:
                rec["qa_verdict_strong"] = {
                    "verdict": "keep",
                    "quality": "medium",
                    "notes": "qa_parse_miss",
                    "model": QA_MODEL,
                }
                stats["kept"] += 1
            else:
                verdict = v.get("verdict", "keep")
                rec["qa_verdict_strong"] = {
                    "verdict": verdict,
                    "quality": v.get("quality", "medium"),
                    "issues": v.get("issues", []),
                    "notes": v.get("notes", ""),
                    "corrected_dimension": v.get("corrected_dimension"),
                    "model": QA_MODEL,
                }

                if verdict == "reject":
                    stats["rejected"] += 1
                elif verdict == "downgrade":
                    rec["auto_label"]["overall_quality"] = "medium"
                    stats["downgraded"] += 1
                else:
                    stats["kept"] += 1

                # Apply dimension correction
                if v.get("corrected_dimension"):
                    rec["auto_label"]["failure_dimension"] = v["corrected_dimension"]

        stats["tokens"] += result.get("tokens", 0)
        progress_state["done"] += len(batch)


async def run_qa(args):
    """Main async QA runner."""
    import aiohttp

    if not SAVED_FILE.exists():
        print("No pipeline_saved_badcases.jsonl found")
        return

    # Load all records
    records = []
    with open(SAVED_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    total = len(records)
    print(f"Loaded {total:,} records for FULL QA scan with {QA_MODEL}")
    print(f"Workers: {args.workers}, Batch size: {args.batch_size}")

    # Backup first
    backup_name = f"pipeline_saved_pre_fullqa_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    backup_path = OUTPUTS / backup_name
    with open(backup_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Backup: {backup_path.name}")

    # Create batches
    bs = args.batch_size
    batches = [records[i:i+bs] for i in range(0, total, bs)]
    print(f"Processing {len(batches):,} batches")

    stats = {"kept": 0, "downgraded": 0, "rejected": 0, "failed": 0, "tokens": 0}
    progress_state = {"done": 0}
    t_start = time.time()

    semaphore = asyncio.Semaphore(args.workers)

    # Write initial progress
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump({"status": "qa_full_scan_running", "done": 0, "total": total, "percent": 0, "model": QA_MODEL}, f)

    connector = aiohttp.TCPConnector(limit=args.workers + 5, limit_per_host=args.workers + 5)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        for bi, batch in enumerate(batches):
            task = asyncio.create_task(qa_batch_async(session, batch, bi, semaphore, stats, progress_state))
            tasks.append(task)

        # Monitor progress
        last_print = 0
        while True:
            done_count = progress_state["done"]
            pct = done_count / total * 100 if total > 0 else 0
            elapsed = time.time() - t_start
            rate = done_count / elapsed if elapsed > 0 else 0
            eta = (total - done_count) / rate / 60 if rate > 0 else 0

            # Update progress file
            with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
                json.dump({
                    "status": "qa_full_scan_running",
                    "done": done_count, "total": total,
                    "percent": round(pct, 1),
                    "rate_per_s": round(rate, 2),
                    "eta_min": round(eta, 1),
                    "elapsed_min": round(elapsed / 60, 1),
                    "model": QA_MODEL,
                    **{k: v for k, v in stats.items()},
                }, f, ensure_ascii=False)

            if done_count - last_print >= 500 or done_count == total:
                print(
                    f"  [{done_count:,}/{total:,}] ({pct:.1f}%) "
                    f"kept={stats['kept']:,} down={stats['downgraded']:,} rej={stats['rejected']:,} fail={stats['failed']:,} "
                    f"| {rate:.1f} rec/s, ETA {eta:.1f} min"
                )
                last_print = done_count

            if done_count >= total:
                break

            # Check if all tasks done
            all_done = all(t.done() for t in tasks)
            if all_done:
                break

            await asyncio.sleep(2)

        # Ensure all tasks complete
        await asyncio.gather(*tasks, return_exceptions=True)

    # Final stats
    done_count = progress_state["done"]
    elapsed = time.time() - t_start

    # Separate kept vs rejected
    kept_records = []
    rejected_records = []
    for rec in records:
        qv = rec.get("qa_verdict_strong", {})
        if qv.get("verdict") == "reject":
            rejected_records.append(rec)
        else:
            kept_records.append(rec)

    # Write back
    with open(SAVED_FILE, "w", encoding="utf-8") as f:
        for r in kept_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if rejected_records:
        with open(REJECTED_FILE, "a", encoding="utf-8") as f:
            for r in rejected_records:
                r["rejection_reason"] = f"qa_full_scan_rejected_by_{QA_MODEL}"
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Log
    LOG_DIR.mkdir(exist_ok=True)
    with open(QA_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "type": "full_scan",
            "input_count": total,
            "model": QA_MODEL,
            "stats": stats,
            "workers": args.workers,
            "batch_size": args.batch_size,
            "elapsed_min": round(elapsed / 60, 1),
            "speed_rec_per_s": round(total / elapsed, 1) if elapsed > 0 else 0,
        }, ensure_ascii=False) + "\n")

    # Final progress
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "status": "qa_full_scan_done",
            "done": total, "total": total, "percent": 100,
            "model": QA_MODEL,
            "elapsed_min": round(elapsed / 60, 1),
            **stats,
        }, f, ensure_ascii=False)

    print(f"\n{'='*70}")
    print(f"FULL QA SCAN DONE in {elapsed/60:.1f} min ({QA_MODEL})")
    print(f"  Total scanned:  {total:,}")
    print(f"  Kept:           {stats['kept']:,}")
    print(f"  Downgraded:     {stats['downgraded']:,}")
    print(f"  Rejected:       {stats['rejected']:,}")
    print(f"  LLM failures:   {stats['failed']:,}")
    print(f"  Total tokens:   {stats['tokens']:,}")
    print(f"  Speed:          {total/elapsed:.1f} rec/s")
    print(f"  pipeline_saved_badcases.jsonl: {len(kept_records):,} records")
    print(f"  pipeline_rejected_cases.jsonl: +{len(rejected_records):,} records")
    print(f"  Backup: {backup_path.name}")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="Full QA scan with strong model")
    parser.add_argument("--workers", type=int, default=15, help="Concurrent async workers")
    parser.add_argument("--batch-size", type=int, default=20, help="Records per LLM call")
    args = parser.parse_args()

    asyncio.run(run_qa(args))


if __name__ == "__main__":
    main()
