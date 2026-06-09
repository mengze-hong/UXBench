"""
Generate benchmark_badcases.jsonl — a filtered, benchmark-ready subset of
pipeline_saved_badcases.jsonl.

Filter criteria:
  1. overall_quality in ('high', 'medium')      — annotation quality gate
  2. needs_context == False                      — query must be self-contained
  3. failure_dimension not in ('其他', None)     — must have a specific failure type
  4. is_duplicate != True                        — no duplicates
  5. needs_image != True                         — text-only (no image dependency)
  6. source_query.message length >= 5 chars      — non-garbage query
  7. judge_average >= 3.5                        — minimum labeling confidence
  8. dislike_turn present in full_history        — bad response must be recoverable

Output schema per record:
  cid, source_query, selected_history, agent_response_full, failure_dimension,
  scenario, overall_quality, judge_average, needs_context, severity_tier,
  explanation, agent_response_preview, signal_type, signal_confidence,
  dislike_turn_id, selected_turn_indices

  agent_response_full: the complete text of the failing AI response
    (from full_history[dislike_turn_id].message; replaces the truncated preview)

Usage:
  python generate_benchmark.py
  python generate_benchmark.py --min-judge-avg 4.0  # stricter quality
  python generate_benchmark.py --quality high        # only high quality
"""

import json, sys, io, argparse
from pathlib import Path
from collections import Counter

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

HERE = Path(__file__).parent
OUTPUTS = HERE.parent / "outputs"
SAVED_FILE = OUTPUTS / "pipeline_saved_badcases.jsonl"
BENCHMARK_FILE = OUTPUTS / "benchmark_badcases.jsonl"

EXCLUDE_DIMS = {"其他", None}


def extract_full_response(r: dict) -> str:
    """Extract the complete failing AI response from full_history by dislike_turn_id."""
    dislike_id = r.get("auto_label", {}).get("dislike_turn_id")
    if dislike_id is None:
        return ""
    full_hist = r.get("full_history", [])
    for turn in full_hist:
        if turn.get("turn_index") == dislike_id:
            return turn.get("message", "") or ""
    return ""


def build_benchmark_entry(r: dict) -> dict:
    """Extract benchmark-relevant fields from a full pipeline record."""
    al = r.get("auto_label", {})
    sq = r.get("source_query", {})
    qv = r.get("qa_verdict_strong", {})

    return {
        "cid": r.get("cid"),
        "save_time": r.get("save_time"),
        # Query
        "source_query": {
            "turn_index": sq.get("turn_index"),
            "message": sq.get("message", ""),
        },
        "selected_turn_indices": r.get("selected_turn_indices", []),
        "selected_history": r.get("selected_history", []),
        # The failing AI response (complete, not truncated)
        "agent_response_full": extract_full_response(r),
        # Failure characterization
        "failure_dimension": al.get("failure_dimension"),
        "failure_dimension_raw": al.get("failure_dimension_raw"),
        "scenario": al.get("scenario"),
        "severity_tier": r.get("severity_tier"),
        # Quality scores
        "overall_quality": al.get("overall_quality"),
        "judge_average": al.get("judge_average"),
        "judge_scores": al.get("judge_scores"),
        "judge_audit": al.get("judge_audit"),
        # Context flags
        "needs_context": al.get("needs_context"),
        "needs_image": r.get("needs_image", False),
        "is_duplicate": r.get("is_duplicate", False),
        # Annotation metadata
        "explanation": al.get("explanation"),
        "agent_response_preview": al.get("agent_response_preview"),
        "signal_type": al.get("signal_type"),
        "signal_confidence": al.get("signal_confidence"),
        "dislike_turn_id": al.get("dislike_turn_id"),
        "representativeness": al.get("representativeness"),
        "confidence": al.get("confidence"),
        # QA validation
        "qa_verdict": qv.get("verdict"),
        "qa_issues": qv.get("issues", []),
        "qa_notes": qv.get("notes", ""),
        # Reclassify metadata (if applicable)
        "failure_dimension_reclassified": al.get("failure_dimension_reclassified", False),
        "reclassify_confidence": al.get("reclassify_confidence"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-judge-avg", type=float, default=3.5,
                        help="Minimum judge_average score (default 3.5)")
    parser.add_argument("--quality", choices=["high", "medium", "both"], default="both",
                        help="Filter by overall_quality (default: both high+medium)")
    parser.add_argument("--min-query-len", type=int, default=5,
                        help="Minimum source_query message length (default 5)")
    parser.add_argument("--no-dedup", action="store_true",
                        help="Skip duplicate filter (include is_duplicate=True)")
    args = parser.parse_args()

    allowed_quality = {"high", "medium"}
    if args.quality == "high":
        allowed_quality = {"high"}
    elif args.quality == "medium":
        allowed_quality = {"medium"}

    # Load records
    records = []
    errors = 0
    with open(SAVED_FILE, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                errors += 1

    print(f"Loaded {len(records):,} records  ({errors} parse errors skipped)")

    # Apply filters
    stats = {
        "total": len(records),
        "fail_quality": 0,
        "fail_needs_context": 0,
        "fail_dimension": 0,
        "fail_duplicate": 0,
        "fail_needs_image": 0,
        "fail_query_len": 0,
        "fail_judge_avg": 0,
        "fail_no_response": 0,
        "fail_wrong_cutoff": 0,
        "passed": 0,
    }

    passed = []
    for r in records:
        al = r.get("auto_label", {})

        # 1. quality gate
        if al.get("overall_quality") not in allowed_quality:
            stats["fail_quality"] += 1
            continue

        # 2. context independence
        if al.get("needs_context") is not False:
            stats["fail_needs_context"] += 1
            continue

        # 3. specific failure dimension
        if al.get("failure_dimension") in EXCLUDE_DIMS:
            stats["fail_dimension"] += 1
            continue

        # 4. duplicate filter
        if not args.no_dedup and r.get("is_duplicate") is True:
            stats["fail_duplicate"] += 1
            continue

        # 5. image dependency
        if r.get("needs_image") is True:
            stats["fail_needs_image"] += 1
            continue

        # 6. query length
        msg = r.get("source_query", {}).get("message", "") or ""
        if len(msg) < args.min_query_len:
            stats["fail_query_len"] += 1
            continue

        # 7. judge score (also filter None — unannotated records are unsafe)
        avg = al.get("judge_average")
        if avg is None or avg < args.min_judge_avg:
            stats["fail_judge_avg"] += 1
            continue

        # 8. must have a recoverable full bad response
        if not extract_full_response(r):
            stats["fail_no_response"] += 1
            continue

        # 9. wrong-cutoff: selected_history must not extend to or past source_query turn
        hist = r.get("selected_history", [])
        if hist:
            sq_idx = r.get("source_query", {}).get("turn_index", 0)
            last_hist_turn = max(t.get("turn_index", -1) for t in hist)
            if last_hist_turn >= sq_idx:
                stats["fail_wrong_cutoff"] += 1
                continue

        passed.append(r)

    stats["passed"] = len(passed)

    print(f"\nFilter results:")
    print(f"  total:              {stats['total']:>8,}")
    print(f"  fail quality:       {stats['fail_quality']:>8,}")
    print(f"  fail needs_context: {stats['fail_needs_context']:>8,}")
    print(f"  fail dimension:     {stats['fail_dimension']:>8,}")
    print(f"  fail duplicate:     {stats['fail_duplicate']:>8,}")
    print(f"  fail needs_image:   {stats['fail_needs_image']:>8,}")
    print(f"  fail query_len:     {stats['fail_query_len']:>8,}")
    print(f"  fail judge_avg:     {stats['fail_judge_avg']:>8,}")
    print(f"  fail no_response:   {stats['fail_no_response']:>8,}")
    print(f"  fail wrong_cutoff:  {stats['fail_wrong_cutoff']:>8,}")
    print(f"  PASSED:             {stats['passed']:>8,}")

    # Dimension distribution in passed set
    dim_counter = Counter(r.get("auto_label", {}).get("failure_dimension") for r in passed)
    print(f"\nFailure dimension distribution ({len(passed):,} records):")
    for k, v in dim_counter.most_common():
        pct = v / len(passed) * 100
        print(f"  {k or 'None':20s}: {v:6,}  ({pct:.1f}%)")

    # Quality distribution
    qc = Counter(r.get("auto_label", {}).get("overall_quality") for r in passed)
    print(f"\nQuality distribution:")
    for k, v in qc.most_common():
        print(f"  {k or 'None':10s}: {v:6,}")

    # Write benchmark file
    with open(BENCHMARK_FILE, "w", encoding="utf-8") as f:
        for r in passed:
            entry = build_benchmark_entry(r)
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\nWritten: {BENCHMARK_FILE}")
    print(f"Records: {len(passed):,}")

    # Summary stats file
    import time
    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_file": SAVED_FILE.name,
        "output_file": BENCHMARK_FILE.name,
        "filter_params": {
            "min_judge_avg": args.min_judge_avg,
            "quality": args.quality,
            "min_query_len": args.min_query_len,
            "no_dedup": args.no_dedup,
        },
        "counts": stats,
        "dim_distribution": dict(dim_counter.most_common()),
        "quality_distribution": dict(qc.most_common()),
    }
    summary_file = OUTPUTS / "benchmark_generation_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Summary: {summary_file.name}")


if __name__ == "__main__":
    main()
