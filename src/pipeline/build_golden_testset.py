"""
Golden Test Set Builder.

Reads pipeline output (saved_auto.jsonl) and applies a 5-stage filter pipeline to produce
a high-quality, diverse, solvable golden test set.

Stages:
  1. Quality Gate       — hard score thresholds
  2. Solvability Filter — exclude structurally-unsolvable failures
  3. Difficulty Filter  — exclude trivial / unannotatable cases
  4. Deduplication      — one best case per similar query per failure type
  5. Stratified Sample  — every failure_type covered, balanced to TARGET

Usage:
  python build_golden_testset.py
  python build_golden_testset.py --input outputs/saved_auto.jsonl --output outputs/golden_testset.jsonl --target 2000
  python build_golden_testset.py --dry-run
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# Paths (defaults relative to this script's location)
# Override with --input / --output at runtime
# ─────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
DEFAULT_INPUT  = SCRIPT_DIR.parent / "outputs" / "pipeline_saved_badcases.jsonl"
DEFAULT_OUTPUT = SCRIPT_DIR.parent / "outputs" / "golden_testset.jsonl"

TARGET = 2000
MIN_PER_FAILURE_TYPE = 5   # guaranteed floor per failure type
MAX_FAILURE_TYPE_SHARE = 0.40  # no single type > 40% of final set

# ─────────────────────────────────────────────────────────────
# Stage 1 — Quality Gate thresholds
# ─────────────────────────────────────────────────────────────
QUALITY_OVERALL = "high"          # overall_quality must be "high"
QUALITY_AVG_MIN = 4.0             # judge_average >= 4.0
QUALITY_DIM_MIN = 3               # every judge dimension score >= 3
SIGNAL_CONF_ALLOWED = {"high", "medium"}  # exclude "low" confidence signals

# ─────────────────────────────────────────────────────────────
# Stage 2 — Solvability: failure types NOT fixable by model optimization
# ─────────────────────────────────────────────────────────────
UNSOLVABLE_FAILURE_TYPES = {
    "系统/功能拒绝",
    "安全限制",
    "安全屏蔽",
    "default回复",
    "功能不支持",
    "系统默认回复",
    "系统错误",
    "服务不可用",
}

# Source query patterns indicating safety-flagged or structurally unsolvable content
UNSOLVABLE_QUERY_PATTERNS = re.compile(
    r"违法|违禁|制造炸|合成毒|毒品|自杀|自残|色情|裸体|性爱|"
    r"系统(默认|错误|故障)|服务(不可用|维护中)|无法(访问|连接)",
    re.IGNORECASE,
)

# Minimum source query length (chars) — too short = malformed / too vague
MIN_QUERY_LEN = 15

# Minimum explanation length — too short = weak annotation
MIN_EXPLANATION_LEN = 40

# ─────────────────────────────────────────────────────────────
# Stage 3 — Difficulty calibration
# ─────────────────────────────────────────────────────────────
# Exclude over-easy: one-off, non-generalizable failures
EXCLUDE_REPRESENTATIVENESS = {"low"}
# Exclude if user query is too vague for the case to be a useful test
QUERY_COMPLETENESS_MIN = 3    # judge_scores.query_completeness >= 3
# Require real model response (not a stub/system message)
RESPONSE_QUALITY_MIN = 3      # judge_scores.response_quality >= 3


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def _get_al(record: dict) -> dict:
    return record.get("auto_label") or {}


def _get_scores(al: dict) -> dict:
    return al.get("judge_scores") or {}


def _get_query_text(record: dict) -> str:
    """Extract query text from source_query (may be str or dict with 'message' key)."""
    sq = record.get("source_query")
    if sq is None:
        return ""
    if isinstance(sq, str):
        return sq.strip()
    if isinstance(sq, dict):
        return (sq.get("message") or "").strip()
    return str(sq).strip()


def _query_dedup_key(record: dict) -> str:
    """Dedup key: first 60 chars of source_query, normalised."""
    q = _get_query_text(record)
    # collapse whitespace + lowercase for loose matching
    q = re.sub(r"\s+", " ", q).lower()
    return q[:60]


# ─────────────────────────────────────────────────────────────
# Stage 1 — Quality Gate
# ─────────────────────────────────────────────────────────────
def passes_quality_gate(record: dict) -> bool:
    al = _get_al(record)
    scores = _get_scores(al)

    if al.get("overall_quality") != QUALITY_OVERALL:
        return False
    avg = al.get("judge_average") or 0
    if avg < QUALITY_AVG_MIN:
        return False
    if scores:
        if any(v < QUALITY_DIM_MIN for v in scores.values() if isinstance(v, (int, float))):
            return False
    sig_conf = al.get("signal_confidence", "")
    if sig_conf and sig_conf not in SIGNAL_CONF_ALLOWED:
        return False
    return True


# ─────────────────────────────────────────────────────────────
# Stage 2 — Solvability Filter
# ─────────────────────────────────────────────────────────────
def passes_solvability(record: dict) -> bool:
    al = _get_al(record)

    # Unsolvable failure type
    ft = (al.get("failure_type") or al.get("failure_dimension") or "").strip()
    if ft in UNSOLVABLE_FAILURE_TYPES:
        return False

    # Source query checks
    query = _get_query_text(record)
    if len(query) < MIN_QUERY_LEN:
        return False
    if UNSOLVABLE_QUERY_PATTERNS.search(query):
        return False

    # Weak annotation
    explanation = (al.get("explanation") or "").strip()
    if len(explanation) < MIN_EXPLANATION_LEN:
        return False

    return True


# ─────────────────────────────────────────────────────────────
# Stage 3 — Difficulty Calibration
# ─────────────────────────────────────────────────────────────
def passes_difficulty(record: dict) -> bool:
    al = _get_al(record)
    scores = _get_scores(al)

    # Exclude non-generalizable (too specific / one-off)
    rep = (al.get("representativeness") or "").lower()
    if rep in EXCLUDE_REPRESENTATIVENESS:
        return False

    # Exclude too-vague queries
    qc = scores.get("query_completeness")
    if qc is not None and isinstance(qc, (int, float)) and qc < QUERY_COMPLETENESS_MIN:
        return False

    # Require real response
    rq = scores.get("response_quality")
    if rq is not None and isinstance(rq, (int, float)) and rq < RESPONSE_QUALITY_MIN:
        return False

    return True


# ─────────────────────────────────────────────────────────────
# Stage 4 — Deduplication (within each failure type bucket)
# ─────────────────────────────────────────────────────────────
def deduplicate(records: list[dict]) -> list[dict]:
    """
    Within each failure_type × dedup_key group, keep only the record
    with the highest judge_average.
    """
    # Group: failure_type → dedup_key → list[record]
    buckets: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        al = _get_al(r)
        ft = (al.get("failure_type") or al.get("failure_dimension") or "unknown").strip()
        dk = _query_dedup_key(r)
        buckets[ft][dk].append(r)

    kept = []
    for ft, dk_map in buckets.items():
        for dk, group in dk_map.items():
            # Keep the record with the highest judge_average
            best = max(group, key=lambda x: (_get_al(x).get("judge_average") or 0))
            kept.append(best)

    return kept


# ─────────────────────────────────────────────────────────────
# Stage 5 — Stratified Sampling
# ─────────────────────────────────────────────────────────────
def stratified_sample(records: list[dict], target: int, min_per_type: int = MIN_PER_FAILURE_TYPE) -> list[dict]:
    """
    Allocate target slots across failure_types:
      - Every type gets at least MIN_PER_FAILURE_TYPE slots (or all it has if fewer)
      - Remaining budget distributed proportionally to type frequency
      - Single type capped at MAX_FAILURE_TYPE_SHARE × target
      - Within each type: sort by judge_average DESC, then take top N
      - Within each type: try to include ≥1 record per scenario (7 scenarios)
      - If total < target after stratified pass, fill from highest-scoring pool-wide remainders
    """
    # Group by failure_type
    by_type: dict[str, list] = defaultdict(list)
    for r in records:
        al = _get_al(r)
        ft = (al.get("failure_type") or al.get("failure_dimension") or "unknown").strip()
        by_type[ft].append(r)

    # Sort within each type by judge_average DESC
    for ft in by_type:
        by_type[ft].sort(key=lambda x: _get_al(x).get("judge_average") or 0, reverse=True)

    n_types = len(by_type)
    floor = min_per_type
    floor_total = sum(min(floor, len(v)) for v in by_type.values())
    remaining_budget = max(0, target - floor_total)

    # Proportional allocation of remaining budget
    total_records = sum(len(v) for v in by_type.values())
    alloc: dict[str, int] = {}
    for ft, recs in by_type.items():
        base = min(floor, len(recs))
        prop = int(remaining_budget * len(recs) / total_records) if total_records > 0 else 0
        alloc[ft] = base + prop

    # Cap any single type at MAX_FAILURE_TYPE_SHARE
    cap = int(target * MAX_FAILURE_TYPE_SHARE)
    for ft in alloc:
        alloc[ft] = min(alloc[ft], cap, len(by_type[ft]))

    # Within each type, ensure scenario diversity first
    SCENARIOS = {
        "产品与服务咨询", "信息与知识查询", "娱乐消遣",
        "私密与生活决策辅助", "创意内容与生成", "办公与效率", "情绪与心理支持",
    }

    selected: list[dict] = []
    used_cids: set[str] = set()
    per_type_remainders: dict[str, list] = {}

    for ft, recs in by_type.items():
        n = alloc[ft]
        if n <= 0:
            per_type_remainders[ft] = recs
            continue

        # Scenario-aware selection: pick one from each scenario first (if available)
        scenario_picks: list[dict] = []
        covered_scenarios: set[str] = set()
        fallback: list[dict] = []

        for r in recs:
            sc = (_get_al(r).get("scenario") or "").strip()
            if sc in SCENARIOS and sc not in covered_scenarios and len(scenario_picks) < len(SCENARIOS):
                scenario_picks.append(r)
                covered_scenarios.add(sc)
            else:
                fallback.append(r)

        # Combine: scenario picks first, then remaining top-scored
        candidate_pool = scenario_picks + fallback
        picked = []
        for r in candidate_pool:
            if len(picked) >= n:
                break
            cid = r.get("cid", "")
            if cid not in used_cids:
                picked.append(r)
                used_cids.add(cid)

        selected.extend(picked)
        # Store unpicked as remainders for gap-filling
        picked_cids = {r.get("cid") for r in picked}
        per_type_remainders[ft] = [r for r in recs if r.get("cid") not in picked_cids]

    # Gap fill: if total < target, pull from highest-scoring remainders pool-wide
    if len(selected) < target:
        all_remainders = []
        for recs in per_type_remainders.values():
            all_remainders.extend(recs)
        all_remainders.sort(key=lambda x: _get_al(x).get("judge_average") or 0, reverse=True)

        for r in all_remainders:
            if len(selected) >= target:
                break
            cid = r.get("cid", "")
            if cid not in used_cids:
                selected.append(r)
                used_cids.add(cid)

    # Trim to exact target if over
    selected = selected[:target]
    return selected


# ─────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────
def build_golden_testset(
    input_path: Path,
    output_path: Path,
    target: int = TARGET,
    dry_run: bool = False,
    min_per_type: int = MIN_PER_FAILURE_TYPE,
) -> None:
    print(f"Reading: {input_path}")
    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # ── Load ──────────────────────────────────────────────────
    records: list[dict] = []
    parse_errors = 0
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                parse_errors += 1
    n_total = len(records)
    print(f"Loaded {n_total:,} records ({parse_errors} parse errors ignored)")

    # ── Stage 1: Quality Gate ─────────────────────────────────
    after_q = [r for r in records if passes_quality_gate(r)]
    print(f"After quality gate:    {len(after_q):,}  (dropped {n_total - len(after_q):,})")

    # ── Stage 2: Solvability ──────────────────────────────────
    after_s = [r for r in after_q if passes_solvability(r)]
    print(f"After solvability:     {len(after_s):,}  (dropped {len(after_q) - len(after_s):,})")

    # ── Stage 3: Difficulty ───────────────────────────────────
    after_d = [r for r in after_s if passes_difficulty(r)]
    print(f"After difficulty:      {len(after_d):,}  (dropped {len(after_s) - len(after_d):,})")

    # ── Stage 4: Deduplication ────────────────────────────────
    after_dedup = deduplicate(after_d)
    print(f"After dedup:           {len(after_dedup):,}  (dropped {len(after_d) - len(after_dedup):,})")

    if len(after_dedup) < target:
        print(f"WARNING: only {len(after_dedup):,} candidates after filtering — target {target} may not be reached.")

    # ── Stage 5: Stratified Sample ────────────────────────────
    final = stratified_sample(after_dedup, target, min_per_type=min_per_type)
    print(f"Final selected:        {len(final):,}")

    # ── Coverage report ───────────────────────────────────────
    failure_types = defaultdict(int)
    scenarios = defaultdict(int)
    avg_scores = []
    for r in final:
        al = _get_al(r)
        ft = (al.get("failure_type") or al.get("failure_dimension") or "unknown").strip()
        failure_types[ft] += 1
        sc = (al.get("scenario") or "unknown").strip()
        scenarios[sc] += 1
        avg = al.get("judge_average") or 0
        avg_scores.append(avg)

    overall_avg = sum(avg_scores) / len(avg_scores) if avg_scores else 0

    print(f"\n{'='*60}")
    print("  Golden Test Set Summary")
    print(f"{'='*60}")
    print(f"  Total input:           {n_total:,}")
    print(f"  After quality gate:    {len(after_q):,}")
    print(f"  After solvability:     {len(after_s):,}")
    print(f"  After difficulty:      {len(after_d):,}")
    print(f"  After dedup:           {len(after_dedup):,}")
    print(f"  Final selected:        {len(final):,}")
    print(f"  Avg judge_average:     {overall_avg:.3f}")
    print(f"  Failure type coverage: {len(failure_types)} types")
    print(f"  Scenario coverage:     {len(scenarios)} scenarios")
    print()
    print("  Distribution by failure_type:")
    for ft, cnt in sorted(failure_types.items(), key=lambda x: -x[1]):
        pct = cnt / len(final) * 100 if final else 0
        print(f"    {ft[:35]:<35}  {cnt:4d} ({pct:5.1f}%)")
    print()
    print("  Distribution by scenario:")
    for sc, cnt in sorted(scenarios.items(), key=lambda x: -x[1]):
        pct = cnt / len(final) * 100 if final else 0
        print(f"    {sc[:35]:<35}  {cnt:4d} ({pct:5.1f}%)")
    print(f"{'='*60}")

    if dry_run:
        print("\n[DRY RUN] No output written.")
        return

    # ── Write output ──────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f_out:
        for rank, r in enumerate(final):
            al = _get_al(r)
            ft = (al.get("failure_type") or al.get("failure_dimension") or "unknown").strip()

            # Compute per-type rank
            ft_rank = sum(
                1 for prev in final[:rank]
                if (_get_al(prev).get("failure_type") or _get_al(prev).get("failure_dimension") or "unknown").strip() == ft
            ) + 1

            out = {
                "cid": r.get("cid"),
                "save_time": r.get("save_time"),
                "source_query": r.get("source_query"),
                "selected_history": r.get("selected_history"),
                "auto_label": al,
                "_golden_meta": {
                    "selection_rank": rank + 1,
                    "failure_type_bucket_rank": ft_rank,
                    "passed_stages": ["quality", "solvability", "difficulty", "dedup", "stratified"],
                },
            }
            f_out.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"\nSaved to: {output_path}")
    print(f"Records:  {len(final):,}")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Build a golden test set from saved bad case pipeline output."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input JSONL path")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSONL path")
    parser.add_argument("--target", type=int, default=TARGET, help=f"Target set size (default {TARGET})")
    parser.add_argument("--min-per-type", type=int, default=MIN_PER_FAILURE_TYPE,
                        help=f"Min cases per failure type (default {MIN_PER_FAILURE_TYPE})")
    parser.add_argument("--dry-run", action="store_true", help="Print stats only, don't write output")
    args = parser.parse_args()

    build_golden_testset(args.input, args.output,
                         target=args.target,
                         dry_run=args.dry_run,
                         min_per_type=args.min_per_type)


if __name__ == "__main__":
    main()
