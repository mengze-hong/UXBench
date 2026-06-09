"""
Post-hoc quality enhancement for saved_auto.jsonl.
Applies P0-P5 fixes WITHOUT modifying the pipeline or any running processes.

Reads saved_auto.jsonl → applies fixes → writes back (preserving all original fields).

P0: Tag image-dependent queries (needs_image=true)
P1: Normalize failure_dimension (preserve raw in failure_dimension_raw)
P2: Add needs_context flag (from judge query_completeness score)
P3: Add severity_tier (from judge severity score)
P4: Deduplicate near-identical source queries within same cid
P5: Fix source > dislike ordering (15 cases)

Safety: backs up the file first, validates record count after.
"""

import json, sys, io, re, shutil
from pathlib import Path
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HERE = Path(__file__).parent
SAVED_FILE = HERE.parent / "outputs" / "saved_auto.jsonl"
BACKUP_FILE = HERE.parent / "outputs" / f"saved_auto_backup_{datetime.now().strftime('%H%M%S')}.jsonl"

# ═══════════════════════════════════════════════════════
# P0: Image-dependency detection
# ═══════════════════════════════════════════════════════
IMAGE_KEYWORDS = [
    "图中", "图片", "照片", "这张图", "这个图", "上传了图片",
    "解答图中", "看图", "图上", "图里", "截图", "图一", "图二",
    "图片1", "图片2", "帮我看看这", "识别图", "这张",
]
_IMAGE_RE = re.compile("|".join(re.escape(k) for k in IMAGE_KEYWORDS))

def detect_image_dependency(rec: dict) -> bool:
    sq = (rec.get("source_query", {}).get("message", "") or "")
    if _IMAGE_RE.search(sq):
        return True
    # Also check if history has image-upload patterns
    for m in (rec.get("full_history", []) or [])[:5]:
        if isinstance(m, dict):
            msg = (m.get("message", "") or "")
            if "用户上传了图片" in msg or "[图片]" in msg:
                return True
    return False


# ═══════════════════════════════════════════════════════
# P1: Dimension normalization (preserves raw)
# Now uses the shared dim_normalize module as single source of truth
# ═══════════════════════════════════════════════════════
from dim_normalize import normalize_dimension as normalize_dim, get_canonical_dims, CANONICAL_DIMS


# ═══════════════════════════════════════════════════════
# P2: needs_context flag
# ═══════════════════════════════════════════════════════
def compute_needs_context(rec: dict) -> bool:
    scores = rec.get("auto_label", {}).get("judge_scores") or {}
    qc = scores.get("query_completeness", 5) or 5
    return qc <= 2


# ═══════════════════════════════════════════════════════
# P3: severity_tier
# ═══════════════════════════════════════════════════════
def compute_severity_tier(rec: dict) -> str:
    scores = rec.get("auto_label", {}).get("judge_scores") or {}
    sev = scores.get("severity", 3) or 3
    if sev >= 4:
        return "critical"
    elif sev >= 3:
        return "moderate"
    else:
        return "mild"


# ═══════════════════════════════════════════════════════
# P5: Fix source > dislike ordering
# ═══════════════════════════════════════════════════════
def fix_ordering(rec: dict) -> bool:
    """Fix cases where source_query turn > dislike turn. Returns True if fixed."""
    al = rec.get("auto_label", {})
    sq = rec.get("source_query", {})
    sq_tid = sq.get("turn_index")
    dt_id = al.get("dislike_turn_id")

    if sq_tid is None or dt_id is None:
        return False
    try:
        sq_tid = int(sq_tid)
        dt_id = int(dt_id)
    except (ValueError, TypeError):
        return False
    if sq_tid <= dt_id:
        return False

    # Find the user turn right before dislike turn
    history = rec.get("full_history", [])
    new_source = None
    for m in history:
        if not isinstance(m, dict):
            continue
        tid = m.get("turn_index", -1)
        try:
            tid = int(tid)
        except (ValueError, TypeError):
            continue
        if tid >= dt_id:
            break
        if m.get("role") == "user":
            new_source = {"turn_index": tid, "message": (m.get("message", "") or "")}

    if new_source:
        rec["source_query_original"] = dict(sq)  # preserve original
        rec["source_query"] = new_source
        return True
    return False


# ═══════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════
def main():
    if not SAVED_FILE.exists():
        print("No saved_auto.jsonl found")
        return

    # Load
    records = []
    with open(SAVED_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    original_count = len(records)
    print(f"Loaded {original_count} records")

    # Backup
    shutil.copy2(SAVED_FILE, BACKUP_FILE)
    print(f"Backup: {BACKUP_FILE}")

    stats = {
        "p0_image_tagged": 0,
        "p1_dim_normalized": 0,
        "p2_needs_context": 0,
        "p3_severity_critical": 0,
        "p3_severity_moderate": 0,
        "p3_severity_mild": 0,
        "p4_deduped": 0,
        "p5_ordering_fixed": 0,
    }

    # ── P0: Image dependency ──
    for r in records:
        needs_img = detect_image_dependency(r)
        r["needs_image"] = needs_img
        if needs_img:
            stats["p0_image_tagged"] += 1

    # ── P1: Dimension normalization (PRESERVE raw) ──
    # IMPORTANT: Always re-normalize from the RAW dimension, not from the
    # previously-normalized value. This ensures updated rules apply properly.
    for r in records:
        al = r.get("auto_label", {})
        # Get the original raw dimension (prefer failure_dimension_raw if exists)
        raw = al.get("failure_dimension_raw") or al.get("failure_dimension", "?")
        normalized = normalize_dim(raw)
        # Always store raw
        al["failure_dimension_raw"] = raw
        if al.get("failure_dimension") != normalized:
            al["failure_dimension"] = normalized
            stats["p1_dim_normalized"] += 1

    # ── P2: needs_context flag ──
    for r in records:
        nc = compute_needs_context(r)
        r["needs_context"] = nc
        if nc:
            stats["p2_needs_context"] += 1

    # ── P3: severity_tier ──
    for r in records:
        tier = compute_severity_tier(r)
        r["severity_tier"] = tier
        stats[f"p3_severity_{tier}"] += 1

    # ── P4: Deduplicate within same cid ──
    # Group by cid
    by_cid = defaultdict(list)
    for i, r in enumerate(records):
        by_cid[r.get("cid", "")].append((i, r))

    # Mark duplicates instead of deleting — tag is_duplicate + duplicate_of
    # This preserves ALL records for later reject pattern analysis
    to_mark_dup = {}  # idx -> kept_idx (the one we keep)
    for cid, group in by_cid.items():
        if len(group) <= 1:
            continue
        marked = set()
        for a in range(len(group)):
            if group[a][0] in marked:
                continue
            for b in range(a + 1, len(group)):
                if group[b][0] in marked:
                    continue
                sq_a = (group[a][1].get("source_query", {}).get("message", "") or "")[:100]
                sq_b = (group[b][1].get("source_query", {}).get("message", "") or "")[:100]
                dt_a = group[a][1].get("auto_label", {}).get("dislike_turn_id")
                dt_b = group[b][1].get("auto_label", {}).get("dislike_turn_id")
                st_a = group[a][1].get("source_query", {}).get("turn_index")
                st_b = group[b][1].get("source_query", {}).get("turn_index")
                # Only dedup if same failure turn + same source turn + similar query text
                # Different turn = different bad case, even if query text is identical
                if (dt_a == dt_b and st_a == st_b
                        and sq_a and sq_b
                        and SequenceMatcher(None, sq_a, sq_b).ratio() > 0.90):
                    ja = group[a][1].get("auto_label", {}).get("judge_average") or 0
                    jb = group[b][1].get("auto_label", {}).get("judge_average") or 0
                    if ja >= jb:
                        to_mark_dup[group[b][0]] = group[a][0]
                        marked.add(group[b][0])
                    else:
                        to_mark_dup[group[a][0]] = group[b][0]
                        marked.add(group[a][0])

    # Apply duplicate flags (NOT deleting — just marking)
    for i, r in enumerate(records):
        if i in to_mark_dup:
            r["is_duplicate"] = True
            r["duplicate_of_index"] = to_mark_dup[i]
        else:
            r["is_duplicate"] = False

    stats["p4_deduped"] = len(to_mark_dup)

    # ── P5: Fix ordering ──
    for r in records:
        if fix_ordering(r):
            stats["p5_ordering_fixed"] += 1

    # Write back ALL records (nothing deleted)
    with open(SAVED_FILE, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    final_count = len(records)
    active_count = sum(1 for r in records if not r.get("is_duplicate"))

    # Report
    print(f"\n{'='*60}")
    print(f"✅ Quality Enhancement Complete")
    print(f"{'='*60}")
    print(f"  Total records: {final_count} (ALL preserved)")
    print(f"  Active (non-duplicate): {active_count}")
    print(f"  Marked as duplicate: {len(to_mark_dup)}")
    print()
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Dimension distribution (active records only)
    active_records = [r for r in records if not r.get("is_duplicate")]
    dims = Counter(r.get("auto_label", {}).get("failure_dimension", "?") for r in active_records)
    print(f"\n  Dimensions ({len(dims)} categories, active only):")
    for d, n in dims.most_common():
        print(f"    {d}: {n}")

    # Severity distribution
    print(f"\n  Severity tiers (active only):")
    tiers = Counter(r.get("severity_tier", "?") for r in active_records)
    for t, n in tiers.most_common():
        print(f"    {t}: {n} ({n/active_count*100:.1f}%)")

    print(f"\n  Image-dependent: {stats['p0_image_tagged']} ({stats['p0_image_tagged']/final_count*100:.1f}%)")
    print(f"  Needs context: {stats['p2_needs_context']} ({stats['p2_needs_context']/final_count*100:.1f}%)")
    print(f"{'='*60}")

    # Write stats_cache.json for dashboard (avoids 1.26GB scan each load)
    _write_stats_cache(active_records, active_count)
    print("  📊 stats_cache.json written for dashboard")


def _write_stats_cache(records, count):
    """Write a lightweight stats cache for dashboard consumption."""
    from collections import Counter
    dims = Counter()
    signals = Counter()
    qualities = Counter()
    sentiments = Counter()
    severities = Counter()
    n_img = n_ctx = n_qa = n_qa_keep = n_has_raw = 0
    judge_sum = conf_sum = 0.0

    for r in records:
        al = r.get("auto_label", {})
        dims[al.get("failure_dimension", "?")] += 1
        signals[al.get("signal_type", "?")] += 1
        qualities[al.get("overall_quality", "?")] += 1
        sentiments[al.get("sentiment", "?")] += 1
        severities[r.get("severity_tier", "?")] += 1
        if r.get("needs_image"): n_img += 1
        if r.get("needs_context"): n_ctx += 1
        if r.get("qa_verdict"):
            n_qa += 1
            if r.get("qa_verdict", {}).get("verdict") == "keep":
                n_qa_keep += 1
        if al.get("failure_dimension_raw"): n_has_raw += 1
        judge_sum += (al.get("judge_average") or 0)
        conf_sum += float(al.get("confidence") or 0)

    cache = {
        "count": count,
        "dims": dict(dims), "signals": dict(signals),
        "qualities": dict(qualities), "sentiments": dict(sentiments),
        "severities": dict(severities),
        "n_img": n_img, "n_ctx": n_ctx, "n_qa": n_qa,
        "n_qa_keep": n_qa_keep, "n_has_raw": n_has_raw,
        "judge_sum": judge_sum, "conf_sum": conf_sum,
    }
    cache_path = HERE.parent / "outputs" / "stats_cache.json"
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
