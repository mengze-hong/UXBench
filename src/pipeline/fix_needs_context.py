"""
Fix 1: Writeback needs_context to auto_label based on qa_verdict_strong.issues.

Logic:
  - If qa_verdict_strong.issues contains 'incomplete_context_dependency'
    OR issues contains 'needs_context' or 'needs_context_dependency'
    → auto_label.needs_context = True
  - Otherwise → auto_label.needs_context = False

Also fix: downgrade overall_quality for records where verdict==downgrade
  (already handled in qa_full_scan but let's double-check)

Usage:
  python fix_needs_context.py
"""

import json, sys, io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HERE = Path(__file__).parent
OUTPUTS = HERE.parent / "outputs"
SAVED_FILE = OUTPUTS / "pipeline_saved_badcases.jsonl"

# Issues that imply needs_context=True
NC_ISSUES = {
    "incomplete_context_dependency",
    "needs_context",
    "needs_context_dependency",
}


def main():
    if not SAVED_FILE.exists():
        print("No pipeline_saved_badcases.jsonl found")
        return

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

    n_nc_true = 0
    n_nc_false = 0
    n_downgrade_fixed = 0

    for rec in records:
        al = rec.setdefault("auto_label", {})
        qv = rec.get("qa_verdict_strong", {})
        issues = set(qv.get("issues") or [])

        # needs_context writeback
        nc = bool(issues & NC_ISSUES)
        al["needs_context"] = nc
        if nc:
            n_nc_true += 1
        else:
            n_nc_false += 1

        # Ensure downgrade verdict is reflected in overall_quality
        if qv.get("verdict") == "downgrade" and al.get("overall_quality") == "high":
            al["overall_quality"] = "medium"
            n_downgrade_fixed += 1

    print(f"  needs_context=True:  {n_nc_true:,}")
    print(f"  needs_context=False: {n_nc_false:,}")
    print(f"  downgrade quality fixes: {n_downgrade_fixed:,}")

    # Write back
    tmp = SAVED_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(SAVED_FILE)
    print(f"\nWritten: {SAVED_FILE.name}")


if __name__ == "__main__":
    main()
