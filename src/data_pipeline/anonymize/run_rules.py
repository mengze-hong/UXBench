#!/usr/bin/env python3
"""Layer 1: rule-based anonymization for JSONL datasets."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from anonymize.pii_rules import anonymize_record, summarize_changes


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rule-based anonymization")
    parser.add_argument("--input", required=True, help="Input JSONL")
    parser.add_argument("--output", required=True, help="Output JSONL")
    parser.add_argument("--changelog", default="", help="Optional changelog JSONL path")
    parser.add_argument("--report", default="", help="Optional report JSON path")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    changelog_path = Path(args.changelog) if args.changelog else out_path.with_suffix(".anonymize_changelog.jsonl")
    report_path = Path(args.report) if args.report else out_path.with_suffix(".anonymize_report.json")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    changelog_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    modified = 0
    pii_replacements = 0
    logs: list[dict] = []
    t0 = time.time()

    with out_path.open("w", encoding="utf-8") as out_f:
        for rec in iter_jsonl(in_path):
            total += 1
            anon, changes = anonymize_record(rec)
            out_f.write(json.dumps(anon, ensure_ascii=False) + "\n")
            if changes:
                modified += 1
                pii_replacements += len(changes)
                logs.append({"cid": rec.get("cid", ""), "n_changes": len(changes), "changes": changes})

    with changelog_path.open("w", encoding="utf-8") as log_f:
        for item in logs:
            log_f.write(json.dumps(item, ensure_ascii=False) + "\n")

    elapsed = time.time() - t0
    report = {
        "method": "rule_based",
        "timestamp": datetime.now().isoformat(),
        "input": str(in_path),
        "output": str(out_path),
        "total_records": total,
        "modified_records": modified,
        "total_pii_replacements": pii_replacements,
        "modification_rate": f"{(modified / total * 100) if total else 0:.2f}%",
        "elapsed_seconds": round(elapsed, 2),
        "pii_type_distribution": summarize_changes(logs),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
