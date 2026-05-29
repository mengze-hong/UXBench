"""Recompute the Task 1 (UX Judge) leaderboard from frozen response files.

Run:
    python -m src.experiments.task1_ux_judge.compute_leaderboard

Reproduces `leaderboard.md` bit-exactly from `responses/*.jsonl`.

Metric definitions (matching the paper / leaderboard.md):
  * For each model file, count predictions where `verdict == ground_truth`
    over the 1,000 Good (gt=+1) and 1,000 Bad (gt=-1) instances.
  * Good% = correct_good / 1000   (recall on Good)
  * Bad%  = correct_bad  / 1000   (recall on Bad)
  * Avg%  = (Good% + Bad%) / 2
  * Verdicts column = #(verdict in {+1,-1}) / 2000
"""
from __future__ import annotations

from pathlib import Path

from src.experiments._common import display_name, load_jsonl, render_table


HERE = Path(__file__).resolve().parent
RESPONSES_DIR = HERE / "responses"
OUT_FILE = HERE / "leaderboard.md"

N_TOTAL = 2000
N_PER_CLASS = 1000


def score_file(path: Path) -> dict:
    rows = load_jsonl(path)
    n_good_correct = 0
    n_bad_correct  = 0
    n_valid_verdict = 0
    for r in rows:
        gt = r.get("ground_truth")
        v  = r.get("verdict")
        if v in (1, -1):
            n_valid_verdict += 1
        if gt == 1 and v == 1:
            n_good_correct += 1
        elif gt == -1 and v == -1:
            n_bad_correct += 1
    good_pct = n_good_correct / N_PER_CLASS * 100
    bad_pct  = n_bad_correct  / N_PER_CLASS * 100
    avg_pct  = (good_pct + bad_pct) / 2
    return {
        "model_file":   path.stem,
        "model":        display_name(path.stem),
        "good_pct":     good_pct,
        "bad_pct":      bad_pct,
        "avg_pct":      avg_pct,
        "verdicts":     f"{n_valid_verdict}/{N_TOTAL}",
    }


def main() -> None:
    files = sorted(RESPONSES_DIR.glob("*.jsonl"))
    rows = [score_file(p) for p in files]
    rows.sort(key=lambda x: x["avg_pct"], reverse=True)

    table = render_table(
        headers=("Rank", "Model", "Good%", "Bad%", "Avg%", "Verdicts"),
        aligns=("l", "l", "r", "r", "r", "l"),
        rows=[
            (i + 1, r["model"], f"{r['good_pct']:.1f}%", f"{r['bad_pct']:.1f}%",
             f"{r['avg_pct']:.1f}%", r["verdicts"])
            for i, r in enumerate(rows)
        ],
    )

    out = (
        "# UXBench Task 1 (UX Judge) — Leaderboard\n\n"
        f"> N = {N_TOTAL:,} ({N_PER_CLASS:,} Good + {N_PER_CLASS:,} Bad) "
        f"\u00b7 {len(rows)} paper-listed foundation models\n\n"
        "- **Good%** = recall on Good samples (model verdicts / 1000)\n"
        "- **Bad%** = recall on Bad samples (model verdicts / 1000)\n"
        "- **Avg%** = (Good% + Bad%) / 2 \u2014 higher is more balanced\n\n"
        + table
        + "\n"
    )

    print(out)
    print(f"\n[wrote] {OUT_FILE}")
    OUT_FILE.write_text(out, encoding="utf-8")


if __name__ == "__main__":
    main()
