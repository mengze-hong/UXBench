"""Recompute the Task 3 (UX Recovery) leaderboard from frozen GRM judge files.

Run:
    python -m src.experiments.task3_ux_recovery.compute_leaderboard

Reproduces `leaderboard.md` bit-exactly from `judge/judge_*.jsonl`.

Metric definitions (matching the paper / leaderboard.md):
  * Good% = #(verdict == +1, grm_ok == True) / N_TOTAL
  * Valid / Total = #(grm_ok == True) / N_TOTAL
"""
from __future__ import annotations

from pathlib import Path

from src.experiments._common import display_name, load_jsonl, render_table


HERE = Path(__file__).resolve().parent
JUDGE_DIR = HERE / "judge"
OUT_FILE = HERE / "leaderboard.md"

N_TOTAL = 500


def score_file(path: Path) -> dict:
    rows = load_jsonl(path)
    n_good = 0
    n_valid = 0
    for r in rows:
        if not r.get("grm_ok", False):
            continue
        n_valid += 1
        if r.get("verdict") == 1:
            n_good += 1

    stem = path.stem
    if stem.startswith("judge_"):
        model_stem = stem[len("judge_"):]
    else:
        model_stem = stem

    good_pct = n_good / N_TOTAL * 100
    return {
        "model_file":   model_stem,
        "model":        display_name(model_stem),
        "good_pct":     good_pct,
        "valid_total":  f"{n_valid}/{N_TOTAL}",
    }


def main() -> None:
    files = sorted(JUDGE_DIR.glob("judge_*.jsonl"))
    rows = [score_file(p) for p in files]
    rows.sort(key=lambda x: x["good_pct"], reverse=True)

    table = render_table(
        headers=("Rank", "Model", "Good%", "Valid / Total"),
        aligns=("l", "l", "r", "l"),
        rows=[
            (i + 1, r["model"], f"{r['good_pct']:.1f}%", r["valid_total"])
            for i, r in enumerate(rows)
        ],
    )

    out = (
        "# UXBench Task 3 (UX Recovery) — Leaderboard\n\n"
        f"> N = {N_TOTAL} \u00b7 {len(rows)} paper-listed foundation models "
        "\u00b7 judged by the trained pointwise GRM\n\n"
        + table
        + "\n"
    )

    print(out)
    print(f"\n[wrote] {OUT_FILE}")
    OUT_FILE.write_text(out, encoding="utf-8")


if __name__ == "__main__":
    main()
