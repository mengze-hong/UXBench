# experiments/results/

Aggregated evaluation results and leaderboard files.

## What These Files Are

Result files are produced by running `scripts/run_eval.py` for each model and task,
then aggregating into summary files used by the dashboard and the GitHub Pages leaderboard.

## File Descriptions

### `task1_leaderboard.md`

Human-readable Markdown table for **Task 1 — UX Judge** (binary Good/Bad classification).

| Column | Description |
|--------|-------------|
| Rank | Ordered by overall accuracy (descending) |
| Model | Model name |
| Overall Acc | `(bad_acc + good_acc) / 2` — balanced accuracy |
| Bad Acc | Recall on bad cases (ground truth = −1) |
| Good Acc | Recall on good cases (ground truth = +1) |
| F1 | Macro F1 score |
| Latency | Average response time in seconds |

### `task2_leaderboard.json`

Machine-readable JSON for **Task 2 — UX Eval** (multi-axis quality scoring).

Top-level structure:
```json
{
  "updated_at": "2026-06-01T00:00:00",
  "models": [
    {
      "model": "gpt-5",
      "mean_score": 4.12,
      "p50": 4.0,
      "good_pct": 0.73,
      "bad_pct": 0.18,
      "n": 4900,
      "avg_latency_s": 3.2
    }
  ]
}
```

| Field | Description |
|-------|-------------|
| `mean_score` | Mean quality score across all samples (1–5 scale) |
| `p50` | Median quality score |
| `good_pct` | Fraction of samples scored ≥ 4 |
| `bad_pct` | Fraction of samples scored ≤ 2 |
| `n` | Number of evaluated samples |
| `avg_latency_s` | Average call latency in seconds |

## Per-Model Raw Files

Per-model result files live in subdirectories:
```
results/
└── task1/
    └── <model_name>/
        ├── results.jsonl   # per-sample predictions
        └── summary.json    # aggregate metrics
```

Each line in `results.jsonl`:
```json
{"cid": "abc123", "verdict": -1, "expected": "bad", "correct": 1, "latency_s": 2.1}
```
