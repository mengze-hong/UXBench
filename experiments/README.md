# experiments/

Evaluation configurations and result files for UXBench model benchmarking.

## Purpose

This folder organises everything needed to run and track model evaluations:
- **configs/** — YAML configuration files that define tasks, testset paths, metrics, and model lists
- **results/** — per-model output files and aggregated leaderboard summaries

## Structure

```
experiments/
├── configs/
│   └── eval_config.yaml   # main evaluation config (tasks, models, runtime settings)
└── results/
    ├── task1_leaderboard.md      # human-readable Task 1 leaderboard
    └── task2_leaderboard.json    # machine-readable Task 2 results
```

## Using eval_config.yaml

The config is consumed by `scripts/run_eval.py`:

```bash
# Run Task 1 for a specific model
python scripts/run_eval.py \
    --task task1_ux_judge \
    --model gpt-5 \
    --config experiments/configs/eval_config.yaml

# Dry-run (first 10 samples only)
python scripts/run_eval.py --task task1_ux_judge --model claude-opus-4.7 --dry-run
```

Key sections in `eval_config.yaml`:

| Section | Description |
|---------|-------------|
| `task1_ux_judge` | Binary Good/Bad classification task — 1 000 bad + 1 000 good cases |
| `task2_ux_eval` | Multi-axis quality evaluation — 4 900 samples |
| `task3_ux_recovery` | Conversational recovery evaluation — 500 samples |
| `models` | Registry of models to include in batch runs |
| `concurrency` | Number of parallel LLM workers |
| `max_retries` / `timeout` | Retry and timeout settings |

## What Results Files Contain

Each model run produces a directory under `results/task<N>/<model_name>/`:
- `results.jsonl` — per-sample predictions with `cid`, `verdict`, `expected`, `correct`, `latency_s`
- `summary.json` — aggregate metrics (`accuracy`, `bad_acc`, `good_acc`, `f1`, etc.)

The leaderboard files aggregate summaries across all models — see `results/README.md` for format details.

## Adding New Model Results

1. Run the evaluation script with `--model <your-model>` (the model must be listed in `SUPPORTED_MODELS` in `src/utils/config.py`)
2. Results land in `experiments/results/task1/<your-model>/`
3. Update `task1_leaderboard.md` / `task2_leaderboard.json` by running the aggregation script or editing manually
