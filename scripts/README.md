# scripts/

Utility scripts for running UXBench evaluations.

## Scripts

| Script | Description |
|--------|-------------|
| `run_eval.py` | Main evaluation runner — runs a model against a UXBench task and writes results to `experiments/results/` |

## `run_eval.py` — Usage

```bash
# Basic usage: evaluate a model on Task 1 (UX Judge)
python scripts/run_eval.py --task task1_ux_judge --model gpt-5

# Specify number of parallel workers
python scripts/run_eval.py --task task1_ux_judge --model claude-opus-4.7 --workers 10

# Dry-run (first 10 samples only — for testing)
python scripts/run_eval.py --task task1_ux_judge --model gpt-5 --dry-run

# Use a custom config file
python scripts/run_eval.py \
    --task task1_ux_judge \
    --model gemini-2.5-pro \
    --config experiments/configs/eval_config.yaml

# Override the output directory
python scripts/run_eval.py \
    --task task1_ux_judge \
    --model gpt-5 \
    --output /tmp/my_results/
```

### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--task` | Yes | — | Task name: `task1_ux_judge`, `task2_ux_eval`, or `task3_ux_recovery` |
| `--model` | Yes | — | Model key, e.g. `gpt-5`, `claude-opus-4.7` (must be in `SUPPORTED_MODELS`) |
| `--config` | No | `experiments/configs/eval_config.yaml` | Path to evaluation config file |
| `--workers` | No | `5` | Number of parallel LLM call workers |
| `--output` | No | From config | Override output directory |
| `--dry-run` | No | `False` | Process only the first 10 samples per split |

### Output

Results are written to `experiments/results/<task>/<model>/`:
- `results.jsonl` — per-sample predictions
- `summary.json` — aggregate accuracy metrics

### Environment

Set your API credentials before running:

```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_API_BASE="https://api.openai.com/v1"   # optional
```

Or create a `.env` file in the repo root.
