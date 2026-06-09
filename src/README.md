# src/

Source code for the UXBench data pipeline and evaluation utilities.

## Structure

| Directory | Description |
|-----------|-------------|
| `pipeline/` | Data construction pipeline that converts raw interaction logs into benchmark test cases |
| `utils/`    | Shared utilities: LLM client, config, checkpointing, data loading, eval prompt templates |

## Pipeline Stages

| Stage | Script | Description |
|-------|--------|-------------|
| 1 | `pipeline/signals.py` | Signal extraction — reconstruct feedback turns from raw logs |
| 2 | `pipeline/prefilter.py` | Pre-filter — deduplication and quality thresholds |
| 3 | `pipeline/miner.py` | Miner Agent — extract failure/success reasons per turn |
| 4 | `pipeline/judge.py` | Judge Agent — 5-axis quality scoring |
| 5 | `pipeline/qa_full_scan.py` | QA Full Scan — remove duplicates and edge cases |

The main orchestrator is `pipeline/pipeline.py`.
