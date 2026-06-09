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
| 0 | `pipeline/signals.py` | Signal extraction — reconstruct feedback turns from raw logs |
| 1 | `pipeline/prefilter.py` | Pre-filter — deduplication and quality thresholds |
| 2 | `pipeline/miner.py` | Miner Agent — extract failure/success reasons per turn |
| 3 | `pipeline/judge.py` | Judge Agent — 5-axis quality scoring |
| 4 | `pipeline/qa_full_scan.py` | QA Full Scan — remove duplicates and edge cases |
| 5 | `pipeline/build_golden_testset.py` | Golden Test Set — stratified sampling to produce final 7,400 test cases |

The main orchestrator is `pipeline/pipeline.py`.
