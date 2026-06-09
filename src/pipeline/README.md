# src/pipeline/

The 6-stage auto-labeling pipeline that converts raw AI assistant interaction logs into
the UXBench golden test set.

## Purpose

Raw conversation logs are noisy and unannotated. This pipeline applies a series of
automated filtering, LLM-based annotation, and quality-control steps to produce
high-quality, labeled benchmark test cases suitable for evaluating AI assistants.

## Data Format

All inter-stage data is stored as **JSONL** (one JSON object per line, UTF-8 encoded).
Each record is keyed by a `cid` (conversation ID) field and carries conversation fields
(`source_query`, `selected_history`, `agent_response_full`) plus annotations added by
each stage (`failure_dimension`, `judge_scores`, `difficulty`, etc.).

## Pipeline Scripts

| File | Stage | Description |
|------|-------|-------------|
| `pipeline.py` | Orchestrator | Main entry point. Runs all stages sequentially using `ThreadPoolExecutor` for parallel LLM calls. |
| `signals.py` | Stage 0 | **Signal extraction** — parses raw interaction logs and reconstructs feedback turns (dislike signals, explicit complaints). |
| `prefilter.py` | Stage 1 | **Pre-filter** — deduplication by conversation hash and basic quality thresholds (length, language, completeness). |
| `miner.py` | Stage 2 | **Miner Agent** — LLM-powered extraction of failure/success reasons per turn. Outputs `failure_dimension` and `miner_explanation` fields. |
| `judge.py` | Stage 3 | **Judge Agent** — LLM-based 5-axis quality scoring (`query_completeness`, `signal_credibility`, `representativeness`, `severity`, `annotation_clarity`). |
| `qa_full_scan.py` | Stage 4 | **QA Full Scan** — removes near-duplicate cases, edge cases that require missing context/images, and low-quality annotations. |
| `build_golden_testset.py` | Stage 5 | **Golden Test Set** — stratified sampling across failure dimensions and difficulty levels to produce the final 7,400 test cases. |
| `postprocess.py` | Post | Post-processing utilities applied after the main pipeline (field cleanup, normalization). |
| `dim_normalize.py` | Post | Normalizes free-text failure dimension labels to a canonical taxonomy. |
| `quality_enhance.py` | Post | Data quality enhancement: enriches records with additional metadata and quality signals. |
| `critical_filter.py` | Post | Filters out critical edge cases that would make evaluation ambiguous. |
| `reclassify_other_dim.py` | Post | Reclassifies records left in the "Other" dimension bucket into more specific categories. |
| `fix_label_normalization.py` | Post | One-off fix for label normalization inconsistencies found during QA. |
| `fix_needs_context.py` | Post | Identifies and marks records that require external context unavailable to an evaluator. |
| `generate_benchmark.py` | Final | Generates the final benchmark JSONL files from the processed golden set. |
| `qa_agent.py` | Agent | QA agent implementation used by Stage 4 full scan. |
| `trace_runner.py` | Runner | Trace-based runner that replays recorded pipeline traces for debugging. |
| `llm_client.py` | Utility | Pipeline-local LLM client wrapper (thin shim over `utils/llm_client.py`). |

## Running the Pipeline End-to-End

```bash
# From the repo root
python src/pipeline/pipeline.py \
    --input  data/raw/interactions.jsonl \
    --output data/processed/ \
    --workers 10
```

Each stage reads from the previous stage's output directory and writes to the next.
To resume from a checkpoint, re-run the same command — completed records are skipped
automatically via `utils/checkpoint.py`.

## Dependencies

- Python 3.10+
- `requests` for LLM API calls
- `utils.llm_client` / `utils.config` for routing and credentials
- Set `OPENAI_API_KEY` and `OPENAI_API_BASE` in your environment or `.env` file
