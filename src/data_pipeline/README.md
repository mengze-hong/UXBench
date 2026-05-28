# Data Pipeline

Code for constructing the UXBench dataset from raw interaction logs. This directory contains the full multi-stage pipeline but **no raw data** (for privacy).

## Directory Structure

```
data_pipeline/
├── lib/                  # Shared utilities
│   ├── config.py         # API endpoint configuration (set your own keys)
│   ├── llm_client.py     # Unified LLM client with retry & routing
│   ├── data_loader.py    # JSONL loading utilities
│   ├── checkpoint.py     # Checkpoint/resume for long-running jobs
│   └── prompts.py        # Prompt templates for all pipeline stages
├── auto_labeling/        # Multi-stage LLM labeling pipeline
│   ├── core/             # Signal mining, judging, QA, post-processing
│   ├── unified_pipeline/ # End-to-end session processing
│   └── prompts/          # System prompts for each pipeline stage
├── anonymize/            # PII detection and removal
│   ├── pii_rules.py      # Rule-based PII patterns
│   ├── run_rules.py      # Rule-based anonymization pass
│   └── run_llm_deep.py   # LLM-based deep anonymization pass
└── full_pipeline/        # Simplified end-to-end pipeline
```

## Setup

1. Configure your LLM proxy in `lib/config.py` or via environment variables:
   ```bash
   export LLM_API_KEY="your-key"
   export LLM_API_URL="http://your-proxy/v1"
   ```

2. Install dependencies (in addition to the top-level `requirements.txt`):
   ```bash
   pip install openai
   ```

   The data pipeline uses the `openai` Python client. The evaluation pipeline in `src/response_generation` and `src/grm_judge` uses `requests` directly and does not need `openai`.

## Pipeline Stages

1. **Signal Mining** — Extract quality signals from raw conversations
2. **Judge** — LLM-as-judge binary quality classification
3. **QA** — Quality assurance with multi-judge consensus
4. **Dimension Classification** — Categorize failure/success dimensions
5. **Anonymization** — Two-pass PII removal (rules + LLM)
6. **Golden Testset Construction** — Stratified sampling for balanced evaluation
