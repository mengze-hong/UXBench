# src/pipeline/prompts/

System prompt text files used by the three LLM agents in the UXBench data construction pipeline.

> **Note:** All prompts are written in Chinese because they are used to evaluate
> Chinese AI assistant conversations.

## Files

| File | Agent | Stage | Description |
|------|-------|-------|-------------|
| `miner_system.txt` | Miner Agent | Stage 2 | Instructs the LLM to extract failure or success reasons from a conversation. Outputs structured JSON with `failure_dimension`, `signal_type`, `explanation` etc. |
| `judge_system.txt` | Judge Agent | Stage 3 | Instructs the LLM to score a candidate bad case on 5 quality axes (1–5 each): `query_completeness`, `signal_credibility`, `representativeness`, `severity`, `annotation_clarity`. |
| `qa_system.txt` | QA Agent | Stage 4 | Instructs the LLM to validate and filter candidate test cases, checking for ambiguity, unsafe content, and annotation accuracy. |

## Usage

These files are loaded at runtime by `miner.py` and `judge.py`:

```python
PROMPT_DIR = Path(__file__).parent / "prompts"
MINER_SYSTEM = (PROMPT_DIR / "miner_system.txt").read_text(encoding="utf-8")
```

> These are **pipeline** prompts for dataset construction only.
> The main evaluation prompt (`POINTWISE_GRM`) used for Task 1/2/3 is in `src/utils/prompts.py`.
