# src/utils/

Shared utility modules used by both the data pipeline (`src/pipeline/`) and the
evaluation scripts (`scripts/`).

## Modules

| File | Description |
|------|-------------|
| `config.py` | API configuration — reads `OPENAI_API_KEY` / `OPENAI_API_BASE` from environment, exposes `get_route()` and per-model thinking params. |
| `llm_client.py` | Unified LLM API client — auto-routing, exponential-backoff retries, JSON parsing, parallel execution via `ThreadPoolExecutor`. |
| `data_loader.py` | JSONL I/O utilities — streaming iterator, batch loader, testset loader (auto-injects `ground_truth` fields). |
| `checkpoint.py` | Checkpoint / resume support — load completed CIDs, append records, clean invalid lines for incremental runs. |
| `prompts.py` | Shared prompt templates and verdict parsing — `POINTWISE_GRM`, `BINARY_VERDICT_PROMPT`, and all `extract_*` functions. |
| `__init__.py` | Re-exports all public symbols for convenient `from utils import ...` access. |

## Usage Examples

### LLM client

```python
from utils.llm_client import call_llm, run_parallel

# Single call (auto-routes url/key from config)
result = call_llm(
    [{"role": "user", "content": "hello"}],
    model="gpt-5.1",
)
if result.ok:
    print(result.content)

# Parallel calls
tasks = [
    (cid, messages, "gpt-5.1", {})
    for cid, messages in batch
]
results = run_parallel(tasks, workers=10)
```

### Config

```python
from utils.config import get_route, get_thinking_params

url, key, cookie = get_route("deepseek-r1")
extra = get_thinking_params("deepseek-r1")  # {"reasoning_effort": "high"}
```

### Data loading

```python
from utils.data_loader import load_jsonl, load_testset

records = load_jsonl("data/testset/bad_1k.jsonl")
all_records = load_testset("data/testset/bad_1k.jsonl", "data/testset/good_1k.jsonl")
# all_records[i]["ground_truth"] is -1 (bad) or 1 (good)
```

### Checkpointing

```python
from utils.checkpoint import load_done_cids, append_record
from threading import Lock

done = load_done_cids("output/results.jsonl")
lock = Lock()

for rec in records:
    if rec["cid"] in done:
        continue
    # ... process ...
    append_record("output/results.jsonl", result, lock=lock)
```

### Prompt templates

```python
from utils.prompts import build_judge_prompt, extract_verdict

prompt = build_judge_prompt(history, query, response)
verdict, note = extract_verdict(model_output, reasoning_text)
# verdict: 1 (good) / -1 (bad) / None (unparseable)
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | Bearer token for the LLM API |
| `OPENAI_API_BASE` | No | Base URL (default: `https://api.openai.com/v1`) |

A `.env` file in the repo root is automatically loaded via `python-dotenv`.
