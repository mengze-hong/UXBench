# UXBench Evaluation Pipeline (`src/`)

Two-step evaluation: **generate responses** → **judge with GRM**.

## Quick Start

```bash
# Step 1: Generate responses from your model
python src/response_generation/generate_responses.py \
    --task task2 \
    --model your-model-name \
    --endpoint http://your-llm-api:8000/v1/chat/completions \
    --api-key YOUR_KEY \
    --workers 10

# Step 2: Judge responses with GRM
python src/grm_judge/run_grm_judge.py \
    --responses outputs/task2/your_model_name.jsonl \
    --testset src/uxbench-dataset/ux_eval_demo_200.jsonl \
    --endpoint http://your-grm-server:8021/v1/chat/completions \
    --workers 80
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_ENDPOINT` | `http://localhost:8000/v1/chat/completions` | Target LLM API |
| `LLM_API_KEY` | `EMPTY` | API key for target LLM |
| `GRM_ENDPOINT` | `http://localhost:8021/v1/chat/completions` | GRM vLLM endpoint |
| `GRM_MODEL` | `pointwise_grm_ux` | GRM model name |
| `GRM_API_KEY` | `EMPTY` | GRM auth token |

## Pipeline Overview

```
                    ┌─────────────────┐
  testset.jsonl ──▶│  generate_      │──▶ responses/<model>.jsonl
  (queries)        │  responses.py   │    (model outputs)
                    └─────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
  responses/*.jsonl│  run_grm_       │──▶ judge_results/judge_<model>.jsonl
  + testset.jsonl ─│  judge.py       │    (verdict + score)
                    └─────────────────┘
```

## Output Format

### Response file (`responses/<model>.jsonl`)
```json
{"cid": "uxbench_ux_eval_0001", "model": "gpt-4o", "generated_response": "...", "latency_s": 2.1}
```

### Judge file (`judge_results/judge_<model>.jsonl`)
```json
{"cid": "uxbench_ux_eval_0001", "judged_model": "gpt-4o", "verdict": 1, "score": 0.82, "latency_s": 0.3}
```

- `verdict`: 1 = good, -1 = bad, 0 = failed
- `score`: P("good") / (P("good") + P("bad")), range [0, 1]

## GRM Deployment

The GRM (Generative Reward Model) is a 20B-parameter model served via vLLM:

```bash
vllm serve /path/to/grm_checkpoint \
    --served-model-name pointwise_grm_ux \
    --host 0.0.0.0 --port 8021 \
    --tensor-parallel-size 8 \
    --max-model-len 16384 \
    --trust-remote-code
```

GRM checkpoint is available upon request for academic use.

## Features

- **Checkpoint/resume**: Both scripts skip already-completed cids
- **Parallel execution**: Configurable worker count for throughput
- **Truncation retry**: GRM judge retries with shorter input on 400 errors
- **OpenAI-compatible**: Works with any OpenAI API-compatible endpoint
