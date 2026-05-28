# UXBench experiments

This directory contains all evaluation outputs reported in the paper.

```
experiments/
├── task1_ux_judge/      27 models × 2,000 records  (Good/Bad classification)
├── task2_ux_eval/       27 models × 4,900 records  (UX-quality generation)
├── task3_ux_recovery/   27 models ×   500 records  (failure-recovery generation)
└── ablations/           paper §A.1 – §A.7
```

Each task directory contains:

* `responses/<model>.jsonl` — one file per evaluated model, each line a single
  prediction with the unified CID (`uxbench_ux_judge_NNNN`,
  `uxbench_ux_eval_NNNN`, `uxbench_ux_recovery_NNNN`). The same CID identifies
  the same underlying user turn across all 27 models for that task.
* `judge/judge_<model>.jsonl` (Tasks 2 / 3 only) — pointwise GRM verdicts on
  each generated response.
* `leaderboard.md` — paper-aligned ranked summary of the 27 models, generated
  directly from the released `responses/` (Task 1) or `judge/` (Tasks 2/3) files.

## The 27 paper-listed models

| Family | Models |
|--------|--------|
| Google Gemini | Gemini 2.5 Flash, 2.5 Pro, 3.0 Flash, 3.1 Pro |
| OpenAI        | GPT-5, GPT-5 mini, GPT-5.1, GPT-5.2, GPT-5.5 |
| Anthropic     | Claude Sonnet 4.5, Opus 4.5, Opus 4.6, Opus 4.7 |
| DeepSeek      | DeepSeek R1, V3, V3.2, V4 Pro |
| ByteDance     | Doubao Seed 1.6, 2.0 Lite, 2.0 Pro |
| Others        | Hunyuan 3, GLM-5, GLM-5.1, Kimi K2.5, K2.6, MiniMax M2.5, Qwen3.6-Plus |

Each model file in every task contains exactly the canonical N rows (2,000 /
4,900 / 500), with no duplicates and no records outside the canonical test
set.

## What is *not* released

To protect both the queries themselves and the providers' generation traces,
the following fields are stripped from every released line:

* `query` (the user turn)
* `reasoning_content` / `api_reasoning_content` / chain-of-thought traces
* `failed_response` / `user_complaint` / scenario / dimension metadata used to
  construct Task 3 prompts
* any internal context, message history, or system prompt fields

Only the model-generated answer (`generated_response`) and the judge fields
needed to reproduce the leaderboard are kept. For Task 1 we additionally keep
`ground_truth` and `difficulty`, which come from the public Good/Bad labels.

## Reproducing the leaderboards

Each `leaderboard.md` is a deterministic function of the released `*.jsonl`
files in that task directory. The numbers reproduce the paper's main results
table (Table 1) for the 27 paper-listed models.

## Ablations

See `ablations/README.md` for the per-experiment details. The ablation
directories (`E1`–`E8`) map one-to-one to the paper's appendix sections.
