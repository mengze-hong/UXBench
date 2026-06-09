# UXBench Task 1 — UX Judge Leaderboard

> Last updated: 2026-05-21 · N=2,000 (Good 1,000 + Bad 1,000) · 26 frontier models

- **Good-Acc** = recall on 1,000 liked conversations
- **Bad-Acc** = recall on 1,000 disliked conversations
- **Avg-Acc** = (Good-Acc + Bad-Acc) / 2 — rewards balanced discrimination

| Rank | Model | Org | Good-Acc | Bad-Acc | Avg-Acc |
|-----:|-------|-----|--------:|-------:|-------:|
| 1 | Claude Opus 4.7 | Anthropic | 89.1% | 61.5% | 75.3% |
| 2 | GPT-5.2 | OpenAI | 85.0% | 65.1% | 75.0% |
| 3 | GPT-5.5 | OpenAI | 92.7% | 55.7% | 74.2% |
| 4 | GPT-5 | OpenAI | 89.5% | 56.2% | 72.9% |
| 5 | GPT-5.1 | OpenAI | 94.8% | 50.1% | 72.5% |
| 6 | Claude Opus 4.6 | Anthropic | 92.6% | 51.5% | 72.0% |
| 7 | Gemini 3.1 Pro | Google | 91.6% | 49.3% | 70.4% |
| 8 | Claude Sonnet 4.5 | Anthropic | 89.6% | 49.0% | 69.3% |
| 9 | Kimi K2.6 | Moonshot | 96.1% | 41.2% | 68.7% |
| 10 | GLM-5.1 | Zhipu AI | 96.1% | 40.9% | 68.5% |
| 11 | Gemini 3.0 Flash | Google | 97.7% | 37.1% | 67.4% |
| 12 | Claude Opus 4.5 | Anthropic | 96.7% | 36.7% | 66.7% |
| 13 | GLM-5 | Zhipu AI | 96.9% | 36.4% | 66.7% |
| 14 | Qwen3.6-Plus | Alibaba | 96.8% | 34.9% | 65.8% |
| 15 | GPT-5 mini | OpenAI | 93.4% | 36.9% | 65.2% |
| 16 | Kimi K2.5 | Moonshot | 96.8% | 32.7% | 64.8% |
| 17 | DeepSeek V4 Pro | DeepSeek | 97.4% | 31.7% | 64.5% |
| 18 | DeepSeek V3.2 | DeepSeek | 95.7% | 33.3% | 64.5% |
| 19 | Hunyuan 3 | Tencent | 95.6% | 33.1% | 64.3% |
| 20 | Gemini 2.5 Pro | Google | 96.8% | 28.7% | 62.8% |
| 21 | Doubao Seed 2.0 Pro | ByteDance | 98.8% | 22.9% | 60.8% |
| 22 | Gemini 2.5 Flash | Google | 97.5% | 21.2% | 59.4% |
| 23 | DeepSeek R1 | DeepSeek | 98.7% | 18.3% | 58.5% |
| 24 | Doubao Seed 1.6 | ByteDance | 99.1% | 16.2% | 57.6% |
| 25 | Doubao Seed 2.0 Lite | ByteDance | 98.7% | 16.0% | 57.4% |
| 26 | DeepSeek V3 | DeepSeek | 99.7% | 11.6% | 55.6% |

---

## Evaluation Protocol

Evaluation framework: `POINTWISE_GRM` prompt (`src/utils/prompts.py`).
Output: "好" (good) or "差" (bad), mapped to verdict = 1 / −1.
