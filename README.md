# 🌟 UXBench: Benchmarking User Experience in AI Assistants

The first user-centric benchmark grounded in **real user feedback signals** (👍 / 👎) collected from a deployed mainstream AI assistant.

---

## 📚 The three tasks

| # | Task | What the model is asked to do | N |
|---|------|-------------------------------|--:|
| 🧑‍⚖️ **1** | **UX Judge** | Read a (user-turn, response) pair and decide whether the response is **Good** or **Bad** from a UX perspective | **2,000** |
| ✍️ **2** | **UX Eval** | Generate a response to a real user query that real users found hard to satisfy in the wild | **4,900** |
| 🛟 **3** | **UX Recovery** | Given a failed assistant response and a user-complaint follow-up turn, write a recovery response the user would accept | **500** |

> **Total**: 7,400 unique instances, 27 models, **3 tasks** — 27 × (2,000 + 4,900 + 500) = **199,800** main-track LLM responses, plus 27 × (4,900 + 500) = **145,800** pointwise GRM judgments on Tasks 2 / 3.

---

## 🗂️ Repository layout

```text
uxbench-v1.0-public/
├── README.md                          ← you are here
├── LICENSE                            ← non-commercial research license
├── requirements.txt
└── src/
    ├── README.md
    │
    ├── 📦 uxbench-dataset/            
    │   ├── README.md
    │   ├── DATASET_STATISTICS.md
    │   ├── uxbench_task1_judge_bad_1k.jsonl
    │   ├── uxbench_task1_judge_good_1k.jsonl
    │   ├── uxbench_task2_eval_4900.jsonl
    │   └── uxbench_task3_recovery_500.jsonl   
    │
    ├── 🛠️ data_pipeline/              ← thumbs-up/down → cleaned query splits
    │                                    (§3.4 Dataset Construction)
    ├── 🚀 response_generation/        ← unified call_llm harness for 27 models
    │                                    (§B.1 Model Selection & Inference)
    ├── 🧠 grm_judge/                  ← trained pointwise GRM (judge for Tasks 2/3)
    │                                    (§B.2 Training Generative Reward Model)
    ├── 📊 figures/                    ← plotting code
    │
    └── 🧪 experiments/                ← all benchmarked artefacts (frozen)
        ├── README.md
        │
        ├── task1_ux_judge/            27 × 2,000 records  ← §4.1
        │   ├── responses/
        │   └── leaderboard.md
        │
        ├── task2_ux_eval/             27 × 4,900 records  ← §4.2
        │   ├── responses/
        │   ├── judge/                 27 GRM judge files
        │   └── leaderboard.md
        │
        ├── task3_ux_recovery/         27 × 500 records    ← §4.3
        │   ├── responses/
        │   ├── judge/                 27 GRM judge files
        │   └── leaderboard.md
        │
        └── ablations/                 ← Appendix C (Full Experimental Results)
            ├── README.md
            ├── E1_binary_vs_threeclass/           ← §C.1 Binary vs. Three-Class
            ├── E2_prompting_strategies_task1/     ← §C.4 Impact of Prompting Strategies
            ├── E3_reasoning_efforts_task1/        ← §C.5 Impact of Reasoning Efforts
            ├── E3_reasoning_efforts_task2/        ← §C.5 Impact of Reasoning Efforts
            ├── E3_reasoning_efforts_task3/        ← §C.5 Impact of Reasoning Efforts
            ├── E4_bias_analysis/                  ← §C.2 Pointwise Bias
            ├── E4_pairwise_vs_pointwise/          ← §C.2 Pairwise Bias
            ├── E5_human_alignment/                ← §C.6 GRM Alignment with Humans
            ├── E6_cross_benchmark_english/        ← §C.8 Cross-Benchmark Generalization
            ├── E7_system_prompt_task2/            ← §C.4 (Task 2 system prompts)
            └── E8_recovery_prompt_strategy_task3/ ← §C.7 Failure Recovery Strategies
```

---

## 📐 Data schemas

Every released `*.jsonl` is one JSON object per line.

### 🧑‍⚖️ Task 1 — `task1_ux_judge/responses/<model>.jsonl`
```json
{
  "cid":          "uxbench_ux_judge_0001",
  "model":        "claude-opus-4.7",
  "ground_truth": 1,            // +1 = Good (👍), -1 = Bad (👎)
  "difficulty":   "medium",     // easy / medium / hard
  "verdict":      1,            // model's predicted label
  "score":        0.92,
  "tokens":       412,
  "latency_s":    3.51
}
```

### ✍️ Task 2 — `task2_ux_eval/responses/<model>.jsonl`
```json
{
  "cid":                "uxbench_ux_eval_0001",
  "model":              "gpt-5.5",
  "generated_response": "<the model's answer>",
  "tokens":             812,
  "latency_s":          5.93
}
```

### ⚖️ GRM judge — `task{2,3}_*/judge/judge_<model>.jsonl`
```json
{
  "cid":          "uxbench_ux_eval_0001",
  "judged_model": "gpt-5.5",
  "verdict":      1,            // GRM verdict (+1 Good / -1 Bad)
  "score":        0.78,
  "good_logprob": -0.24,
  "bad_logprob":  -1.65,
  "grm_ok":       true,
  "latency_s":    0.42
}
```

---

## 📅 Maintenance

UXBench is a **living benchmark**: the data is updated on a **bi-monthly basis** so that the test set keeps tracking the evolving distribution of real user queries. Each release is versioned (this is `v1.0-public`).


---

## 📜 License

The dataset and code are released for **non-commercial research use only** under the terms in [`LICENSE`](LICENSE).