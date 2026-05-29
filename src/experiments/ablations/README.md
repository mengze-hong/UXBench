# UXBench ablations

This directory contains all paper-cited ablation experiments. Each `E*`
sub-directory maps one-to-one to a section in the paper appendix.

| Dir | Paper §  | Question studied |
|-----|----------|------------------|
| `E1_binary_vs_threeclass`            | §C.1     | Three-way verdict (Good / Bad / Neutral) vs. the binary task — does adding a Neutral option let weaker models "escape" the hard cases? |
| `E2_prompting_strategies_task1`      | §C.4     | Effect of different system-prompt phrasings on Task 1 judging accuracy |
| `E3_reasoning_efforts_task1`         | §C.5     | Reduced / disabled thinking budget on Task 1 |
| `E3_reasoning_efforts_task2`         | §C.5     | Reduced / disabled thinking budget on Task 2 generation |
| `E3_reasoning_efforts_task3`         | §C.5     | Reduced / disabled thinking budget on Task 3 recovery |
| `E4_bias_analysis`                   | §C.2     | Pointwise judge bias: how much do Gemini 2.5 Flash / Gemini 3.1 Pro / GPT-5.2 favour their own family? |
| `E4_pairwise_vs_pointwise`           | §C.2     | Pairwise self-preference bias: do strong models pick their own response when used as a pairwise judge? |
| `E5_human_alignment`                 | §C.6     | Human alignment of the trained pointwise GRM (GRM Alignment with Human Annotation) |
| `E6_cross_benchmark_english`         | §C.8     | Cross-benchmark generalisation on English data: AlpacaEval, ArenaHard, WildBench |
| `E7_system_prompt_task2`             | §C.4     | Four different "response" system prompts on Task 2 (helpful / no-system / ux-focused / empathetic) |
| `E8_recovery_prompt_strategy_task3`  | §C.7     | Four recovery-prompt strategies on Task 3 (normal / CoT / dimension-aware / critique) |

## CID alignment

Every record in every ablation file uses the same unified CID scheme as the
main release:

* Task 1 ablations  → `uxbench_ux_judge_NNNN`
* Task 2 ablations  → `uxbench_ux_eval_NNNN`
* Task 3 ablations  → `uxbench_ux_recovery_NNNN`

This means that, for example, `uxbench_ux_eval_0123` in
`task2_ux_eval/responses/gpt_5_5.jsonl` (main result) refers to the **same**
underlying user turn as `uxbench_ux_eval_0123` in
`ablations/E3_reasoning_efforts_task2/responses/gpt_5_5_low.jsonl` (low-effort
ablation) and as `uxbench_ux_eval_0123` in any
`ablations/E4_bias_analysis/judge_*/gpt_5_5.jsonl` (bias ablation).

## Schema

The ablation files reuse the same minimal schemas as the main release; see
the parent `experiments/README.md` for details. Records whose source CID is
not in the canonical UXBench test set have been removed.

## Cross-benchmark notes (E6)

The `E6_cross_benchmark_english/` directory uses the original public
benchmark IDs (AlpacaEval, ArenaHard, WildBench) for `cid` rather than the
UXBench unified CIDs, since these are external benchmarks.
