# UXBench Dataset Statistics & Schema

> Dataset composition and taxonomy details. Raw user data is not included for privacy.

---

## Overview

| Task | Splits | Description |
|------|--------|-------------|
| Task 1: UX Judge | BAD / GOOD | Binary classification — predict whether a response satisfies the user |
| Task 2: UX Eval | Failure queries | Comparative model ranking on user-failure queries |
| Task 3: UX Recovery | Failed interactions | Open-ended generation: recover gracefully after a failed response |

**Source**: Interaction logs from a mainstream AI assistant, filtered through a multi-stage pipeline with LLM-based quality control and human validation.

---

## Data Schema

### Task 1: UX Judge
```json
{
  "cid": "uuid",
  "ground_truth": "1 (good) | -1 (bad)",
  "signal_type": "like | dislike | explicit_praise | deep_dive",
  "scenario": "str (8 categories)",
  "difficulty": "easy | medium | hard | very_hard",
  "history": "[{role, content}, ...]",
  "query": "str",
  "scene_l1": "str", "scene_l2": "str",
  "sector_l1": "str", "sector_l2": "str"
}
```

### Task 2: UX Eval
```json
{
  "cid": "uuid",
  "query": "str (the failure query)",
  "messages": "[{role, content}, ...] (full context for generation)",
  "dimension": "str (failure dimension)",
  "scenario": "str",
  "difficulty": "str",
  "scene_l1": "str", "intent_l1": "str", "sector_l1": "str"
}
```

### Task 3: UX Recovery
```json
{
  "cid": "uuid",
  "history": "[{role, content}, ...]",
  "user_complaint": "str",
  "failed_response": "str",
  "failure_dimension": "str (14 categories)",
  "scenario": "str"
}
```

---

## Taxonomy

### Failure Dimensions (10 categories)

From Task 1 BAD split. Also used in Task 2 and Task 3.

| Dimension | Share | Description |
|-----------|------:|-------------|
| Verbosity / Redundancy | 34.3% | Unnecessarily long or repetitive responses |
| Task Incompleteness | 24.8% | Request not fully fulfilled |
| Intent Misunderstanding | 11.5% | Misinterpreted what the user wanted |
| Factual Error | 11.0% | Contains incorrect facts |
| Information Reliability Issue | 10.1% | Unreliable or unverifiable claims |
| Instruction / Format Failure | 2.8% | Did not follow explicit instructions |
| Insufficiently Informative | 2.2% | Too brief or lacks useful detail |
| Emotional Tone Mismatch | 2.0% | Inappropriate emotional register |
| Safety / Refusal Issue | 0.9% | Over-refusal or safety misfire |
| System / Technical Error | 0.4% | Technical malfunction |

### Success Dimensions (8 categories)

From Task 1 GOOD split.

| Dimension | Share | Description |
|-----------|------:|-------------|
| Accurate Answering | 17.4% | Correct and precise response |
| Knowledge Depth | 15.4% | Demonstrates deep domain knowledge |
| Comprehensive Detail | 13.7% | Thorough coverage of the topic |
| Problem Solving | 12.6% | Effectively solves the user's problem |
| Practical Guidance / Actionability | 12.1% | Provides actionable next steps |
| Creative Generation | 11.6% | High-quality creative output |
| Task Completion | 9.0% | Fully completes the requested task |
| Empathetic Support | 8.2% | Provides emotional support appropriately |

### Recovery Strategies (Task 3, 6 categories)

Strategy labels annotated on model recovery responses (opening behavior).

| Strategy | Share | Definition |
|----------|------:|------------|
| Apology | 45.98% | Apologizes, admits fault, or takes responsibility |
| Agreement | 28.64% | Validates the complaint or agrees the user is right |
| Error Diagnosis | 14.67% | Explains the error or states an information boundary |
| Humor | 4.81% | Uses humor or de-escalation to repair rapport |
| Direct Fix | 3.63% | Directly provides the correction or next step |
| Clarification | 1.56% | Asks what was wrong or what information is needed |

---

## Scenario Distribution (8 categories)

Relative share within each task split.

| Scenario | Task1 BAD | Task1 GOOD | Task2 | Task3 |
|----------|----------:|-----------:|------:|------:|
| Information & Knowledge Seeking            | 52.9% | 58.5% | 37.3% | 41.0% |
| Personal Assistance & Emotional Support    | 19.9% |  6.3% | 14.1% | 19.2% |
| Creative Content & Entertainment           | 10.9% |  6.7% | 13.3% | 15.0% |
| Casual Chat & Companionship                |  4.5% | 16.9% | 14.4% | 11.0% |
| Learning & Educational Support             |  6.8% |  1.3% |  3.7% |  6.4% |
| Productivity & Office Efficiency           |  4.0% |  1.0% |  3.0% |  4.4% |
| Product & Service Inquiry                  |  1.0% |  9.3% | 14.1% |  3.0% |

---

## Conversation History Length (Task 2)

| Turns | Share |
|-------|------:|
| 2     | 56.9% |
| 4     | 21.0% |
| 6     | 11.1% |
| 8     |  5.5% |
| 10+   |  5.5% |

---

## Additional Notes

- **Multi-level taxonomy**: Each instance is labeled with 2-level scene (`scene_l1/l2`), intent (`intent_l1/l2`), and domain sector (`sector_l1/l2`) covering 83 fine-grained categories
- **Quality control**: All instances pass a 3-judge LLM quality gate (overall_quality ≥ "medium", representativeness ≥ "medium")
- **Human validation**: A randomly sampled subset is verified by 3 annotators with >92% agreement (see paper §5.6)
- **Dynamic updates**: The benchmark is designed for bi-monthly refresh to mitigate data contamination
