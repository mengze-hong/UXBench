# UXBench Dataset

The UXBench evaluation dataset.

## Files

| File | Records | Description |
|------|---------|-------------|
| `ux_eval_demo_200.jsonl` | 200 | Demonstration subset of the Task 2 (UX Eval) test set |
| `DATASET_STATISTICS.md` | — | Full schema, taxonomy, and category distributions |

The full dataset (across all three tasks) is available for academic research under the [UXBench Research-Only License](../../LICENSE). Contact the authors via the channels specified in the associated publication.

## Schema (`ux_eval_demo_200.jsonl`)

Each line is a JSON object:

```json
{
  "cid": "uxbench_ux_eval_XXXX",
  "scenario": "str (8 categories)",
  "difficulty": "easy | medium | hard | very_hard | null",
  "dimension": "str (failure dimension)",
  "signal_type": "like | dislike",
  "history_turns": "int (number of prior turns)",
  "overall_quality": "high | medium",
  "representativeness": "high | medium",
  "scene_l1": "str (top-level scene)",
  "scene_l2": "str (sub-scene)",
  "intent_l1": "str (top-level intent)",
  "intent_l2": "str (sub-intent)",
  "sector_l1": "str (top-level domain)",
  "sector_l2": "str (sub-domain)",
  "query": "str (the user query)"
}
```

See [`DATASET_STATISTICS.md`](DATASET_STATISTICS.md) for the full category list and distributions.

## Quick Load

```python
import json

with open("ux_eval_demo_200.jsonl", "r", encoding="utf-8") as f:
    samples = [json.loads(line) for line in f]

print(f"Loaded {len(samples)} samples")
print(f"Scenarios: {sorted({s['scenario'] for s in samples})}")
```

## Privacy

All records are **anonymized** prior to release:

- PII removed via rule-based + LLM-deep pipeline (see `../data_pipeline/anonymize/`)
- Raw conversations are **not released** — only the anonymized query and structured metadata
