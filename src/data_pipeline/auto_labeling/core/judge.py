"""
Stage 2: Quality Judge Agent.
Scores candidate bad cases on 5 axes, decides high/medium/low.
Uses a DIFFERENT model family from the Miner (gpt-5.1) to reduce bias.

OPTIMIZATION: Batch-judges ALL candidates from one dialogue in a single LLM call
(instead of one call per candidate). Cuts judge time by ~2.5x.
"""

import json
from pathlib import Path
from llm_client import call_llm, parse_json_output
from signals import format_dialogue_for_llm

PROMPT_DIR = Path(__file__).parent.parent / "prompts"
JUDGE_SYSTEM = (PROMPT_DIR / "judge_system.txt").read_text(encoding="utf-8")

# gpt-5.2: fast (2.62s), different family from Miner (gpt-5.1 is same family but different size)
# Use gemini as cross-family validation
DEFAULT_JUDGE_MODEL = "gpt-5.2"
FALLBACK_JUDGE_MODEL = "gemini-3-pro-preview"


def _summarize_candidate(c: dict) -> dict:
    """Extract key fields from a candidate for the judge prompt."""
    return {
        "candidate_index": c.get("_idx", 0),
        "dislike_turn_id": c.get("dislike_turn_id"),
        "source_query_turn_id": c.get("source_query_turn_id"),
        "source_query_text": c.get("source_query_text"),
        "agent_response_preview": (c.get("agent_response_preview") or "")[:200],
        "failure_dimension": c.get("failure_dimension"),
        "scenario": c.get("scenario"),
        "signal_type": c.get("signal_type"),
        "signal_confidence": c.get("signal_confidence"),
        "sentiment": c.get("sentiment"),
        "complaint_snippet": c.get("complaint_snippet"),
        "explanation": c.get("explanation"),
        "user_reaction_after_failure": c.get("user_reaction_after_failure"),
        "recovery_attempted": c.get("recovery_attempted"),
        "recovery_successful": c.get("recovery_successful"),
        "representativeness": c.get("representativeness"),
        "confidence": c.get("confidence"),
    }


def judge_badcases_batch(record: dict, enrichment: dict, candidates: list, model: str = DEFAULT_JUDGE_MODEL) -> dict:
    """
    Batch-judge ALL candidates from one dialogue in a single LLM call.

    Returns dict:
    {
      "cid": str,
      "model": str,
      "latency_s": float,
      "tokens": int,
      "llm_ok": bool,
      "results": [  # one per candidate, in order
        {"parsed": {...}, "parse_ok": bool} or None
      ]
    }
    """
    cid = record.get("cid", "")
    history = record.get("history", [])
    if isinstance(history, str):
        history = json.loads(history)
    dialogue_text = format_dialogue_for_llm(history, enrichment)

    # Tag candidates with index
    for i, c in enumerate(candidates):
        c["_idx"] = i

    candidates_json = json.dumps(
        [_summarize_candidate(c) for c in candidates],
        ensure_ascii=False, indent=2
    )

    batch_system = JUDGE_SYSTEM + """

# 批量评审模式
本次你需要评审多个候选 Bad Case。请对每个候选项分别打分，输出一个 JSON **数组**，数组中每个元素是对应候选项的打分结果（格式同单条打分）。

输出格式：
```json
[
  {
    "candidate_index": 0,
    "scores": {"query_completeness": 4, "signal_credibility": 5, ...},
    "average": 4.4,
    "overall_quality": "high",
    "should_keep": true,
    "audit_notes": "...",
    "suggested_corrections": {"dislike_turn_id": null, "failure_dimension": null, "source_query_turn_id": null, "signal_type": null}
  },
  ...
]
```
"""

    user_prompt = (
        f"# Miner 的初判（{len(candidates)} 个候选 Bad Case）\n"
        f"```json\n{candidates_json}\n```\n\n"
        f"# 完整对话（含信号标注）\n{dialogue_text}\n\n"
        f"请对以上 {len(candidates)} 个候选项分别打分，输出 JSON 数组。"
    )

    llm_res = call_llm(
        messages=[
            {"role": "system", "content": batch_system},
            {"role": "user", "content": user_prompt},
        ],
        model=model,
        max_tokens=2000 + 500 * len(candidates),  # Scale with candidates
        temperature=0.1,
    )

    # Fallback on failure
    if not llm_res.ok and model != FALLBACK_JUDGE_MODEL:
        llm_res = call_llm(
            messages=[
                {"role": "system", "content": batch_system},
                {"role": "user", "content": user_prompt},
            ],
            model=FALLBACK_JUDGE_MODEL,
            max_tokens=2000 + 500 * len(candidates),
            temperature=0.1,
        )
        model = FALLBACK_JUDGE_MODEL

    results = [None] * len(candidates)

    if llm_res.ok:
        parsed, parse_error = parse_json_output(llm_res.content)
        if parsed is not None:
            # Handle both list output and single-object output
            if isinstance(parsed, list):
                for item in parsed:
                    idx = item.get("candidate_index", 0)
                    if 0 <= idx < len(results):
                        results[idx] = {"parsed": item, "parse_ok": True}
            elif isinstance(parsed, dict):
                # Single candidate case — LLM returned a single object
                results[0] = {"parsed": parsed, "parse_ok": True}

    return {
        "cid": cid,
        "model": model,
        "latency_s": llm_res.latency_s,
        "tokens": llm_res.tokens,
        "attempts": llm_res.attempts,
        "llm_ok": llm_res.ok,
        "llm_error": llm_res.error,
        "raw_output": llm_res.content,
        "results": results,
    }


# Keep legacy single-candidate function for backward compatibility
def judge_badcase(record: dict, enrichment: dict, candidate: dict, model: str = DEFAULT_JUDGE_MODEL) -> dict:
    """Legacy: judge a single candidate. Wraps batch function."""
    res = judge_badcases_batch(record, enrichment, [candidate], model=model)
    single = res["results"][0] if res["results"] else None
    return {
        "cid": res["cid"],
        "model": res["model"],
        "latency_s": res["latency_s"],
        "tokens": res["tokens"],
        "attempts": res["attempts"],
        "llm_ok": res["llm_ok"],
        "llm_error": res["llm_error"],
        "raw_output": res["raw_output"],
        "parsed": single["parsed"] if single else None,
        "parse_error": "" if single and single["parse_ok"] else "batch_parse_failed",
    }
