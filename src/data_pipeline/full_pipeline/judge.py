"""Judge stage for candidate bad cases."""

from __future__ import annotations

import json
from pathlib import Path

from config import CONFIG
from full_pipeline.llm_client import call_llm, parse_json_output
from full_pipeline.signals import format_dialogue_for_llm

PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"
JUDGE_SYSTEM = (PROMPT_DIR / "judge_system.txt").read_text(encoding="utf-8")
DEFAULT_JUDGE_MODEL = CONFIG.get("models", {}).get("judge", "gpt-5.2")


def _summarize_candidate(c: dict, idx: int) -> dict:
    return {
        "candidate_index": idx,
        "dislike_turn_id": c.get("dislike_turn_id"),
        "source_query_turn_id": c.get("source_query_turn_id"),
        "source_query_text": c.get("source_query_text"),
        "agent_response_preview": (c.get("agent_response_preview") or "")[:200],
        "failure_dimension": c.get("failure_dimension"),
        "scenario": c.get("scenario"),
        "signal_type": c.get("signal_type"),
        "signal_confidence": c.get("signal_confidence"),
        "sentiment": c.get("sentiment"),
        "explanation": c.get("explanation"),
        "confidence": c.get("confidence"),
    }


def judge_badcases_batch(record: dict, enrichment: dict, candidates: list, model: str = DEFAULT_JUDGE_MODEL) -> dict:
    cid = record.get("cid", "")
    history = record.get("history", [])
    if isinstance(history, str):
        history = json.loads(history)
    dialogue_text = format_dialogue_for_llm(history, enrichment)
    candidates_json = json.dumps([_summarize_candidate(c, i) for i, c in enumerate(candidates)], ensure_ascii=False, indent=2)
    batch_system = JUDGE_SYSTEM + "\n\n请按 JSON 数组返回所有候选项的评审结果。"
    user_prompt = (
        f"# 候选 Bad Case（{len(candidates)} 个）\n```json\n{candidates_json}\n```\n\n"
        f"# 完整对话\n{dialogue_text}\n\n请输出 JSON 数组。"
    )

    llm_res = call_llm(
        messages=[{"role": "system", "content": batch_system}, {"role": "user", "content": user_prompt}],
        model=model,
        max_tokens=2000 + 500 * len(candidates),
        temperature=0.1,
    )

    results = [None] * len(candidates)
    if llm_res.ok:
        parsed, _ = parse_json_output(llm_res.content)
        if isinstance(parsed, list):
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                idx = int(item.get("candidate_index", 0))
                if 0 <= idx < len(results):
                    results[idx] = {"parsed": item, "parse_ok": True}
        elif isinstance(parsed, dict) and len(candidates) == 1:
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
