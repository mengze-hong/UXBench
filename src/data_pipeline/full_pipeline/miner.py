"""Bad case miner (LLM stage)."""

from __future__ import annotations

import json
from pathlib import Path

from config import CONFIG
from full_pipeline.llm_client import call_llm, parse_json_output
from full_pipeline.signals import format_dialogue_for_llm

PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"
MINER_SYSTEM = (PROMPT_DIR / "miner_system.txt").read_text(encoding="utf-8")

DEFAULT_MINER_MODEL = CONFIG.get("models", {}).get("miner", "gpt-5.1")


def mine_badcases(record: dict, enrichment: dict, model: str = DEFAULT_MINER_MODEL) -> dict:
    cid = record.get("cid", "")
    history = record.get("history", [])
    if isinstance(history, str):
        history = json.loads(history)

    dialogue_text = format_dialogue_for_llm(history, enrichment)
    signal_summary = {
        "unliked_turns": enrichment.get("unliked_turns", []),
        "explicit_complaints": enrichment.get("explicit_complaints", []),
        "has_interrupted": enrichment.get("has_interrupted", False),
        "turn_count": enrichment.get("turn_count", len(history)),
    }
    user_prompt = (
        f"# 对话信号摘要\n```json\n{json.dumps(signal_summary, ensure_ascii=False, indent=2)}\n```\n\n"
        f"# 完整对话\n{dialogue_text}\n\n请严格按系统提示输出 JSON。"
    )

    llm_res = call_llm(
        messages=[{"role": "system", "content": MINER_SYSTEM}, {"role": "user", "content": user_prompt}],
        model=model,
        max_tokens=4000,
        temperature=0.1,
    )
    parsed = None
    parse_error = ""
    if llm_res.ok:
        parsed, parse_error = parse_json_output(llm_res.content)

    return {
        "cid": cid,
        "model": model,
        "latency_s": llm_res.latency_s,
        "tokens": llm_res.tokens,
        "attempts": llm_res.attempts,
        "llm_ok": llm_res.ok,
        "llm_error": llm_res.error,
        "raw_output": llm_res.content,
        "parsed": parsed,
        "parse_error": parse_error,
    }
