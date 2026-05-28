"""
Stage 1: Bad Case Miner Agent.
Feeds a dialogue + ALL signals (including user reactions, recovery chains) into an LLM,
extracts ALL candidate bad cases (both strong and weak signal types).
"""

import json
from pathlib import Path
from llm_client import call_llm, parse_json_output
from signals import format_dialogue_for_llm

PROMPT_DIR = Path(__file__).parent.parent / "prompts"
MINER_SYSTEM = (PROMPT_DIR / "miner_system.txt").read_text(encoding="utf-8")

# Use gpt-5.1 for speed (1.14s avg latency), fallback to gemini-3-pro
DEFAULT_MINER_MODEL = "gpt-5.1"
FALLBACK_MINER_MODEL = "gemini-3-pro-preview"


def mine_badcases(record: dict, enrichment: dict, model: str = DEFAULT_MINER_MODEL) -> dict:
    """
    Run the Miner on one dialogue record.
    Now includes user reactions + recovery chain signals for comprehensive mining.
    """
    cid = record.get("cid", "")
    history = record.get("history", [])
    if isinstance(history, str):
        history = json.loads(history)

    # Render the dialogue with ALL signal markers
    dialogue_text = format_dialogue_for_llm(history, enrichment)

    # Comprehensive signal summary
    signal_summary = {
        "unliked_turns": enrichment["unliked_turns"],
        "explicit_complaints": [
            {"turn": c["turn_index"], "trigger_asst_turn": c["triggered_asst_turn_id"], "snippet": c["snippet"]}
            for c in enrichment["explicit_complaints"]
        ],
        "dissatisfied_turns": enrichment.get("dissatisfied_turns", []),
        "user_reactions": [
            {"asst_turn": r["asst_turn_id"], "reaction": r["reaction"],
             "dissatisfied": r["dissatisfied"], "details": r["details"]}
            for r in enrichment.get("user_reactions", []) if r["dissatisfied"]
        ],
        "recovery_chains": [
            {"failure_turn": c["failure_turn_id"], "outcome": c["recovery_outcome"],
             "chain_length": c["chain_length"]}
            for c in enrichment.get("recovery_chains", [])
        ],
        "reaction_summary": enrichment.get("reaction_summary", {}),
        "has_interrupted": enrichment["has_interrupted"],
        "turn_count": enrichment["turn_count"],
        "total_dissatisfied": enrichment.get("total_dissatisfied", 0),
    }

    user_prompt = (
        f"# 本次对话信号摘要\n"
        f"```json\n{json.dumps(signal_summary, ensure_ascii=False, indent=2)}\n```\n\n"
        f"# 完整对话（含信号标注）\n{dialogue_text}\n\n"
        f"请严格按系统提示的 JSON 格式输出。注意：不仅关注被点踩的轮次，也要关注 repeat/rephrase/abandon/dismiss 等行为信号暴露的 AI 问题。"
    )

    llm_res = call_llm(
        messages=[
            {"role": "system", "content": MINER_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        model=model,
        max_tokens=4000,
        temperature=0.1,
    )

    # If primary model fails, try fallback
    if not llm_res.ok and model != FALLBACK_MINER_MODEL:
        llm_res = call_llm(
            messages=[
                {"role": "system", "content": MINER_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            model=FALLBACK_MINER_MODEL,
            max_tokens=4000,
            temperature=0.1,
        )
        model = FALLBACK_MINER_MODEL

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
