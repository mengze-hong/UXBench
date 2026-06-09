"""
Single-record processor — used by both pipeline.py (batch mode) and the dashboard (live demo).

Runs the full pipeline on ONE dialogue and returns a detailed trace of every stage,
so the dashboard can display inputs, intermediate outputs, and final decisions.
"""

import json
import time
from datetime import datetime

from signals import enrich_dialogue, format_dialogue_for_llm
from prefilter import prefilter
from miner import mine_badcases, DEFAULT_MINER_MODEL, MINER_SYSTEM
from judge import judge_badcase, DEFAULT_JUDGE_MODEL, JUDGE_SYSTEM


def process_one_traced(
    record: dict,
    miner_model: str = DEFAULT_MINER_MODEL,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    progress_callback=None,
) -> dict:
    """
    Run the full pipeline on one record with detailed stage-by-stage trace.

    Returns a dict:
    {
      "cid": str,
      "decision": "saved" | "deleted" | "rejected" | "partial",
      "started_at": str,
      "duration_s": float,
      "stages": {
        "enrich":    { input, output, duration_s },
        "prefilter": { input, output, duration_s, keep, reason },
        "miner":     { input, prompt, raw_output, parsed, duration_s, tokens, model },
        "judge":     [ { candidate, prompt, raw_output, parsed, duration_s, tokens, model }, ... ],
      },
      "final_badcases": [...]
    }
    """
    def notify(stage, msg):
        if progress_callback:
            try: progress_callback(stage, msg)
            except Exception: pass

    cid = record.get("cid", "")
    history = record.get("history", [])
    t_start = time.time()

    trace = {
        "cid": cid,
        "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "decision": "",
        "duration_s": 0.0,
        "stages": {},
        "final_badcases": [],
    }

    # ── Stage 1: Enrichment ──
    notify("enrich", "computing signals...")
    t0 = time.time()
    enrichment = enrich_dialogue(history)
    dialogue_formatted = format_dialogue_for_llm(history, enrichment)
    trace["stages"]["enrich"] = {
        "duration_s": round(time.time() - t0, 3),
        "turn_count": enrichment["turn_count"],
        "unliked_turns": enrichment["unliked_turns"],
        "explicit_complaints": enrichment["explicit_complaints"],
        "has_interrupted": enrichment["has_interrupted"],
        "turns_signals": enrichment["turns"],
        "dialogue_formatted": dialogue_formatted,
    }

    # ── Stage 2: Pre-filter ──
    notify("prefilter", "applying rules...")
    t0 = time.time()
    keep, reason = prefilter(record, enrichment)
    trace["stages"]["prefilter"] = {
        "duration_s": round(time.time() - t0, 3),
        "keep": keep,
        "reason": reason,
    }
    if not keep:
        trace["decision"] = "deleted"
        trace["deleted_reason"] = "prefilter:" + reason
        trace["duration_s"] = round(time.time() - t_start, 3)
        return trace

    # ── Stage 3: Miner ──
    notify("miner", f"calling {miner_model}...")
    miner_out = mine_badcases(record, enrichment, model=miner_model)

    # Build user prompt used by miner (for display)
    signal_summary = {
        "unliked_turns": enrichment["unliked_turns"],
        "explicit_complaints": [
            {"turn": c["turn_index"], "trigger_asst_turn": c["triggered_asst_turn_id"], "snippet": c["snippet"]}
            for c in enrichment["explicit_complaints"]
        ],
        "has_interrupted": enrichment["has_interrupted"],
        "turn_count": enrichment["turn_count"],
    }
    miner_user_prompt = (
        f"# 本次对话信号摘要\n```json\n{json.dumps(signal_summary, ensure_ascii=False, indent=2)}\n```\n\n"
        f"# 完整对话（含信号标注）\n{dialogue_formatted}\n\n"
        f"请严格按系统提示的 JSON 格式输出。"
    )

    trace["stages"]["miner"] = {
        "model": miner_model,
        "system_prompt": MINER_SYSTEM,
        "user_prompt": miner_user_prompt,
        "raw_output": miner_out["raw_output"],
        "parsed": miner_out["parsed"],
        "parse_error": miner_out["parse_error"],
        "duration_s": miner_out["latency_s"],
        "tokens": miner_out["tokens"],
        "attempts": miner_out["attempts"],
        "llm_ok": miner_out["llm_ok"],
        "llm_error": miner_out["llm_error"],
    }

    if not miner_out["llm_ok"] or miner_out["parsed"] is None:
        trace["decision"] = "deleted"
        trace["deleted_reason"] = "miner_failed: " + (miner_out["llm_error"] or miner_out["parse_error"])
        trace["duration_s"] = round(time.time() - t_start, 3)
        return trace

    parsed = miner_out["parsed"]
    usable = parsed.get("usable", False)
    badcases = parsed.get("badcases", []) or []
    trace["stages"]["miner"]["usable"] = usable
    trace["stages"]["miner"]["n_candidates"] = len(badcases)
    trace["stages"]["miner"]["reject_reason"] = parsed.get("reject_reason", "")

    if not usable or not badcases:
        trace["decision"] = "deleted"
        trace["deleted_reason"] = "miner_rejected: " + (parsed.get("reject_reason", "") or "no_badcases")
        trace["duration_s"] = round(time.time() - t_start, 3)
        return trace

    # ── Stage 4: Judge each candidate ──
    trace["stages"]["judge"] = []
    n_kept = 0
    for idx, bc in enumerate(badcases):
        notify("judge", f"judging candidate {idx+1}/{len(badcases)}...")
        judge_out = judge_badcase(record, enrichment, bc, model=judge_model)

        # Build the judge user prompt for display
        candidate_summary = json.dumps({
            "dislike_turn_id": bc.get("dislike_turn_id"),
            "source_query_turn_id": bc.get("source_query_turn_id"),
            "source_query_text": bc.get("source_query_text"),
            "failure_dimension": bc.get("failure_dimension"),
            "scenario": bc.get("scenario"),
            "sentiment": bc.get("sentiment"),
            "complaint_snippet": bc.get("complaint_snippet"),
            "explanation": bc.get("explanation"),
            "representativeness": bc.get("representativeness"),
            "confidence": bc.get("confidence"),
        }, ensure_ascii=False, indent=2)
        judge_user_prompt = (
            f"# Miner 的初判\n```json\n{candidate_summary}\n```\n\n"
            f"# 完整对话（含信号标注）\n{dialogue_formatted}\n\n"
            f"请严格按系统提示的 JSON 格式输出打分。"
        )

        entry = {
            "idx": idx,
            "candidate": bc,
            "model": judge_model,
            "system_prompt": JUDGE_SYSTEM,
            "user_prompt": judge_user_prompt,
            "raw_output": judge_out["raw_output"],
            "parsed": judge_out["parsed"],
            "parse_error": judge_out["parse_error"],
            "duration_s": judge_out["latency_s"],
            "tokens": judge_out["tokens"],
            "llm_ok": judge_out["llm_ok"],
            "llm_error": judge_out["llm_error"],
            "final_saved": False,
        }

        if judge_out["llm_ok"] and judge_out["parsed"] is not None:
            j = judge_out["parsed"]
            if j.get("should_keep"):
                entry["final_saved"] = True
                n_kept += 1
                trace["final_badcases"].append({
                    "candidate": bc,
                    "judge": j,
                })
        trace["stages"]["judge"].append(entry)

    trace["decision"] = "saved" if n_kept > 0 else "rejected"
    trace["n_kept"] = n_kept
    trace["n_candidates"] = len(badcases)
    trace["duration_s"] = round(time.time() - t_start, 3)
    return trace
