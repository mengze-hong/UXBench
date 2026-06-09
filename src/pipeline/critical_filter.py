"""
Critical Filter: Classify saved bad cases into two tracks.

Track A — Service Recovery Assessment (明确失败):
  - User EXPLICITLY expressed dissatisfaction (complaint/repeat/rephrase)
  - The failure is clear and unambiguous
  - Output format: full dialogue up to failure turn → agent response blanked → let diverse models fill
  
Track B — Prediction Model Training (隐性不满):
  - User showed implicit dissatisfaction (abandon/short_dismiss/skip_read) or just disliked
  - The failure is debatable / requires inference
  - Output format: full dialogue with labels for training prediction models

Runs on saved_auto.jsonl, outputs two new files. Does NOT modify any existing files.
"""

import json, sys, io
from pathlib import Path
from collections import Counter
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HERE = Path(__file__).parent
OUTPUTS = HERE.parent / "outputs"
SAVED = OUTPUTS / "saved_auto.jsonl"

# Output files (new, independent)
TRACK_A_FILE = OUTPUTS / "track_a_recovery_assessment.jsonl"
TRACK_B_FILE = OUTPUTS / "track_b_prediction_training.jsonl"
SUMMARY_FILE = OUTPUTS / "track_classification_summary.json"

# Track A criteria: explicit failure signals
EXPLICIT_SIGNALS = {"explicit_complaint", "repeat_query", "rephrase_query", "repeat", "rephrase"}
# Also include dislike + explicit sentiment
EXPLICIT_SENTIMENTS = {"explicit"}

# Track B: everything else (implicit signals)
IMPLICIT_SIGNALS = {"dislike", "abandon", "short_dismiss", "skip_read", "no_followup"}


def classify_record(rec: dict) -> str:
    """Classify a record into Track A or Track B."""
    al = rec.get("auto_label", {})
    signal = al.get("signal_type", "")
    sentiment = al.get("sentiment", "")
    confidence = al.get("confidence", 0)
    judge_avg = al.get("judge_average") or 0
    
    # Track A: Explicit failure
    # 1. Signal is explicit complaint / repeat / rephrase
    if signal in EXPLICIT_SIGNALS:
        return "A"
    
    # 2. Dislike + explicit sentiment (user said something about it)
    if signal == "dislike" and sentiment == "explicit":
        return "A"
    
    # 3. Dislike + very high judge scores (failure is unambiguous)
    if signal == "dislike" and judge_avg >= 4.5 and confidence and float(confidence) >= 0.9:
        return "A"
    
    # Everything else → Track B
    return "B"


def build_track_a_entry(rec: dict) -> dict:
    """
    Build Track A entry: dialogue context up to failure, agent response blanked.
    Format ready for diverse model evaluation.
    """
    al = rec.get("auto_label", {})
    history = rec.get("full_history", [])
    dt_id = al.get("dislike_turn_id")
    try:
        dt_id = int(dt_id) if dt_id is not None else None
    except (ValueError, TypeError):
        dt_id = None
    sq = rec.get("source_query", {})
    
    # Build context: everything up to and including the source query
    context_turns = []
    failed_response = ""
    for m in history:
        if not isinstance(m, dict):
            continue
        try:
            tid = int(m.get("turn_index", -1))
        except (ValueError, TypeError):
            continue
        
        if dt_id is not None and tid == dt_id and m.get("role") == "assistant":
            # This is the failed response — record it but blank it in context
            failed_response = (m.get("message", "") or "")[:3000]
            context_turns.append({
                "turn_index": tid,
                "role": "assistant",
                "content": "[TO_BE_FILLED_BY_MODEL]",
                "_original_response": failed_response,
            })
        elif tid <= (dt_id or 999):
            context_turns.append({
                "turn_index": tid,
                "role": m.get("role", ""),
                "content": (m.get("message", "") or "")[:2000],
            })
    
    # Also include turns after failure for recovery analysis
    recovery_turns = []
    past_failure = False
    for m in history:
        if not isinstance(m, dict):
            continue
        try:
            tid = int(m.get("turn_index", -1))
        except (ValueError, TypeError):
            continue
        if dt_id is not None and tid == dt_id:
            past_failure = True
            continue
        if past_failure:
            recovery_turns.append({
                "turn_index": tid,
                "role": m.get("role", ""),
                "content": (m.get("message", "") or "")[:1000],
            })
            if len(recovery_turns) >= 6:
                break
    
    return {
        "track": "A",
        "cid": rec.get("cid", ""),
        "use_case": "service_recovery_assessment",
        "source_query": sq.get("message", ""),
        "source_query_turn_id": sq.get("turn_index"),
        "failure_turn_id": dt_id,
        "failure_dimension": al.get("failure_dimension", ""),
        "failure_dimension_raw": al.get("failure_dimension_raw", ""),
        "scenario": al.get("scenario", ""),
        "signal_type": al.get("signal_type", ""),
        "sentiment": al.get("sentiment", ""),
        "explanation": al.get("explanation", ""),
        "judge_average": al.get("judge_average", 0),
        "overall_quality": al.get("overall_quality", ""),
        # Context for model evaluation
        "dialogue_context": context_turns,
        "original_failed_response": failed_response,
        "recovery_turns": recovery_turns,
        "has_recovery": len(recovery_turns) > 0,
    }


def build_track_b_entry(rec: dict) -> dict:
    """
    Build Track B entry: full dialogue with behavior labels for prediction training.
    """
    al = rec.get("auto_label", {})
    history = rec.get("full_history", [])
    enrichment = rec.get("enrichment", {})
    sq = rec.get("source_query", {})
    dt_id = al.get("dislike_turn_id")
    try:
        dt_id = int(dt_id) if dt_id is not None else None
    except (ValueError, TypeError):
        dt_id = None
    
    # Build labeled turns
    labeled_turns = []
    reactions = {r["asst_turn_id"]: r for r in enrichment.get("user_reactions", [])}
    
    for m in history:
        if not isinstance(m, dict):
            continue
        try:
            tid = int(m.get("turn_index", -1))
        except (ValueError, TypeError):
            continue
        role = m.get("role", "")
        
        turn_entry = {
            "turn_index": tid,
            "role": role,
            "content": (m.get("message", "") or "")[:2000],
            "is_unliked": int(m.get("is_unliked", 0)) == 1,
        }
        
        # Add reaction label for assistant turns
        if role == "assistant" and tid in reactions:
            r = reactions[tid]
            turn_entry["user_reaction"] = r.get("reaction", "")
            turn_entry["user_dissatisfied"] = r.get("dissatisfied", False)
        
        # Mark the failure turn
        turn_entry["is_failure_turn"] = tid == dt_id
        
        labeled_turns.append(turn_entry)
    
    return {
        "track": "B",
        "cid": rec.get("cid", ""),
        "use_case": "dissatisfaction_prediction_training",
        "source_query": sq.get("message", ""),
        "failure_turn_id": dt_id,
        "failure_dimension": al.get("failure_dimension", ""),
        "scenario": al.get("scenario", ""),
        "signal_type": al.get("signal_type", ""),
        "signal_confidence": al.get("signal_confidence", ""),
        "sentiment": al.get("sentiment", ""),
        "explanation": al.get("explanation", ""),
        "judge_average": al.get("judge_average", 0),
        "overall_quality": al.get("overall_quality", ""),
        "labeled_turns": labeled_turns,
        "reaction_summary": enrichment.get("reaction_summary", {}),
    }


def main():
    records = [json.loads(l) for l in open(SAVED, "r", encoding="utf-8") if l.strip()]
    print(f"Loaded {len(records)} saved bad cases")
    
    track_a = []
    track_b = []
    
    for rec in records:
        track = classify_record(rec)
        if track == "A":
            track_a.append(build_track_a_entry(rec))
        else:
            track_b.append(build_track_b_entry(rec))
    
    # Write outputs
    with open(TRACK_A_FILE, "w", encoding="utf-8") as f:
        for r in track_a:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    with open(TRACK_B_FILE, "w", encoding="utf-8") as f:
        for r in track_b:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    # Stats
    a_dims = Counter(r["failure_dimension"] for r in track_a)
    b_dims = Counter(r["failure_dimension"] for r in track_b)
    a_signals = Counter(r["signal_type"] for r in track_a)
    b_signals = Counter(r["signal_type"] for r in track_b)
    a_quality = Counter(r["overall_quality"] for r in track_a)
    b_quality = Counter(r["overall_quality"] for r in track_b)
    a_recovery = sum(1 for r in track_a if r["has_recovery"])
    
    summary = {
        "total": len(records),
        "track_a_count": len(track_a),
        "track_b_count": len(track_b),
        "track_a_pct": round(len(track_a) / len(records) * 100, 1),
        "track_b_pct": round(len(track_b) / len(records) * 100, 1),
        "track_a_with_recovery": a_recovery,
        "track_a_dimensions": dict(a_dims.most_common()),
        "track_b_dimensions": dict(b_dims.most_common()),
        "track_a_signals": dict(a_signals.most_common()),
        "track_b_signals": dict(b_signals.most_common()),
        "track_a_quality": dict(a_quality.most_common()),
        "track_b_quality": dict(b_quality.most_common()),
    }
    
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Track A — Service Recovery Assessment (明确失败)")
    print(f"  Cases: {len(track_a)} ({summary['track_a_pct']}%)")
    print(f"  With recovery turns: {a_recovery}")
    print(f"  Quality: {dict(a_quality.most_common())}")
    print(f"  Signals: {dict(a_signals.most_common())}")
    print(f"  Top dims: {dict(a_dims.most_common(5))}")
    print()
    print(f"Track B — Prediction Model Training (隐性不满)")
    print(f"  Cases: {len(track_b)} ({summary['track_b_pct']}%)")
    print(f"  Quality: {dict(b_quality.most_common())}")
    print(f"  Signals: {dict(b_signals.most_common())}")
    print(f"  Top dims: {dict(b_dims.most_common(5))}")
    print(f"{'='*60}")
    print(f"\nOutput files:")
    print(f"  {TRACK_A_FILE}")
    print(f"  {TRACK_B_FILE}")
    print(f"  {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
