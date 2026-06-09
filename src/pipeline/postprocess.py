"""
Post-processing: Transform saved_auto.jsonl → three downstream JSONL files.

1) uxbench_testcases.jsonl     — for evaluating different models (U2)
2) uxbench_behavior_labels.jsonl — for training dissatisfaction prediction (U1)
3) uxbench_recovery_chains.jsonl — for studying failure recovery (U3)

Pure Python, no LLM calls. Fast.
"""

import json
import sys
import io
from pathlib import Path
from datetime import datetime
from collections import Counter

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HERE = Path(__file__).parent
ROOT = HERE.parent
OUTPUTS = ROOT / "outputs"
SAVED_FILE = OUTPUTS / "saved_auto.jsonl"

# Output files
TC_FILE = OUTPUTS / "uxbench_testcases.jsonl"
BL_FILE = OUTPUTS / "uxbench_behavior_labels.jsonl"
RC_FILE = OUTPUTS / "uxbench_recovery_chains.jsonl"


def load_saved() -> list:
    if not SAVED_FILE.exists():
        return []
    lines = []
    with open(SAVED_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    lines.append(json.loads(line))
                except Exception:
                    pass
    return lines


def get_msg_by_turn(history: list, turn_id) -> dict:
    """Find message by turn_index in history."""
    for m in history:
        if m.get("turn_index") == turn_id:
            return m
    return {}


def extract_context_before(history: list, turn_id) -> list:
    """Get all turns strictly before turn_id as context."""
    ctx = []
    for m in history:
        if m.get("turn_index", 0) >= turn_id:
            break
        ctx.append({
            "role": m.get("role", ""),
            "content": (m.get("message", "") or "")[:2000],
            "turn_index": m.get("turn_index"),
        })
    return ctx


def generate_testcases(records: list) -> list:
    """
    U2: Generate benchmark test cases.
    Each bad case becomes one test case with source_query + context + reference_response.
    """
    testcases = []
    tc_id = 0
    for rec in records:
        cid = rec.get("cid", "")
        history = rec.get("full_history", [])
        if isinstance(history, str):
            history = json.loads(history)
        auto = rec.get("auto_label", {})

        # Source query
        sq = rec.get("source_query", {})
        sq_turn = sq.get("turn_index", -1)
        sq_text = sq.get("message", "")

        # Dislike turn (the failed AI response)
        dt_id = auto.get("dislike_turn_id")
        dt_msg = get_msg_by_turn(history, dt_id)
        agent_response = (dt_msg.get("message", "") or "")[:3000]

        if not sq_text or not agent_response:
            continue

        # Context before source query
        context = extract_context_before(history, sq_turn)

        tc_id += 1
        testcases.append({
            "id": f"tc_{tc_id:05d}",
            "cid": cid,
            "source_query": sq_text,
            "source_query_turn_id": sq_turn,
            "history_context": context,
            "reference_response": agent_response,
            "reference_response_turn_id": dt_id,
            "failure_dimension": auto.get("failure_dimension", ""),
            "failure_explanation": auto.get("explanation", ""),
            "scenario": auto.get("scenario", ""),
            "signal_type": auto.get("signal_type", "dislike"),
            "signal_confidence": auto.get("signal_confidence", "high"),
            "sentiment": auto.get("sentiment", ""),
            "quality_tier": auto.get("overall_quality", ""),
            "judge_average": auto.get("judge_average", 0),
            "representativeness": auto.get("representativeness", ""),
        })

    return testcases


def generate_behavior_labels(records: list) -> list:
    """
    U1: Generate per-AI-turn behavior labels.
    For EVERY assistant turn in saved dialogues, label whether user was dissatisfied.
    """
    labels = []
    bl_id = 0

    # Collect info about which turns are failures
    for rec in records:
        cid = rec.get("cid", "")
        history = rec.get("full_history", [])
        if isinstance(history, str):
            history = json.loads(history)
        auto = rec.get("auto_label", {})
        enrichment = rec.get("enrichment", {})

        dt_id = auto.get("dislike_turn_id")
        reactions = enrichment.get("user_reactions", [])
        reaction_by_asst = {r["asst_turn_id"]: r for r in reactions}

        for i, msg in enumerate(history):
            if msg.get("role") != "assistant":
                continue

            tid = msg.get("turn_index")
            agent_text = (msg.get("message", "") or "")[:2000]

            # Find preceding user query
            prev_query = ""
            for j in range(i - 1, -1, -1):
                if history[j].get("role") == "user":
                    prev_query = (history[j].get("message", "") or "")[:1000]
                    break

            # Context before this turn
            context = extract_context_before(history, tid)

            # Determine satisfaction
            is_unliked = int(msg.get("is_unliked", 0)) == 1
            reaction = reaction_by_asst.get(tid, {})
            dissatisfied = reaction.get("dissatisfied", False) or is_unliked
            reaction_type = reaction.get("reaction", "unknown")
            reaction_details = reaction.get("details", "")

            # Is this the specifically flagged failure turn?
            is_flagged_failure = tid == dt_id

            signals = []
            if is_unliked:
                signals.append("dislike")
            if reaction_type == "complain":
                signals.append("explicit_complaint")
            if reaction_type in ("repeat", "rephrase", "short_dismiss", "abandon", "skip_read"):
                signals.append(reaction_type)

            bl_id += 1
            labels.append({
                "id": f"bl_{bl_id:05d}",
                "cid": cid,
                "turn_id": tid,
                "agent_response": agent_text,
                "preceding_query": prev_query,
                "history_context": context[-6:],  # Last 3 turns of context to save space
                "user_satisfied": not dissatisfied,
                "dissatisfaction_signals": signals,
                "user_reaction": reaction_type,
                "reaction_details": reaction_details,
                "is_unliked": is_unliked,
                "is_flagged_failure": is_flagged_failure,
                "failure_dimension": auto.get("failure_dimension", "") if is_flagged_failure else "",
            })

    return labels


def generate_recovery_chains(records: list) -> list:
    """
    U3: Generate failure recovery chain records.
    """
    chains = []
    rc_id = 0

    for rec in records:
        cid = rec.get("cid", "")
        history = rec.get("full_history", [])
        if isinstance(history, str):
            history = json.loads(history)
        auto = rec.get("auto_label", {})
        enrichment = rec.get("enrichment", {})

        rec_chains = enrichment.get("recovery_chains", [])
        if not rec_chains:
            continue

        for chain in rec_chains:
            fail_tid = chain["failure_turn_id"]
            fail_msg = get_msg_by_turn(history, fail_tid)
            if not fail_msg:
                continue

            rc_id += 1
            chains.append({
                "id": f"rc_{rc_id:05d}",
                "cid": cid,
                "failure_turn_id": fail_tid,
                "failure_response": (fail_msg.get("message", "") or "")[:2000],
                "failure_reaction": chain.get("failure_reaction", ""),
                "failure_dimension": auto.get("failure_dimension", "") if fail_tid == auto.get("dislike_turn_id") else "",
                "recovery_chain": chain.get("recovery_chain", []),
                "recovery_outcome": chain.get("recovery_outcome", ""),
                "chain_length": chain.get("chain_length", 0),
            })

    return chains


def write_jsonl(path: Path, records: list):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    records = load_saved()
    print(f"Loaded {len(records)} saved records from {SAVED_FILE}")

    if not records:
        print("No records to process.")
        return

    # Generate
    testcases = generate_testcases(records)
    behavior_labels = generate_behavior_labels(records)
    recovery_chains = generate_recovery_chains(records)

    # Write
    write_jsonl(TC_FILE, testcases)
    write_jsonl(BL_FILE, behavior_labels)
    write_jsonl(RC_FILE, recovery_chains)

    # Stats
    print(f"\n{'='*60}")
    print(f"U2: uxbench_testcases.jsonl       → {len(testcases)} test cases")
    print(f"U1: uxbench_behavior_labels.jsonl  → {len(behavior_labels)} turn labels")
    print(f"U3: uxbench_recovery_chains.jsonl  → {len(recovery_chains)} recovery chains")
    print(f"{'='*60}")

    if testcases:
        dims = Counter(t["failure_dimension"] for t in testcases)
        signals = Counter(t["signal_type"] for t in testcases)
        tiers = Counter(t["quality_tier"] for t in testcases)
        print(f"\nTestcase dimensions: {dict(dims)}")
        print(f"Testcase signal types: {dict(signals)}")
        print(f"Testcase quality tiers: {dict(tiers)}")

    if behavior_labels:
        sat = sum(1 for b in behavior_labels if b["user_satisfied"])
        dis = sum(1 for b in behavior_labels if not b["user_satisfied"])
        print(f"\nBehavior labels: {sat} satisfied / {dis} dissatisfied")

    if recovery_chains:
        outcomes = Counter(c["recovery_outcome"] for c in recovery_chains)
        print(f"\nRecovery outcomes: {dict(outcomes)}")


if __name__ == "__main__":
    main()
