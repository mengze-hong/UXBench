#!/usr/bin/env python3
"""
run_eval.py  —  UXBench evaluation runner.

Task 1 · UX Judge:
    Judges pre-existing responses in the test set using any OpenAI-compatible model.
    The model acts as a pointwise judge (outputs "好"/"差").

Task 2 · UX Eval  (response generation only):
    Generates responses for each query in the test set.
    Scoring is done separately via: scripts/grm_judge/run_grm_judge_task2.py

Task 3 · UX Recovery  (recovery generation only):
    Generates recovery responses for each failed interaction.
    Scoring is done separately via: scripts/grm_judge/run_grm_judge_task3.py

Usage:
    # Task 1 — judge pre-existing responses
    python scripts/run_eval.py --task task1_ux_judge --model claude-opus-4.7

    # Task 2 — generate responses (run GRM judge separately after)
    python scripts/run_eval.py --task task2_ux_eval --model claude-opus-4.7 --workers 10

    # Task 3 — generate recovery responses (run GRM judge separately after)
    python scripts/run_eval.py --task task3_ux_recovery --model claude-opus-4.7

    # Dry run (10 samples only)
    python scripts/run_eval.py --task task1_ux_judge --model claude-opus-4.7 --dry-run

Full pipeline for Task 2 / Task 3:
    Step 1 — generate:
        python scripts/run_eval.py --task task2_ux_eval --model <model>
        # outputs: experiments/results/task2/<model>/results.jsonl

    Step 2 — GRM judge (requires local vLLM GRM server, see scripts/grm_judge/README.md):
        cp experiments/results/task2/<model>/results.jsonl \\
           experiments/results/task2/responses/<model_key>.jsonl
        python scripts/grm_judge/run_grm_judge_task2.py --models <model_key>
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

# Add src to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from utils.config import get_route, API_KEY, API_URL  # noqa: F401
from utils.data_loader import load_jsonl
from utils.llm_client import call_llm
from utils.prompts import build_judge_prompt, extract_verdict, POINTWISE_GRM
from utils.checkpoint import load_done_cids, append_record


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="UXBench evaluation runner")
    p.add_argument("--task",    required=True,
                   choices=["task1_ux_judge", "task2_ux_eval", "task3_ux_recovery"])
    p.add_argument("--model",   required=True,
                   help="Model name as used in API calls (e.g. claude-opus-4.7)")
    p.add_argument("--config",  default="experiments/configs/eval_config.yaml")
    p.add_argument("--workers", type=int, default=5,
                   help="Number of parallel LLM calls (default: 5)")
    p.add_argument("--output",  default=None,
                   help="Override output directory")
    p.add_argument("--dry-run", action="store_true",
                   help="Run on first 10 samples only (sanity check)")
    return p.parse_args()


def load_config(config_path: str) -> dict:
    import yaml
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Shared helpers ────────────────────────────────────────────────────────────

def _run_parallel(samples, worker_fn, out_file, lock, workers, dry_run):
    """Generic parallel loop with progress printing and incremental writes."""
    if dry_run:
        samples = samples[:10]
        print(f"  [dry-run] capped at {len(samples)} samples")

    total     = len(samples)
    completed = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(worker_fn, s): s for s in samples}
        for future in as_completed(futures):
            record = future.result()
            append_record(out_file, record, lock=lock)
            completed += 1
            if completed % 100 == 0 or completed == total:
                print(f"    {completed}/{total} done")

    return load_jsonl(out_file)


# ── Task 1 · UX Judge ────────────────────────────────────────────────────────

def _judge_task1(sample: dict, model: str) -> dict:
    """
    Judge one Task 1 sample using the Pointwise GRM prompt.

    Data fields used:
        selected_history    : prior conversation turns {role, message/content, ...}
        source_query        : {turn_index, message} — the current user query
        agent_response_full : AI response to evaluate (pre-existing in test set)
        ground_truth        : -1 (bad) or 1 (good) — already in data
    """
    history  = sample.get("selected_history", [])
    sq       = sample.get("source_query", {})
    query    = sq.get("message", "") if isinstance(sq, dict) else str(sq)
    response = sample.get("agent_response_full", "")

    prompt_text = build_judge_prompt(
        history=history,
        query=query,
        response=response,
        template=POINTWISE_GRM,
    )
    result = call_llm([{"role": "user", "content": prompt_text}], model=model)

    if result.ok:
        verdict, parse_note = extract_verdict(result.content, result.reasoning_content)
    else:
        verdict, parse_note = None, "__llm_error__"

    return {
        **sample,
        "predicted":  verdict,
        "raw_output": result.content if result.ok else result.error,
        "parse_note": parse_note,
        "ok":         verdict is not None,
    }


def run_task1(args, cfg: dict):
    """
    Task 1: UX Judge.

    Judges whether each pre-existing AI response in the test set is
    good (1) or bad (-1). Uses POINTWISE_GRM prompt with the specified model.

    Metric: Avg-Acc = (Good-Acc + Bad-Acc) / 2
    """
    print(f"[Task 1] UX Judge · model={args.model} · workers={args.workers}")

    bad_path   = ROOT / cfg["task1_ux_judge"]["testset"]
    good_path  = ROOT / cfg["task1_ux_judge"]["testset_good"]
    output_dir = Path(args.output or ROOT / cfg["task1_ux_judge"]["output_dir"]) / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file   = output_dir / "results.jsonl"

    bad_samples  = load_jsonl(bad_path)
    good_samples = load_jsonl(good_path)

    # ground_truth is already in the data; only set as fallback if missing
    for s in bad_samples:
        if s.get("ground_truth") is None:
            s["ground_truth"] = -1
    for s in good_samples:
        if s.get("ground_truth") is None:
            s["ground_truth"] = 1

    all_samples = bad_samples + good_samples

    # Resume: skip already-completed cids
    done_cids = load_done_cids(out_file)
    if done_cids:
        before      = len(all_samples)
        all_samples = [s for s in all_samples if s.get("cid") not in done_cids]
        print(f"  Resume: skipped {before - len(all_samples)} / {before} (already done)")

    n_bad  = sum(1 for s in all_samples if s.get("ground_truth") == -1)
    n_good = sum(1 for s in all_samples if s.get("ground_truth") ==  1)
    print(f"  Evaluating {len(all_samples)} samples ({n_bad} bad + {n_good} good)...")

    lock   = Lock()
    worker = lambda s: _judge_task1(s, model=args.model)
    all_results = _run_parallel(all_samples, worker, out_file, lock, args.workers, args.dry_run)

    # Metrics
    bad_results  = [r for r in all_results if r.get("ground_truth") == -1]
    good_results = [r for r in all_results if r.get("ground_truth") ==  1]

    bad_acc  = sum(1 for r in bad_results  if r.get("predicted") == -1) / max(len(bad_results),  1) * 100
    good_acc = sum(1 for r in good_results if r.get("predicted") ==  1) / max(len(good_results), 1) * 100
    avg_acc  = (bad_acc + good_acc) / 2

    summary = {
        "model":    args.model,
        "task":     "task1_ux_judge",
        "n_bad":    len(bad_results),
        "n_good":   len(good_results),
        "good_acc": round(good_acc, 2),
        "bad_acc":  round(bad_acc,  2),
        "avg_acc":  round(avg_acc,  2),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Results → {output_dir}")
    print(f"  Good-Acc: {good_acc:.1f}%  Bad-Acc: {bad_acc:.1f}%  Avg-Acc: {avg_acc:.1f}%")
    return summary


# ── Task 2 · UX Eval (generation only) ───────────────────────────────────────

def _generate_task2(sample: dict, model: str) -> dict:
    """
    Generate a response for one Task 2 sample.

    Data fields used:
        messages : full conversation context (system + prior turns)
        query    : the final user query the model must respond to

    NOTE: Scoring (Good%) is done separately by the Pointwise GRM.
          See scripts/grm_judge/run_grm_judge_task2.py
    """
    messages = sample.get("messages", [])
    query    = sample.get("query", "")

    result = call_llm(messages, model=model, temperature=0.7, max_tokens=2048)

    return {
        **sample,
        "model":              model,
        "generated_response": result.content if result.ok else None,
        "ok":                 result.ok,
        "error":              result.error if not result.ok else None,
    }


def run_task2(args, cfg: dict):
    """
    Task 2: UX Eval — response generation.

    Generates a response for each query in the test set.
    Output is saved to experiments/results/task2/<model>/results.jsonl

    To score the generated responses, run the GRM judge next:
        cp experiments/results/task2/<model>/results.jsonl \\
           experiments/results/task2/responses/<model_key>.jsonl
        python scripts/grm_judge/run_grm_judge_task2.py --models <model_key>
    """
    print(f"[Task 2] UX Eval · model={args.model} · workers={args.workers}")
    print( "         (generation only — run GRM judge separately for scoring)")

    testset_path = ROOT / cfg["task2_ux_eval"]["testset"]
    output_dir   = Path(args.output or ROOT / cfg["task2_ux_eval"]["output_dir"]) / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file     = output_dir / "results.jsonl"

    samples   = load_jsonl(testset_path)
    done_cids = load_done_cids(out_file)
    if done_cids:
        before  = len(samples)
        samples = [s for s in samples if s.get("cid") not in done_cids]
        print(f"  Resume: skipped {before - len(samples)} / {before}")

    print(f"  Generating responses for {len(samples)} samples...")

    lock   = Lock()
    worker = lambda s: _generate_task2(s, model=args.model)
    results = _run_parallel(samples, worker, out_file, lock, args.workers, args.dry_run)

    n_ok   = sum(1 for r in results if r.get("ok"))
    n_fail = len(results) - n_ok

    print(f"\n  Results → {out_file}")
    print(f"  Generated: {n_ok} ok, {n_fail} failed")
    print( "  Next step: run GRM judge — see scripts/grm_judge/README.md")


# ── Task 3 · UX Recovery (generation only) ───────────────────────────────────

def _generate_task3(sample: dict, model: str) -> dict:
    """
    Generate a recovery response for one Task 3 sample.

    Data fields used:
        history         : full dialogue history up to the failure point
        user_complaint  : the user's complaint/dissatisfaction message
        failed_response : the original bad AI response (context for recovery)

    NOTE: Scoring (Recovery Rate) is done separately by the Pointwise GRM.
          See scripts/grm_judge/run_grm_judge_task3.py
    """
    history       = sample.get("history", [])
    failed        = sample.get("failed_response", "")
    complaint     = sample.get("user_complaint", "")

    # No system prompt — mirrors the original experimental setup.
    # Build: history turns → failed response (assistant) → user complaint.
    messages = []
    for turn in history:
        role    = turn.get("role", "user")
        content = turn.get("message") or turn.get("content", "")
        messages.append({"role": role, "content": content})
    if failed:
        messages.append({"role": "assistant", "content": failed})
    messages.append({"role": "user", "content": complaint})

    result = call_llm(messages, model=model, temperature=0.7, max_tokens=2048)

    return {
        **sample,
        "model":             model,
        "recovery_response": result.content if result.ok else None,
        "ok":                result.ok,
        "error":             result.error if not result.ok else None,
    }


def run_task3(args, cfg: dict):
    """
    Task 3: UX Recovery — recovery response generation.

    Generates a recovery response for each failed interaction in the test set.
    Output is saved to experiments/results/task3/<model>/results.jsonl

    To score the generated responses, run the GRM judge next:
        cp experiments/results/task3/<model>/results.jsonl \\
           experiments/results/task3/responses/<model_key>.jsonl
        python scripts/grm_judge/run_grm_judge_task3.py --models <model_key>
    """
    print(f"[Task 3] UX Recovery · model={args.model} · workers={args.workers}")
    print( "         (generation only — run GRM judge separately for scoring)")

    testset_path = ROOT / cfg["task3_ux_recovery"]["testset"]
    output_dir   = Path(args.output or ROOT / cfg["task3_ux_recovery"]["output_dir"]) / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file     = output_dir / "results.jsonl"

    samples   = load_jsonl(testset_path)
    done_cids = load_done_cids(out_file)
    if done_cids:
        before  = len(samples)
        samples = [s for s in samples if s.get("cid") not in done_cids]
        print(f"  Resume: skipped {before - len(samples)} / {before}")

    print(f"  Generating recovery responses for {len(samples)} samples...")

    lock   = Lock()
    worker = lambda s: _generate_task3(s, model=args.model)
    results = _run_parallel(samples, worker, out_file, lock, args.workers, args.dry_run)

    n_ok   = sum(1 for r in results if r.get("ok"))
    n_fail = len(results) - n_ok

    print(f"\n  Results → {out_file}")
    print(f"  Generated: {n_ok} ok, {n_fail} failed")
    print( "  Next step: run GRM judge — see scripts/grm_judge/README.md")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg  = load_config(args.config)

    if args.task == "task1_ux_judge":
        run_task1(args, cfg)
    elif args.task == "task2_ux_eval":
        run_task2(args, cfg)
    elif args.task == "task3_ux_recovery":
        run_task3(args, cfg)


if __name__ == "__main__":
    main()
