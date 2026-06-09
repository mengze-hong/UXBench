"""
run_main_judge.py
=================
Run Pointwise GRM judge on Task 2 (UX Eval) model responses.

For each model, reads generated responses from experiments/results/task2/responses/<model>.jsonl
and scores them using the locally-served Pointwise GRM (vLLM).

Requires the GRM model to be served locally. See docs/GRM_SETUP.md for setup instructions.
推理配置（prompt template / payload / logprobs 解析 / score 公式 / verdict / 截断重试）
与 ablation_grm_judge/run_ablation_judge.py 完全一致。

Data format:
  - testset: uxbench_task2_eval_4900.jsonl — field "history" is list[dict] with {role, content/message}
  - responses/<model>.jsonl — fields: {cid, model, generated_response, ...}

Usage:
    python scripts/grm_judge/run_grm_judge_task2.py
    python scripts/grm_judge/run_grm_judge_task2.py --endpoint http://localhost:8021/v1/chat/completions
    python scripts/grm_judge/run_grm_judge_task2.py --workers 80 --smoke 10
    python scripts/grm_judge/run_grm_judge_task2.py --models claude_opus_47 glm_5_1
"""
import argparse
import ast
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock

import requests

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ── 路径 ─────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
TESTSET_FILE = SCRIPT_DIR.parent.parent / "data" / "uxbench_task2_eval_4900.jsonl"
RESP_DIR = SCRIPT_DIR.parent.parent / "experiments" / "results" / "task2" / "responses"
RESULTS_DIR = SCRIPT_DIR.parent.parent / "experiments" / "results" / "task2" / "grm_judge"
LOG_DIR = RESULTS_DIR / "logs"


# ── Pointwise GRM 配置 ───────────────────────────────────────────────
GRM_MODEL = "pointwise_grm_ux_v031"
GRM_AUTH_TOKEN = os.environ.get("GRM_AUTH_TOKEN", "")
GRM_PORT = 8021

# no_proxy: set via environment if needed
# e.g. export no_proxy="localhost,127.0.0.1"

# ── Pointwise GRM Prompt Template ────────────────────────────────────
POINTWISE_GRM_TEMPLATE = """# Role
你是一位用户体验评估专家。你的任务是判断AI助手的回复是否能够令用户满意。

# Task
请根据以下对话上下文和AI助手的回复，判断该回复是否能让用户满意。
评估时应考虑：回复是否准确、完整、有帮助，是否正确理解了用户意图，表达是否清晰恰当。

# Input Data
## 历史对话
{context}
## 用户问题
{prompt}
## AI助手回复
{response_a}

# Evaluation Criteria
1. 正确性：回复是否包含事实性错误
2. 完整性：回复是否充分回答了用户的问题
3. 意图理解：AI是否正确理解了用户的需求
4. 表达质量：回复是否清晰、简洁、格式恰当（不冗余啰嗦）
5. 实用性：回复对用户是否有实际帮助

# Output Format
请仅输出一个字："好" 或 "差"。
- 输出"好"代表：该回复能够令用户满意
- 输出"差"代表：该回复无法令用户满意

# Constraint
不要输出任何解释、分析或额外的标点符号，只输出最终的判定结果。

# Final Answer"""


# ── 全局 logger ──────────────────────────────────────────────────────
_LOG_FH = None
_LOG_LOCK = Lock()


def log(msg: str):
    """Log to both stdout and run log file."""
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    if _LOG_FH is not None:
        with _LOG_LOCK:
            _LOG_FH.write(line + "\n")
            _LOG_FH.flush()


# ── GRM 调用 ─────────────────────────────────────────────────────────
@dataclass
class GRMResult:
    ok: bool
    score: float = None
    content: str = ""
    good_logprob: float = None
    bad_logprob: float = None
    latency_s: float = 0.0
    error: str = ""


def _parse_history(raw):
    """Parse history field: handles list[dict] or JSON string."""
    if not raw:
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return []
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
    return []


def build_grm_prompt(query_rec: dict, response_text: str,
                     max_history_turns: int = 0,
                     max_response_chars: int = 0) -> str:
    """Build the Pointwise GRM judge prompt for a single sample."""
    history = _parse_history(query_rec.get("history", []))
    user_query = query_rec.get("query", "") or ""

    if max_history_turns > 0 and len(history) > max_history_turns:
        history = history[-max_history_turns:]
    if max_response_chars > 0 and len(response_text) > max_response_chars:
        response_text = response_text[:max_response_chars] + "...[截断]"

    context_lines = []
    for turn in history:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role", "")
        msg = turn.get("content", turn.get("message", "")) or ""
        role_label = "用户" if role == "user" else "AI"
        context_lines.append(f"{role_label}：{msg}")
    context = "\n".join(context_lines) if context_lines else "无"

    return POINTWISE_GRM_TEMPLATE.format(
        context=context,
        prompt=user_query,
        response_a=response_text,
    )


def call_pointwise_grm(prompt: str, endpoint: str, model_name: str = GRM_MODEL,
                       max_retries: int = 3, timeout: int = 120) -> GRMResult:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GRM_AUTH_TOKEN}",
    }
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 4,
        "output_seq_len": 4,
        "stream": False,
        "logprobs": True,
        "top_logprobs": 5,
        "enable_thinking": False,
        "enable_enhancement": False,
    }

    last_err = ""
    t0 = time.time()
    for attempt in range(1, max_retries + 1):
        t0 = time.time()
        try:
            resp = requests.post(endpoint, headers=headers, json=payload,
                                 timeout=timeout, verify=False)
            latency = time.time() - t0

            if resp.status_code == 200:
                data = resp.json()
                choices = data.get("choices", [])
                if not choices:
                    return GRMResult(ok=False, latency_s=latency, error="no_choices")

                choice = choices[0]
                msg = choice.get("message", {})
                content = (msg.get("content") or msg.get("reasoning_content") or "").strip()

                # Parse logprobs (skip whitespace-only token positions).
                good_logprob = None
                bad_logprob = None
                logprobs_data = choice.get("logprobs", {})
                if logprobs_data and "content" in logprobs_data:
                    for token_info in logprobs_data["content"]:
                        emitted = (token_info.get("token") or "").strip()
                        if not emitted:
                            continue
                        top_logprobs = token_info.get("top_logprobs", [])
                        for tlp in top_logprobs:
                            token = tlp.get("token", "").strip()
                            if token == "好" and good_logprob is None:
                                good_logprob = tlp["logprob"]
                            elif token == "差" and bad_logprob is None:
                                bad_logprob = tlp["logprob"]
                        if good_logprob is not None or bad_logprob is not None:
                            break

                score = None
                if good_logprob is not None and bad_logprob is not None:
                    good_prob = math.exp(good_logprob)
                    bad_prob = math.exp(bad_logprob)
                    score = good_prob / (good_prob + bad_prob)
                elif good_logprob is not None:
                    score = math.exp(good_logprob)
                elif bad_logprob is not None:
                    score = 1.0 - math.exp(bad_logprob)

                return GRMResult(
                    ok=True, score=score, content=content,
                    good_logprob=good_logprob, bad_logprob=bad_logprob,
                    latency_s=latency,
                )

            elif resp.status_code == 429:
                last_err = f"rate_limited_{resp.status_code}"
                time.sleep(2 * attempt + 1)
            else:
                last_err = f"http_{resp.status_code}: {resp.text[:200]}"
                if resp.status_code == 400:
                    return GRMResult(ok=False, latency_s=time.time() - t0, error=last_err)
                time.sleep(1.5 * attempt)

        except requests.Timeout:
            last_err = "timeout"
            time.sleep(2 * attempt)
        except Exception as e:
            last_err = f"exception: {str(e)[:150]}"
            time.sleep(1.5 * attempt)

    return GRMResult(ok=False, latency_s=time.time() - t0, error=last_err)


# ── 单条 judge（带截断重试）─────────────────────────────────────────
def judge_one(cid: str, query_rec: dict, resp_text: str,
              model_tag: str, endpoint: str, grm_model_name: str) -> dict:
    truncation_levels = [
        (0, 0),
        (6, 4000),
        (3, 2000),
    ]

    result = None
    for max_hist, max_resp in truncation_levels:
        prompt = build_grm_prompt(
            query_rec, resp_text,
            max_history_turns=max_hist, max_response_chars=max_resp,
        )
        result = call_pointwise_grm(prompt=prompt, endpoint=endpoint, model_name=grm_model_name)

        if not result.ok and result.error and "http_400" in result.error:
            continue
        break

    verdict = 0
    if result and result.ok:
        c = (result.content or "").strip()
        if c and c[0] == "好":
            verdict = 1
        elif c and c[0] == "差":
            verdict = -1

    return {
        "cid": cid,
        "judged_model": model_tag,
        "verdict": verdict,
        "score": result.score if result and result.ok else None,
        "content": result.content if result and result.ok else "",
        "good_logprob": result.good_logprob if result and result.ok else None,
        "bad_logprob": result.bad_logprob if result and result.ok else None,
        "grm_ok": result.ok if result else False,
        "latency_s": result.latency_s if result else 0,
        "error": result.error if result else "no_result",
    }


# ── 单模型处理 ───────────────────────────────────────────────────────
def judge_model(model_tag: str, cid2query: dict, endpoint: str,
                grm_model_name: str, workers: int, smoke: int) -> dict:
    resp_path = RESP_DIR / f"{model_tag}.jsonl"
    out_path = RESULTS_DIR / f"judge_{model_tag}.jsonl"

    # Load model responses
    responses = {}
    with open(resp_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            responses[d["cid"]] = d

    # Load existing results for resume
    done_cids = set()
    results = []
    if out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                    done_cids.add(r["cid"])
                    results.append(r)
                except Exception:
                    pass

    # Build todo list (skip already-done)
    todo = []
    skipped_no_resp = 0
    skipped_no_query = 0
    for cid, resp_rec in sorted(responses.items()):
        if cid in done_cids:
            continue
        query_rec = cid2query.get(cid)
        if not query_rec:
            skipped_no_query += 1
            continue
        resp_text = resp_rec.get("generated_response", "") or ""
        if not resp_text.strip():
            skipped_no_resp += 1
            continue
        todo.append((cid, query_rec, resp_text))

    if smoke > 0:
        todo = todo[:smoke]

    if skipped_no_query or skipped_no_resp:
        log(f"  [{model_tag}] skipped: no_query={skipped_no_query} no_resp={skipped_no_resp}")

    if not todo:
        log(f"  [{model_tag}] All done ({len(results)} records)")
        return compute_model_metrics(results, model_tag)

    log(f"  [{model_tag}] {len(todo)} to judge (done={len(done_cids)}, workers={workers})")
    t0 = time.time()
    lock = Lock()
    good_count = [0]
    bad_count = [0]
    fail_count = [0]

    fout = open(out_path, "a", encoding="utf-8")

    def do_one(item):
        cid, query_rec, resp_text = item
        rec = judge_one(cid, query_rec, resp_text, model_tag, endpoint, grm_model_name)

        with lock:
            results.append(rec)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            if rec["verdict"] == 1:
                good_count[0] += 1
            elif rec["verdict"] == -1:
                bad_count[0] += 1
            else:
                fail_count[0] += 1

            total_done = good_count[0] + bad_count[0] + fail_count[0]
            if total_done % 200 == 0:
                elapsed = time.time() - t0
                speed = total_done / elapsed if elapsed > 0 else 0
                valid = good_count[0] + bad_count[0]
                good_pct = (good_count[0] / valid) if valid else 0
                remaining = (len(todo) - total_done) / speed if speed > 0 else 0
                log(f"    [{model_tag}] {total_done}/{len(todo)}  "
                    f"good={good_count[0]} bad={bad_count[0]} fail={fail_count[0]}  "
                    f"good%={good_pct:.1%}  {speed:.1f}/s  ETA≈{remaining:.0f}s")

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(do_one, item) for item in todo]
            for fut in as_completed(futs):
                try:
                    fut.result()
                except Exception as e:
                    with lock:
                        fail_count[0] += 1
                        log(f"    [{model_tag}][worker exception] {e}")
    finally:
        fout.close()

    elapsed = time.time() - t0
    log(f"  [{model_tag}] DONE: {len(todo)} in {elapsed:.1f}s, "
        f"good={good_count[0]} bad={bad_count[0]} fail={fail_count[0]}")
    return compute_model_metrics(results, model_tag)


# ── 指标计算 ─────────────────────────────────────────────────────────
def compute_model_metrics(results: list, model_tag: str) -> dict:
    total = len(results)
    good_results = [r for r in results if r.get("verdict") == 1]
    bad_results = [r for r in results if r.get("verdict") == -1]
    fail_results = [r for r in results if r.get("verdict", 0) == 0]

    n_good = len(good_results)
    n_bad = len(bad_results)
    n_fail = len(fail_results)
    n_valid = n_good + n_bad

    if n_valid == 0:
        return {
            "model": model_tag,
            "n_total": total,
            "n_valid": 0,
            "n_fail": n_fail,
            "error": "no_valid",
        }

    good_pct = n_good / n_valid
    bad_pct = n_bad / n_valid

    scores = [r["score"] for r in results if r.get("score") is not None]
    score_mean = sum(scores) / len(scores) if scores else None
    score_p50 = None
    if scores:
        ss = sorted(scores)
        score_p50 = ss[len(ss) // 2]

    good_scores = [r["score"] for r in good_results if r.get("score") is not None]
    bad_scores = [r["score"] for r in bad_results if r.get("score") is not None]

    latencies = [r["latency_s"] for r in results if r.get("latency_s", 0) > 0]
    avg_lat = sum(latencies) / len(latencies) if latencies else 0

    return {
        "model": model_tag,
        "n_total": total,
        "n_valid": n_valid,
        "n_fail": n_fail,
        "n_good": n_good,
        "n_bad": n_bad,
        "good_pct": round(good_pct, 4),
        "bad_pct": round(bad_pct, 4),
        "score_mean": round(score_mean, 4) if score_mean is not None else None,
        "score_p50": round(score_p50, 4) if score_p50 is not None else None,
        "good_score_mean": round(sum(good_scores) / len(good_scores), 4) if good_scores else None,
        "bad_score_mean": round(sum(bad_scores) / len(bad_scores), 4) if bad_scores else None,
        "avg_latency_s": round(avg_lat, 3),
    }


# ── Leaderboard 更新 ─────────────────────────────────────────────────
def update_leaderboard(all_metrics: dict, leaderboard_path: Path):
    """更新实时leaderboard文件"""
    if not all_metrics:
        return
    
    # Sort by Good%
    ranked_binary = sorted(
        all_metrics.items(),
        key=lambda kv: kv[1].get("good_pct") if kv[1].get("good_pct") is not None else -1,
        reverse=True,
    )
    
    # Sort by ScoreMean
    ranked_score = sorted(
        all_metrics.items(),
        key=lambda kv: kv[1].get("score_mean") if kv[1].get("score_mean") is not None else -1,
        reverse=True,
    )
    
    leaderboard_content = f"""# UXBench Leaderboard - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Binary Verdict Ranking (Good%)
| Rank | Model | Good% | Bad% | Fail | Score Mean | Score P50 | Valid N |
|------|-------|-------|------|------|------------|-----------|---------|
"""
    
    for rank, (model_tag, m) in enumerate(ranked_binary, 1):
        good_pct = f"{m.get('good_pct', 0):.1%}" if m.get("good_pct") is not None else "N/A"
        bad_pct = f"{m.get('bad_pct', 0):.1%}" if m.get("bad_pct") is not None else "N/A"
        fail = str(m.get("n_fail", 0))
        score_mean = f"{m.get('score_mean', 0):.4f}" if m.get("score_mean") is not None else "N/A"
        score_p50 = f"{m.get('score_p50', 0):.4f}" if m.get("score_p50") is not None else "N/A"
        n_valid = str(m.get("n_valid", 0))
        
        leaderboard_content += f"| {rank} | {model_tag} | {good_pct} | {bad_pct} | {fail} | {score_mean} | {score_p50} | {n_valid} |\n"
    
    leaderboard_content += f"""

## Continuous Score Ranking (Score Mean)
| Rank | Model | Score Mean | Score P50 | Good% | Bad% | Valid N |
|------|-------|------------|-----------|-------|------|---------|
"""
    
    for rank, (model_tag, m) in enumerate(ranked_score, 1):
        score_mean = f"{m.get('score_mean', 0):.4f}" if m.get("score_mean") is not None else "N/A"
        score_p50 = f"{m.get('score_p50', 0):.4f}" if m.get("score_p50") is not None else "N/A"
        good_pct = f"{m.get('good_pct', 0):.1%}" if m.get("good_pct") is not None else "N/A"
        bad_pct = f"{m.get('bad_pct', 0):.1%}" if m.get("bad_pct") is not None else "N/A"
        n_valid = str(m.get("n_valid", 0))
        
        leaderboard_content += f"| {rank} | {model_tag} | {score_mean} | {score_p50} | {good_pct} | {bad_pct} | {n_valid} |\n"
    
    leaderboard_content += f"""

## Model Completion Status
| Model | Status | Completion Time |
|-------|--------|-----------------|
"""
    
    for model_tag, m in all_metrics.items():
        status = "✅ Completed" if m.get("n_total", 0) > 0 and m.get("error") is None else "❌ Error" if m.get("error") else "⏳ Running"
        completion_time = datetime.now().strftime('%H:%M:%S')
        leaderboard_content += f"| {model_tag} | {status} | {completion_time} |\n"
    
    with open(leaderboard_path, "w", encoding="utf-8") as f:
        f.write(leaderboard_content)
    
    log(f"Leaderboard updated: {leaderboard_path}")


# ── Main ─────────────────────────────────────────────────────────────
def discover_models() -> list:
    """List all *.jsonl in responses/ as candidate model tags."""
    all_models = sorted([p.stem for p in RESP_DIR.glob("*.jsonl")])
    # Exclude models that should not be judged
    excluded_models = ["gpt_oss_120b", "volc_deepseek_v3_2"]
    filtered_models = [model for model in all_models if model not in excluded_models]
    return filtered_models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://localhost:8021/v1/chat/completions")
    parser.add_argument("--grm-model", default=GRM_MODEL)
    parser.add_argument("--workers", type=int, default=80)
    parser.add_argument("--smoke", type=int, default=0)
    parser.add_argument("--models", nargs="+", default=None,
                        help="Subset of model tags to judge (default: all in responses/)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Open run log file
    global _LOG_FH
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"run_{run_ts}.log"
    _LOG_FH = open(log_path, "a", encoding="utf-8")

    all_models = discover_models()
    if args.models:
        models = [m for m in args.models if m in all_models]
        missing = [m for m in args.models if m not in all_models]
        if missing:
            log(f"WARNING: unknown models ignored: {missing}")
    else:
        models = all_models

    log("=" * 70)
    log("UXBench Main Experiment GRM Judge")
    log(f"Endpoint: {args.endpoint}")
    log(f"GRM Model: {args.grm_model}")
    log(f"Workers: {args.workers}")
    log(f"Smoke: {args.smoke if args.smoke else 'off (full)'}")
    log(f"Models to judge: {len(models)}")
    log(f"  -> {models}")
    log(f"Results dir: {RESULTS_DIR}")
    log(f"Run log: {log_path}")
    log("=" * 70)

    # Connectivity test
    log("\nTesting endpoint with full GRM prompt...")
    test_prompt = build_grm_prompt(
        {"history": [], "query": "你好"},
        "你好！有什么可以帮你的吗？",
    )
    test_result = call_pointwise_grm(test_prompt, endpoint=args.endpoint, model_name=args.grm_model)
    if test_result.ok:
        log(f"  OK! content=\"{test_result.content}\"  score={test_result.score}  latency={test_result.latency_s:.2f}s")
    else:
        log(f"  FAILED: {test_result.error}")
        log("  Continuing anyway...")

    # Load testset (queries)
    log(f"\nLoading testset from {TESTSET_FILE.name}...")
    cid2query = {}
    with open(TESTSET_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            cid2query[d["cid"]] = d
    log(f"  Loaded {len(cid2query)} queries")

    # Run GRM judge for each model
    all_metrics = {}
    overall_t0 = time.time()
    leaderboard_path = RESULTS_DIR / "leaderboard.md"
    
    for i, model_tag in enumerate(models, 1):
        log(f"\n{'='*60}")
        log(f"[{i}/{len(models)}] Judging model: {model_tag}")
        log(f"{'='*60}")
        try:
            m = judge_model(model_tag, cid2query, args.endpoint, args.grm_model,
                            args.workers, args.smoke)
            all_metrics[model_tag] = m
        except Exception as e:
            log(f"  EXCEPTION while judging {model_tag}: {e}")
            all_metrics[model_tag] = {"model": model_tag, "error": str(e)}

        # Save metrics after each model (crash-safe)
        metrics_path = RESULTS_DIR / "main_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, ensure_ascii=False, indent=2)
        
        # Update leaderboard after each model completes
        update_leaderboard(all_metrics, leaderboard_path)

    overall_elapsed = time.time() - overall_t0
    log(f"\nALL DONE in {overall_elapsed:.1f}s ({overall_elapsed/60:.1f} min)")

    # ── Summary ranking ──
    def _row(short, m):
        good_pct  = f"{m['good_pct']:.1%}"           if m.get("good_pct") is not None else "N/A"
        bad_pct   = f"{m['bad_pct']:.1%}"            if m.get("bad_pct") is not None else "N/A"
        fail      = str(m.get("n_fail", 0))
        scr_mean  = f"{m['score_mean']:.4f}"         if m.get("score_mean") is not None else "N/A"
        scr_p50   = f"{m['score_p50']:.4f}"          if m.get("score_p50") is not None else "N/A"
        n         = str(m.get("n_valid", 0))
        return (f"{short:<28} {good_pct:>8} {bad_pct:>8} {fail:>5} "
                f"{scr_mean:>10} {scr_p50:>9} {n:>6}")

    header = (f"{'Model':<28} {'Good%':>8} {'Bad%':>8} {'Fail':>5} "
              f"{'ScoreMean':>10} {'ScoreP50':>9} {'N':>6}")
    bar = "-" * len(header)

    log(f"\n{'='*len(header)}")
    log("SUMMARY — View 1: Binary verdict ranking (sorted by Good%)")
    log(f"{'='*len(header)}")
    log(header)
    log(bar)
    ranked_binary = sorted(
        all_metrics.items(),
        key=lambda kv: kv[1].get("good_pct") if kv[1].get("good_pct") is not None else -1,
        reverse=True,
    )
    for model_tag, m in ranked_binary:
        log(_row(model_tag, m))

    log(f"\n{'='*len(header)}")
    log("SUMMARY — View 2: Continuous P(好)/(P(好)+P(差)) ranking (sorted by ScoreMean)")
    log(f"{'='*len(header)}")
    log(header)
    log(bar)
    ranked_score = sorted(
        all_metrics.items(),
        key=lambda kv: kv[1].get("score_mean") if kv[1].get("score_mean") is not None else -1,
        reverse=True,
    )
    for model_tag, m in ranked_score:
        log(_row(model_tag, m))

    metrics_path = RESULTS_DIR / "main_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)
    log(f"\nMetrics saved to {metrics_path}")
    log("Done!")

    # Final leaderboard update
    update_leaderboard(all_metrics, leaderboard_path)
    log(f"Final leaderboard saved to {leaderboard_path}")

    if _LOG_FH:
        _LOG_FH.close()


if __name__ == "__main__":
    main()
