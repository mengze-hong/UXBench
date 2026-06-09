"""
run_task3_judge.py
==================
Run Pointwise GRM judge on Task 3 (UX Recovery) model responses.

Difference from Task 2: each response record already contains the complaint context
(user_complaint, failed_response) so no separate testset file is needed.
The prompt frames: [failed AI response → user complaint] as history,
and grades the recovery response.

Usage:
    python scripts/grm_judge/run_grm_judge_task3.py
    python scripts/grm_judge/run_grm_judge_task3.py --workers 60 --models gpt_5 glm_5_1
    python scripts/grm_judge/run_grm_judge_task3.py --smoke 5
"""
import argparse
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
RESP_DIR = SCRIPT_DIR.parent.parent / "experiments" / "results" / "task3" / "responses"
RESULTS_DIR = SCRIPT_DIR.parent.parent / "experiments" / "results" / "task3" / "grm_judge"
LOG_DIR = RESULTS_DIR / "logs"

# ── Pointwise GRM 配置 ───────────────────────────────────────────────
GRM_MODEL = "pointwise_grm_ux_v031"
GRM_AUTH_TOKEN = os.environ.get("GRM_AUTH_TOKEN", "")
GRM_PORT = 8021

# no_proxy: set via environment if needed

# ── Pointwise GRM Prompt Template (与 task2 一致) ────────────────────
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


# ── logger ───────────────────────────────────────────────────────────
_LOG_FH = None
_LOG_LOCK = Lock()


def log(msg: str):
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


def build_grm_prompt(rec: dict,
                     max_failed_chars: int = 0,
                     max_complaint_chars: int = 0,
                     max_response_chars: int = 0) -> str:
    """构建 GRM judge prompt，针对 task3 投诉挽回场景。

    历史对话仅放上一轮的 AI 失败回复；用户问题用 user_complaint；
    AI助手回复用 generated_response (即 recovery response)。
    """
    failed_response = rec.get("failed_response", "") or ""
    user_complaint = rec.get("user_complaint", "") or ""
    response_text = rec.get("generated_response", "") or ""

    if max_failed_chars > 0 and len(failed_response) > max_failed_chars:
        failed_response = failed_response[:max_failed_chars] + "...[截断]"
    if max_complaint_chars > 0 and len(user_complaint) > max_complaint_chars:
        user_complaint = user_complaint[:max_complaint_chars] + "...[截断]"
    if max_response_chars > 0 and len(response_text) > max_response_chars:
        response_text = response_text[:max_response_chars] + "...[截断]"

    if failed_response.strip():
        context = f"AI：{failed_response}"
    else:
        context = "无"

    return POINTWISE_GRM_TEMPLATE.format(
        context=context,
        prompt=user_complaint,
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
def judge_one(rec: dict, model_tag: str, endpoint: str, grm_model_name: str) -> dict:
    truncation_levels = [
        (0, 0, 0),           # 不截断
        (4000, 1000, 4000),  # 中等截断
        (2000, 500, 2000),   # 重度截断
    ]

    result = None
    for max_failed, max_complaint, max_resp in truncation_levels:
        prompt = build_grm_prompt(
            rec,
            max_failed_chars=max_failed,
            max_complaint_chars=max_complaint,
            max_response_chars=max_resp,
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
        "cid": rec["cid"],
        "judged_model": model_tag,
        "verdict": verdict,
        "score": result.score if result and result.ok else None,
        "content": result.content if result and result.ok else "",
        "good_logprob": result.good_logprob if result and result.ok else None,
        "bad_logprob": result.bad_logprob if result and result.ok else None,
        "grm_ok": result.ok if result else False,
        "latency_s": result.latency_s if result else 0,
        "error": result.error if result else "no_result",
        "failure_dimension": rec.get("failure_dimension", ""),
        "scenario": rec.get("scenario", ""),
    }


# ── 单模型处理 ───────────────────────────────────────────────────────
def judge_model(model_tag: str, endpoint: str, grm_model_name: str,
                workers: int, smoke: int) -> dict:
    resp_path = RESP_DIR / f"{model_tag}.jsonl"
    out_path = RESULTS_DIR / f"judge_{model_tag}.jsonl"

    # Load responses
    records = []
    with open(resp_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                d = json.loads(line)
                records.append(d)
            except Exception:
                pass

    # Load existing results (resume)
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

    # Build todo
    todo = []
    skipped_no_resp = 0
    for rec in records:
        if rec.get("cid") in done_cids:
            continue
        resp_text = rec.get("generated_response", "") or ""
        if not resp_text.strip() or "__ERROR__" in resp_text:
            skipped_no_resp += 1
            continue
        todo.append(rec)

    if smoke > 0:
        todo = todo[:smoke]

    if skipped_no_resp:
        log(f"  [{model_tag}] skipped: no_resp={skipped_no_resp}")

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

    def do_one(rec):
        out = judge_one(rec, model_tag, endpoint, grm_model_name)
        with lock:
            results.append(out)
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            fout.flush()
            if out["verdict"] == 1:
                good_count[0] += 1
            elif out["verdict"] == -1:
                bad_count[0] += 1
            else:
                fail_count[0] += 1
            total_done = good_count[0] + bad_count[0] + fail_count[0]
            if total_done % 100 == 0:
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
            futs = [pool.submit(do_one, r) for r in todo]
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
        return {"model": model_tag, "n_total": total, "n_valid": 0,
                "n_fail": n_fail, "error": "no_valid"}

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

    # by failure_dimension
    from collections import defaultdict
    dim_stats = defaultdict(lambda: {"good": 0, "bad": 0})
    for r in results:
        if r.get("verdict") not in (1, -1):
            continue
        dim = r.get("failure_dimension") or "unknown"
        if r["verdict"] == 1:
            dim_stats[dim]["good"] += 1
        else:
            dim_stats[dim]["bad"] += 1

    return {
        "model": model_tag,
        "n_total": total,
        "n_valid": n_valid,
        "n_fail": n_fail,
        "n_good": n_good,
        "n_bad": n_bad,
        "good_pct": round(n_good / n_valid, 4),
        "bad_pct": round(n_bad / n_valid, 4),
        "score_mean": round(score_mean, 4) if score_mean is not None else None,
        "score_p50": round(score_p50, 4) if score_p50 is not None else None,
        "good_score_mean": round(sum(good_scores) / len(good_scores), 4) if good_scores else None,
        "bad_score_mean": round(sum(bad_scores) / len(bad_scores), 4) if bad_scores else None,
        "avg_latency_s": round(avg_lat, 3),
        "by_dimension": {dim: dict(cnt) for dim, cnt in dim_stats.items()},
    }


# ── Leaderboard ──────────────────────────────────────────────────────
def update_leaderboard(all_metrics: dict, leaderboard_path: Path):
    if not all_metrics:
        return
    ranked_binary = sorted(
        all_metrics.items(),
        key=lambda kv: kv[1].get("good_pct") if kv[1].get("good_pct") is not None else -1,
        reverse=True,
    )
    ranked_score = sorted(
        all_metrics.items(),
        key=lambda kv: kv[1].get("score_mean") if kv[1].get("score_mean") is not None else -1,
        reverse=True,
    )

    out = [
        f"# Task3 Failure-Recovery GRM Leaderboard - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        "## Binary Verdict Ranking (Good%)\n",
        "| Rank | Model | Good% | Bad% | Fail | Score Mean | Score P50 | Valid N |\n",
        "|------|-------|-------|------|------|------------|-----------|---------|\n",
    ]
    for rank, (model_tag, m) in enumerate(ranked_binary, 1):
        good_pct = f"{m.get('good_pct', 0):.1%}" if m.get("good_pct") is not None else "N/A"
        bad_pct = f"{m.get('bad_pct', 0):.1%}" if m.get("bad_pct") is not None else "N/A"
        fail = str(m.get("n_fail", 0))
        score_mean = f"{m.get('score_mean', 0):.4f}" if m.get("score_mean") is not None else "N/A"
        score_p50 = f"{m.get('score_p50', 0):.4f}" if m.get("score_p50") is not None else "N/A"
        n_valid = str(m.get("n_valid", 0))
        out.append(f"| {rank} | {model_tag} | {good_pct} | {bad_pct} | {fail} | {score_mean} | {score_p50} | {n_valid} |\n")

    out.append("\n## Continuous Score Ranking (Score Mean)\n")
    out.append("| Rank | Model | Score Mean | Score P50 | Good% | Bad% | Valid N |\n")
    out.append("|------|-------|------------|-----------|-------|------|---------|\n")
    for rank, (model_tag, m) in enumerate(ranked_score, 1):
        score_mean = f"{m.get('score_mean', 0):.4f}" if m.get("score_mean") is not None else "N/A"
        score_p50 = f"{m.get('score_p50', 0):.4f}" if m.get("score_p50") is not None else "N/A"
        good_pct = f"{m.get('good_pct', 0):.1%}" if m.get("good_pct") is not None else "N/A"
        bad_pct = f"{m.get('bad_pct', 0):.1%}" if m.get("bad_pct") is not None else "N/A"
        n_valid = str(m.get("n_valid", 0))
        out.append(f"| {rank} | {model_tag} | {score_mean} | {score_p50} | {good_pct} | {bad_pct} | {n_valid} |\n")

    with open(leaderboard_path, "w", encoding="utf-8") as f:
        f.writelines(out)
    log(f"Leaderboard updated: {leaderboard_path}")


# ── Main ─────────────────────────────────────────────────────────────
def discover_models() -> list:
    return sorted([p.stem for p in RESP_DIR.glob("*.jsonl")])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://localhost:8021/v1/chat/completions")
    parser.add_argument("--grm-model", default=GRM_MODEL)
    parser.add_argument("--workers", type=int, default=60)
    parser.add_argument("--smoke", type=int, default=0)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--resp-dir", default=None,
                        help="Response directory (overrides default RESP_DIR)")
    args = parser.parse_args()

    # Override RESP_DIR if --resp-dir is provided
    global RESP_DIR
    if args.resp_dir:
        RESP_DIR = Path(args.resp_dir)
        log(f"  Overriding RESP_DIR -> {RESP_DIR}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    global _LOG_FH
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"run_{run_ts}.log"
    _LOG_FH = open(log_path, "a", encoding="utf-8")

    all_models = discover_models()
    models = [m for m in (args.models or all_models) if m in all_models]

    log("=" * 70)
    log("UXBench Task3 Failure-Recovery GRM Judge")
    log(f"Endpoint: {args.endpoint}")
    log(f"GRM Model: {args.grm_model}")
    log(f"Workers: {args.workers}")
    log(f"Smoke: {args.smoke if args.smoke else 'off (full)'}")
    log(f"Models to judge: {len(models)} -> {models}")
    log(f"Resp dir: {RESP_DIR}")
    log(f"Results dir: {RESULTS_DIR}")
    log(f"Run log: {log_path}")
    log("=" * 70)

    log("\nTesting endpoint with full GRM prompt...")
    test_prompt = build_grm_prompt({
        "failed_response": "我不知道。",
        "user_complaint": "你这回答太敷衍了",
        "generated_response": "抱歉刚才回答得不够认真，我重新整理一下，给你更具体的内容。",
    })
    test_result = call_pointwise_grm(test_prompt, endpoint=args.endpoint, model_name=args.grm_model)
    if test_result.ok:
        log(f"  OK! content=\"{test_result.content}\"  score={test_result.score}  latency={test_result.latency_s:.2f}s")
    else:
        log(f"  FAILED: {test_result.error}")
        log("  Continuing anyway...")

    all_metrics = {}
    overall_t0 = time.time()
    leaderboard_path = RESULTS_DIR / "leaderboard.md"

    for i, model_tag in enumerate(models, 1):
        log(f"\n{'='*60}")
        log(f"[{i}/{len(models)}] Judging model: {model_tag}")
        log(f"{'='*60}")
        try:
            m = judge_model(model_tag, args.endpoint, args.grm_model,
                            args.workers, args.smoke)
            all_metrics[model_tag] = m
        except Exception as e:
            log(f"  EXCEPTION while judging {model_tag}: {e}")
            all_metrics[model_tag] = {"model": model_tag, "error": str(e)}

        metrics_path = RESULTS_DIR / "task3_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, ensure_ascii=False, indent=2)
        update_leaderboard(all_metrics, leaderboard_path)

    overall_elapsed = time.time() - overall_t0
    log(f"\nALL DONE in {overall_elapsed:.1f}s ({overall_elapsed/60:.1f} min)")

    header = (f"{'Model':<28} {'Good%':>8} {'Bad%':>8} {'Fail':>5} "
              f"{'ScoreMean':>10} {'ScoreP50':>9} {'N':>6}")
    log(f"\n{'='*len(header)}")
    log("SUMMARY — Binary verdict ranking (sorted by Good%)")
    log(f"{'='*len(header)}")
    log(header)
    log("-" * len(header))
    ranked = sorted(all_metrics.items(),
                    key=lambda kv: kv[1].get("good_pct") if kv[1].get("good_pct") is not None else -1,
                    reverse=True)
    for model_tag, m in ranked:
        good_pct = f"{m.get('good_pct',0):.1%}" if m.get("good_pct") is not None else "N/A"
        bad_pct = f"{m.get('bad_pct',0):.1%}" if m.get("bad_pct") is not None else "N/A"
        fail = str(m.get("n_fail", 0))
        scr_mean = f"{m.get('score_mean',0):.4f}" if m.get("score_mean") is not None else "N/A"
        scr_p50 = f"{m.get('score_p50',0):.4f}" if m.get("score_p50") is not None else "N/A"
        n = str(m.get("n_valid", 0))
        log(f"{model_tag:<28} {good_pct:>8} {bad_pct:>8} {fail:>5} {scr_mean:>10} {scr_p50:>9} {n:>6}")

    update_leaderboard(all_metrics, leaderboard_path)
    log(f"Final leaderboard saved to {leaderboard_path}")
    if _LOG_FH:
        _LOG_FH.close()


if __name__ == "__main__":
    main()
