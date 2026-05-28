"""
run_grm_judge.py
================
Judge model responses using a Generative Reward Model (GRM).

The GRM outputs a binary verdict ("好"=good / "差"=bad) and a continuous score
derived from logprobs: score = P(好) / (P(好) + P(差)).

Supports:
  - OpenAI-compatible vLLM endpoint
  - Checkpoint/resume (skips already-judged cids)
  - Configurable concurrency
  - Truncation retry for long inputs

Usage:
    python run_grm_judge.py --responses responses/gpt_5.jsonl --testset testset.jsonl
    python run_grm_judge.py --responses-dir responses/ --testset testset.jsonl --workers 80
    python run_grm_judge.py --responses responses/gpt_5.jsonl --smoke 10
"""
import argparse
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import requests


# ── Configuration (override via env vars or CLI) ────────────────────
GRM_ENDPOINT = os.environ.get("GRM_ENDPOINT", "http://localhost:8021/v1/chat/completions")
GRM_MODEL = os.environ.get("GRM_MODEL", "pointwise_grm_ux")
GRM_API_KEY = os.environ.get("GRM_API_KEY", "EMPTY")

SCRIPT_DIR = Path(__file__).resolve().parent


# ── Pointwise GRM Prompt Template ───────────────────────────────────
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


# ── GRM Call ────────────────────────────────────────────────────────
def call_grm(prompt_text: str, endpoint: str, model: str, api_key: str) -> dict:
    """Call GRM and extract verdict + score from logprobs."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt_text}],
        "max_tokens": 1,
        "temperature": 0.0,
        "logprobs": True,
        "top_logprobs": 20,
    }

    t0 = time.time()
    try:
        resp = requests.post(endpoint, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        choice = data["choices"][0]
        content = choice["message"]["content"].strip()

        # Extract logprobs for scoring
        good_logprob = None
        bad_logprob = None

        logprobs_data = choice.get("logprobs", {})
        if logprobs_data and "content" in logprobs_data:
            top_lps = logprobs_data["content"][0].get("top_logprobs", [])
            for item in top_lps:
                token = item.get("token", "")
                lp = item.get("logprob", -100)
                if "好" in token and good_logprob is None:
                    good_logprob = lp
                elif "差" in token and bad_logprob is None:
                    bad_logprob = lp

        # Compute score
        score = None
        if good_logprob is not None and bad_logprob is not None:
            p_good = math.exp(good_logprob)
            p_bad = math.exp(bad_logprob)
            score = p_good / (p_good + p_bad)

        # Verdict
        verdict = 0
        if "好" in content:
            verdict = 1
        elif "差" in content:
            verdict = -1

        return {
            "ok": True,
            "verdict": verdict,
            "score": round(score, 4) if score else None,
            "good_logprob": good_logprob,
            "bad_logprob": bad_logprob,
            "latency_s": round(time.time() - t0, 3),
            "error": "",
        }
    except requests.HTTPError as e:
        # Status-only error string — never include raw exception (would leak endpoint URL)
        status = e.response.status_code if e.response is not None else 0
        err = f"HTTP {status}"
    except requests.Timeout:
        err = "timeout"
    except requests.ConnectionError:
        err = "connection_error"
    except Exception as e:
        err = type(e).__name__
    return {
        "ok": False,
        "verdict": 0,
        "score": None,
        "good_logprob": None,
        "bad_logprob": None,
        "latency_s": round(time.time() - t0, 3),
        "error": err,
    }


def build_grm_prompt(query_rec: dict, response_text: str,
                     max_history_turns: int = 6,
                     max_response_chars: int = 4000) -> str:
    """Build the GRM judge prompt from query record and model response."""
    history = query_rec.get("history", [])
    if isinstance(history, str):
        try:
            history = json.loads(history)
        except:
            history = []

    user_query = query_rec.get("query", "")

    if max_history_turns > 0 and len(history) > max_history_turns:
        history = history[-max_history_turns:]
    if max_response_chars > 0 and len(response_text) > max_response_chars:
        response_text = response_text[:max_response_chars] + "...[truncated]"

    context_lines = []
    for turn in history:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role", "")
        msg = turn.get("content", "") or ""
        role_label = "用户" if role == "user" else "AI"
        context_lines.append(f"[{role_label}]: {msg}")

    context = "\n".join(context_lines) if context_lines else "(无历史对话)"

    return POINTWISE_GRM_TEMPLATE.format(
        context=context,
        prompt=user_query,
        response_a=response_text,
    )


def judge_file(response_file: Path, testset: dict, output_dir: Path,
               endpoint: str, model: str, api_key: str,
               workers: int = 80, smoke: int = 0):
    """Judge all responses in a file."""
    model_name = response_file.stem
    out_file = output_dir / f"judge_{model_name}.jsonl"

    # Load responses
    responses = {}
    with open(response_file, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                responses[rec["cid"]] = rec

    # Load existing results
    done_cids = set()
    if out_file.exists():
        with open(out_file, "r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    done_cids.add(json.loads(line)["cid"])

    todo = [(cid, resp) for cid, resp in responses.items() if cid not in done_cids]
    if smoke > 0:
        todo = todo[:smoke]

    print(f"  {model_name}: total={len(responses)}, done={len(done_cids)}, todo={len(todo)}")

    if not todo:
        return

    lock = Lock()
    with open(out_file, "a", encoding="utf-8") as fh:
        def process(item):
            cid, resp_rec = item
            query_rec = testset.get(cid, {})
            response_text = resp_rec.get("generated_response", "")

            prompt = build_grm_prompt(query_rec, response_text)
            result = call_grm(prompt, endpoint, model, api_key)

            # Retry with truncation on 400 (likely context-length error)
            if not result["ok"] and result.get("error", "") == "HTTP 400":
                prompt = build_grm_prompt(query_rec, response_text,
                                         max_history_turns=3, max_response_chars=2000)
                result = call_grm(prompt, endpoint, model, api_key)

            output = {
                "cid": cid,
                "judged_model": resp_rec.get("model", model_name),
                **result,
            }
            return output

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(process, item): item for item in todo}
            completed = 0
            for future in as_completed(futures):
                rec = future.result()
                with lock:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fh.flush()
                completed += 1
                if completed % 100 == 0:
                    print(f"    {model_name}: {completed}/{len(todo)}")


def main():
    parser = argparse.ArgumentParser(description="GRM Judge for UXBench")
    parser.add_argument("--responses", type=str, help="Single response file to judge")
    parser.add_argument("--responses-dir", type=str, help="Directory of response files")
    parser.add_argument("--testset", type=str, required=True, help="Testset JSONL (with query + history)")
    parser.add_argument("--output-dir", type=str, default="judge_results")
    parser.add_argument("--endpoint", default=GRM_ENDPOINT)
    parser.add_argument("--model", default=GRM_MODEL)
    parser.add_argument("--api-key", default=GRM_API_KEY)
    parser.add_argument("--workers", type=int, default=80)
    parser.add_argument("--smoke", type=int, default=0)
    args = parser.parse_args()

    # Load testset as dict keyed by cid
    testset = {}
    with open(args.testset, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                testset[rec["cid"]] = rec

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.responses:
        judge_file(Path(args.responses), testset, output_dir,
                   args.endpoint, args.model, args.api_key,
                   args.workers, args.smoke)
    elif args.responses_dir:
        for f in sorted(Path(args.responses_dir).glob("*.jsonl")):
            judge_file(f, testset, output_dir,
                       args.endpoint, args.model, args.api_key,
                       args.workers, args.smoke)
    else:
        parser.error("Provide either --responses or --responses-dir")


if __name__ == "__main__":
    main()
