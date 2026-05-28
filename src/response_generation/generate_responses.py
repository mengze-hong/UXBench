"""
generate_responses.py
=====================
Generate model responses for UXBench Task 2 (UX Eval) and Task 3 (UX Recovery).

For each query in the testset, call a target LLM to generate a response.
Supports checkpoint/resume (skips already-completed cids).

Usage:
    python generate_responses.py --task task2 --model gpt-5 --endpoint http://localhost:8000/v1
    python generate_responses.py --task task3 --model gpt-5 --endpoint http://localhost:8000/v1
    python generate_responses.py --task task2 --model gpt-5 --smoke 10  # test with 10 samples
"""
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ENDPOINT = os.environ.get("LLM_ENDPOINT", "http://localhost:8000/v1/chat/completions")
DEFAULT_API_KEY = os.environ.get("LLM_API_KEY", "EMPTY")


def load_testset(task: str) -> list:
    """Load testset JSONL for the given task."""
    data_dir = SCRIPT_DIR.parent / "uxbench-dataset"
    if task == "task2":
        f = data_dir / "ux_eval_demo_200.jsonl"
    elif task == "task3":
        f = data_dir / "ux_recovery_demo.jsonl"  # if available
    else:
        raise ValueError(f"Unknown task: {task}")

    if not f.exists():
        raise FileNotFoundError(f"Testset not found: {f}")

    records = []
    with open(f, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                records.append(json.loads(line))
    return records


def build_messages_task2(record: dict) -> list:
    """Build chat messages for Task 2 (response generation)."""
    messages = []
    # Add history if available
    history = record.get("history") or record.get("messages") or []
    for msg in history:
        if isinstance(msg, dict) and "role" in msg:
            messages.append({"role": msg["role"], "content": msg.get("content", "")})
    # Add the final user query
    query = record.get("query", "")
    if query and (not messages or messages[-1].get("content") != query):
        messages.append({"role": "user", "content": query})
    return messages


def build_messages_task3(record: dict) -> list:
    """Build chat messages for Task 3 (failure recovery)."""
    messages = []
    history = record.get("history", [])
    for msg in history:
        if isinstance(msg, dict) and "role" in msg:
            messages.append({"role": msg["role"], "content": msg.get("content", "")})
    # Add the failed response
    failed = record.get("failed_response", "")
    if failed:
        messages.append({"role": "assistant", "content": failed})
    # Add user complaint
    complaint = record.get("user_complaint", "")
    if complaint:
        messages.append({"role": "user", "content": complaint})
    return messages


def call_llm(messages: list, endpoint: str, api_key: str, model: str,
             max_tokens: int = 2048, temperature: float = 0.7) -> dict:
    """Call an OpenAI-compatible API endpoint."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    t0 = time.time()
    try:
        resp = requests.post(endpoint, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        reasoning = data["choices"][0]["message"].get("reasoning_content", "")
        tokens = data.get("usage", {}).get("completion_tokens", 0)
        return {
            "generated_response": content,
            "reasoning_content": reasoning,
            "latency_s": round(time.time() - t0, 2),
            "tokens": tokens,
            "error": "",
        }
    except requests.HTTPError as e:
        # Log status code only; do NOT include str(e) which would leak the endpoint URL
        status = e.response.status_code if e.response is not None else "unknown"
        err = f"HTTP {status}"
    except requests.Timeout:
        err = "timeout"
    except requests.ConnectionError:
        err = "connection_error"
    except Exception as e:
        err = type(e).__name__
    return {
        "generated_response": "",
        "reasoning_content": "",
        "latency_s": round(time.time() - t0, 2),
        "tokens": 0,
        "error": err,
    }


def generate(task: str, model: str, endpoint: str, api_key: str,
             workers: int = 10, smoke: int = 0, output_dir: Path = None):
    """Main generation loop with checkpoint/resume."""
    records = load_testset(task)
    if smoke > 0:
        records = records[:smoke]

    if output_dir is None:
        output_dir = SCRIPT_DIR / "outputs" / task
    output_dir.mkdir(parents=True, exist_ok=True)

    model_slug = model.replace("/", "_").replace("-", "_")
    out_file = output_dir / f"{model_slug}.jsonl"

    # Load existing results for resume
    done_cids = set()
    if out_file.exists():
        with open(out_file, "r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    done_cids.add(json.loads(line)["cid"])

    todo = [r for r in records if r["cid"] not in done_cids]
    print(f"Task: {task} | Model: {model} | Total: {len(records)} | Done: {len(done_cids)} | Todo: {len(todo)}")

    if not todo:
        print("All done!")
        return

    build_fn = build_messages_task2 if task == "task2" else build_messages_task3

    with open(out_file, "a", encoding="utf-8") as fh:
        def process(rec):
            messages = build_fn(rec)
            result = call_llm(messages, endpoint, api_key, model)
            result["cid"] = rec["cid"]
            result["model"] = model
            return result

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(process, r): r for r in todo}
            completed = 0
            for future in as_completed(futures):
                result = future.result()
                fh.write(json.dumps(result, ensure_ascii=False) + "\n")
                fh.flush()
                completed += 1
                if completed % 50 == 0:
                    print(f"  Progress: {completed}/{len(todo)}")

    print(f"Done. Output: {out_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate LLM responses for UXBench")
    parser.add_argument("--task", required=True, choices=["task2", "task3"])
    parser.add_argument("--model", required=True, help="Model name (passed to API)")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="API endpoint URL")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="API key")
    parser.add_argument("--workers", type=int, default=10, help="Parallel workers")
    parser.add_argument("--smoke", type=int, default=0, help="Limit to N samples for testing")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else None
    generate(args.task, args.model, args.endpoint, args.api_key,
             args.workers, args.smoke, output_dir)


if __name__ == "__main__":
    main()
