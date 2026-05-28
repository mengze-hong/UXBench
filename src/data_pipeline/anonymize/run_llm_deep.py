#!/usr/bin/env python3
"""Optional Layer 1.5: LLM deep anonymization on top of rule-based output."""

from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from config import CONFIG

SYSTEM_PROMPT = """你是数据脱敏专家。识别文本中个人可识别信息(PII)并给出同类型替换。
输出 JSON：
{"has_pii": true, "items": [{"original":"原文本","type":"pii类型","replacement":"替换文本"}]}
或
{"has_pii": false}
只输出 JSON。"""


def parse_json_from_text(content: str) -> dict:
    content = (content or "").strip()
    if content.startswith("```"):
        content = content.strip("`")
        content = content.replace("json", "", 1).strip()
    try:
        return json.loads(content)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", content)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return {"has_pii": False, "parse_error": True}


def call_llm(text: str, model: str, timeout: int) -> dict:
    url = (CONFIG.get("llm", {}).get("api_url") or "").strip()
    key = (CONFIG.get("llm", {}).get("api_key") or "").strip()
    if not url or not key:
        return {"has_pii": False, "error": "missing_api_config"}
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text[:6000]}],
        "max_tokens": 900,
        "temperature": 0.0,
    }
    for attempt in range(3):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if resp.status_code == 200:
                content = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")
                return parse_json_from_text(content)
            if resp.status_code == 429:
                time.sleep(2 * (attempt + 1))
                continue
            time.sleep(1)
        except Exception:
            time.sleep(1.2 * (attempt + 1))
    return {"has_pii": False, "error": "api_failed"}


def apply_replacements(text: str, items: list[dict]) -> str:
    out = text
    for item in items:
        o = item.get("original", "")
        r = item.get("replacement", "")
        if o and r and o in out:
            out = out.replace(o, r)
    return out


def build_text_for_check(rec: dict) -> str:
    chunks: list[str] = []

    # Common top-level text fields across Task1/2/3
    for field in (
        "query",
        "prompt",
        "user_query",
        "user_complaint",
        "failed_response",
        "agent_response_full",
        "liked_response_full",
        "explanation",
        "system_prompt",
    ):
        v = rec.get(field)
        if isinstance(v, str) and v.strip():
            chunks.append(f"[{field}] {v[:2500]}")

    sq = rec.get("source_query", {})
    if isinstance(sq, dict):
        msg = sq.get("message")
        if isinstance(msg, str) and msg.strip():
            chunks.append("[source_query] " + msg[:1500])

    # Conversation arrays
    for arr_field in ("selected_history", "full_history", "history", "messages"):
        arr = rec.get(arr_field)
        if not isinstance(arr, list):
            continue
        for turn in arr[-8:]:
            if not isinstance(turn, dict):
                continue
            msg = turn.get("message") or turn.get("content") or ""
            if not isinstance(msg, str) or not msg.strip():
                continue
            role = turn.get("role", "turn")
            chunks.append(f"[{arr_field}:{role}] {msg[:600]}")

    # Fallback: compact dump of record tail if still empty
    if not chunks:
        chunks.append(json.dumps(rec, ensure_ascii=False)[:4000])

    return "\n".join(chunks)


def process_one(rec: dict, model: str, timeout: int) -> tuple[dict, list[dict]]:
    check_text = build_text_for_check(rec)
    result = call_llm(check_text, model=model, timeout=timeout)
    if not result.get("has_pii"):
        return rec, []
    items = result.get("items", [])
    if not items:
        return rec, []
    anon = json.loads(json.dumps(rec))
    if isinstance(anon.get("source_query"), dict) and isinstance(anon["source_query"].get("message"), str):
        anon["source_query"]["message"] = apply_replacements(anon["source_query"]["message"], items)
    for field in (
        "query",
        "prompt",
        "user_query",
        "user_complaint",
        "failed_response",
        "agent_response_full",
        "liked_response_full",
        "explanation",
        "system_prompt",
    ):
        if isinstance(anon.get(field), str):
            anon[field] = apply_replacements(anon[field], items)
    for arr_field in ("selected_history", "full_history", "history"):
        if isinstance(anon.get(arr_field), list):
            for turn in anon[arr_field]:
                if isinstance(turn.get("message"), str):
                    turn["message"] = apply_replacements(turn["message"], items)
                if isinstance(turn.get("content"), str):
                    turn["content"] = apply_replacements(turn["content"], items)
    return anon, items


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM deep anonymization")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--model", default="gpt-5.1")
    parser.add_argument("--changelog", default="")
    args = parser.parse_args()

    api_url = (CONFIG.get("llm", {}).get("api_url") or "").strip()
    api_key = (CONFIG.get("llm", {}).get("api_key") or "").strip()
    if not api_url or not api_key:
        raise SystemExit(
            "Missing LLM API config: please set config.json or PIPELINE_API_URL/PIPELINE_API_KEY before running deep anonymization."
        )

    in_path = Path(args.input)
    out_path = Path(args.output)
    changelog = Path(args.changelog) if args.changelog else out_path.with_suffix(".llm_changelog.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    results = [None] * len(records)
    changes = []
    modified = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_one, rec, args.model, args.timeout): i for i, rec in enumerate(records)}
        for fut in as_completed(futures):
            i = futures[fut]
            anon, items = fut.result()
            results[i] = anon
            if items:
                modified += 1
                changes.append({"cid": anon.get("cid", ""), "n_items": len(items), "items": items})

    with out_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with changelog.open("w", encoding="utf-8") as f:
        for c in changes:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "method": "llm_deep",
                "input": str(in_path),
                "output": str(out_path),
                "total_records": len(records),
                "modified_records": modified,
                "workers": args.workers,
                "model": args.model,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
