"""LLM client for open-source pipeline."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any

import requests

from config import CONFIG


@dataclass
class LLMResult:
    ok: bool
    content: str = ""
    tokens: int = 0
    latency_s: float = 0.0
    error: str = ""
    model: str = ""
    attempts: int = 1


def call_llm(
    messages: list[dict],
    model: str,
    max_tokens: int = 4096,
    temperature: float = 0.1,
    max_retries: int = 3,
) -> LLMResult:
    url = (CONFIG.get("llm", {}).get("api_url") or "").strip()
    key = (CONFIG.get("llm", {}).get("api_key") or "").strip()
    timeout = int(CONFIG.get("llm", {}).get("timeout_seconds", 180))
    if not url or not key:
        return LLMResult(ok=False, error="missing_api_config", model=model)

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
    last_err = "unknown"

    for attempt in range(1, max_retries + 1):
        t0 = time.time()
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            latency = time.time() - t0
            if resp.status_code == 200:
                data = resp.json()
                choices = data.get("choices", [])
                if not choices:
                    last_err = "no_choices"
                    continue
                msg = choices[0].get("message", {})
                content = (msg.get("content") or msg.get("reasoning_content") or "").strip()
                tokens = data.get("usage", {}).get("total_tokens", 0)
                return LLMResult(
                    ok=True,
                    content=content,
                    tokens=tokens,
                    latency_s=round(latency, 3),
                    model=model,
                    attempts=attempt,
                )
            last_err = f"http_{resp.status_code}: {resp.text[:160]}"
            time.sleep(1.5 * attempt)
        except Exception as e:
            last_err = str(e)[:200]
            time.sleep(1.5 * attempt)

    return LLMResult(ok=False, error=last_err, model=model, attempts=max_retries)


def parse_json_output(text: str) -> tuple[Any, str]:
    if not text:
        return None, "empty_output"
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        return json.loads(cleaned), ""
    except Exception:
        pass
    for start, end in (("{", "}"), ("[", "]")):
        i = cleaned.find(start)
        if i < 0:
            continue
        depth = 0
        in_str = False
        esc = False
        for j in range(i, len(cleaned)):
            c = cleaned[j]
            if esc:
                esc = False
                continue
            if c == "\\":
                esc = True
                continue
            if c == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if c == start:
                depth += 1
            elif c == end:
                depth -= 1
                if depth == 0:
                    snippet = cleaned[i : j + 1]
                    try:
                        return json.loads(snippet), ""
                    except Exception as e:
                        return None, f"json_decode_fail: {str(e)[:120]}"
        break
    return None, "no_json_found"
