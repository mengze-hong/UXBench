"""
UXBench unified LLM API client.

Consolidates all LLM call capabilities used across the project:
- Multi-endpoint automatic routing (configured via utils.config)
- Exponential-backoff retries
- Robust JSON parsing (fence stripping + bracket matching + bare-integer detection)
- ThreadPoolExecutor parallel calls

Usage:
    from utils.llm_client import call_llm, call_llm_simple, parse_json_output, run_parallel

    # Auto-routed call (looks up url/key from config table by model name)
    result = call_llm([{"role": "user", "content": "hello"}], model="gpt-5.1")

    # Explicit url/key (for backward compatibility)
    content, reasoning = call_llm_simple(messages, model="gpt-5.5", url=URL, api_key=KEY)
"""

import json
import re
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable

from utils.config import get_route


# ── Data structures ───────────────────────────────────────────────────────────


@dataclass
class LLMResult:
    """Result object returned by a single LLM API call."""
    ok: bool
    content: str = ""
    reasoning_content: str = ""
    tokens: int = 0
    latency_s: float = 0.0
    status: str = ""
    error: str = ""
    model: str = ""
    attempts: int = 1


# ── Core call function ────────────────────────────────────────────────────────


def call_llm(
    messages: list[dict],
    model: str = "gpt-5.1",
    url: str | None = None,
    api_key: str | None = None,
    cookie: str | None = None,
    max_tokens: int = 4096,
    temperature: float | None = 0.2,
    max_retries: int = 3,
    timeout: int = 180,
    **extra,
) -> LLMResult:
    """
    Single LLM API call with retry and auto-routing.

    Parameters
    ----------
    messages   : OpenAI-format messages list
    model      : Model name
    url        : API endpoint (None = auto-lookup from config)
    api_key    : Bearer token (None = auto-lookup from config)
    cookie     : Optional Cookie header
    max_tokens : Maximum tokens to generate
    temperature: Sampling temperature (None = omit from payload, for models that don't accept it)
    max_retries: Maximum number of retry attempts
    timeout    : Per-request timeout in seconds
    **extra    : Additional payload params (e.g. reasoning_effort, thinking, etc.)

    Returns
    -------
    LLMResult object
    """
    # Auto-route if url/key not provided
    if url is None or api_key is None:
        _url, _key, _cookie = get_route(model)
        url = url or _url
        api_key = api_key or _key
        cookie = cookie if cookie is not None else _cookie

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    if cookie:
        headers["Cookie"] = cookie

    payload: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        **extra,
    }
    if temperature is not None:   # None = omit; some models (e.g. Claude Bedrock) don't accept temperature
        payload["temperature"] = temperature

    last_err = ""
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

                msg_obj = choices[0]["message"]
                content = msg_obj.get("content") or ""
                reasoning_content = msg_obj.get("reasoning_content") or msg_obj.get("thinking") or ""

                # Fallback: some thinking models return empty content; answer is at end of reasoning_content
                if not content.strip() and reasoning_content:
                    content = reasoning_content

                tokens = data.get("usage", {}).get("total_tokens", 0)
                return LLMResult(
                    ok=True,
                    content=content,
                    reasoning_content=reasoning_content,
                    tokens=tokens,
                    latency_s=round(latency, 3),
                    status="OK",
                    model=model,
                    attempts=attempt,
                )

            elif resp.status_code == 429:
                last_err = f"rate_limited_{resp.status_code}"
                time.sleep(2 * attempt + 1)
            else:
                last_err = f"http_{resp.status_code}: {resp.text[:200]}"
                time.sleep(1.5 * attempt)

        except requests.Timeout:
            last_err = "timeout"
            time.sleep(2 * attempt)
        except Exception as e:
            last_err = f"exception: {str(e)[:150]}"
            time.sleep(1.5 * attempt)

    return LLMResult(
        ok=False, status="FAIL", error=last_err,
        model=model, attempts=max_retries,
        latency_s=0.0,
    )


def call_llm_simple(
    messages: list[dict],
    model: str,
    url: str | None = None,
    api_key: str | None = None,
    cookie: str | None = None,
    max_tokens: int = 12000,
    max_retries: int = 5,
    timeout: int = 180,
    **extra,
) -> tuple[str, str]:
    """
    Simplified LLM call returning a (content, reasoning_content) tuple.

    Returns
    -------
    (content, reasoning_content)
        On failure, content = "__ERROR__: ..."
    """
    result = call_llm(
        messages, model=model, url=url, api_key=api_key, cookie=cookie,
        max_tokens=max_tokens, max_retries=max_retries, timeout=timeout, **extra,
    )
    if result.ok:
        return result.content, result.reasoning_content
    return f"__ERROR__: {result.error}", ""


# ── JSON parsing ──────────────────────────────────────────────────────────────


def parse_json_output(text: str) -> tuple[Any, str]:
    """
    Robustly extract JSON from LLM output.

    Handles:
    - Pure JSON
    - ```json ... ``` fenced blocks
    - Prefix text followed by JSON
    - Trailing commas
    - Bare integers ("1" / "-1")

    Returns
    -------
    (parsed_obj, error_str)  — error is "" on success
    """
    if not text:
        return None, "empty_output"

    cleaned = text.strip()

    # Strategy 1: strip markdown fence
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    # Strategy 2: direct full parse
    try:
        return json.loads(cleaned), ""
    except Exception:
        pass

    # Strategy 3: find first balanced { ... } or [ ... ]
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        first = cleaned.find(start_char)
        if first < 0:
            continue
        depth = 0
        in_str = False
        esc = False
        for i in range(first, len(cleaned)):
            c = cleaned[i]
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
            if c == start_char:
                depth += 1
            elif c == end_char:
                depth -= 1
                if depth == 0:
                    snippet = cleaned[first:i + 1]
                    try:
                        return json.loads(snippet), ""
                    except Exception as e:
                        return None, f"json_decode_fail: {str(e)[:100]}"
        break

    # Strategy 4: bare integer detection ("1", "-1", "0")
    stripped = cleaned.strip().strip('"').strip("'")
    if stripped in ("1", "-1", "0"):
        return {"verdict": int(stripped)}, "bare_int"

    # Strategy 5: reverse-scan for {"verdict": ...} pattern
    verdict_match = re.search(r'\{\s*"verdict"\s*:\s*(-?[01])\s*\}', cleaned)
    if verdict_match:
        return {"verdict": int(verdict_match.group(1))}, "regex_verdict"

    return None, "no_json_found"


# ── Parallel execution ────────────────────────────────────────────────────────


def run_parallel(
    tasks: list[tuple],
    workers: int = 5,
    on_done: Callable | None = None,
) -> dict:
    """
    Execute multiple LLM calls in parallel.

    Parameters
    ----------
    tasks   : list of (key, messages, model, kwargs_dict)
    workers : maximum concurrency
    on_done : optional callback (key, LLMResult) invoked after each completed call

    Returns
    -------
    {key: LLMResult}
    """
    results = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {
            pool.submit(call_llm, msgs, model=model, **kwargs): key
            for key, msgs, model, kwargs in tasks
        }
        for future in as_completed(future_map):
            key = future_map[future]
            try:
                result = future.result()
            except Exception as e:
                result = LLMResult(ok=False, error=str(e), status="EXCEPTION")
            results[key] = result
            if on_done:
                on_done(key, result)
    return results
