"""
UXBench unified LLM API client.

Features:
- Multi-endpoint auto-routing
- Multiple API key support
- reasoning_content fallback (thinking models)
- Exponential backoff retry
- Robust JSON parsing (fence stripping + bracket matching + bare integer detection)
- ThreadPoolExecutor parallel calls

Usage:
    from lib.llm_client import call_llm, call_llm_simple, parse_json_output, run_parallel

    # 自动路由版（根据 model 名查 config 表）
    result = call_llm([{"role": "user", "content": "hello"}], model="gpt-5.1")

    # 显式指定 url/key（兼容旧代码）
    content, reasoning = call_llm_simple(messages, model="gpt-5.5", url=PRIMARY_URL, api_key=KEY)
"""

import json
import re
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable

from lib.config import get_route


# ── 数据结构 ──────────────────────────────────────────────────────────────────


@dataclass
class LLMResult:
    """LLM 调用结果。"""
    ok: bool
    content: str = ""
    reasoning_content: str = ""
    tokens: int = 0
    latency_s: float = 0.0
    status: str = ""
    error: str = ""
    model: str = ""
    attempts: int = 1


# ── 核心调用函数 ──────────────────────────────────────────────────────────────


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
    单次 LLM API 调用（带重试和自动路由）。

    Parameters
    ----------
    messages   : OpenAI 格式 messages 列表
    model      : 模型名称
    url        : API endpoint（None 则自动从 config 查找）
    api_key    : Bearer token（None 则自动查找）
    cookie     : 可选 Cookie 头
    max_tokens : 最大生成 token 数
    temperature: 采样温度
    max_retries: 最大重试次数
    timeout    : 单次请求超时秒数
    **extra    : 额外 payload 参数（如 reasoning_effort, thinking 等）

    Returns
    -------
    LLMResult 对象
    """
    # 自动路由
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
    if temperature is not None:   # None = 不传，适合 Claude Bedrock 等不接受 temperature 的模型
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

                # Fallback: thinking 模型有时 content 为空，答案在 reasoning_content 末尾
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
    简化版 LLM 调用（兼容 api_client.py 接口）。

    Returns
    -------
    (content, reasoning_content)
        失败时 content = "__ERROR__: ..."
    """
    result = call_llm(
        messages, model=model, url=url, api_key=api_key, cookie=cookie,
        max_tokens=max_tokens, max_retries=max_retries, timeout=timeout, **extra,
    )
    if result.ok:
        return result.content, result.reasoning_content
    return f"__ERROR__: {result.error}", ""


# ── JSON 解析 ─────────────────────────────────────────────────────────────────


def parse_json_output(text: str) -> tuple[Any, str]:
    """
    从 LLM 输出中健壮地提取 JSON。

    处理场景：
    - 纯 JSON
    - ```json ... ``` 围栏
    - 前缀文本 + JSON
    - 尾部逗号
    - 裸整数 "1" / "-1"

    Returns
    -------
    (parsed_obj, error_str)  成功时 error 为 ""
    """
    if not text:
        return None, "empty_output"

    cleaned = text.strip()

    # Strategy 1: 剥离 markdown fence
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    # Strategy 2: 直接完整解析
    try:
        return json.loads(cleaned), ""
    except Exception:
        pass

    # Strategy 3: 寻找第一个平衡的 { ... } 或 [ ... ]
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

    # Strategy 4: 裸整数识别（如 "1", "-1"）
    stripped = cleaned.strip().strip('"').strip("'")
    if stripped in ("1", "-1", "0"):
        return {"verdict": int(stripped)}, "bare_int"

    # Strategy 5: 反向搜索 {"verdict": ...} 模式
    verdict_match = re.search(r'\{\s*"verdict"\s*:\s*(-?[01])\s*\}', cleaned)
    if verdict_match:
        return {"verdict": int(verdict_match.group(1))}, "regex_verdict"

    return None, "no_json_found"


# ── 并行执行 ──────────────────────────────────────────────────────────────────


def run_parallel(
    tasks: list[tuple],
    workers: int = 5,
    on_done: Callable | None = None,
) -> dict:
    """
    并行执行多个 LLM 调用。

    Parameters
    ----------
    tasks   : list of (key, messages, model, kwargs_dict)
    workers : 最大并发数
    on_done : 回调函数 (key, LLMResult)，每完成一个调用时触发

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
