"""
UXBench unified API configuration.

All LLM API endpoints, keys, and model routing are managed here.
To add a new model, simply add one entry to MODEL_ROUTES.
"""

import os

# ── Endpoints ─────────────────────────────────────────────────────────────────
# Replace with your own LLM proxy endpoints
SECONDARY_URL   = os.environ.get("LLM_SECONDARY_URL", "http://YOUR_LLM_PROXY/v1/chat/completions")
PRIMARY_URL  = os.environ.get("LLM_PRIMARY_URL", "http://YOUR_LLM_PROXY/v1/chat/completions")
LLM_URL  = os.environ.get("LLM_API_URL", "http://YOUR_LLM_PROXY/v1")

# ── API Keys ──────────────────────────────────────────────────────────────────
API_KEY_V1 = os.environ.get("LLM_API_KEY", "YOUR_API_KEY")
API_KEY_V2 = os.environ.get("LLM_API_KEY_V2", "YOUR_API_KEY")

# ── Cookie (some endpoints require session cookies) ───────────────────────────
API_V2_COOKIE = os.environ.get("LLM_API_COOKIE", "")

# ── Model routing table: model_name -> (url, api_key, cookie_or_None) ────────
MODEL_ROUTES: dict[str, tuple[str, str, str | None]] = {
    # Add your model routes here. Example:
    # "gpt-4o":            (PRIMARY_URL, API_KEY_V1, None),
    # "claude-opus-4.7":   (PRIMARY_URL, API_KEY_V2, None),
    # "deepseek-r1":       (SECONDARY_URL,  API_KEY_V1, None),
}

# ── Thinking parameters: model_name -> extra_params dict ─────────────────────
THINKING_PARAMS: dict[str, dict] = {
    # Add thinking/reasoning parameters for specific models. Example:
    # "deepseek-v4-pro": {"reasoning_effort": "high"},
    # "gpt-5.5":         {"reasoning_effort": "high"},
}


def get_route(model: str) -> tuple[str, str, str | None]:
    """
    Auto-select (url, key, cookie) based on model name.

    Lookup priority:
    1. MODEL_ROUTES exact match
    2. Default fallback → LLM_URL + V1 key
    """
    if model in MODEL_ROUTES:
        return MODEL_ROUTES[model]
    return LLM_URL, API_KEY_V1, None


def get_thinking_params(model: str) -> dict:
    """返回模型的 thinking extra 参数（无则返回空 dict）。"""
    return THINKING_PARAMS.get(model, {})
