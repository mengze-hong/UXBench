"""
UXBench API Configuration.

Set the following environment variables (or use a .env file):
  OPENAI_API_BASE  - Your LLM API base URL (OpenAI-compatible)
  OPENAI_API_KEY   - Your API key
"""

import os
import warnings
from dotenv import load_dotenv

load_dotenv()

# ── Single API endpoint ───────────────────────────────────────────────────────
API_BASE = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
API_URL  = f"{API_BASE.rstrip('/')}/chat/completions"
API_KEY  = os.environ.get("OPENAI_API_KEY", "")

if not API_KEY:
    warnings.warn(
        "OPENAI_API_KEY is not set. "
        "Please configure it in your environment or .env file.",
        RuntimeWarning,
        stacklevel=2,
    )

# ── Model list (all use the same endpoint + key) ──────────────────────────────
SUPPORTED_MODELS = [
    # Task 1 / UX Judge
    "claude-opus-4.7",
    "claude-opus-4.6",
    "claude-opus-4.5",
    "claude-sonnet-4.5",
    "gpt-5.5",
    "gpt-5.2",
    "gpt-5.1",
    "gpt-5",
    "gpt-5-mini",
    "gpt-4o",
    "gemini-3.1-pro",
    "gemini-3.0-flash",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "deepseek-v4-pro",
    "deepseek-v3.2",
    "deepseek-v3",
    "deepseek-r1",
    "glm-5.1",
    "glm-5",
    "kimi-k2.6",
    "kimi-k2.5",
    "qwen3.6-plus",
    "hunyuan-3",
    "doubao-seed-2.0-pro",
    "doubao-seed-2.0-lite",
    "doubao-seed-1.6",
]

# ── Thinking / reasoning params per model (optional) ─────────────────────────
THINKING_PARAMS: dict[str, dict] = {
    "doubao-seed-2.0-pro":   {"reasoning": {"effort": "high"}},
    "hunyuan-3":             {"reasoning_effort": "high", "temperature": 0.9, "top_p": 1.0},
    "deepseek-v4-pro":       {"reasoning_effort": "high"},
    "glm-5.1":               {"thinking": {"type": "enabled"}},
    "gpt-5.5":               {"reasoning_effort": "high"},
    "gemini-3.1-pro":        {"reasoning_effort": "high"},
}


def get_route(model: str) -> tuple[str, str, None]:
    """Return (api_url, api_key, cookie=None) for the given model."""
    return API_URL, API_KEY, None


def get_thinking_params(model: str) -> dict:
    """Return extra reasoning params for a model (empty dict if none)."""
    return THINKING_PARAMS.get(model, {})
