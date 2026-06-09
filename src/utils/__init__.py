"""
UXBench shared utilities — core reusable modules for the project.

Re-exports the most commonly used symbols from the utils sub-modules
so callers can do ``from utils import load_jsonl, call_llm`` etc.
"""

from utils.config import API_KEY, API_URL, get_route, get_thinking_params
from utils.data_loader import load_jsonl, iter_jsonl, load_testset, count_jsonl
from utils.llm_client import call_llm, call_llm_simple, parse_json_output, run_parallel, LLMResult
from utils.checkpoint import load_done_cids, load_done_with_clean, append_record, load_judge_done
from utils.prompts import (
    POINTWISE_GRM,
    BINARY_VERDICT_PROMPT,
    build_judge_prompt,
    build_v1_user_message,
    extract_verdict,
    extract_binary_verdict,
)

__all__ = [
    # config
    "API_KEY", "API_URL", "get_route", "get_thinking_params",
    # data_loader
    "load_jsonl", "iter_jsonl", "load_testset", "count_jsonl",
    # llm_client
    "call_llm", "call_llm_simple", "parse_json_output", "run_parallel", "LLMResult",
    # checkpoint
    "load_done_cids", "load_done_with_clean", "append_record", "load_judge_done",
    # prompts
    "POINTWISE_GRM", "BINARY_VERDICT_PROMPT",
    "build_judge_prompt", "build_v1_user_message",
    "extract_verdict", "extract_binary_verdict",
]
