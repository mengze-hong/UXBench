"""
Backward-compatible LLM client wrapper for src/pipeline/.

The canonical implementation lives in src/utils/llm_client.py.
This file re-exports every public symbol so that all pipeline modules
using `from llm_client import ...` continue to work unchanged.
"""

import sys
from pathlib import Path

# Ensure src/ (the package root) is on the path so `utils.*` can be found
_SRC_ROOT = Path(__file__).resolve().parents[1]   # …/src
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

# Re-export all public symbols from the canonical utils implementation
from utils.llm_client import (  # noqa: E402, F401
    LLMResult,
    call_llm,
    call_llm_simple,
    parse_json_output,
    run_parallel,
)
from utils.config import (  # noqa: E402, F401
    API_URL,
    API_KEY,
    get_route,
    get_thinking_params,
)
