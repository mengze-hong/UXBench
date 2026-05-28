"""
Backward-compatible LLM client wrapper.

真正实现已移至项目根目录 lib/llm_client.py，此文件作为向后兼容 wrapper，
确保所有 `from llm_client import ...` 的旧代码无需修改即可正常运行。
"""

import sys
from pathlib import Path

# 将项目根目录加入 sys.path，以便 import lib.*
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Re-export 所有公开接口
from lib.llm_client import (  # noqa: E402, F401
    LLMResult,
    call_llm,
    call_llm_simple,
    parse_json_output,
    run_parallel,
)
from lib.config import (  # noqa: E402, F401
    SECONDARY_URL,
    PRIMARY_URL,
    LLM_URL as API_URL,
    API_KEY_V1 as _DEFAULT_KEY,
    get_route,
)  # Endpoint aliases for backward compatibility
