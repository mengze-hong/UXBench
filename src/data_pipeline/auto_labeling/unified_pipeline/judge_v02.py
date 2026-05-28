"""Single-call LLM judge for UXBench-style train rows (lite pipeline)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

CORE = Path(__file__).resolve().parents[1] / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from llm_client import call_llm, parse_json_output  # type: ignore


def _trunc(s: str, n: int = 7000) -> str:
    s = s or ""
    return s if len(s) <= n else s[:n] + "\n...[截断]"


BAD_SYSTEM = """你是数据质检与标注审核员。输入为一条「点踩」对话样本（用户提问 + 上文 + AI回复）。
请输出**一个 JSON 对象**（不要 markdown），字段必须齐全：
{
  "failure_dimension": "8类之一：理解偏差|信息未给出|信息可靠性差|图文显示偏差|实用性差|指令遵循失败|信息量不足|冗余啰嗦|系统功能错误|任务未完成|回答无效/无用",
  "failure_dimension_raw": "从回复中概括的原始短语",
  "scenario": "7类之一：信息/知识咨询|创作写作|商品/服务咨询|学习/教育辅助|私人健康/个人生活|情绪/心理支持|私密与生活决策辅助|其他",
  "severity_tier": "moderate 或 severe 或 null",
  "explanation": "一句话说明为何该回复应被点踩（中文）",
  "signal_confidence": "high|medium|low",
  "representativeness": "high|medium|low",
  "needs_context": false,
  "needs_image": false,
  "is_duplicate": false,
  "judge_scores": {
    "query_completeness": 1-5整数,
    "signal_credibility": 1-5整数,
    "representativeness": 1-5整数,
    "severity": 1-5整数,
    "annotation_clarity": 1-5整数
  },
  "judge_audit": "简短审计说明",
  "overall_quality": "high|medium|low — 指这条「训练/评测样本」本身是否清晰可用，不是指用户对AI是否满意",
  "confidence": 0-1小数,
  "qa_verdict": "keep 或 delete",
  "qa_issues": [],
  "qa_notes": "一句质检备注"
}
若样本噪声大、信号不可信、或不适合做训练数据，qa_verdict 用 delete。"""


GOOD_SYSTEM = """你是数据质检与标注审核员。输入为一条「点赞」对话样本。
请输出**一个 JSON 对象**（不要 markdown），字段必须齐全：
{
  "success_dimension": "8类之一：准确回答|全面详细|知识深度|实用性/操作指导|解决问题|任务完成|创意生成|共情支持",
  "scenario": "同上 bad 的 scenario 分类",
  "sentiment": "implicit 或 explicit",
  "explanation": "一句话说明为何该回复值得点赞（中文）",
  "tags": ["success_testcase"] 或 ["success_testcase_explicit"],
  "signal_confidence": "high|medium|low",
  "representativeness": "high|medium|low",
  "praise_snippet": null 或 用户好评片段字符串,
  "judge_average": 1-5数字（综合质量分）,
  "overall_quality": "high|medium|low — 指样本是否适合作为高质量训练数据（标注是否清晰），不是点赞强度",
  "qa_verdict": {
    "verdict": "keep 或 delete",
    "quality": "high|medium|low",
    "issues": [],
    "notes": "一句备注",
    "corrected_dimension": null
  }
}
若样本不适合做高质量 good case，qa_verdict.verdict 用 delete。"""


def judge_bad_case(case: dict, *, model: str) -> tuple[dict[str, Any] | None, str]:
    user_block = {
        "cid": case.get("cid"),
        "source_query": case.get("source_query"),
        "selected_history": _trunc(json.dumps(case.get("selected_history") or [], ensure_ascii=False)),
        "agent_response": _trunc(case.get("agent_response_full") or ""),
    }
    messages = [
        {"role": "system", "content": BAD_SYSTEM},
        {"role": "user", "content": json.dumps(user_block, ensure_ascii=False)},
    ]
    r = call_llm(messages, model=model, max_tokens=1200, temperature=0.15, timeout=120)
    if not r.ok:
        return None, r.error or "llm_fail"
    parsed, err = parse_json_output(r.content)
    if err or not isinstance(parsed, dict):
        return None, err or "parse_fail"
    parsed["_model"] = model
    return parsed, ""


def judge_good_case(case: dict, *, model: str) -> tuple[dict[str, Any] | None, str]:
    user_block = {
        "cid": case.get("cid"),
        "source_query": case.get("source_query"),
        "selected_history": _trunc(json.dumps(case.get("selected_history") or [], ensure_ascii=False)),
        "agent_response": _trunc(case.get("liked_response_full") or ""),
    }
    messages = [
        {"role": "system", "content": GOOD_SYSTEM},
        {"role": "user", "content": json.dumps(user_block, ensure_ascii=False)},
    ]
    r = call_llm(messages, model=model, max_tokens=1200, temperature=0.15, timeout=120)
    if not r.ok:
        return None, r.error or "llm_fail"
    parsed, err = parse_json_output(r.content)
    if err or not isinstance(parsed, dict):
        return None, err or "parse_fail"
    parsed["_model"] = model
    return parsed, ""
