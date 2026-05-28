"""
UXBench 共享 Prompt 模板 + verdict 解析函数。

包含：
- POINTWISE_GRM: 评测 prompt（判断好/差）
- BINARY_VERDICT_PROMPT: 二元评测 prompt（判断 1/-1 JSON）
- build_judge_prompt(): 将对话填充进模板
- extract_verdict(): 解析好/差（增强版，支持 reasoning fallback）
- extract_binary_verdict(): 解析 JSON verdict

Usage:
    from lib.prompts import POINTWISE_GRM, build_judge_prompt, extract_verdict
"""


# ── 评测 Prompt（输出"好"或"差"）─────────────────────────────────────────

POINTWISE_GRM = """# Role
你是一位用户体验评估专家。你的任务是判断AI助手的回复是否能够令用户满意。
# Task
请根据以下对话上下文和AI助手的回复，判断该回复是否能让用户满意。评估时应考虑：回复是否准确、完整、有帮助，是否正确理解了用户意图，表达是否清晰恰当。
# Input Data
## 历史对话
{context}
## 用户问题
{prompt}
## AI助手回复
{response}
# Evaluation Criteria
1. 正确性：回复是否包含事实性错误
2. 完整性：回复是否充分回答了用户的问题
3. 意图理解：AI是否正确理解了用户的需求
4. 表达质量：回复是否清晰、简洁、格式恰当（不冗余啰嗦）
5. 实用性：回复对用户是否有实际帮助
# Output Format
请仅输出一个字："好" 或 "差"。
- 输出"好"代表：该回复能够令用户满意
- 输出"差"代表：该回复无法令用户满意
# Constraint
不要输出任何解释、分析或额外的标点符号，只输出最终的判定结果。"""


# ── v0.1 评测 Prompt（输出 JSON verdict）──────────────────────────────────────

BINARY_VERDICT_PROMPT = """你是一个 AI 回复质量评判员，专门评估中文 AI 助手（OrganizationAI Assistant）的回复质量。

## 任务说明
你将收到一段多轮对话历史，以及当前轮用户的提问和 AI 的回复。
请判断：**AI 在当前轮的回复，用户是否满意？**

## 评判标准
请聚焦于内容质量本身，具体包括：
- 是否准确回答了用户的问题（无事实错误、无逻辑问题）
- 是否真正满足了用户的核心需求（任务是否完成）
- 是否存在明显的意图理解偏差或任务未完成情况

## 注意事项（防止误判）
- 回复中使用 emoji、加粗、标题、列表等格式装饰，**不代表回复质量高**
- 回复长度长，**不代表回复质量高**（可能冗余啰嗦）
- 对话历史较长，**不代表当前轮回复质量高**
- 请仅根据当前轮的回复内容是否真正满足用户当前提问来判断

## 输出格式
只输出一个 JSON，不要任何解释：
{"verdict": <-1 或 1>}

- 1 = 满意（回复质量良好，用户不会对此轮回复产生负面反馈）
- -1 = 不满意（回复存在明显质量问题，用户可能会点踩或产生负面反馈）"""


# ── Prompt 填充函数 ───────────────────────────────────────────────────────────


def build_judge_prompt(
    history: list[dict],
    query: str,
    response: str,
    template: str | None = None,
) -> str:
    """
    将对话历史、用户问题、AI 回复填充进 prompt 模板。

    Parameters
    ----------
    history  : list of {"role": "user"/"assistant", "content"/"message": "..."}
    query    : 当前用户问题
    response : AI 生成的回复文本
    template : 自定义模板（默认 POINTWISE_GRM）

    Returns
    -------
    填充后的 prompt 字符串
    """
    if template is None:
        template = POINTWISE_GRM

    ctx_lines = []
    for turn in (history or []):
        role = "用户" if turn.get("role") == "user" else "AI"
        content = turn.get("content") or turn.get("message", "")
        ctx_lines.append(f"**{role}：** {content}")
    context = "\n".join(ctx_lines) if ctx_lines else "（无历史对话）"

    return template.format(context=context, prompt=query, response=response)


def build_v1_user_message(record: dict) -> str:
    """
    将 v0.1 格式的 testset 记录组装成 user message（用于 BINARY_VERDICT_PROMPT）。
    """
    history = record.get("selected_history", [])
    source_query = record.get("source_query", {})
    agent_response = record.get("agent_response_full", "")

    lines = ["## 对话历史"]
    for turn in history:
        role = turn.get("role", "")
        msg = turn.get("message", "")
        role_label = "用户" if role == "user" else "AI"
        lines.append(f"**{role_label}：** {msg}")

    lines.append("\n## 当前轮")
    lines.append(f"**用户提问：**\n{source_query.get('message', '')}")
    lines.append(f"\n**AI 回复：**\n{agent_response}")

    return "\n\n".join(lines)


# ── Verdict 解析函数 ──────────────────────────────────────────────────────────


def extract_verdict(content: str, reasoning: str = "") -> tuple[int | None, str]:
    """
    从模型输出中解析好/差判断（增强版，支持 reasoning fallback）。

    解析顺序（content 优先，content 失败时 fallback 到 reasoning 末尾 500 字）：
    1. 精确匹配 "好" / "差"
    2. 以 "好" / "差" 开头/结尾
    3. 最后一行含 "好" / "差"
    4. rfind("好") vs rfind("差")（排除"不好"）
    5. 关键词兜底："不满意" → 差；"令用户满意" → 好

    Returns
    -------
    (verdict, parse_note)
        verdict    : 1（好）/ -1（差）/ None（无法解析）
        parse_note : 解析方式描述
    """
    reasoning_tail = reasoning[-500:] if reasoning else ""

    for text, src in [(content, "content"), (reasoning_tail, "reasoning")]:
        if not text:
            continue
        s = text.strip()

        if s == "好":
            return 1, f"exact_{src}"
        if s == "差":
            return -1, f"exact_{src}"
        if s.startswith("好"):
            return 1, f"starts_{src}"
        if s.startswith("差"):
            return -1, f"starts_{src}"
        if s.endswith("好"):
            return 1, f"ends_{src}"
        if s.endswith("差"):
            return -1, f"ends_{src}"

        last = s.splitlines()[-1].strip() if s.splitlines() else ""
        if "好" in last and "差" not in last and "不好" not in last:
            return 1, f"last_line_{src}"
        if "差" in last and "好" not in last:
            return -1, f"last_line_{src}"

        lh = text.rfind("好")
        lc = text.rfind("差")
        if lh >= 1 and text[lh - 1] == "不":
            lh = -1
        if lh > lc >= 0 or (lh >= 0 and lc < 0):
            return 1, f"rfind_{src}"
        if lc > lh >= 0 or (lc >= 0 and lh < 0):
            return -1, f"rfind_{src}"

        if "不满意" in text or "无法令用户满意" in text:
            return -1, f"kw_bad_{src}"
        if "令用户满意" in text and "不令用户满意" not in text:
            return 1, f"kw_good_{src}"

    return None, "cannot_parse"


def extract_binary_verdict(text: str) -> tuple[int | None, str]:
    """
    从 LLM 输出中解析 JSON verdict（1 或 -1）。

    解析策略（渐进式）：
    1. 完整 JSON 解析（含 fence 剥离）
    2. 正则提取 "verdict": N
    3. 裸整数识别

    Returns
    -------
    (verdict, parse_note)
    """
    import re

    if not text:
        return None, "empty"

    cleaned = text.strip()

    # Strip fence
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    # Strategy 1: JSON parse
    import json
    try:
        obj = json.loads(cleaned)
        v = obj.get("verdict")
        if v in (1, -1, 0):
            return int(v), "json_full"
    except Exception:
        pass

    # Strategy 2: regex
    m = re.search(r'"verdict"\s*:\s*(-?[01])', cleaned)
    if m:
        return int(m.group(1)), "regex"

    # Strategy 3: 反向扫描
    m = re.search(r'\{\s*"verdict"\s*:\s*(-?[01])\s*\}', text)
    if m:
        return int(m.group(1)), "reverse_scan"

    # Strategy 4: 裸整数
    stripped = cleaned.strip().strip('"').strip("'")
    if stripped in ("1", "-1"):
        return int(stripped), "bare_int"

    return None, "no_verdict_found"
