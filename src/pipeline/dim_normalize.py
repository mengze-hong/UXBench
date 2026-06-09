"""
Shared dimension normalization rules.
Used by both pipeline.py (at write time) and quality_enhance.py (post-hoc).

Maps LLM-generated free-text failure dimensions to canonical categories.
Always preserves the original label in `failure_dimension_raw`.

Design principle:
  - LLM freely generates fine-grained labels (发散)
  - This module aggregates them into canonical categories (聚合)
  - Categories are DATA-DRIVEN, not a fixed number — new categories can emerge
  - If a sub-category has enough data and semantic independence, it stays as first-class
  - "其他" should capture < 3% of records
"""

import re

# ── Canonical categories ──────────────────────────────────────────
# This is NOT a fixed set. New categories can be added as the data evolves.
# The principle is: each category should be (1) semantically distinct,
# (2) actionable for downstream tasks, and (3) have ≥5 records.
CANONICAL_DIMS = {
    # Core content quality
    "冗余/啰嗦",           # verbose, repetitive, low info density
    "任务未完成",           # incomplete, not actionable, mission not accomplished
    "事实性错误",           # factual errors, hallucination, contradiction
    "信息可靠性不足",       # uncertain claims, outdated info, no evidence

    # Understanding & alignment
    "意图识别偏差",         # misunderstand user intent, answer off-topic
    "需求澄清不足",        # didn't ask clarifying questions, assumed wrong
    "上下文遗漏/记忆",     # lost context across turns, forgot constraints ← upgraded from sub

    # Interaction & delivery
    "格式/结构不当",        # format/structure issues, length mismatch ← upgraded from sub
    "指令遵循失败",         # didn't follow explicit constraints
    "情感/语气失当",        # wrong tone, lacks empathy
    "风格/受众不匹配",     # style mismatch, audience level wrong ← upgraded from sub

    # Service quality
    "预期落差/过度拒答",    # unnecessary refusal, overly cautious
    "服务恢复失败",         # failed to recover after user complaint/repeat ← upgraded from sub
    "信息不充分/泛化",     # too generic, not specific enough ← upgraded from sub

    # System & safety
    "安全/合规问题",        # safety, privacy, medical advice, copyright
    "系统/服务中断",        # system error, rendering failure, placeholder leak

    # Meta
    "对话管理不当",         # multi-turn strategy issues, unnecessary confirmations
    "其他",                 # truly uncategorizable (target: <3%)
}

# ── Rules: ordered from specific to general ──────────────────────────
# Each rule: (regex_pattern, target_category)
# The first matching rule wins, so more specific patterns go first.
_RULES = [
    # System & infrastructure (most specific first)
    (r"中断|无输出|无响应|服务中断|系统.*失败|不可用|渲染错误|占位符|输出异常|"
     r"系统.*标记.*泄露|格式错误.*系统|工具痕迹|系统.*格式错误|多模态缺失|"
     r"空回复|空响应|模板.*泄露|模板.*残留|渲染异常", "系统/服务中断"),

    # Safety & compliance
    (r"安全|合规|版权|隐私|性化|冲突降级|分诊|医疗安全|年龄.*适配|"
     r"表达不当.*血腥|不当.*协助|代写作业", "安全/合规问题"),

    # Service recovery failure (BEFORE 预期落差 — more specific)
    (r"服务恢复|服务补救|恢复失败|恢复不当|恢复不足|未响应情绪|"
     r"未根据.*反馈.*调整|自适应|未承认错误|未对齐.*关切|"
     r"服务.*体验问题|对话策略.*自适应|多轮不自适应|复读", "服务恢复失败"),

    # Conversation management
    (r"对话管理|多余确认|交互不当|交互不足|继续输出导致|套路化追问|过度追问|"
     r"响应迟缓|交互策略|回答策略", "对话管理不当"),

    # Factual errors (before reliability — more severe)
    (r"事实.*错|幻觉|虚假|虚构|计算错|编造|前后不一致|自相矛盾|伪引用|不实|"
     r"事实性.*前提|内容一致性.*幻觉|编造数据|过度武断|逻辑矛盾|不准确.*不严谨|"
     r"引用.*不清|事实.*不当|事实性.*偏差|误导性|概念性错误|定义不严谨|"
     r"事实性.*可靠性|物理世界.*错误|尺度理解偏差|量级.*偏差|推理错误|"
     r"伪造|不当内容.*攻击|偏见.*贬损|不严谨.*误导|逻辑性错误.*擅自|引用不当", "事实性错误"),

    # Excessive refusal / expectation gap
    (r"拒答|拒绝|过度拒|过度谨慎|预期落差|能力.*边界|过度承诺|推诿|不合理拒|"
     r"拒答体验|合规拒答|帮助不足|过度安全|安全策略不当|过度澄清.*未主动产出|"
     r"过度回避|预期管理|预期不匹配|能力受限.*替代方案|替代方案不足|"
     r"过度保守|回避结论", "预期落差/过度拒答"),

    # Context & memory (standalone — NOT folded into intent)
    (r"上下文遗漏|上下文丢失|上下文跟踪|记忆.*处理|设定跟随|一致性.*设定|一致性.*记忆|"
     r"指代消解|未对齐约束|对话一致性|未更新关键信息|口径不一致|依据不透明|"
     r"未利用上下文|上下文利用不足|上下文.*记忆.*边界|上下文.*角色.*跟踪|"
     r"对话承接不足|上下文约束", "上下文遗漏/记忆"),

    # Intent misunderstanding
    (r"意图.*偏差|意图.*识别|答非所问|答非所需|跑题|误解.*意图|误读.*意图|未对齐.*意图|"
     r"聚焦不足|未命中|偏离|歧义|指代不明|误判情境|意图理解|"
     r"未对齐用户|未对齐细节|理解偏差|理解错误|需求理解偏差|语境不匹配|"
     r"未跟进用户意图|未按预期执行|意图对齐|任务对齐|未满足.*意图|未按意图|"
     r"需求匹配错误|推荐不当|不对焦", "意图识别偏差"),

    # Instruction following (explicit constraints)
    (r"指令|遵循|未遵|未按.*指令|未按.*约束|不遵从|未按指令|未遵循|"
     r"角色.*语境不匹配|未按部门|约束未满足|约束校验|约束.*遵循|约束.*验证|"
     r"关键约束|约束对齐", "指令遵循失败"),

    # Format & structure (standalone — NOT folded into verbose)
    (r"格式.*不[匹当]|格式不匹配|格式问题|格式错误|结构化不足|结构不佳|结构问题|"
     r"篇幅控制|长度失控|过度扩写|字数控制|信息组织|未突出结论|不够直接|"
     r"未按.*格式|格式.*约束|格式.*要素|范围.*不满足|轻微不符合格式|"
     r"结论不[清突]|结构混乱|排版|体裁不匹配|未直接回答|信息.*优先级|"
     r"未分层|长度不匹配|格式/呈现|不符合格式|回答不符合格式|"
     r"表达不够简洁.*结论|结构不当.*结论|未结论优先|创意质量", "格式/结构不当"),

    # Clarification (didn't ask / assumed wrong)
    (r"澄清|追问|需求.*不足|假设不当|过度假设|臆测|泛化推断|信息收集不足|"
     r"未确认版本|粒度控制|不够针对|过度泛化|未澄清|任务范围|"
     r"未抓住关键担忧|量化判断标准|未充分利用.*补充信息|针对性不足", "需求澄清不足"),

    # Style / audience mismatch (standalone — NOT folded into tone)
    (r"风格.*不[匹当符]|受众不匹配|风格不匹配|本地化|可用性不足|密度.*不匹配|"
     r"过度.*中立|回避立场|过度价值中立|叙事密度|"
     r"未按偏好|推荐质量.*相关性|用户期望不匹配|口吻偏教科书|"
     r"风格.*格式不符|不符合用户偏好|偏颇|缺乏辩证|对齐失败.*价值观|"
     r"表达不够地道|立场偏差|不够中立", "风格/受众不匹配"),

    # Tone & empathy (emotional dimension)
    (r"情感|语气|共情|说教|拟人|不专业|机械|冒犯|不当.*语气|缺少共情|模板化表达|"
     r"阳刚|语气不当|情感响应|未满足.*期待|共情不足|表达方式不当|"
     r"表达不当|沟通.*同理心|危机安抚|同理心不足", "情感/语气失当"),

    # Insufficient / generic (standalone — NOT folded into 任务未完成)
    (r"信息不充分|不够具体|泛泛|空泛|不够落地|不够深入|缺乏针对性|"
     r"过度泛化回答|不够个性|模板化|内容质量|表述不严谨|"
     r"不可执行|内容不完整|遗漏|部分满足|泛化不足|"
     r"证据不足|内容不严谨|规范性不足|可操作性不足|"
     r"回答缺少依据|可解释性不足|信息增益有限|受限信息.*处理|"
     r"回答过于笼统|缺乏具体性|生成质量.*创意不足|"
     r"内容不具体|缺少可操作|信息不足.*举例|"
     r"解释不充分|缺少依据|不够实用", "信息不充分/泛化"),

    # Reliability / uncertainty handling
    (r"信息.*可靠|过时|不确定性|论证不足|依据不足|过度自信|过度绝对|条件不足|"
     r"无依据|时间.*知识边界|不确定性.*处理|过度自信.*概率|可信度|"
     r"信息不足时.*区间判断|时效性|时间.*指代|过度确定|缺少限定|过度推断|"
     r"缺乏依据|事实性.*推断|推断不准|概念混淆|边界不清", "信息可靠性不足"),

    # Task incomplete (core — mission not accomplished)
    (r"任务未完成|未完成|未覆盖|未抓住重点|决策支持不足|回答无效|无用|没有实质|"
     r"未给出.*结论|未按需|效率低|未充分解决|未充分满足|"
     r"未聚焦|概括偏题|覆盖不足|摘要.*偏题|"
     r"未满足需求|未满足任务|不相关", "任务未完成"),

    # Verbose / redundant (broad catch — LAST among content rules)
    (r"冗余|冗长|啰嗦|重复|信息.*密度低|信息密度不匹配|不够直达|内部.*细节|思维链|元叙述|内部规划|"
     r"元信息|提示泄露|系统提示|暴露|循环生成|生成失控|可读性|术语|表达不佳|"
     r"表达不清|含糊|优先级不当|未提供有效替代|过长|不简洁|信息密度|"
     r"过度医疗化|未按需简化|不够聚焦|扩展过多|扩写|过度展开|过度复杂|信息噪音|"
     r"输出不够简洁|过度回答|未按要求简洁|过度扩写|verbosity", "冗余/啰嗦"),

    # English dimension names
    (r"query_misinterpretation|misinterpretation|misunderstanding", "意图识别偏差"),

    # Stragglers that can be captured
    (r"未满足.*需求|未满足用户|未解决问题|任务完成度|未按隐含需求", "任务未完成"),
    (r"可执行性不足|可执行性差|未匹配用户水平", "信息不充分/泛化"),
    (r"信息过载|未收敛", "冗余/啰嗦"),
    (r"恢复策略", "服务恢复失败"),
    (r"风格.*约束", "风格/受众不匹配"),
    (r"事实性/.*准确|事实性/.*引用", "事实性错误"),
    (r"上下文/.*指代|指代.*处理失败", "上下文遗漏/记忆"),

    # Catch-all
    (r"无明显失败|可不收录|非失败", "其他"),
]

_COMPILED = [(re.compile(p, re.IGNORECASE), c) for p, c in _RULES]


def normalize_dimension(raw: str) -> str:
    """Map a raw LLM-generated dimension to a canonical category.
    
    The canonical set is NOT fixed — it grows as new meaningful categories emerge.
    Pipeline: raw text → direct match → regex → fallback to "其他".
    """
    if not raw or raw == "?":
        return "其他"
    # Direct match to canonical
    if raw in CANONICAL_DIMS:
        return raw
    # Regex rules
    for regex, cat in _COMPILED:
        if regex.search(raw):
            return cat
    return "其他"


def get_canonical_dims():
    """Return the set of canonical dimension names."""
    return CANONICAL_DIMS.copy()
