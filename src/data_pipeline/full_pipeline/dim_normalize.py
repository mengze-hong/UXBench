"""Normalize failure dimensions to canonical labels."""

from __future__ import annotations

import re

_RULES = [
    (r"事实|幻觉|编造|错误|不实", "事实性错误"),
    (r"意图|答非所问|跑题|误解", "意图识别偏差"),
    (r"冗余|啰嗦|重复|过长", "冗余/啰嗦"),
    (r"任务未完成|未完成|遗漏|部分满足", "任务未完成"),
    (r"可靠|过时|依据不足|不确定", "信息可靠性不足"),
    (r"格式|结构|排版", "格式/结构不当"),
    (r"语气|共情|冒犯|情感", "情感/语气失当"),
    (r"拒答|预期落差|过度保守", "预期落差/过度拒答"),
]

_COMPILED = [(re.compile(p, re.IGNORECASE), cat) for p, cat in _RULES]


def normalize_dimension(raw: str) -> str:
    if not raw or raw == "?":
        return "其他"
    for regex, cat in _COMPILED:
        if regex.search(raw):
            return cat
    return raw if len(raw) <= 20 else "其他"
