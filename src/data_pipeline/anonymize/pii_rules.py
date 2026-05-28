"""Rule-based PII anonymization helpers."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter

PII_PATTERNS = {
    "phone": re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
    "id_card": re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)"),
    "email": re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"),
    "bank_card": re.compile(r"(?<!\d)\d{16,19}(?!\d)"),
    "ip_address": re.compile(r"(?<!\d)(?:(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)(?!\d)"),
    "detailed_address": re.compile(r"[\u4e00-\u9fa5]{2,8}(?:省|自治区)[\u4e00-\u9fa5]{2,8}(?:市|州)[\u4e00-\u9fa5]{2,10}(?:区|县|镇)"),
    "street_address": re.compile(r"[\u4e00-\u9fa5]{2,10}(?:路|街|道|巷|弄)\d{1,5}号"),
}


def _consistent_hash(original: str, pii_type: str) -> int:
    key = f"{pii_type}:{original}"
    return int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16)


def _replace_phone(match_str: str) -> str:
    h = _consistent_hash(match_str, "phone")
    prefix = ["138", "139", "150", "151", "186", "187"][h % 6]
    suffix = f"{h % 10000:04d}"
    return f"{prefix}****{suffix}"


def _replace_id_card(match_str: str) -> str:
    return match_str[:6] + "********" + match_str[-4:]


def _replace_email(match_str: str) -> str:
    h = _consistent_hash(match_str, "email")
    return f"user_{h % 10000:04d}@example.com"


def _replace_bank_card(match_str: str) -> str:
    return match_str[:4] + "****" + match_str[-4:]


def _replace_ip(match_str: str) -> str:
    h = _consistent_hash(match_str, "ip")
    return f"192.168.{h % 256}.{(h >> 8) % 256}"


def _replace_detailed_address(match_str: str) -> str:
    return "[地址已脱敏]"


def _replace_street_address(match_str: str) -> str:
    return "[街道地址已脱敏]"


REPLACERS = {
    "phone": _replace_phone,
    "id_card": _replace_id_card,
    "email": _replace_email,
    "bank_card": _replace_bank_card,
    "ip_address": _replace_ip,
    "detailed_address": _replace_detailed_address,
    "street_address": _replace_street_address,
}


def anonymize_text(text: str) -> tuple[str, list[dict]]:
    if not text:
        return text, []
    result = text
    changes: list[dict] = []
    for pii_type, pattern in PII_PATTERNS.items():
        matches = list(pattern.finditer(result))
        for m in reversed(matches):
            original = m.group()
            replacement = REPLACERS[pii_type](original)
            result = result[: m.start()] + replacement + result[m.end() :]
            changes.append({"type": pii_type, "original": original, "replacement": replacement, "position": m.start()})
    return result, changes


def _apply_on_history(hist: list, field_prefix: str) -> tuple[list, list[dict]]:
    out = []
    all_changes: list[dict] = []
    for i, turn in enumerate(hist):
        new_turn = dict(turn)
        msg = turn.get("message")
        content = turn.get("content")
        if isinstance(msg, str):
            new_msg, changes = anonymize_text(msg)
            new_turn["message"] = new_msg
            for c in changes:
                c["field"] = f"{field_prefix}[{i}].message"
            all_changes.extend(changes)
        if isinstance(content, str):
            new_content, changes = anonymize_text(content)
            new_turn["content"] = new_content
            for c in changes:
                c["field"] = f"{field_prefix}[{i}].content"
            all_changes.extend(changes)
        out.append(new_turn)
    return out, all_changes


def anonymize_record(rec: dict) -> tuple[dict, list[dict]]:
    anon = dict(rec)
    all_changes: list[dict] = []

    sq = anon.get("source_query")
    if isinstance(sq, dict) and isinstance(sq.get("message"), str):
        new_msg, changes = anonymize_text(sq["message"])
        sq2 = dict(sq)
        sq2["message"] = new_msg
        anon["source_query"] = sq2
        for c in changes:
            c["field"] = "source_query.message"
        all_changes.extend(changes)

    for field in ("agent_response_full", "liked_response_full", "explanation", "system_prompt"):
        if isinstance(anon.get(field), str):
            new_text, changes = anonymize_text(anon[field])
            anon[field] = new_text
            for c in changes:
                c["field"] = field
            all_changes.extend(changes)

    if isinstance(anon.get("selected_history"), list):
        anon["selected_history"], changes = _apply_on_history(anon["selected_history"], "selected_history")
        all_changes.extend(changes)

    if isinstance(anon.get("full_history"), list):
        anon["full_history"], changes = _apply_on_history(anon["full_history"], "full_history")
        all_changes.extend(changes)

    if isinstance(anon.get("history"), list):
        anon["history"], changes = _apply_on_history(anon["history"], "history")
        all_changes.extend(changes)

    if isinstance(anon.get("user_profile"), dict):
        up = dict(anon["user_profile"])
        sensitive_keys = ["name", "姓名", "phone", "电话", "手机", "email", "邮箱", "address", "地址", "身份证", "id_card", "qq", "wechat", "微信号"]
        for key in list(up.keys()):
            key_lower = str(key).lower()
            if any(sk in key_lower for sk in sensitive_keys):
                old_val = up.get(key)
                if old_val not in (None, "", "null"):
                    up[key] = "[已脱敏]"
                    all_changes.append({"field": f"user_profile.{key}", "type": "profile_field", "original": str(old_val)[:80], "replacement": "[已脱敏]"})
        up_text = json.dumps(up, ensure_ascii=False)
        up_new, up_changes = anonymize_text(up_text)
        try:
            anon["user_profile"] = json.loads(up_new)
        except Exception:
            anon["user_profile"] = up
        for c in up_changes:
            c["field"] = "user_profile"
        all_changes.extend(up_changes)

    return anon, all_changes


def summarize_changes(change_logs: list[dict]) -> dict:
    counter = Counter()
    for entry in change_logs:
        for ch in entry.get("changes", []):
            counter[ch.get("type", "unknown")] += 1
    return dict(counter)
