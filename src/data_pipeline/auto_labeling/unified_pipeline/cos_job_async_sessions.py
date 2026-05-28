"""
COS 异步任务：从整包 JSON 拼 session（与本地 unified_pipeline 行为对齐）。

与旧版区别：
- assistant 的「深度思考 / 工具卡片」与「最终回复」**拆开**：
  - `history[].content` = 仅最终正文（与 `session_builder.raw_row_to_turn` 一致）
  - `history[].reasoning` / `reasoning_len` = 思考链 + 工具/引用卡片摘要（若有）

请在运行环境中实现 `download_cos_file` / `upload_cos_file`（或改为 from 你们平台模块 import）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# 本地 unified_pipeline（COS 上需保证 PYTHONPATH 含项目根）
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from auto_labeling.unified_pipeline.session_builder import (
    assistant_reasoning_and_plain_text,
    get_ability_label,
    get_record_id,
    safe_float,
    safe_int,
    safe_str,
)


def download_cos_file(source_file: str) -> bytes | str:
    """由平台注入：从 COS 下载原始 JSON/JSONL 文本或 bytes。"""
    raise RuntimeError("请替换为平台的 download_cos_file(source_file)")


def upload_cos_file(args: dict[str, Any], output_data: str) -> None:
    """由平台注入：上传结果到 COS。"""
    raise RuntimeError("请替换为平台的 upload_cos_file(args, output_data)")


async def main(args: dict[str, Any]) -> dict[str, Any]:
    try:
        file_data = download_cos_file(args["source_file"])
        if isinstance(file_data, bytes):
            file_data = file_data.decode("utf-8")
        data = json.loads(file_data) if isinstance(file_data, str) else file_data

        if isinstance(data, dict):
            items = data.get("outputs", [])
        elif isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], dict) and "outputs" in data[0]:
                items = data[0].get("outputs", [])
            else:
                items = data
        else:
            items = []

        conversations: dict[str, list[dict[str, Any]]] = {}
        user_profiles: dict[str, dict[str, Any]] = {}

        for item in items:
            if not isinstance(item, dict):
                continue

            cid = safe_str(item.get("cid")).strip()
            if not cid:
                continue

            atom_ability = safe_str(item.get("atom_ability"))
            ability_type, involves_image_processing = get_ability_label(atom_ability)

            if cid not in user_profiles:
                user_profiles[cid] = {
                    "cid": cid,
                    "age": safe_str(item.get("age")),
                    "gender": safe_str(item.get("gender")),
                    "occupation": safe_str(item.get("occupation")),
                    "industry": safe_str(item.get("industry")),
                    "education": safe_str(item.get("education")),
                    "nine_groups": safe_str(item.get("nine_groups")),
                    "interest_tags": item.get("interest_tags", ""),
                    "openid": safe_str(item.get("openid")),
                    "register_day": safe_str(item.get("register_day")),
                    "ftime": safe_str(item.get("ftime")),
                }

            user_content = safe_str(item.get("modelprompt")).strip()
            assistant_reasoning, assistant_content = assistant_reasoning_and_plain_text(
                item,
                answer_max_chars=int(args.get("answer_max_chars", 120_000)),
            )

            turn: dict[str, Any] = {
                "record_id": get_record_id(item),
                "cid": cid,
                "convidx": safe_int(item.get("convidx")),
                "chainid": safe_str(item.get("chainid")),
                "regeneateidx": safe_int(item.get("regeneateidx")),
                "regenerateidx": safe_int(item.get("regenerateidx")),
                "ftime": safe_str(item.get("ftime")),
                "user_content": user_content,
                "assistant_content": assistant_content,
                "assistant_reasoning": assistant_reasoning,
                "copy_cnt": safe_int(item.get("copy_cnt")),
                "like_cnt": safe_int(item.get("like_cnt")),
                "unlike_cnt": safe_int(item.get("unlike_cnt")),
                "share_cnt": safe_int(item.get("share_cnt")),
                "click_regen_cnt": safe_int(item.get("click_regen_cnt")),
                "click_picture_cnt": safe_int(item.get("click_picture_cnt")),
                "click_picture_save_cnt": safe_int(item.get("click_picture_save_cnt")),
                "stopgenerate": safe_int(item.get("stopgenerate")),
                "timecosttotal": safe_float(item.get("timecosttotal")),
                "mainbody_len": safe_int(item.get("mainbody_len")),
                "has_attachment": safe_int(item.get("has_attachment")),
                "promptcreatetime": safe_str(item.get("promptcreatetime")),
                "answercreatetime": safe_str(item.get("answercreatetime")),
                "atom_ability": atom_ability,
                "real_ability": safe_str(item.get("real_ability")),
                "output_category1": safe_str(item.get("output_category1")),
                "mas_intent_v2_1st": safe_str(item.get("mas_intent_v2_1st")),
                "badcase_type": safe_str(item.get("badcase_type")),
                "ability_type": ability_type,
                "involves_image_processing": involves_image_processing,
                "app_url": safe_str(item.get("app_url")),
                "chatscene": safe_str(item.get("chatscene")),
                "plat_type": safe_str(item.get("plat_type")),
                "systemprompt": safe_str(item.get("systemprompt")),
                "modelprompt": safe_str(item.get("modelprompt")),
                "answer_msg": item.get("answer_msg", ""),
                "mas_intent": safe_str(item.get("mas_intent")),
                "mas_1nd_intent": safe_str(item.get("mas_1nd_intent")),
                "mas_1nd_sub_intent": safe_str(item.get("mas_1nd_sub_intent")),
                "mas_intent_v2_2nd": safe_str(item.get("mas_intent_v2_2nd")),
                "is_online_search": safe_str(item.get("is_online_search")),
                "search_type": safe_str(item.get("search_type")),
                "is_search": safe_str(item.get("is_search")),
                "is_has_citation": safe_str(item.get("is_has_citation")),
                "realpluginid": safe_str(item.get("realpluginid")),
                "is_deep_seek": safe_str(item.get("is_deep_seek")),
                "interest_tags": safe_str(item.get("interest_tags")),
                "output_category2": safe_str(item.get("output_category2")),
                "output_category3": safe_str(item.get("output_category3")),
                "chatmodel_type": safe_str(item.get("chatmodel_type")),
                "product_type": safe_str(item.get("product_type")),
                "terminal": safe_str(item.get("terminal")),
                "has_feedback": 1
                if (safe_int(item.get("like_cnt")) > 0 or safe_int(item.get("unlike_cnt")) > 0)
                else 0,
                "is_liked": 1 if safe_int(item.get("like_cnt")) > 0 else 0,
                "is_unliked": 1 if safe_int(item.get("unlike_cnt")) > 0 else 0,
            }

            conversations.setdefault(cid, []).append(turn)

        result: list[dict[str, Any]] = []

        for cid, turns in conversations.items():
            if not turns:
                continue

            turns_sorted = sorted(
                turns,
                key=lambda x: (
                    safe_int(x.get("convidx", 0)),
                    safe_int(x.get("regenerateidx", 0)),
                    safe_int(x.get("regeneateidx", 0)),
                    safe_str(x.get("ftime", "")),
                    safe_str(x.get("promptcreatetime", "")),
                ),
            )

            history: list[dict[str, Any]] = []

            for turn in turns_sorted:
                common_stats = {
                    k: v
                    for k, v in turn.items()
                    if k
                    not in (
                        "user_content",
                        "assistant_content",
                        "assistant_reasoning",
                    )
                }

                if safe_str(turn["user_content"]).strip():
                    history.append(
                        {
                            "role": "user",
                            "content": turn["user_content"],
                            "content_len": len(turn["user_content"]),
                            "turn_stats": dict(common_stats),
                        }
                    )

                if safe_str(turn["assistant_content"]).strip():
                    entry: dict[str, Any] = {
                        "role": "assistant",
                        "content": turn["assistant_content"],
                        "content_len": len(turn["assistant_content"]),
                        "turn_stats": dict(common_stats),
                    }
                    rs = safe_str(turn.get("assistant_reasoning", "")).strip()
                    if rs:
                        entry["reasoning"] = rs
                        entry["reasoning_len"] = len(rs)
                    history.append(entry)

            if not history:
                continue

            alternation_issues = 0
            for i in range(1, len(history)):
                if history[i]["role"] == history[i - 1]["role"]:
                    alternation_issues += 1

            session_involves_image_processing = (
                1
                if any(
                    safe_int(msg.get("turn_stats", {}).get("involves_image_processing", 0)) == 1
                    for msg in history
                )
                else 0
            )

            session_ability_types = sorted(
                {safe_str(msg.get("turn_stats", {}).get("ability_type", "unknown")) for msg in history}
            )

            session_stats: dict[str, Any] = {
                "turn_count": len(turns_sorted),
                "message_count": len(history),
                "user_message_count": sum(1 for h in history if h["role"] == "user"),
                "assistant_message_count": sum(1 for h in history if h["role"] == "assistant"),
                "total_like_cnt": sum(safe_int(t.get("like_cnt", 0)) for t in turns_sorted),
                "total_unlike_cnt": sum(safe_int(t.get("unlike_cnt", 0)) for t in turns_sorted),
                "total_copy_cnt": sum(safe_int(t.get("copy_cnt", 0)) for t in turns_sorted),
                "total_share_cnt": sum(safe_int(t.get("share_cnt", 0)) for t in turns_sorted),
                "total_click_regen_cnt": sum(safe_int(t.get("click_regen_cnt", 0)) for t in turns_sorted),
                "total_click_picture_cnt": sum(safe_int(t.get("click_picture_cnt", 0)) for t in turns_sorted),
                "total_click_picture_save_cnt": sum(
                    safe_int(t.get("click_picture_save_cnt", 0)) for t in turns_sorted
                ),
                "total_stopgenerate_cnt": sum(safe_int(t.get("stopgenerate", 0)) for t in turns_sorted),
                "total_timecosttotal": sum(safe_float(t.get("timecosttotal", 0)) for t in turns_sorted),
                "total_mainbody_len": sum(safe_int(t.get("mainbody_len", 0)) for t in turns_sorted),
                "total_content_len": sum(safe_int(h.get("content_len", 0)) for h in history),
                "has_feedback": 1 if any(safe_int(t.get("has_feedback", 0)) == 1 for t in turns_sorted) else 0,
                "alternation_issues": alternation_issues,
                "session_start_time": turns_sorted[0].get("promptcreatetime", ""),
                "session_end_time": turns_sorted[-1].get("answercreatetime", ""),
            }

            result.append(
                {
                    "cid": cid,
                    "history": history,
                    "user_profile": user_profiles.get(cid, {}),
                    "session_involves_image_processing": session_involves_image_processing,
                    "session_ability_types": session_ability_types,
                    "session_stats": session_stats,
                }
            )

        output_data = json.dumps(result, ensure_ascii=False)
        upload_cos_file(args, output_data)

        return {"status": 0, "message": f"文件处理成功，共输出 {len(result)} 个会话"}

    except Exception as e:
        return {"status": 1, "message": f"处理失败: {str(e)}"}
