"""`may data/` 目录约定（整理后子路径），供各脚本默认参数复用。"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MAY_ROOT = PROJECT_ROOT / "may data"
MAY_RAW = MAY_ROOT / "raw"
MAY_SESSIONS = MAY_ROOT / "sessions"
MAY_TRAIN_FROM_SESSIONS = MAY_ROOT / "train_from_sessions"
MAY_LOGS = MAY_ROOT / "logs"
MAY_LEGACY_V02 = MAY_ROOT / "legacy_v02"

# 原始五月 JSONL（文件名含中文）
DEFAULT_DISLIKE_JSONL = MAY_RAW / "点踩-五月1-10.jsonl"
DEFAULT_LIKE_JSONL = MAY_RAW / "点赞-五月1-10.jsonl"

DEFAULT_BAD_SESSIONS_JSONL = MAY_SESSIONS / "bad_sessions_may.jsonl"
DEFAULT_GOOD_SESSIONS_JSONL = MAY_SESSIONS / "good_sessions_may.jsonl"

DEFAULT_BAD_TRAIN_FROM_SESSIONS = MAY_TRAIN_FROM_SESSIONS / "may_bad_train_from_sessions.jsonl"
DEFAULT_GOOD_TRAIN_FROM_SESSIONS = MAY_TRAIN_FROM_SESSIONS / "may_good_train_from_sessions.jsonl"
