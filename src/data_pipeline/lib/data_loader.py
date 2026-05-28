"""
UXBench 数据加载工具。

提供 JSONL 文件的流式读取、批量加载和测试集加载功能。

Usage:
    from lib.data_loader import load_jsonl, iter_jsonl, load_testset
"""

import json
from pathlib import Path
from typing import Iterator


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    """
    流式逐行读取 JSONL 文件（生成器，内存友好）。

    Yields
    ------
    dict — 每行解析后的 JSON 对象
    """
    path = Path(path)
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def load_jsonl(path: str | Path, limit: int = 0) -> list[dict]:
    """
    加载 JSONL 文件到列表。

    Parameters
    ----------
    path  : 文件路径
    limit : 若 > 0，只加载前 N 条

    Returns
    -------
    list[dict]
    """
    records = []
    for rec in iter_jsonl(path):
        records.append(rec)
        if limit > 0 and len(records) >= limit:
            break
    return records


def load_testset(
    bad_path: str | Path,
    good_path: str | Path,
    limit: int = 0,
) -> list[dict]:
    """
    加载 BAD + GOOD 测试集，自动补充 ground_truth 字段。

    Parameters
    ----------
    bad_path  : BAD testset JSONL 文件路径
    good_path : GOOD testset JSONL 文件路径
    limit     : 若 > 0，每个文件只取前 N 条

    Returns
    -------
    合并后的 list[dict]，每条包含 ground_truth 字段
    """
    records = []

    for rec in iter_jsonl(bad_path):
        if rec.get("ground_truth") is None:
            rec["ground_truth"] = -1
        records.append(rec)
        if limit > 0 and len(records) >= limit:
            break

    count = 0
    for rec in iter_jsonl(good_path):
        if rec.get("ground_truth") is None:
            rec["ground_truth"] = 1
        records.append(rec)
        count += 1
        if limit > 0 and count >= limit:
            break

    return records


def count_jsonl(path: str | Path) -> int:
    """快速计数 JSONL 文件行数（不解析 JSON）。"""
    path = Path(path)
    count = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                count += 1
    return count
