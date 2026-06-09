"""
UXBench data loading utilities.

Provides streaming, batch, and testset loading for JSONL files.

Usage:
    from utils.data_loader import load_jsonl, iter_jsonl, load_testset
"""

import json
from pathlib import Path
from typing import Iterator


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    """
    Stream a JSONL file line by line (generator — memory-efficient).

    Yields
    ------
    dict — parsed JSON object for each line
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
    Load a JSONL file into a list.

    Parameters
    ----------
    path  : File path
    limit : If > 0, load only the first N records

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
    Load BAD + GOOD testsets, automatically injecting ground_truth fields.

    Parameters
    ----------
    bad_path  : Path to the BAD testset JSONL file
    good_path : Path to the GOOD testset JSONL file
    limit     : If > 0, load only the first N records from each file

    Returns
    -------
    Merged list[dict] where each record contains a ground_truth field
    (−1 for bad, +1 for good)
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
    """Fast line count for a JSONL file (no JSON parsing)."""
    path = Path(path)
    count = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                count += 1
    return count
