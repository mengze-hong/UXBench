"""Load raw items from JSONL or wrapped JSON (COS / batch job shape)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


def parse_top_level_payload(data: Any) -> list[dict]:
    """
    Match legacy job semantics:
    - dict with key outputs -> list
    - list whose first element is dict with outputs
    - list of dict rows
    """
    if isinstance(data, dict):
        items = data.get("outputs")
        if isinstance(items, list):
            return [x for x in items if isinstance(x, dict)]
        return [data] if data else []
    if isinstance(data, list):
        if data and isinstance(data[0], dict) and "outputs" in data[0]:
            items = data[0].get("outputs") or []
            return [x for x in items if isinstance(x, dict)]
        return [x for x in data if isinstance(x, dict)]
    return []


def iter_raw_items_from_path(path: Path, *, max_rows: int = 0) -> Iterator[dict]:
    """
    Stream JSONL rows (one JSON object per line). Does not load the full file.

    If the file is a single JSON array/object instead of JSONL, put it through an
    offline converter — full-document parse is intentionally not done here to
    avoid OOM on multi-GB exports.
    """
    n = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row
                n += 1
                if max_rows and n >= max_rows:
                    return


def load_items_from_bytes_or_str(file_data: bytes | str) -> list[dict]:
    """For in-memory COS-style payloads (used by session_builder.build_sessions_from_payload)."""
    if isinstance(file_data, bytes):
        file_data = file_data.decode("utf-8", errors="replace")
    data = json.loads(file_data) if isinstance(file_data, str) else file_data
    return parse_top_level_payload(data)
