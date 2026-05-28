"""Deduplicate raw rows by message id (messageid / amessageid / …)."""

from __future__ import annotations

from typing import Any, Iterator

MESSAGE_ID_KEYS = (
    "messageid",
    "message_id",
    "MessageID",
    "umessageid",
    "amessageid",
)


def resolve_dedupe_key_field(preferred: str) -> str | None:
    if preferred and preferred != "auto":
        return preferred
    return None


def get_message_id(row: dict, key_field: str | None) -> str:
    if key_field:
        v = row.get(key_field)
        if v not in (None, ""):
            return str(v).strip()
        return ""
    for k in MESSAGE_ID_KEYS:
        v = row.get(k)
        if v not in (None, ""):
            return str(v).strip()
    tid = str(row.get("traceid") or row.get("traceID") or "").strip()
    conv = str(row.get("convidx") or "").strip()
    cid = str(row.get("cid") or "").strip()
    if tid and conv:
        return f"{tid}:{conv}"
    if cid and conv:
        return f"{cid}:{conv}"
    return ""


def _ts_tuple(row: dict) -> tuple:
    """Sortable time: prefer answercreatetime, then promptcreatetime, then ftime string."""
    a = str(row.get("answercreatetime") or "")
    p = str(row.get("promptcreatetime") or "")
    f = str(row.get("ftime") or "")
    return (a, p, f)


def _pick_row(group: list[dict], *, keep: str) -> dict:
    rev = keep != "first"
    return sorted(group, key=_ts_tuple, reverse=rev)[0]


def dedupe_rows(rows: list[dict], *, key_field: str | None, keep: str = "last") -> tuple[list[dict], dict]:
    """
    Returns (deduped_rows, stats).
    keep: 'last' keeps max timestamp, 'first' keeps min.
    """
    buckets: dict[str, list[dict]] = {}
    no_id: list[dict] = []
    for r in rows:
        mid = get_message_id(r, key_field)
        if not mid:
            no_id.append(r)
            continue
        buckets.setdefault(mid, []).append(r)

    out: list[dict] = []
    duplicate_groups = 0
    removed = 0
    for _mid, group in buckets.items():
        if len(group) == 1:
            out.append(group[0])
            continue
        duplicate_groups += 1
        removed += len(group) - 1
        out.append(_pick_row(group, keep=keep))

    out.extend(no_id)
    stats = {
        "input_rows": len(rows),
        "output_rows": len(out),
        "duplicate_groups": duplicate_groups,
        "dropped_within_group": removed,
        "no_id_rows": len(no_id),
    }
    return out, stats


def dedupe_merge_from_iter(
    rows: Iterator[dict],
    *,
    key_field: str | None,
    keep: str = "last",
) -> tuple[list[dict], dict]:
    """
    流式去重：每个 message id 只保留一条最佳记录，不把同一 id 的多条副本整表放进内存。
    无 id 的行无法去重，全部原样保留（仍占内存）。
    """
    best: dict[str, tuple[tuple, dict]] = {}
    no_id: list[dict] = []
    input_rows = 0
    duplicate_groups = 0
    dropped_within_group = 0
    ids_with_dup: set[str] = set()

    for r in rows:
        input_rows += 1
        mid = get_message_id(r, key_field)
        if not mid:
            no_id.append(r)
            continue
        t = _ts_tuple(r)
        if mid not in best:
            best[mid] = (t, r)
            continue
        if mid not in ids_with_dup:
            ids_with_dup.add(mid)
            duplicate_groups += 1
        dropped_within_group += 1
        old_t, old_r = best[mid]
        if keep == "first":
            if t < old_t:
                best[mid] = (t, r)
        else:
            if t > old_t:
                best[mid] = (t, r)

    out = [best[k][1] for k in best]
    out.extend(no_id)
    stats = {
        "input_rows": input_rows,
        "output_rows": len(out),
        "duplicate_groups": duplicate_groups,
        "dropped_within_group": dropped_within_group,
        "no_id_rows": len(no_id),
    }
    return out, stats
