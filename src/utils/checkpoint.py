"""
UXBench checkpoint / resume utilities.

Supports incremental evaluation and generation scripts by persisting
completed record IDs to disk, preventing redundant re-processing.

Usage:
    from utils.checkpoint import load_done_cids, append_record, load_judge_done
"""

import json
from pathlib import Path
from threading import Lock


def load_done_cids(path: str | Path) -> set[str]:
    """
    Load the set of completed cids from an existing results JSONL file.

    A record is considered complete if it has a non-None value in either
    the 'predicted' field (run_eval.py output) or the legacy 'verdict' field.

    Parameters
    ----------
    path : Path to the results JSONL file

    Returns
    -------
    set of cid strings
    """
    path = Path(path)
    done = set()
    if not path.exists():
        return done

    for line in path.read_text("utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            cid = r.get("cid")
            # Accept either field name used by different eval scripts
            finished = r.get("predicted") is not None or r.get("verdict") is not None
            if cid and finished:
                done.add(cid)
        except Exception:
            pass

    return done


def load_done_with_clean(path: str | Path) -> tuple[set[str], list[str]]:
    """
    Load checkpoint state from a results file, also returning cleaned valid lines.

    Filters out: __ERROR__ lines, empty-response lines, and duplicate cid lines.

    Parameters
    ----------
    path : Path to the output JSONL file

    Returns
    -------
    (done_cids, clean_lines)
        done_cids  : set — successfully completed cids
        clean_lines: list — valid JSONL lines (suitable for rewriting the file)
    """
    path = Path(path)
    done_cids: set = set()
    clean_lines: list = []

    if not path.exists():
        return done_cids, clean_lines

    raw = path.read_bytes().decode("utf-8", errors="replace")
    seen: set = set()

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            cid = r.get("cid")
            # Check whether the record has a valid (non-error) response
            resp = (r.get("generated_response") or r.get("content") or "").strip()
            verdict = r.get("verdict")
            has_valid = (resp and "__ERROR__" not in resp) or verdict is not None
            if has_valid and cid and cid not in seen:
                seen.add(cid)
                clean_lines.append(line)
        except Exception:
            pass

    done_cids = seen
    return done_cids, clean_lines


def rewrite_clean(path: str | Path, lines: list[str]) -> None:
    """
    Atomically overwrite a file with a clean set of JSONL lines (UTF-8, one line per entry).
    """
    path = Path(path)
    content = "\n".join(lines) + ("\n" if lines else "")
    path.write_text(content, encoding="utf-8")


def append_record(path: str | Path, record: dict, lock: Lock | None = None) -> None:
    """
    Append a single JSON record to a JSONL file.

    Parameters
    ----------
    path   : Target file path
    record : Dictionary to serialize and write
    lock   : Optional threading.Lock for thread-safe concurrent writes
    """
    path = Path(path)
    line = json.dumps(record, ensure_ascii=False) + "\n"

    if lock:
        with lock:
            with path.open("a", encoding="utf-8") as f:
                f.write(line)
    else:
        with path.open("a", encoding="utf-8") as f:
            f.write(line)


def load_judge_done(path: str | Path) -> dict[str, dict]:
    """
    Load checkpoint state from a judge results file.
    Only records with a non-None verdict are included.

    Returns
    -------
    dict[cid -> record]
    """
    path = Path(path)
    done: dict = {}
    if not path.exists():
        return done

    for line in path.read_text("utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            if r.get("verdict") is not None:
                done[r["cid"]] = r
        except Exception:
            pass

    return done
