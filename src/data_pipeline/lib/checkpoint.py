"""
UXBench 断点续跑工具。

支持评测脚本和生成脚本的断点续跑（checkpoint / resume），
避免重复跑已完成的记录。

Usage:
    from lib.checkpoint import load_done_cids, append_record, load_judge_done
"""

import json
from pathlib import Path
from threading import Lock


def load_done_cids(path: str | Path) -> set[str]:
    """
    从结果 JSONL 文件中加载已完成的 cid 集合。

    只保留 verdict 不为 None 的记录（即有效结果）。

    Parameters
    ----------
    path : 结果 JSONL 文件路径

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
            if cid and r.get("verdict") is not None:
                done.add(cid)
        except Exception:
            pass

    return done


def load_done_with_clean(path: str | Path) -> tuple[set[str], list[str]]:
    """
    从结果文件中加载断点状态，同时清理无效行。

    过滤掉：__ERROR__ 行、空响应行、重复 cid 行。

    Parameters
    ----------
    path : 输出 JSONL 文件路径

    Returns
    -------
    (done_cids, clean_lines)
        done_cids  : set — 已成功完成的 cid
        clean_lines: list — 合法的 JSONL 行
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
            # 检查是否有有效响应
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
    将干净行列表原子覆盖写回文件（UTF-8，每行末尾换行）。
    """
    path = Path(path)
    content = "\n".join(lines) + ("\n" if lines else "")
    path.write_text(content, encoding="utf-8")


def append_record(path: str | Path, record: dict, lock: Lock | None = None) -> None:
    """
    追加一条 JSON 记录到 JSONL 文件。

    Parameters
    ----------
    path   : 目标文件路径
    record : 要写入的字典
    lock   : 可选 threading.Lock（多线程安全写入）
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
    从 judge 结果文件中加载断点状态。
    只保留 verdict 不为 None 的记录。

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
