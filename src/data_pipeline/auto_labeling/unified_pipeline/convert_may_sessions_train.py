#!/usr/bin/env python3
"""
将 `may data/sessions/` 下 bad/good_sessions_may.jsonl（完整 session 行）
转为与 UXBench `bad_train_15k.jsonl` / `good_train_15k.jsonl` 同键、同顺序的训练行。

流程：session → session_to_core_record →（默认）extract_full_session_train_case
每 session 一行、``selected_history`` 仅为**信号轮 user 之前**的上下文（与 lite 截断规则一致），
**且必须存在显式点踩/点赞轮**，否则跳过；
加 ``--lite-signal-rows`` 则退回 extract_lite_cases（可多行 per session）→（可选）sanitize / outlier
→ build_*_train_row + 中性 Judge 占位 → 按参考键序写出。

默认输入在 `may data/sessions/`，输出在 `may data/train_from_sessions/`。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from auto_labeling.unified_pipeline.aligned_schema import build_bad_train_row, build_good_train_row
from auto_labeling.unified_pipeline.export_may_unfiltered_v02 import (
    NEUTRAL_BAD_JUDGE,
    NEUTRAL_GOOD_JUDGE,
    _ordered_bad,
    _ordered_good,
    _outlier_check,
    _sanitize_lite_case,
    system_prompt_from_rich,
)
from auto_labeling.unified_pipeline.lite_extract import extract_full_session_train_case, extract_lite_cases
from auto_labeling.unified_pipeline.may_data_paths import (
    DEFAULT_BAD_SESSIONS_JSONL,
    DEFAULT_BAD_TRAIN_FROM_SESSIONS,
    DEFAULT_GOOD_SESSIONS_JSONL,
    DEFAULT_GOOD_TRAIN_FROM_SESSIONS,
)
from auto_labeling.unified_pipeline.session_builder import session_to_core_record

Mode = Literal["bad", "good"]


def _apply_system_prompt(case: dict, rich_session: dict, *, sanitize: bool) -> None:
    from auto_labeling.unified_pipeline.text_sanitize import strip_decorative_text

    sp = system_prompt_from_rich(rich_session)
    case["_system_prompt"] = strip_decorative_text(sp) if sanitize else sp


def convert_sessions_jsonl(
    in_path: Path,
    mode: Mode,
    out_path: Path,
    *,
    save_tag: str,
    full_session_one_row: bool,
    one_per_session: bool,
    do_sanitize: bool,
    do_outlier: bool,
    max_history_messages: int,
    max_agent_chars: int,
    max_any_message_chars: int,
    max_total_chars: int,
    max_sessions: int,
) -> dict[str, Any]:
    drop_reasons: Counter[str] = Counter()
    n_sessions = 0
    n_lite = 0
    n_written = 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with in_path.open("r", encoding="utf-8") as inf, out_path.open("w", encoding="utf-8") as outf:
        for line in inf:
            if max_sessions and n_sessions >= max_sessions:
                break
            line = line.strip()
            if not line:
                continue
            session = json.loads(line)
            if not isinstance(session, dict):
                continue
            n_sessions += 1

            core = session_to_core_record(session)
            if full_session_one_row:
                cases = extract_full_session_train_case(core, mode)
            else:
                cases = extract_lite_cases(
                    core,
                    mode,
                    one_per_session=one_per_session,
                    include_full_history=False,
                )
            n_lite += len(cases)

            for c in cases:
                _apply_system_prompt(c, session, sanitize=do_sanitize)
                if do_sanitize:
                    _sanitize_lite_case(c, mode=mode)
                if do_outlier:
                    ok, reason = _outlier_check(
                        c,
                        mode=mode,
                        max_history_messages=max_history_messages,
                        max_agent_chars=max_agent_chars,
                        max_any_message_chars=max_any_message_chars,
                        max_total_chars=max_total_chars,
                    )
                    if not ok:
                        drop_reasons[reason] += 1
                        continue

                sp = c.get("_system_prompt") or ""
                if mode == "bad":
                    row = build_bad_train_row(
                        c,
                        dict(NEUTRAL_BAD_JUDGE),
                        system_prompt=sp,
                        save_time=save_tag,
                        use_judge_fallback=False,
                    )
                    outf.write(json.dumps(_ordered_bad(row), ensure_ascii=False) + "\n")
                else:
                    row = build_good_train_row(
                        c,
                        dict(NEUTRAL_GOOD_JUDGE),
                        system_prompt=sp,
                        judge_model="no_llm_judge",
                        save_time=save_tag,
                        use_judge_fallback=False,
                    )
                    outf.write(json.dumps(_ordered_good(row), ensure_ascii=False) + "\n")
                n_written += 1

    return {
        "input": str(in_path),
        "output": str(out_path),
        "mode": mode,
        "full_session_one_row": full_session_one_row,
        "sessions_read": n_sessions,
        "lite_cases": n_lite,
        "written": n_written,
        "dropped_outliers": int(sum(drop_reasons.values())),
        "drop_reasons": dict(drop_reasons),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bad-sessions", type=Path, default=DEFAULT_BAD_SESSIONS_JSONL)
    p.add_argument("--good-sessions", type=Path, default=DEFAULT_GOOD_SESSIONS_JSONL)
    p.add_argument(
        "--output-bad-train",
        type=Path,
        default=DEFAULT_BAD_TRAIN_FROM_SESSIONS,
    )
    p.add_argument(
        "--output-good-train",
        type=Path,
        default=DEFAULT_GOOD_TRAIN_FROM_SESSIONS,
    )
    p.add_argument("--one-per-session", action="store_true")
    p.add_argument(
        "--lite-signal-rows",
        action="store_true",
        help="仅导出显式点踩/点赞轮（可多行 per session）；默认每 session 一行且含完整对话",
    )
    p.add_argument("--no-sanitize", action="store_true", help="不做 Markdown/Emoji 规则清洗")
    p.add_argument("--no-outlier", action="store_true", help="不按长度/history 条数过滤")
    p.add_argument("--max-history-messages", type=int, default=36)
    p.add_argument("--max-agent-chars", type=int, default=14_000)
    p.add_argument("--max-any-message-chars", type=int, default=14_000)
    p.add_argument("--max-total-chars", type=int, default=80_000)
    p.add_argument("--max-sessions", type=int, default=0, help="只处理前 N 个 session，0 表示全量")
    args = p.parse_args()

    save_tag = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    do_sanitize = not args.no_sanitize
    do_outlier = not args.no_outlier

    full_session = not args.lite_signal_rows
    report: dict[str, Any] = {
        "started": save_tag,
        "sanitize": do_sanitize,
        "outlier_filter": do_outlier,
        "full_session_one_row": full_session,
        "bad": convert_sessions_jsonl(
            Path(args.bad_sessions),
            "bad",
            Path(args.output_bad_train),
            save_tag=save_tag,
            full_session_one_row=full_session,
            one_per_session=args.one_per_session,
            do_sanitize=do_sanitize,
            do_outlier=do_outlier,
            max_history_messages=args.max_history_messages,
            max_agent_chars=args.max_agent_chars,
            max_any_message_chars=args.max_any_message_chars,
            max_total_chars=args.max_total_chars,
            max_sessions=args.max_sessions or 0,
        ),
        "good": convert_sessions_jsonl(
            Path(args.good_sessions),
            "good",
            Path(args.output_good_train),
            save_tag=save_tag,
            full_session_one_row=full_session,
            one_per_session=args.one_per_session,
            do_sanitize=do_sanitize,
            do_outlier=do_outlier,
            max_history_messages=args.max_history_messages,
            max_agent_chars=args.max_agent_chars,
            max_any_message_chars=args.max_any_message_chars,
            max_total_chars=args.max_total_chars,
            max_sessions=args.max_sessions or 0,
        ),
    }

    rep_path = Path(args.output_bad_train).with_name("may_sessions_to_train_report.json")
    rep_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
