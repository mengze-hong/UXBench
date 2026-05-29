"""Shared helpers for the three UXBench leaderboard scripts.

Defines:
  - PRETTY_NAMES: file-stem -> paper display name
  - PAPER_27:     the canonical 27 model order from the paper
  - load_jsonl:   tolerant jsonl reader
  - render_table: leaderboard markdown table renderer
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List


# Canonical mapping: response/judge file stem -> paper display name.
# Keep these *exactly* in sync with leaderboard.md.
PRETTY_NAMES: Dict[str, str] = {
    "claude_opus_4_5":          "Claude Opus 4.5",
    "claude_opus_4_6":          "Claude Opus 4.6",
    "claude_opus_4_7":          "Claude Opus 4.7",
    "claude_sonnet_4_5":        "Claude Sonnet 4.5",
    "deepseek_r1":              "DeepSeek R1",
    "deepseek_v3":              "DeepSeek V3",
    "deepseek_v3_2":            "DeepSeek V3.2",
    "deepseek_v4_pro":          "DeepSeek V4 Pro",
    "doubao_seed_1_6":          "Doubao Seed 1.6",
    "doubao_seed_2_0_lite":     "Doubao Seed 2.0 Lite",
    "doubao_seed_2_0_pro":      "Doubao Seed 2.0 Pro",
    "gemini_2_5_flash":         "Gemini 2.5 Flash",
    "gemini_2_5_pro":           "Gemini 2.5 Pro",
    "gemini_3_0_flash":         "Gemini 3.0 Flash",
    "gemini_3_flash_preview":   "Gemini 3.0 Flash",   # alias used in some files
    "gemini_3_1_pro_preview":   "Gemini 3.1 Pro",
    "gemini_3_pro_preview":     "Gemini 3.1 Pro",     # alias used in some files
    "glm_5":                    "GLM-5",
    "glm_5_1":                  "GLM-5.1",
    "gpt_5":                    "GPT-5",
    "gpt_5_1":                  "GPT-5.1",
    "gpt_5_2":                  "GPT-5.2",
    "gpt_5_5":                  "GPT-5.5",
    "gpt_5_mini":               "GPT-5 mini",
    "hunyuan_3_preview":        "Hunyuan 3",
    "kimi_k2_5":                "Kimi K2.5",
    "kimi_k2_6":                "Kimi K2.6",
    "minimax_m2_5":             "MiniMax M2.5",
    "qwen3_6_plus":             "Qwen3.6-Plus",
}


def display_name(stem: str) -> str:
    """Return the paper-aligned display name for a model file stem."""
    return PRETTY_NAMES.get(stem, stem)


def load_jsonl(path: Path) -> List[dict]:
    """Read a JSONL file, skipping blank/malformed lines silently."""
    rows: List[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def render_table(
    headers: Iterable[str],
    aligns:  Iterable[str],
    rows:    Iterable[Iterable[str]],
) -> str:
    """Render a markdown table. `aligns` items are 'l', 'r', or 'c'."""
    headers = list(headers)
    aligns  = list(aligns)
    align_row = []
    for a in aligns:
        if a == "r":
            align_row.append("---:")
        elif a == "c":
            align_row.append(":---:")
        else:
            align_row.append("---")
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(align_row) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)
