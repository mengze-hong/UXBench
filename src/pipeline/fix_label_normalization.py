"""
Fix label normalization issues in pipeline_saved_badcases.jsonl:

1. scenario: 繁体 → 简体, 错别字修正, 非标准类别 → 最近似标准类别
2. failure_dimension: 非标准维度名 → 标准维度名

Canonical scenario set (from miner_system.txt):
  产品与服务咨询 / 创意内容与生成 / 信息与知识查询 / 办公与效率
  私密与生活决策辅助 / 情绪与心理支持 / 娱乐消遣

Canonical failure_dimension set (from data majority):
  冗余啰嗦 / 任务未完成 / 意图理解偏差 / 事实性错误 / 信息可靠性不足
  系统技术错误 / 需求澄清不足 / 指令遵循失败 / 情感语气失当 / 信息不充分
  过度拒绝 / 格式结构问题 / 安全合规问题 / 其他

Usage:
  python fix_label_normalization.py
  python fix_label_normalization.py --dry-run
"""

import json, sys, io, argparse
from pathlib import Path
from collections import Counter

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HERE = Path(__file__).parent
OUTPUTS = HERE.parent / "outputs"
SAVED_FILE = OUTPUTS / "pipeline_saved_badcases.jsonl"

# ── Canonical sets ────────────────────────────────────────────────────
SCENARIO_CANONICAL = {
    "产品与服务咨询",
    "创意内容与生成",
    "信息与知识查询",
    "办公与效率",
    "私密与生活决策辅助",
    "情绪与心理支持",
    "娱乐消遣",
}

DIM_CANONICAL = {
    "冗余啰嗦", "任务未完成", "意图理解偏差", "事实性错误",
    "信息可靠性不足", "系统技术错误", "需求澄清不足", "指令遵循失败",
    "情感语气失当", "信息不充分", "过度拒绝", "格式结构问题",
    "安全合规问题", "其他",
}

# ── Mapping rules ─────────────────────────────────────────────────────
SCENARIO_MAP = {
    # 繁体 → 简体
    "信息與知識查询": "信息与知识查询",
    "信息與知识查询": "信息与知识查询",
    "信息與知識查詢": "信息与知识查询",
    "產品與服務咨詢": "产品与服务咨询",
    "產品與服務咨询": "产品与服务咨询",
    # 错别字
    "娱乐消散": "娱乐消遣",
    "娱乐消消": "娱乐消遣",
    # 非标准类别 → 最近似标准类别
    "教育与学习辅导": "信息与知识查询",     # 教育查询 → 信息与知识查询
    "职场与服务咨询": "产品与服务咨询",      # 职场咨询 → 产品与服务咨询
    "生活服务咨询":   "私密与生活决策辅助",  # 生活服务 → 私密与生活决策辅助
    "生活决策辅助":   "私密与生活决策辅助",  # 缩写版本
    "出行与路线咨询": "信息与知识查询",      # 路线查询 → 信息与知识查询
    "其他":           "信息与知识查询",      # 3条，归到最大类
}

DIM_MAP = {
    # 过度承诺类 → 过度拒绝（expectation management 同一家族）
    "过度承诺":           "过度拒绝",
    "过度承诺/能力错觉":  "过度拒绝",
    "能力边界说明不当":   "过度拒绝",
    # 复合标签 → 拆分到主维度
    "过度推断+缺乏澄清":      "需求澄清不足",
    "信息一致性问题":          "事实性错误",
    "事实性错误/逻辑一致性":   "事实性错误",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="只打印变更，不写文件")
    args = parser.parse_args()

    records = []
    errors = 0
    with open(SAVED_FILE, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                errors += 1
    print(f"Loaded {len(records):,} records  ({errors} parse errors skipped)")

    scenario_fixes = Counter()
    dim_fixes = Counter()

    for r in records:
        al = r.get("auto_label", {})

        # ── Fix scenario ──
        sc = al.get("scenario")
        if sc and sc not in SCENARIO_CANONICAL:
            new_sc = SCENARIO_MAP.get(sc)
            if new_sc:
                scenario_fixes[f"{sc!r} → {new_sc!r}"] += 1
                if not args.dry_run:
                    al["scenario"] = new_sc
            else:
                scenario_fixes[f"{sc!r} → [UNMAPPED]"] += 1

        # ── Fix failure_dimension ──
        fd = al.get("failure_dimension")
        if fd and fd not in DIM_CANONICAL:
            new_fd = DIM_MAP.get(fd)
            if new_fd:
                dim_fixes[f"{fd!r} → {new_fd!r}"] += 1
                if not args.dry_run:
                    al["failure_dimension"] = new_fd
                    # preserve raw
                    if not al.get("failure_dimension_raw"):
                        al["failure_dimension_raw"] = fd
            else:
                dim_fixes[f"{fd!r} → [UNMAPPED]"] += 1

    # Report
    print(f"\nScenario fixes ({sum(scenario_fixes.values())} records):")
    for k, v in scenario_fixes.most_common():
        print(f"  {k}: {v}")

    print(f"\nFailure dimension fixes ({sum(dim_fixes.values())} records):")
    for k, v in dim_fixes.most_common():
        print(f"  {k}: {v}")

    if args.dry_run:
        print("\n[DRY RUN] No files written.")
        return

    # Write back
    tmp = SAVED_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(SAVED_FILE)
    print(f"\nWritten: {SAVED_FILE.name}")
    print(f"Total changes: {sum(scenario_fixes.values()) + sum(dim_fixes.values())} records")


if __name__ == "__main__":
    main()
