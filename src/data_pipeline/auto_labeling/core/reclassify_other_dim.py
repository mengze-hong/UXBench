"""
Reclassify 720 records with failure_dimension='其他' to specific dimensions.

Strategy:
  - Batch 10 records per LLM call
  - Use context: source_query + agent_response_preview + explanation + qa notes
  - Target: one of the 13 canonical failure dimensions
  - Fall back to '其他' only if truly impossible to categorize
  - 20 concurrent workers (sync ThreadPoolExecutor)

Usage:
  python reclassify_other_dim.py
  python reclassify_other_dim.py --dry-run   # print first batch only
"""

import json, sys, io, time, argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

HERE = Path(__file__).parent
OUTPUTS = HERE.parent / "outputs"
SAVED_FILE = OUTPUTS / "pipeline_saved_badcases.jsonl"
LOG_DIR = HERE.parent / "logs"
LOG_FILE = LOG_DIR / "reclassify_other_log.jsonl"

sys.path.insert(0, str(HERE))
from llm_client import call_llm, parse_json_output

MODEL = "claude-sonnet-4.5"

VALID_DIMS = {
    "冗余啰嗦", "任务未完成", "意图理解偏差", "事实性错误",
    "信息可靠性不足", "信息不充分", "指令遵循失败", "格式结构问题",
    "情感语气失当", "过度拒绝", "需求澄清不足", "安全合规问题",
    "系统技术错误", "其他",
}

SYSTEM_PROMPT = '你是数据标注员。对每条记录，从可用维度列表中选出最合适的一个，输出到 new_dimension 字段。即使判断失败不明显也必须选最接近的维度，绝不能输出列表外的词。\n\n可用维度（14个，必须精确匹配其中一个）：冗余啰嗦、任务未完成、意图理解偏差、事实性错误、信息可靠性不足、信息不充分、指令遵循失败、格式结构问题、情感语气失当、过度拒绝、需求澄清不足、安全合规问题、系统技术错误、其他\n\n输出格式：JSON数组，字段为 index、new_dimension、confidence、reasoning。不输出任何其他内容。'


def _parse_array(text: str):
    """Extract a JSON array from LLM output, handles markdown fences."""
    import re as _re
    if not text:
        return None
    # Strip markdown fence
    m = _re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    cleaned = m.group(1).strip() if m else text.strip()
    # Try direct parse
    try:
        result = json.loads(cleaned)
        if isinstance(result, list):
            return result
    except Exception:
        pass
    # Try to find the outermost [ ... ]
    first = cleaned.find('[')
    if first < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(first, len(cleaned)):
        c = cleaned[i]
        if esc:
            esc = False; continue
        if c == '\\':
            esc = True; continue
        if c == '"':
            in_str = not in_str; continue
        if in_str:
            continue
        if c == '[':
            depth += 1
        elif c == ']':
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(cleaned[first:i + 1])
                except Exception:
                    return None
    return None


def reclassify_batch(batch: list) -> list:
    """
    batch: list of {index, record} dicts
    Returns list of {index, new_dimension, confidence, reasoning}
    """
    items = []
    for b in batch:
        r = b["record"]
        al = r.get("auto_label", {})
        sq = r.get("source_query", {})
        qv = r.get("qa_verdict_strong", {})
        items.append({
            "index": b["index"],
            "source_query": (sq.get("message", "") or "")[:200],
            "agent_response_preview": (al.get("agent_response_preview", "") or "")[:300],
            "explanation": (al.get("explanation", "") or "")[:200],
            "qa_notes": (qv.get("notes", "") or "")[:150],
            "qa_issues": qv.get("issues", []),
        })

    user_msg = (
        f'请对以下 {len(items)} 条记录重新分类失败维度，每条输出 index + new_dimension + confidence + reasoning：\n\n'
        f'{json.dumps(items, ensure_ascii=False, indent=1)}\n\n'
        '输出示例（严格按此格式）：\n'
        '[{"index": 119, "new_dimension": "冗余啰嗦", "confidence": 0.9, "reasoning": "一句话理由"}, ...]\n\n'
        '现在输出完整JSON数组：'
    )

    result = call_llm(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        model=MODEL,
        max_tokens=150 * len(batch) + 200,
        temperature=0.1,
        max_retries=3,
        timeout=90,
    )

    if not result.ok:
        return [{"index": b["index"], "new_dimension": "其他", "confidence": 0.0,
                 "reasoning": f"llm_failed: {result.error}"} for b in batch]

    # Use robust array-first parser (parse_json_output may return dict instead of list)
    parsed = _parse_array(result.content)
    if not parsed:
        return [{"index": b["index"], "new_dimension": "其他", "confidence": 0.0,
                 "reasoning": f"parse_failed: {result.content[:100]}"} for b in batch]

    # Normalize field names: some models return 'failure_dimension' instead of 'new_dimension'
    for item in parsed:
        if "new_dimension" not in item and "failure_dimension" in item:
            item["new_dimension"] = item.pop("failure_dimension")

    return parsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print first batch only")
    parser.add_argument("--workers", type=int, default=20, help="Concurrent workers")
    parser.add_argument("--batch-size", type=int, default=10, help="Records per LLM call")
    args = parser.parse_args()

    # Load all records
    records = []
    with open(SAVED_FILE, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                pass

    # Find "其他" indices
    other_indices = [i for i, r in enumerate(records)
                     if r.get("auto_label", {}).get("failure_dimension") == "其他"]
    print(f"Total records: {len(records):,}")
    print(f"Records with failure_dimension=其他: {len(other_indices):,}")

    if not other_indices:
        print("Nothing to reclassify.")
        return

    # Build batches
    bs = args.batch_size
    batches = []
    for i in range(0, len(other_indices), bs):
        chunk = other_indices[i:i+bs]
        batches.append([{"index": idx, "record": records[idx]} for idx in chunk])

    print(f"Batches: {len(batches)} × {bs} records, workers={args.workers}")

    if args.dry_run:
        print("\n[DRY RUN] First batch input:")
        for b in batches[0]:
            al = b["record"].get("auto_label", {})
            sq = b["record"].get("source_query", {})
            print(f"  [{b['index']}] query={sq.get('message','')[:60]!r}")
        print("\nRunning first batch...")
        results = reclassify_batch(batches[0])
        for r in results:
            print(f"  [{r['index']}] → {r.get('new_dimension')} (conf={r.get('confidence')}) {r.get('reasoning','')[:60]}")
        return

    # Run parallel
    t0 = time.time()
    all_results = {}  # global_index → new_dimension

    stats = {"ok": 0, "fallback": 0, "invalid_dim": 0}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(reclassify_batch, batch): batch for batch in batches}
        done = 0
        for fut in as_completed(futures):
            done += 1
            batch = futures[fut]
            try:
                res_list = fut.result()
            except Exception as e:
                print(f"  EXCEPTION in batch: {e}")
                res_list = []

            # Map results back to global indices
            res_by_local = {}
            for res in res_list:
                res_by_local[res.get("index")] = res

            for b in batch:
                idx = b["index"]
                res = res_by_local.get(idx)
                if res is None:
                    # Missing — keep 其他
                    all_results[idx] = {"new_dimension": "其他", "confidence": 0.0, "reasoning": "missing"}
                    stats["fallback"] += 1
                else:
                    nd = res.get("new_dimension", "其他")
                    if nd not in VALID_DIMS:
                        stats["invalid_dim"] += 1
                        nd = "其他"
                    all_results[idx] = {"new_dimension": nd, "confidence": res.get("confidence", 0.8),
                                        "reasoning": res.get("reasoning", "")}
                    stats["ok"] += 1

            if done % 10 == 0 or done == len(batches):
                elapsed = time.time() - t0
                rate = (done * bs) / elapsed if elapsed > 0 else 0
                print(f"  [{done}/{len(batches)} batches] {rate:.1f} rec/s  ok={stats['ok']} fallback={stats['fallback']}")

    elapsed = time.time() - t0
    print(f"\nReclassification done in {elapsed:.1f}s")

    # Dimension distribution
    from collections import Counter
    dim_counter = Counter(v["new_dimension"] for v in all_results.values())
    print("\nNew dimension distribution:")
    for k, v in dim_counter.most_common():
        print(f"  {k}: {v}")

    # Apply to records
    for idx, res in all_results.items():
        records[idx]["auto_label"]["failure_dimension"] = res["new_dimension"]
        records[idx]["auto_label"]["failure_dimension_reclassified"] = True
        records[idx]["auto_label"]["reclassify_confidence"] = res["confidence"]
        records[idx]["auto_label"]["reclassify_reasoning"] = res["reasoning"]

    # Write back
    tmp = SAVED_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(SAVED_FILE)
    print(f"\nWritten: {SAVED_FILE.name}")

    # Log
    LOG_DIR.mkdir(exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_reclassified": len(all_results),
            "stats": stats,
            "dim_distribution": dict(dim_counter),
            "elapsed_s": round(elapsed, 1),
        }, ensure_ascii=False) + "\n")
    print(f"Log: {LOG_FILE.name}")


if __name__ == "__main__":
    main()
