"""
UXBench data pipeline — 6-stage auto-labeling pipeline.

Converts raw interaction logs into benchmark test cases through:
  Stage 0  signals.py              — signal extraction from raw logs
  Stage 1  prefilter.py            — deduplication and quality filtering
  Stage 2  miner.py                — Miner LLM agent (extract failure/success reasons)
  Stage 3  judge.py                — Judge LLM agent (5-axis quality scoring)
  Stage 4  qa_full_scan.py         — QA full scan (remove duplicates and edge cases)
  Stage 5  build_golden_testset.py — stratified sampling → final golden test set

Entry point: pipeline.py (ThreadPool orchestrator).
"""
