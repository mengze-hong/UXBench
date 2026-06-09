"""
UXBench data pipeline — 5-stage auto-labeling pipeline.

Converts raw interaction logs into benchmark test cases through:
  Stage 1  signals.py              — signal extraction from raw logs
  Stage 2  prefilter.py            — deduplication and quality filtering
  Stage 3  miner.py                — Miner LLM agent (extract failure/success reasons)
  Stage 4  judge.py                — Judge LLM agent (5-axis quality scoring)
  Stage 5  qa_full_scan.py         — QA full scan (remove duplicates and edge cases)

Entry point: pipeline.py (ThreadPool orchestrator).
"""
