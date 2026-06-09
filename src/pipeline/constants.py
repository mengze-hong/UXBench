"""
Pipeline constants — file names, directory names, and worker defaults.

Centralises all path/name constants so pipeline modules stay DRY.
Extracted from the original pipeline code (previously imported from
the now-removed `lib/` tree).
"""


class DislikeConstants:
    """File names for the dislike-signal pipeline outputs."""

    # Raw input data (relative to project root) — use --input to override
    INPUT_FILENAME = "uxbench_raw_interactions.jsonl"

    # Pipeline output files (written into SharedConstants.OUTPUTS_DIR)
    SAVED_FILENAME     = "saved_auto.jsonl"
    DELETED_FILENAME   = "deleted_auto.jsonl"
    REJECTED_FILENAME  = "rejected_auto.jsonl"

    # After QA full-scan the saved file is renamed to this legacy name
    LEGACY_SAVED_FILENAME = "pipeline_saved_badcases.jsonl"

    # Per-run log
    RUN_LOG_FILENAME = "run_log.jsonl"

    # Incremental resume cache (list of all processed cids)
    PROCESSED_CIDS_FILENAME = "processed_cids.json"


class SharedConstants:
    """Shared directory names and runtime defaults."""

    # Output directories (relative to src/)
    OUTPUTS_DIR = "outputs"
    LOGS_DIR    = "logs"
    PROMPTS_DIR = "prompts"

    # Default concurrency
    DEFAULT_WORKERS = 5

    # Progress file written by each pipeline stage
    PROGRESS_FILENAME = "progress.json"
