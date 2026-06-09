# tools/

Developer tooling for the UXBench project.

## Contents

| Directory | Description |
|-----------|-------------|
| `dashboard/` | Interactive web dashboard for exploring the testset and viewing the leaderboard |

## Running the Dashboard

**Prerequisites:** Python 3.10+, `fastapi`, `uvicorn`

```bash
pip install fastapi uvicorn

# From the repo root
uvicorn tools.dashboard.app:app --host 0.0.0.0 --port 8512 --reload
```

Then open `http://localhost:8512` in your browser.

## What the Dashboard Shows

The dashboard is a single-page FastAPI application with six tabs:

| Tab | Description |
|-----|-------------|
| 👍 Good — Distributions | Statistical breakdown of the good-case testset (dimensions, difficulty, signal types, etc.) |
| 👍 Good — Data | Browsable record viewer with search, filters, and detail pane |
| 👎 Bad — Distributions | Statistical breakdown of the bad-case testset (including 5-axis judge scores) |
| 👎 Bad — Data | Browsable record viewer for bad cases |
| 🏆 Leaderboard | Binary and 3-class classification leaderboard across all evaluated models |
| 🧑‍⚖️ Blind Eval | Interactive blind evaluation — judge conversations without seeing ground-truth labels |

## Dependencies

```
fastapi
uvicorn
```

Data files are downloaded from HuggingFace (`mengze-hong/UXBench`) and placed under
`data/` in the repo root. See the main README for download instructions.
