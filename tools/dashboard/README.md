# tools/dashboard/

FastAPI + HTML single-file dashboard for exploring the UXBench testset and leaderboard.

## Running

```bash
# From the repo root
uvicorn tools.dashboard.app:app --host 0.0.0.0 --port 8512 --reload
```

Open `http://localhost:8512`.

## Architecture

`app.py` is a self-contained FastAPI application that:

1. Loads testset JSONL files into memory at startup
2. Serves a single HTML page (`/`) with embedded CSS and JavaScript
3. Exposes REST API endpoints (`/api/stats`, `/api/records`, `/api/detail`, `/api/leaderboard`, `/api/blind/*`) consumed by the frontend

## Key Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main dashboard HTML |
| `/api/stats` | GET | Aggregated statistics for `?mode=bad\|good` |
| `/api/records` | GET | Paginated, filtered record list |
| `/api/detail` | GET | Full record detail by `cid` |
| `/api/exclude` | POST | Move a case to the excluded list |
| `/api/leaderboard` | GET | Binary classification leaderboard |
| `/api/leaderboard_3class` | GET | 3-class judge performance leaderboard |
| `/api/blind/session` | GET | Current blind eval session state |
| `/api/blind/judge` | POST | Submit a judgment |
| `/api/blind/reset` | GET | Archive session and resample |

## Configuration

Data paths default to `data/` in the repo root. Override with the `DATA_DIR` env var:

```bash
export DATA_DIR=/path/to/your/data
uvicorn tools.dashboard.app:app --port 8512
```

Download data files from HuggingFace first — see the repo README "Download Dataset" section.
