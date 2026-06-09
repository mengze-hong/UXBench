# docs/

GitHub Pages source for the UXBench project website and interactive leaderboard.

## Structure

```
docs/
├── index.html          # Main landing page and leaderboard
├── css/
│   └── style.css       # Page styles
├── js/
│   └── leaderboard.js  # Dynamic leaderboard table (loads JSON, renders rankings)
└── img/                # Web assets (figures, logos)
```

## Developing Locally

Serve from the `docs/` directory with Python's built-in HTTP server:

```bash
cd docs
python -m http.server 8000
```

Then open `http://localhost:8000` in your browser.

No build step is required — the page is plain HTML/CSS/JS.

## How the Leaderboard Works

`leaderboard.js` fetches `experiments/results/task2_leaderboard.json` (or a relative
path configured in the script) and dynamically renders the model rankings table.
To update the leaderboard, update the JSON file and push — GitHub Pages will serve
the new data automatically.

## Deployment

GitHub Pages should be configured to serve from the **`docs/` folder** on the `main`
branch (Settings → Pages → Source: Deploy from a branch → `main` / `docs`).
