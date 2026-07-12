# FairML Results Dashboard

Interactive Streamlit dashboard for the fairness-aware training experiment:
headline results, 8-model comparison, accuracy↔fairness tradeoff scatter, and
interactive α×β heatmaps, for both COMPAS and German Credit.

This folder is **self-contained** — the result CSVs are bundled in `data/`,
so the deployed app needs no PyTorch, no fairlearn, and never retrains
anything. Total dependencies: streamlit, pandas, plotly.

## Run locally

From the repo root:

```bash
.venv/bin/pip install streamlit plotly        # one time
.venv/bin/streamlit run new_experiment/dashboard/app.py
```

Opens at http://localhost:8501.

## Deploy to Streamlit Community Cloud (free, shareable link)

Streamlit Cloud deploys from a GitHub repo. Easiest path: push **just this
folder** as its own small public repo.

1. Create the repo and push (from this folder):

   ```bash
   cd new_experiment/dashboard
   git init
   git add .
   git commit -m "FairML results dashboard"
   gh repo create fairml-dashboard --public --source=. --push
   ```

   (No `gh` CLI? Create an empty repo named `fairml-dashboard` on github.com,
   then `git remote add origin https://github.com/<your-username>/fairml-dashboard.git`
   and `git push -u origin main`.)

2. Go to **https://share.streamlit.io** → sign in with GitHub → **Create app**
   → "Deploy a public app from GitHub".

3. Pick the `fairml-dashboard` repo, branch `main`, main file path **`app.py`**
   → **Deploy**.

4. After ~2 minutes you get a permanent URL like
   `https://<your-username>-fairml-dashboard.streamlit.app` — send that link
   to your professor. Anyone with the link can view it; nothing to install.

## Updating the dashboard after re-running the experiment

```bash
python new_experiment/run_experiment.py --dataset both   # regenerate results
python new_experiment/dashboard/refresh_data.py          # copy CSVs into data/
cd new_experiment/dashboard && git add data && git commit -m "update results" && git push
```

Streamlit Cloud redeploys automatically on every push.

## Files

- `app.py` — the dashboard
- `data/<dataset>/{sweep_results.csv, model_comparison.csv}` — bundled results
- `refresh_data.py` — re-copies CSVs from `../RESULTS/` after a new sweep
- `requirements.txt` — minimal deps for Streamlit Cloud (streamlit, pandas, plotly)
