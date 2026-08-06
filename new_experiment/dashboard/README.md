# FairML Results Dashboard

Interactive Streamlit view of the paper pipeline: headline result, model
comparison with seed error bars and Wilcoxon significance, the
accuracy↔fairness tradeoff, the β cross-effect (group vs. individual
fairness), and interactive α×β heatmaps — for COMPAS, German Credit and
Adult.

Every number shown is a **mean ± sd over the 10 seeds**, read from the
canonical per-seed results in `new_experiment/RESULTS/<dataset>/seed<k>/`.

## Run locally

From the repo root:

```bash
.venv/bin/pip install -r new_experiment/dashboard/requirements.txt   # one time
.venv/bin/streamlit run new_experiment/dashboard/app.py
```

Opens at http://localhost:8501. No retraining happens — the dashboard only
reads result CSVs.

## Where the data comes from

The app tries two sources, in this order:

1. **Live results** — `new_experiment/RESULTS/<dataset>/seed*/` when running
   inside the repo. Always current; this is what you get locally.
2. **Bundled snapshot** — `dashboard/data/<dataset>/` for a standalone
   deploy where `RESULTS/` isn't present.

The sidebar states which source is in use. Live wins, so the dashboard can
never quietly display superseded numbers.

## Deploy to Streamlit Community Cloud (free, shareable link)

Bundle a snapshot first, so the folder stands alone:

```bash
.venv/bin/python new_experiment/dashboard/refresh_data.py
```

Then push **just this folder** as its own small public repo:

```bash
cd new_experiment/dashboard
git init && git add . && git commit -m "FairML results dashboard"
gh repo create fairml-dashboard --public --source=. --push
```

(No `gh` CLI? Create an empty `fairml-dashboard` repo on github.com, then
`git remote add origin https://github.com/<your-username>/fairml-dashboard.git`
and `git push -u origin main`.)

Go to **https://share.streamlit.io** → sign in with GitHub → **Create app** →
pick the repo, branch `main`, main file path **`app.py`** → **Deploy**. After
~2 minutes you get a permanent URL like
`https://<your-username>-fairml-dashboard.streamlit.app` that anyone can open
with nothing to install.

## Updating after re-running the experiment

```bash
.venv/bin/python new_experiment/run_experiment.py --dataset all --seeds 0-9
.venv/bin/python new_experiment/dashboard/refresh_data.py    # only for the deployed copy
```

Locally the dashboard picks up new results on its own (clear Streamlit's cache
with **R** or the ⋮ menu if a page is already open). Streamlit Cloud redeploys
on every push.

## Files

- `app.py` — the dashboard
- `refresh_data.py` — bundles the per-seed CSVs into `data/` for deployment
- `data/<dataset>/{sweep_all_seeds.csv, comparison_all_seeds.csv}` — snapshot
  (generated; absent until you run `refresh_data.py`)
- `requirements.txt` — deps for a standalone deploy (streamlit, pandas,
  plotly, numpy, scipy)
