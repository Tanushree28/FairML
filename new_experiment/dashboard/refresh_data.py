"""Bundle the latest per-seed results into dashboard/data/ so the dashboard
folder is self-contained for deployment (e.g. Streamlit Cloud).

The app reads the live CSVs under new_experiment/RESULTS/ whenever it runs
inside the repo; these bundled copies are only the fallback for a standalone
deploy. Run after re-running the experiment:

    python new_experiment/dashboard/refresh_data.py
"""

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "RESULTS"

# per-seed source file -> bundled concatenation the app reads
BUNDLES = {"sweep_results.csv": "sweep_all_seeds.csv",
           "model_comparison.csv": "comparison_all_seeds.csv"}

for ds_dir in sorted(p for p in RESULTS.iterdir() if p.is_dir() and p.name != "paper"):
    for src_name, out_name in BUNDLES.items():
        files = sorted(ds_dir.glob(f"seed*/{src_name}"))
        if not files:
            print(f"skipped {ds_dir.name}/{out_name}: no seed*/{src_name} found")
            continue
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        dest = HERE / "data" / ds_dir.name
        dest.mkdir(parents=True, exist_ok=True)
        df.to_csv(dest / out_name, index=False)
        print(f"wrote {dest / out_name}  ({len(files)} seeds, {len(df)} rows)")
