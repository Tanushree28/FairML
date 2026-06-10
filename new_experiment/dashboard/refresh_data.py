"""Copy the latest result CSVs from new_experiment/RESULTS/ into
dashboard/data/ so the dashboard folder is self-contained for deployment.

Run after re-running the experiment:
    python new_experiment/dashboard/refresh_data.py
"""

import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "RESULTS"

for ds_dir in sorted(RESULTS.iterdir()):
    if not ds_dir.is_dir():
        continue
    dest = HERE / "data" / ds_dir.name
    dest.mkdir(parents=True, exist_ok=True)
    for csv in ("sweep_results.csv", "model_comparison.csv"):
        src = ds_dir / csv
        if src.exists():
            shutil.copy2(src, dest / csv)
            print(f"copied {src} -> {dest / csv}")
