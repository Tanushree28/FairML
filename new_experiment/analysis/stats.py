"""Seed aggregation and significance testing for the paper pipeline."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

RESULTS = Path(__file__).resolve().parent.parent / "RESULTS"

DPD2 = "DPD (Largest 2 Groups)"
METRICS = ["Accuracy", "Demographic Parity Difference", DPD2,
           "Equalized Odds Difference", "Theil Index", "Gini Coefficient"]


def load_seed_csvs(dataset, kind, results_root=RESULTS):
    """Concat RESULTS/<dataset>/seed*/<kind>.csv into one frame."""
    frames = []
    for f in sorted((results_root / dataset).glob(f"seed*/{kind}.csv")):
        df = pd.read_csv(f)
        if "Seed" not in df.columns:
            df["Seed"] = int(f.parent.name.removeprefix("seed"))
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No {kind}.csv under {results_root / dataset}/seed*/")
    return pd.concat(frames, ignore_index=True)


def holm_correction(pvals):
    """Holm step-down adjusted p-values (same order as input)."""
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def significance_vs_baseline(comp_df, family, baseline_family, metrics=METRICS):
    """Paired Wilcoxon signed-rank across seeds, Holm-corrected over `metrics`."""
    a = comp_df[comp_df["Family"] == family].sort_values("Seed")
    b = comp_df[comp_df["Family"] == baseline_family].sort_values("Seed")
    assert list(a["Seed"]) == list(b["Seed"]), "seed sets must match for a paired test"
    pvals = []
    for m in metrics:
        diff = a[m].to_numpy() - b[m].to_numpy()
        pvals.append(1.0 if np.allclose(diff, 0) else wilcoxon(diff).pvalue)
    return pd.DataFrame({"Metric": metrics, "p": pvals,
                         "p_holm": holm_correction(pvals)})
