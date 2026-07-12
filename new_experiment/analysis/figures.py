"""Paper figures and tables. Every function writes into out_dir and returns the path."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from analysis.stats import (DPD2, METRICS, RESULTS, load_seed_csvs,
                            significance_vs_baseline)

ARCH_LABELS = {"logreg": "Logistic Regression", "mlp": "MLP (64-64)"}
ARCH_COLORS = {"logreg": "tab:blue", "mlp": "tab:orange"}
BASELINE_FAMILIES = [
    "ExpGrad LogReg (Demographic Parity)",
    "ExpGrad LogReg (Equalized Odds)",
    "ThresholdOptimizer LogReg (Demographic Parity)",
    "Reweighing LogReg (Kamiran-Calders)",
    "Random Forest (no fairness)",
]


def pareto_mask(acc, dpd):
    """True where no other point has >= accuracy AND <= DPD (one strict)."""
    acc, dpd = np.asarray(acc), np.asarray(dpd)
    mask = np.ones(len(acc), dtype=bool)
    for i in range(len(acc)):
        dominated = ((acc >= acc[i]) & (dpd <= dpd[i])
                     & ((acc > acc[i]) | (dpd < dpd[i])))
        if dominated.any():
            mask[i] = False
    return mask


def plot_pareto(dataset, out_dir, results_root=RESULTS):
    sweep = load_seed_csvs(dataset, "sweep_results", results_root)
    comp = load_seed_csvs(dataset, "model_comparison", results_root)
    mean_sweep = (sweep.groupby(["Arch", "Alpha", "Beta"], as_index=False)
                  [["Accuracy", DPD2]].mean())

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for arch, g in mean_sweep.groupby("Arch"):
        ax.scatter(g[DPD2], g["Accuracy"], s=16, alpha=0.35,
                   color=ARCH_COLORS[arch], label=f"{ARCH_LABELS[arch]} sweep (seed mean)")
        front = g[pareto_mask(g["Accuracy"].to_numpy(), g[DPD2].to_numpy())].sort_values(DPD2)
        ax.plot(front[DPD2], front["Accuracy"], "-o", lw=2, ms=4,
                color=ARCH_COLORS[arch], label=f"{ARCH_LABELS[arch]} Pareto front")

    agg = comp.groupby("Family")[["Accuracy", DPD2]].agg(["mean", "std"])
    markers = ["D", "s", "^", "v", "P"]
    for fam, mk in zip(BASELINE_FAMILIES, markers):
        if fam not in agg.index:
            continue
        ax.errorbar(agg.loc[fam, (DPD2, "mean")], agg.loc[fam, ("Accuracy", "mean")],
                    xerr=agg.loc[fam, (DPD2, "std")], yerr=agg.loc[fam, ("Accuracy", "std")],
                    fmt=mk, ms=7, capsize=3, label=fam)

    ax.set_xlabel(f"{DPD2}  (lower = fairer)")
    ax.set_ylabel("Accuracy (test)")
    ax.set_title(f"Accuracy–fairness Pareto frontier — {dataset} (mean over seeds)")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    out = Path(out_dir) / "fig1_pareto.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out
