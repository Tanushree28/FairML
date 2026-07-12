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


def plot_beta_cross_effect(dataset, out_dir, results_root=RESULTS, alphas=(0.25, 0.5, 0.75)):
    """The paper's core figure: group (DPD) and individual (Theil) fairness
    as β moves from individual-only (0) to group-only (1), per arch and α."""
    sweep = load_seed_csvs(dataset, "sweep_results", results_root)
    fig, axes = plt.subplots(2, len(alphas), figsize=(4.2 * len(alphas), 7.2),
                             sharex=True, squeeze=False)
    series = [(DPD2, "tab:red", "group fairness (DPD, largest 2)"),
              ("Theil Index", "tab:purple", "individual fairness (Theil)")]
    for r, arch in enumerate(["logreg", "mlp"]):
        for c, a in enumerate(alphas):
            ax = axes[r][c]
            g = sweep[(sweep["Arch"] == arch) & (sweep["Alpha"] == a)]
            st = g.groupby("Beta")[[DPD2, "Theil Index"]].agg(["mean", "std"])
            for metric, color, label in series:
                m = st[(metric, "mean")]
                s = st[(metric, "std")].fillna(0)
                ax.plot(st.index, m, "-o", ms=4, color=color,
                        label=label if (r == 0 and c == 0) else None)
                ax.fill_between(st.index, m - s, m + s, color=color, alpha=0.2)
            ax.set_title(f"{ARCH_LABELS[arch]}, α={a}", fontsize=10)
            if r == 1:
                ax.set_xlabel("β  (0 = individual only, 1 = group only)")
            if c == 0:
                ax.set_ylabel("metric value (test)")
    fig.legend(loc="upper right", fontsize=8)
    fig.suptitle(f"Group vs. individual fairness as β varies — {dataset} "
                 f"(mean ± std over seeds)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = Path(out_dir) / "fig2_beta_cross_effect.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_dpd_vs_theil(dataset, out_dir, results_root=RESULTS):
    sweep = load_seed_csvs(dataset, "sweep_results", results_root)
    fair = sweep[sweep["Alpha"] < 1]
    mean = fair.groupby(["Arch", "Alpha", "Beta"], as_index=False)[[DPD2, "Theil Index"]].mean()
    fig, ax = plt.subplots(figsize=(6.5, 5))
    sc = ax.scatter(mean[DPD2], mean["Theil Index"], c=mean["Beta"],
                    cmap="viridis", s=32)
    fig.colorbar(sc, label="β (group-fairness weight)")
    ax.set_xlabel(DPD2)
    ax.set_ylabel("Theil Index")
    ax.set_title(f"Group vs. individual fairness trade-off — {dataset} (seed means)")
    fig.tight_layout()
    out = Path(out_dir) / "fig2b_dpd_vs_theil.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out
