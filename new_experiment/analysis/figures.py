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


STAR_LEVELS = [(0.001, "***"), (0.01, "**"), (0.05, "*")]
FIG3_METRICS = ["Accuracy", DPD2, "Equalized Odds Difference", "Theil Index"]
PAIRS = [("Logistic Regression baseline", "Fair Logistic Regression"),
         ("MLP (64-64) baseline", "Fair MLP (64-64)")]


def stars(p):
    for thr, s in STAR_LEVELS:
        if p < thr:
            return s
    return "ns"


def plot_fair_vs_baseline(dataset, out_dir, results_root=RESULTS):
    comp = load_seed_csvs(dataset, "model_comparison", results_root)
    # one Holm family per (baseline, fair) pair, across the four plotted metrics
    sig = {ff: significance_vs_baseline(comp, ff, bf, metrics=FIG3_METRICS)
                 .set_index("Metric")["p_holm"]
           for bf, ff in PAIRS}

    fig, axes = plt.subplots(1, len(FIG3_METRICS), figsize=(4.4 * len(FIG3_METRICS), 4.6))
    x = np.arange(len(PAIRS))
    width = 0.38
    for ax, metric in zip(axes, FIG3_METRICS):
        for k, kind in enumerate(["baseline", "fair"]):
            fams = [p[k] for p in PAIRS]
            mean = [comp.loc[comp["Family"] == f, metric].mean() for f in fams]
            std = [comp.loc[comp["Family"] == f, metric].std() for f in fams]
            ax.bar(x + (k - 0.5) * width, mean, width, yerr=std, capsize=3,
                   label=kind if metric == FIG3_METRICS[0] else None)
        for i, (bf, ff) in enumerate(PAIRS):
            top = comp.loc[comp["Family"].isin([bf, ff]), metric].max()
            ax.text(i, top * 1.04, stars(sig[ff][metric]), ha="center", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(["LogReg", "MLP"])
        ax.set_title(metric, fontsize=10)
    fig.legend(loc="upper right")
    fig.suptitle(f"Fairness loss vs. baseline — {dataset} "
                 f"(mean ± std over seeds; Wilcoxon, Holm-corrected)")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = Path(out_dir) / "fig3_fair_vs_baseline.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def master_table(dataset, out_dir, results_root=RESULTS):
    comp = load_seed_csvs(dataset, "model_comparison", results_root)
    rows = []
    for fam, g in comp.groupby("Family", sort=False):
        row = {"Model": fam, "Seeds": len(g)}
        for m in METRICS:
            row[m] = f"{g[m].mean():.3f} ± {g[m].std():.3f}"
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(Path(out_dir) / "table1_master_comparison.csv", index=False)
    (Path(out_dir) / "table1_master_comparison.tex").write_text(df.to_latex(index=False))
    return df


def ablation_table(dataset, out_dir, results_root=RESULTS):
    """No-fairness / SoftGE-only / SoftDP-only / blend, at the seed-mean-best α.
    Best (α, β) picked on validation seed means (same utopia rule as training)."""
    sweep = load_seed_csvs(dataset, "sweep_results", results_root)
    rows = []
    for arch in ["logreg", "mlp"]:
        g = sweep[sweep["Arch"] == arch]
        mean = g.groupby(["Alpha", "Beta"], as_index=False).mean(numeric_only=True)
        fair = mean[(mean["Alpha"] < 1)
                    & mean["Val Positive Rate"].between(0.05, 0.95)].copy()
        fair["dist"] = np.sqrt((1 - fair["Val Accuracy"]) ** 2
                               + fair["Val DPD (Largest 2 Groups)"] ** 2)
        best = fair.sort_values("dist").iloc[0]
        a_star, b_star = best["Alpha"], best["Beta"]
        variants = [("No fairness (α=1)", 1, 0),
                    (f"SoftGE only (α={a_star}, β=0)", a_star, 0),
                    (f"SoftDP only (α={a_star}, β=1)", a_star, 1),
                    (f"Blend (α={a_star}, β={b_star})", a_star, b_star)]
        for name, a, b in variants:
            sel = g[(g["Alpha"] == a) & (g["Beta"] == b)]
            row = {"Arch": ARCH_LABELS[arch], "Variant": name}
            for m in METRICS:
                row[m] = f"{sel[m].mean():.3f} ± {sel[m].std():.3f}"
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(Path(out_dir) / "table2_ablation.csv", index=False)
    (Path(out_dir) / "table2_ablation.tex").write_text(df.to_latex(index=False))
    return df


HEATMAP_METRICS = ["Accuracy", DPD2, "Theil Index"]


def seed_mean_heatmaps(dataset, out_dir, results_root=RESULTS):
    sweep = load_seed_csvs(dataset, "sweep_results", results_root)
    paths = []
    for arch in ["logreg", "mlp"]:
        g = sweep[sweep["Arch"] == arch]
        for metric in HEATMAP_METRICS:
            pv = g.pivot_table(index="Alpha", columns="Beta", values=metric, aggfunc="mean")
            plt.figure(figsize=(8, 6))
            sns.heatmap(pv, annot=True, fmt=".3f",
                        cmap="viridis" if metric == "Accuracy" else "coolwarm",
                        cbar_kws={"label": f"{metric} (test, seed mean)"})
            plt.title(f"{metric} (mean over seeds) — {ARCH_LABELS[arch]} — {dataset}")
            plt.xlabel("β")
            plt.ylabel("α")
            plt.tight_layout()
            out = Path(out_dir) / f"heatmap_{arch}_{metric.replace(' ', '_')}.png"
            plt.savefig(out, dpi=150)
            plt.close()
            paths.append(out)
    return paths


def surrogate_scatter(dataset, out_dir, results_root=RESULTS):
    """Does the differentiable training surrogate track the hard eval metric?"""
    sweep = load_seed_csvs(dataset, "sweep_results", results_root)
    pairs = [("Soft DP", DPD2), ("Soft GE", "Theil Index")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, (soft, hard) in zip(axes, pairs):
        ax.scatter(sweep[soft], sweep[hard], s=10, alpha=0.35)
        r = np.corrcoef(sweep[soft], sweep[hard])[0, 1]
        ax.set_xlabel(f"{soft} (training surrogate)")
        ax.set_ylabel(f"{hard} (evaluation metric)")
        ax.set_title(f"Pearson r = {r:.3f}")
    fig.suptitle(f"Surrogate vs. hard metric — {dataset} (all sweep points, all seeds)")
    fig.tight_layout()
    out = Path(out_dir) / "supp_surrogate_validation.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out
