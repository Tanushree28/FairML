"""Side-by-side evaluation across datasets.

Main: COMPAS, Adult, ACS Public Coverage. Boundary case: ACS Employment (the
group/individual trade-off does not appear). Discussion: Taiwan (low
disparity, large n) and German (n=1000), shown only in selection diagnostics.
"""

import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from analysis.figures import (ARCH_COLORS, ARCH_LABELS, BASELINE_FAMILIES, FIG3_METRICS,
                              PAIRS, pareto_mask, seed_mean_selection, stars)
from analysis.stats import DPD2, RESULTS, holm_correction, load_seed_csvs, significance_vs_baseline

MAIN_DATASETS = ["compas", "adult", "acs_pubcov"]
BOUNDARY_DATASETS = ["acs_employment"]
DISCUSSION_DATASETS = ["taiwan", "german"]
RESULT_DATASETS = MAIN_DATASETS + BOUNDARY_DATASETS
ALL_DATASETS = RESULT_DATASETS + DISCUSSION_DATASETS
REPORT_FIGURES = ["cross_fig1_pareto.png", "cross_fig2_beta.png",
                  "cross_fig3_dissociation.png", "cross_fig4_selection_diagnostic.png"]
DATASET_LABELS = {"compas": "COMPAS (race)", "adult": "Adult (sex)",
                  "taiwan": "Taiwan Credit (sex)", "taiwan_edu": "Taiwan Credit (education)",
                  "acs_employment": "ACS Employment (sex)",
                  "acs_pubcov": "ACS Public Coverage (race)",
                  "german": "German Credit (sex)"}
SHORT_LABELS = {"compas": "COMPAS", "adult": "Adult", "acs_pubcov": "ACS PubCov",
                "acs_employment": "ACS Employ.\n(boundary)", "taiwan": "Taiwan", "german": "German"}
GE2 = "Theil Index"          # stored column name; the metric is GE(2), see IndividualFairness
VAL_DPD2 = "Val DPD (Largest 2 Groups)"
BETA_ALPHA = 0.75            # α at which the β curves are drawn (matches the paper figure)


def _role(name):
    if name in BOUNDARY_DATASETS:
        return "boundary"
    return "discussion" if name in DISCUSSION_DATASETS else "main"


def _title(name):
    return DATASET_LABELS[name] + ("\n(boundary case)" if name in BOUNDARY_DATASETS else "")


def _write_table(df, out_dir, stem):
    df.to_csv(Path(out_dir) / f"{stem}.csv", index=False)
    (Path(out_dir) / f"{stem}.tex").write_text(df.to_latex(index=False, float_format="%.3f"))
    return df


def _fair_nondegenerate(arch_sweep):
    return arch_sweep[(arch_sweep["Alpha"] < 1)
                      & arch_sweep["Val Positive Rate"].between(0.05, 0.95)]


def dataset_profile(datasets, out_dir):
    """Size, base rate, and the label gap between the two largest groups."""
    from data_loading import load_dataset
    rows = []
    for name in datasets:
        ds = load_dataset(name, seed=0)
        y = pd.concat([ds["y_train"], ds["y_val"], ds["y_test"]], ignore_index=True)
        s = pd.concat([ds["sens"][k] for k in ("train", "val", "test")], ignore_index=True)
        top2 = s.value_counts().index[:2]
        rates = y.groupby(s).mean()
        rows.append({
            "Dataset": DATASET_LABELS[name],
            "Role": _role(name),
            "n": len(y),
            "Features (encoded)": ds["input_dim"],
            "Positive rate": y.mean(),
            "Groups compared": f"{top2[0]} vs {top2[1]}",
            "Label gap (pts)": 100 * abs(rates[top2[0]] - rates[top2[1]]),
            "Smaller group in val split": int(ds["sens"]["val"].value_counts()[top2].min()),
        })
    return _write_table(pd.DataFrame(rows), out_dir, "cross_table0_dataset_profile")


def headline_table(datasets, out_dir, results_root=RESULTS):
    """Selected fair model vs its BCE-only baseline, per dataset and architecture."""
    rows = []
    for name in datasets:
        comp = load_seed_csvs(name, "model_comparison", results_root)
        for base_fam, fair_fam in PAIRS:
            sig = significance_vs_baseline(comp, fair_fam, base_fam,
                                           metrics=FIG3_METRICS).set_index("Metric")["p_holm"]
            b = comp[comp["Family"] == base_fam]
            f = comp[comp["Family"] == fair_fam]
            row = {"Dataset": DATASET_LABELS[name], "Role": _role(name),
                   "Arch": "LR" if "Logistic" in base_fam else "MLP"}
            for metric, short in [("Accuracy", "Acc"), (DPD2, "DPD2"), (GE2, "GE2")]:
                row[f"{short} base"] = b[metric].mean()
                row[f"{short} fair"] = f[metric].mean()
                row[f"{short} p_holm"] = sig[metric]
            row["DPD2 change %"] = 100 * (row["DPD2 fair"] - row["DPD2 base"]) / row["DPD2 base"]
            row["Acc cost (pts)"] = 100 * (row["Acc base"] - row["Acc fair"])
            row["Selected beta (mean)"] = f["Beta"].mean()
            rows.append(row)
    return _write_table(pd.DataFrame(rows), out_dir, "cross_table1_headline")


def _paired_change(g, variant, metrics):
    """Per-seed paired contrast of a sweep variant (α, β) against α=1, β=0."""
    base = g[(g["Alpha"] == 1) & (g["Beta"] == 0)].sort_values("Seed")
    var = g[(g["Alpha"] == variant[0]) & (g["Beta"] == variant[1])].sort_values("Seed")
    pvals, out = [], {}
    for m in metrics:
        diff = var[m].to_numpy() - base[m].to_numpy()
        pvals.append(1.0 if np.allclose(diff, 0) else wilcoxon(diff).pvalue)
        out[m] = (base[m].mean(), var[m].mean())
    return out, dict(zip(metrics, holm_correction(pvals)))


def dissociation_table(datasets, out_dir, results_root=RESULTS):
    """Individual-only (β=0) and group-only (β=1) at α*, against the α=1 baseline.
    Holm family = the four FIG3 metrics per variant, as in the paper."""
    rows = []
    for name in datasets:
        sweep = load_seed_csvs(name, "sweep_results", results_root)
        for arch in ["logreg", "mlp"]:
            g = sweep[sweep["Arch"] == arch]
            a_star, _ = seed_mean_selection(g)
            for label, beta in [("individual only (β=0)", 0), ("group only (β=1)", 1)]:
                means, p = _paired_change(g, (a_star, beta), FIG3_METRICS)
                row = {"Dataset": DATASET_LABELS[name], "Role": _role(name),
                       "Arch": "LR" if arch == "logreg" else "MLP", "alpha*": a_star, "Variant": label}
                for metric, short in [(DPD2, "DPD2"), (GE2, "GE2"), ("Accuracy", "Acc")]:
                    base, var = means[metric]
                    row[f"{short} base"] = base
                    row[f"{short} variant"] = var
                    row[f"{short} change %"] = 100 * (var - base) / base
                    row[f"{short} p_holm"] = p[metric]
                rows.append(row)
    return _write_table(pd.DataFrame(rows), out_dir, "cross_table2_dissociation")


def selection_diagnostics(datasets, out_dir, results_root=RESULTS):
    """Can validation-based selection work here? r and SNR follow the paper's
    Table (non-degenerate α<1 configs, one point per seed). The spread ratio is
    the across-config std of seed-mean (1 − val acc) over that of seed-mean val
    DPD2: large values mean the utopia rule effectively ranks by accuracy."""
    rows = []
    for name in datasets:
        sweep = load_seed_csvs(name, "sweep_results", results_root)
        comp = load_seed_csvs(name, "model_comparison", results_root)
        for arch in ["logreg", "mlp"]:
            f = _fair_nondegenerate(sweep[sweep["Arch"] == arch])
            by_cfg = f.groupby(["Alpha", "Beta"])
            means = by_cfg[["Val Accuracy", VAL_DPD2]].mean()
            fair_fam = f"Fair {ARCH_LABELS[arch]}"
            sel = comp[comp["Family"] == fair_fam]
            rows.append({
                "Dataset": DATASET_LABELS[name],
                "Role": _role(name),
                "Arch": "LR" if arch == "logreg" else "MLP",
                "r(val, test) DPD2": np.corrcoef(f[VAL_DPD2], f[DPD2])[0, 1],
                "SNR": means[VAL_DPD2].std() / by_cfg[VAL_DPD2].std().mean(),
                "Spread ratio acc:DPD2": (1 - means["Val Accuracy"]).std() / means[VAL_DPD2].std(),
                "Selected beta (mean)": sel["Beta"].mean(),
                "Distinct selected configs": sel[["Alpha", "Beta"]].drop_duplicates().shape[0],
            })
    return _write_table(pd.DataFrame(rows), out_dir, "cross_table3_selection_diagnostics")


def _draw_pareto(ax, name, results_root):
    sweep = load_seed_csvs(name, "sweep_results", results_root)
    comp = load_seed_csvs(name, "model_comparison", results_root)
    mean_sweep = sweep.groupby(["Arch", "Alpha", "Beta"], as_index=False)[["Accuracy", DPD2]].mean()
    for arch, g in mean_sweep.groupby("Arch"):
        ax.scatter(g[DPD2], g["Accuracy"], s=10, alpha=0.25, color=ARCH_COLORS[arch])
        front = g[pareto_mask(g["Accuracy"].to_numpy(), g[DPD2].to_numpy())].sort_values(DPD2)
        ax.plot(front[DPD2], front["Accuracy"], "-o", lw=1.8, ms=3, color=ARCH_COLORS[arch],
                label=f"{ARCH_LABELS[arch]} Pareto front")
    agg = comp.groupby("Family")[["Accuracy", DPD2]].agg(["mean", "std"])
    for fam, mk in zip(BASELINE_FAMILIES, ["D", "s", "^", "v", "P", "X"]):
        if fam in agg.index:
            ax.errorbar(agg.loc[fam, (DPD2, "mean")], agg.loc[fam, ("Accuracy", "mean")],
                        xerr=agg.loc[fam, (DPD2, "std")], yerr=agg.loc[fam, ("Accuracy", "std")],
                        fmt=mk, ms=6, capsize=2, color="0.25", mfc="white", label=fam)
    # near-degenerate small-α points sit far below; zoom on the competitive region
    fam_acc = agg[("Accuracy", "mean")]
    top = max(mean_sweep["Accuracy"].max(), fam_acc.max())
    ax.set_ylim(min(fam_acc.min(), top - 0.05) - 0.02, top + 0.01)
    ax.set_title(_title(name))
    ax.set_xlabel(f"{DPD2} (lower = fairer)")


def plot_cross_pareto(datasets, out_dir, results_root=RESULTS):
    ncols = 2 if len(datasets) == 4 else len(datasets)
    nrows = -(-len(datasets) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 4.4 * nrows + 1.0), squeeze=False)
    for i, (ax, name) in enumerate(zip(axes.flat, datasets)):
        _draw_pareto(ax, name, results_root)
        if i % ncols == 0:
            ax.set_ylabel("Accuracy (test)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle("Accuracy–fairness frontiers side by side (seed means; baselines ± std)")
    fig.tight_layout(rect=[0, 1.0 / (4.4 * nrows + 1.0), 1, 0.96])
    out = Path(out_dir) / "cross_fig1_pareto.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_cross_beta(datasets, out_dir, results_root=RESULTS, alpha=BETA_ALPHA):
    fig, axes = plt.subplots(2, len(datasets), figsize=(4.6 * len(datasets), 6.6),
                             sharex=True, squeeze=False)
    for c, name in enumerate(datasets):
        sweep = load_seed_csvs(name, "sweep_results", results_root)
        for r, (metric, ylabel) in enumerate([(DPD2, "DPD2 (group)"), (GE2, "GE(2) (individual)")]):
            ax = axes[r][c]
            for arch in ["logreg", "mlp"]:
                st = (sweep[(sweep["Arch"] == arch) & (sweep["Alpha"] == alpha)]
                      .groupby("Beta")[metric].agg(["mean", "std"]))
                ax.plot(st.index, st["mean"], "-o", ms=3, color=ARCH_COLORS[arch],
                        label=ARCH_LABELS[arch] if (r == 0 and c == 0) else None)
                ax.fill_between(st.index, st["mean"] - st["std"].fillna(0),
                                st["mean"] + st["std"].fillna(0), color=ARCH_COLORS[arch], alpha=0.15)
            if r == 0:
                ax.set_title(_title(name))
            else:
                ax.set_xlabel("β (0 = individual only, 1 = group only)")
            if c == 0:
                ax.set_ylabel(ylabel)
    fig.legend(loc="upper right", fontsize=8)
    fig.suptitle(f"Group and individual fairness as β varies, α={alpha} (mean ± std over seeds)")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = Path(out_dir) / "cross_fig2_beta.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_cross_dissociation(table, datasets, out_dir):
    """Absolute change from baseline for β=0 / β=1, per metric and architecture.
    Absolute, not %: Taiwan's near-zero baseline DPD2 makes % changes explode."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), squeeze=False)
    x = np.arange(len(datasets))
    width = 0.38
    colors = {"individual only (β=0)": "tab:purple", "group only (β=1)": "tab:red"}
    for r, arch in enumerate(["LR", "MLP"]):
        for c, short in enumerate(["DPD2", "GE2"]):
            ax = axes[r][c]
            for k, variant in enumerate(colors):
                sub = (table[(table["Arch"] == arch) & (table["Variant"] == variant)]
                       .set_index("Dataset").loc[[DATASET_LABELS[d] for d in datasets]])
                vals = (sub[f"{short} variant"] - sub[f"{short} base"]).to_numpy()
                pos = x + (k - 0.5) * width
                ax.bar(pos, vals, width, color=colors[variant],
                       label=variant if (r == 0 and c == 0) else None)
                for xi, v, p in zip(pos, vals, sub[f"{short} p_holm"]):
                    ax.annotate(stars(p), (xi, v), ha="center", fontsize=8,
                                xytext=(0, 3 if v >= 0 else -10), textcoords="offset points")
            ax.axhline(0, color="0.3", lw=0.8)
            ax.margins(y=0.15)
            ax.set_xticks(x)
            ax.set_xticklabels([SHORT_LABELS[d] for d in datasets], fontsize=9)
            ax.set_title(f"{arch}: {'DPD2 (group)' if short == 'DPD2' else 'GE(2) (individual)'}")
            if c == 0:
                ax.set_ylabel("change vs α=1 baseline (absolute)")
    fig.legend(loc="upper right", fontsize=8)
    fig.suptitle("Double dissociation side by side (at α*, Holm-corrected Wilcoxon)")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = Path(out_dir) / "cross_fig3_dissociation.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_selection_diagnostic(datasets, out_dir, results_root=RESULTS, arch="logreg"):
    ncols = min(3, len(datasets))
    nrows = -(-len(datasets) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.8 * ncols, 3.6 * nrows), squeeze=False)
    for ax in axes.flat[len(datasets):]:
        ax.axis("off")
    for i, (ax, name) in enumerate(zip(axes.flat, datasets)):
        sweep = load_seed_csvs(name, "sweep_results", results_root)
        f = _fair_nondegenerate(sweep[sweep["Arch"] == arch])
        r = np.corrcoef(f[VAL_DPD2], f[DPD2])[0, 1]
        ax.scatter(f[VAL_DPD2], f[DPD2], s=8, alpha=0.35,
                   color="tab:gray" if name in DISCUSSION_DATASETS else "tab:blue")
        hi = max(f[VAL_DPD2].max(), f[DPD2].max()) * 1.05
        ax.plot([0, hi], [0, hi], "--", color="0.4", lw=0.8)
        ax.set_xlim(0, hi)
        ax.set_ylim(0, hi)
        ax.set_title(f"{DATASET_LABELS[name]} [{_role(name)}]\nr = {r:.2f}", fontsize=10)
        ax.set_xlabel("validation DPD2")
        if i % ncols == 0:
            ax.set_ylabel("test DPD2")
    fig.suptitle(f"Does validation DPD2 predict test DPD2? ({ARCH_LABELS[arch]}, non-degenerate configs)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = Path(out_dir) / "cross_fig4_selection_diagnostic.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def run_cross_dataset(out_dir, results_root=RESULTS, with_profile=True, report_fig_dir=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if with_profile:
        dataset_profile(ALL_DATASETS, out_dir)
    headline_table(RESULT_DATASETS, out_dir, results_root)
    diss = dissociation_table(RESULT_DATASETS, out_dir, results_root)
    selection_diagnostics(ALL_DATASETS, out_dir, results_root)
    plot_cross_pareto(RESULT_DATASETS, out_dir, results_root)
    plot_cross_beta(RESULT_DATASETS, out_dir, results_root)
    plot_cross_dissociation(diss, RESULT_DATASETS, out_dir)
    plot_selection_diagnostic(ALL_DATASETS, out_dir, results_root)
    if report_fig_dir is not None:
        report_fig_dir = Path(report_fig_dir)
        report_fig_dir.mkdir(parents=True, exist_ok=True)
        for fig in REPORT_FIGURES:
            shutil.copy2(out_dir / fig, report_fig_dir / fig.removeprefix("cross_"))
    return out_dir
