"""FairML results dashboard.

Interactive view of the paper pipeline: headline result, model comparison
with seed error bars and Wilcoxon significance, the accuracy-fairness
tradeoff, the beta cross-effect, and alpha x beta sweep heatmaps, for
COMPAS, German Credit and Adult.

Data source order: the live per-seed CSVs under new_experiment/RESULTS/ when
running inside the repo, otherwise the bundled copies in dashboard/data/
(written by refresh_data.py for standalone deployment). Live results win so
the dashboard can never silently show superseded numbers.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from scipy.stats import wilcoxon
except ImportError:  # standalone deploy without scipy: significance hidden
    wilcoxon = None

HERE = Path(__file__).resolve().parent

st.set_page_config(page_title="FairML — Fairness-Aware Training Results",
                   page_icon="⚖️", layout="wide")

DATASETS = {
    "COMPAS (recidivism — sensitive attribute: race)": "compas",
    "German Credit (credit risk — sensitive attribute: sex)": "german",
    "Adult (income > 50K — sensitive attribute: sex)": "adult",
    "Taiwan Credit Default (default next month — sensitive attribute: sex)": "taiwan",
}

METRICS = [
    "Accuracy",
    "Demographic Parity Difference",
    "DPD (Largest 2 Groups)",
    "Equalized Odds Difference",
    "Theil Index",
    "Gini Coefficient",
]

FAIRNESS_METRIC = "DPD (Largest 2 Groups)"
LOSS_CAPTION = "Loss = α·BCE + (1−α)·[β·SoftDP + (1−β)·SoftGE]"

ARCH_LABELS = {"logreg": "Logistic Regression", "mlp": "MLP (64-64)"}
PAIRS = [("Logistic Regression baseline", "Fair Logistic Regression"),
         ("MLP (64-64) baseline", "Fair MLP (64-64)")]


def _live_frames(ds_key):
    """Per-seed CSVs from the canonical results tree (running inside the repo)."""
    root = HERE.parent / "RESULTS" / ds_key
    sweeps = sorted(root.glob("seed*/sweep_results.csv"))
    comps = sorted(root.glob("seed*/model_comparison.csv"))
    if not sweeps or not comps:
        return None
    return (pd.concat([pd.read_csv(f) for f in sweeps], ignore_index=True),
            pd.concat([pd.read_csv(f) for f in comps], ignore_index=True),
            f"live results — {len(sweeps)} seed files from RESULTS/{ds_key}/")


def _bundled_frames(ds_key):
    """Copies bundled for standalone deployment (see refresh_data.py)."""
    base = HERE / "data" / ds_key
    sweep, comp = base / "sweep_all_seeds.csv", base / "comparison_all_seeds.csv"
    if sweep.exists() and comp.exists():
        return (pd.read_csv(sweep), pd.read_csv(comp),
                "bundled snapshot — dashboard/data/")
    return None


@st.cache_data
def load_results(ds_key):
    for source in (_live_frames, _bundled_frames):
        frames = source(ds_key)
        if frames is not None:
            return frames
    st.error(
        f"No result CSVs found for '{ds_key}'. Run the experiment first:\n\n"
        "`.venv/bin/python new_experiment/run_experiment.py --dataset all --seeds 0-9`")
    st.stop()


def summarize(comp_df):
    """Mean, sd and seed count per model family."""
    rows = []
    for fam, g in comp_df.groupby("Family", sort=False):
        row = {"Family": fam, "Seeds": len(g)}
        for m in METRICS:
            row[m] = g[m].mean()
            row[f"{m} sd"] = g[m].std()
        rows.append(row)
    return pd.DataFrame(rows)


def paired_p(comp_df, fair_family, baseline_family, metric):
    """Paired Wilcoxon across seeds; None when it cannot be computed."""
    if wilcoxon is None:
        return None
    a = comp_df[comp_df["Family"] == fair_family].sort_values("Seed")
    b = comp_df[comp_df["Family"] == baseline_family].sort_values("Seed")
    if len(a) < 2 or list(a["Seed"]) != list(b["Seed"]):
        return None
    diff = a[metric].to_numpy() - b[metric].to_numpy()
    if np.allclose(diff, 0):
        return 1.0
    return float(wilcoxon(diff).pvalue)


def stars(p):
    if p is None:
        return ""
    for thr, s in [(0.001, "***"), (0.01, "**"), (0.05, "*")]:
        if p < thr:
            return s
    return "ns"


# ----------------------------------------------------------------------------
st.title("⚖️ FairML — Group vs. Individual Fairness Trade-off")
st.markdown(
    "Logistic regression / MLP trained with a **differentiable composite loss** "
    f"blending accuracy and fairness — `{LOSS_CAPTION}` — across three datasets "
    "and **10 random seeds**, compared against standard fairness baselines "
    "(Reweighing, fairlearn ExponentiatedGradient, ThresholdOptimizer) and "
    "unconstrained models."
)

ds_label = st.sidebar.radio("Dataset", list(DATASETS.keys()))
ds_key = DATASETS[ds_label]
sweep_df, comp_df, source_note = load_results(ds_key)
summary = summarize(comp_df)
n_seeds = int(comp_df["Seed"].nunique())

st.sidebar.caption(f"Source: {source_note}")
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Reading the numbers**\n\n"
    f"Every value is the **mean ± sd over {n_seeds} seeds** (test set). "
    "All fairness metrics: **lower = fairer**.\n\n"
    "*DPD (Largest 2 Groups)* compares the two biggest sensitive groups "
    "(COMPAS: African-American vs Caucasian) — the all-groups DPD/EOD on "
    "COMPAS are dominated by groups with only 3–5 test samples, so they "
    "barely move regardless of training."
)

# --- Headline numbers --------------------------------------------------------
st.header("Headline result")

base_row = summary[summary["Family"] == "Logistic Regression baseline"].iloc[0]
fair_row = summary[summary["Family"] == "Fair Logistic Regression"].iloc[0]
p_fair = paired_p(comp_df, "Fair Logistic Regression",
                  "Logistic Regression baseline", FAIRNESS_METRIC)

c1, c2, c3 = st.columns(3)
c1.metric("Fair model accuracy",
          f"{fair_row['Accuracy']:.3f} ± {fair_row['Accuracy sd']:.3f}",
          delta=f"{fair_row['Accuracy'] - base_row['Accuracy']:+.3f} vs baseline")
c2.metric(f"Baseline {FAIRNESS_METRIC}",
          f"{base_row[FAIRNESS_METRIC]:.3f} ± {base_row[f'{FAIRNESS_METRIC} sd']:.3f}")
c3.metric(f"Fair model {FAIRNESS_METRIC}",
          f"{fair_row[FAIRNESS_METRIC]:.3f} ± {fair_row[f'{FAIRNESS_METRIC} sd']:.3f}",
          delta=f"{fair_row[FAIRNESS_METRIC] - base_row[FAIRNESS_METRIC]:+.3f} (lower = fairer)",
          delta_color="inverse")

pct = ((1 - fair_row[FAIRNESS_METRIC] / base_row[FAIRNESS_METRIC]) * 100
       if base_row[FAIRNESS_METRIC] else 0)
sig_text = (f" Paired Wilcoxon across seeds: **p = {p_fair:.4f} ({stars(p_fair)})**."
            if p_fair is not None else "")
st.markdown(
    f"The fairness-trained logistic regression changes the disparity gap by "
    f"**{pct:.0f}%** ({base_row[FAIRNESS_METRIC]:.3f} → {fair_row[FAIRNESS_METRIC]:.3f}) "
    f"for an accuracy change of {fair_row['Accuracy'] - base_row['Accuracy']:+.3f}."
    f"{sig_text} Hyperparameters (α, β) were selected on the validation set only."
)

# --- Model comparison --------------------------------------------------------
st.header("Model comparison (test set)")

metric = st.selectbox("Metric", METRICS, index=METRICS.index(FAIRNESS_METRIC))
plot_df = summary.sort_values(metric, ascending=(metric == "Accuracy"))

fig = px.bar(plot_df, x=metric, y="Family", orientation="h", text=metric,
             error_x=f"{metric} sd",
             color="Family", color_discrete_sequence=px.colors.qualitative.T10)
fig.update_traces(texttemplate="%{text:.3f}", textposition="auto", showlegend=False)
fig.update_layout(
    height=440, margin=dict(l=10, r=40, t=30, b=10),
    xaxis_title=(f"{metric} "
                 f"({'higher = better' if metric == 'Accuracy' else 'lower = fairer'})"),
    yaxis_title=None)
st.plotly_chart(fig, width="stretch")
st.caption(f"Bars are means, whiskers ± sd over {n_seeds} seeds.")

if wilcoxon is not None:
    sig_rows = []
    for baseline_family, fair_family in PAIRS:
        p = paired_p(comp_df, fair_family, baseline_family, metric)
        if p is not None:
            sig_rows.append({"Comparison": f"{fair_family} vs {baseline_family}",
                             "p (Wilcoxon)": round(p, 4), "": stars(p)})
    if sig_rows:
        st.caption("Paired significance for the selected metric "
                   "(fairness-trained model vs. its own baseline, across seeds):")
        st.dataframe(pd.DataFrame(sig_rows), width="stretch", hide_index=True)

with st.expander("Full comparison table (mean ± sd)"):
    display = summary.set_index("Family")[["Seeds"] + METRICS]
    st.dataframe(display.style.format({m: "{:.4f}" for m in METRICS}, na_rep="—"),
                 width="stretch")

# --- Accuracy vs fairness tradeoff ------------------------------------------
st.header("Accuracy ↔ fairness tradeoff")
st.markdown(
    "Every point is one (α, β) setting from the sweep, averaged over seeds. "
    "The ideal corner is **top-left** (high accuracy, low disparity). Stars are "
    "the comparison models."
)

sweep_mean = sweep_df.groupby(["Arch", "Alpha", "Beta"], as_index=False)[METRICS].mean()
sweep_mean["Architecture"] = sweep_mean["Arch"].map(ARCH_LABELS)
fig = px.scatter(sweep_mean, x=FAIRNESS_METRIC, y="Accuracy", color="Architecture",
                 hover_data=["Alpha", "Beta"],
                 color_discrete_sequence=["#4C78A8", "#F58518"])
fig.add_trace(go.Scatter(
    x=summary[FAIRNESS_METRIC], y=summary["Accuracy"], mode="markers+text",
    text=summary["Family"].str.replace(r" \(.*\)", "", regex=True),
    textposition="top center", textfont=dict(size=9),
    marker=dict(symbol="star", size=14, color="#E45756"),
    name="Comparison models"))
fig.update_layout(height=520, xaxis_title=f"{FAIRNESS_METRIC} (lower = fairer)",
                  margin=dict(l=10, r=10, t=30, b=10))
st.plotly_chart(fig, width="stretch")

# --- Beta cross-effect -------------------------------------------------------
st.header("The β knob: group vs. individual fairness")
st.markdown(
    "The project's core question. **β = 0** spends the whole fairness budget on "
    "the *individual* term (SoftGE), **β = 1** on the *group* term (SoftDP). "
    "When the two kinds of fairness are in tension, the red and purple lines "
    "move in opposite directions."
)

bc1, bc2 = st.columns(2)
beta_arch_label = bc1.selectbox("Architecture", list(ARCH_LABELS.values()),
                                key="beta_arch")
beta_arch = [k for k, v in ARCH_LABELS.items() if v == beta_arch_label][0]
alpha_choices = sorted(a for a in sweep_df["Alpha"].unique() if 0 < a < 1)
alpha_pick = bc2.select_slider("α (accuracy weight)", options=alpha_choices,
                               value=alpha_choices[len(alpha_choices) // 2])

beta_slice = sweep_df[(sweep_df["Arch"] == beta_arch) & (sweep_df["Alpha"] == alpha_pick)]
beta_stats = beta_slice.groupby("Beta")[[FAIRNESS_METRIC, "Theil Index"]].agg(["mean", "std"])

fig = go.Figure()
for metric_name, line_color, band_color, label in [
        (FAIRNESS_METRIC, "#E45756", "rgba(228,87,86,0.18)",
         "group fairness (DPD, largest 2)"),
        ("Theil Index", "#8B5CF6", "rgba(139,92,246,0.18)",
         "individual fairness (Theil)")]:
    mean = beta_stats[(metric_name, "mean")]
    sd = beta_stats[(metric_name, "std")].fillna(0)
    fig.add_trace(go.Scatter(x=beta_stats.index, y=mean + sd, mode="lines",
                             line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=beta_stats.index, y=mean - sd, mode="lines",
                             line=dict(width=0), fill="tonexty", fillcolor=band_color,
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=beta_stats.index, y=mean, mode="lines+markers",
                             name=label, line=dict(color=line_color, width=3)))
fig.update_layout(height=460, margin=dict(l=10, r=10, t=50, b=10),
                  xaxis_title="β  (0 = individual only, 1 = group only)",
                  yaxis_title=f"metric value (mean ± sd, {n_seeds} seeds)",
                  title=f"{beta_arch_label}, α = {alpha_pick}")
st.plotly_chart(fig, width="stretch")

# --- Sweep heatmaps ----------------------------------------------------------
st.header("α × β sweep heatmaps")
st.markdown(
    f"`{LOSS_CAPTION}` — **α** weights accuracy (α = 1 means no fairness term); "
    "**β** splits the fairness budget between the group term (SoftDP) and the "
    "individual term (SoftGE). Values are seed means; with the differentiable "
    "loss the knobs have a visible effect (the original detached loss produced "
    "heatmaps that were flat across β)."
)

hc1, hc2 = st.columns(2)
arch_label = hc1.selectbox("Architecture", list(ARCH_LABELS.values()), key="heat_arch")
heat_metric = hc2.selectbox("Heatmap metric", METRICS,
                            index=METRICS.index(FAIRNESS_METRIC), key="heat_metric")

arch_key = [k for k, v in ARCH_LABELS.items() if v == arch_label][0]
pivot = (sweep_mean[sweep_mean["Arch"] == arch_key]
         .pivot_table(index="Alpha", columns="Beta", values=heat_metric))

fig = px.imshow(pivot, text_auto=".3f", aspect="auto",
                color_continuous_scale="Viridis" if heat_metric == "Accuracy" else "RdBu_r",
                labels=dict(x="β (1 = all group fairness)",
                            y="α (1 = no fairness term)",
                            color=heat_metric))
fig.update_layout(height=520, title=f"{heat_metric} — {arch_label} (seed mean, test set)",
                  margin=dict(l=10, r=10, t=50, b=10))
fig.update_xaxes(type="category")
fig.update_yaxes(type="category")
st.plotly_chart(fig, width="stretch")

# --- Methods -----------------------------------------------------------------
with st.expander("Methods & what was fixed"):
    st.markdown(f"""
**Setup.** 60/20/20 train/val/test split re-drawn per seed, StandardScaler +
one-hot encoding. Models: logistic regression and a 64-64 MLP (PyTorch, Adam,
early stopping on validation loss). One model per (architecture, α, β, seed);
**10 seeds** per configuration, so every number here is a mean ± sd.

**The composite loss.** `{LOSS_CAPTION}` where *SoftDP* is the largest gap in
mean predicted probability between sensitive groups (differentiable
demographic parity) and *SoftGE* is the generalized entropy index (alpha = 2)
on the benefit vector *b = 1 + p − y* (Speicher et al., 2018).

**What was fixed vs. the original code.** The original loss converted the
fairness terms to numpy (`.round().detach().numpy()`) before backprop, cutting
them out of the gradient — only the accuracy term ever trained, which is why
earlier heatmaps were flat across β. It also applied sigmoid twice. Both fixed.

**COMPAS label leakage removed.** The earlier feature set included
`duration = end − start`, which correlates −0.78 with the label for mechanical
reasons (re-offending truncates the observation window). COMPAS now uses
ProPublica's standard features and row filters (n = 6172), so accuracy sits at
the literature-typical ~0.67 rather than an inflated ~0.87.

**Metric caveat.** All-groups DPD/EOD take a max–min gap across every
sensitive group; on COMPAS the smallest race groups have 3–5 test samples, so
those metrics are pinned by noise. *DPD (Largest 2 Groups)* restricts to the
two largest groups and is stable across splits.

**Model selection.** Best (α, β) chosen on the **validation set only**: the
point closest to utopia (accuracy = 1, DPD = 0), excluding near-constant
predictors (predicting one class for everyone is trivially "fair"). Test
metrics are reported, never used for selection.

**Comparison baselines.** Reweighing (Kamiran & Calders, pre-processing),
fairlearn `ExponentiatedGradient` (Demographic Parity / Equalized Odds),
`ThresholdOptimizer` (post-processing), plus Random Forest and XGBoost as
unconstrained accuracy references.

**Significance.** Paired Wilcoxon signed-rank across seeds, fairness-trained
model vs. its own baseline. With {n_seeds} seeds the smallest attainable
two-sided p is ≈ 0.002, so `***` (p < 0.001) can never appear.
""")

st.caption("FairML research project — regenerate results with "
           "`python new_experiment/run_experiment.py --dataset all --seeds 0-9`, "
           "then paper figures with `python new_experiment/analysis/run_analysis.py --dataset all`.")
