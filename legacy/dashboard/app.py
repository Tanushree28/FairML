"""FairML results dashboard.

Visualizes the fairness-aware training experiment: model comparison,
alpha x beta sweep heatmaps, and the accuracy-fairness tradeoff, for
COMPAS and German Credit.

Reads the result CSVs bundled in dashboard/data/ (copied from
new_experiment/RESULTS/ — run `python refresh_data.py` after re-running the
experiment). Falls back to ../RESULTS/ when running inside the full repo.
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

HERE = Path(__file__).resolve().parent

st.set_page_config(page_title="FairML — Fairness-Aware Training Results",
                   page_icon="⚖️", layout="wide")

DATASETS = {
    "COMPAS (recidivism — sensitive attribute: race)": "compas",
    "German Credit (credit risk — sensitive attribute: sex)": "german",
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


@st.cache_data
def load_results(ds_key):
    for base in (HERE / "data" / ds_key, HERE.parent / "RESULTS" / ds_key):
        sweep, comp = base / "sweep_results.csv", base / "model_comparison.csv"
        if sweep.exists() and comp.exists():
            return pd.read_csv(sweep), pd.read_csv(comp)
    st.error(f"No result CSVs found for dataset '{ds_key}'. "
             "Run the experiment and refresh_data.py first.")
    st.stop()


# ----------------------------------------------------------------------------
st.title("⚖️ FairML — Fairness-Aware Training on COMPAS & German Credit")
st.markdown(
    "Logistic regression / MLP trained with a **differentiable composite loss** "
    f"blending accuracy and fairness — `{LOSS_CAPTION}` — compared against "
    "standard fairness baselines (fairlearn ExponentiatedGradient, "
    "ThresholdOptimizer) and unconstrained models."
)

ds_label = st.sidebar.radio("Dataset", list(DATASETS.keys()))
ds_key = DATASETS[ds_label]
sweep_df, comp_df = load_results(ds_key)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Reading the fairness metrics**\n\n"
    "All fairness metrics: **lower = fairer**. "
    "*DPD (Largest 2 Groups)* compares the two biggest sensitive groups "
    "(COMPAS: African-American vs Caucasian) — the all-groups DPD/EOD on "
    "COMPAS are dominated by groups with only 3–5 test samples, so they "
    "barely move regardless of training."
)

# --- Headline numbers --------------------------------------------------------
st.header("Headline result")

baseline = comp_df[comp_df["Model"].str.contains("Logistic Regression baseline")].iloc[0]
fair = comp_df[comp_df["Model"].str.startswith("Fair Logistic Regression")].iloc[0]

c1, c2, c3 = st.columns(3)
c1.metric("Baseline accuracy → Fair model accuracy",
          f"{fair['Accuracy']:.3f}",
          delta=f"{fair['Accuracy'] - baseline['Accuracy']:+.3f} vs baseline")
c2.metric(f"Baseline {FAIRNESS_METRIC}", f"{baseline[FAIRNESS_METRIC]:.3f}")
c3.metric(f"Fair model {FAIRNESS_METRIC}",
          f"{fair[FAIRNESS_METRIC]:.3f}",
          delta=f"{fair[FAIRNESS_METRIC] - baseline[FAIRNESS_METRIC]:+.3f} (lower = fairer)",
          delta_color="inverse")

pct = (1 - fair[FAIRNESS_METRIC] / baseline[FAIRNESS_METRIC]) * 100 if baseline[FAIRNESS_METRIC] else 0
st.markdown(
    f"**{fair['Model']}** reduces the disparity gap by **{pct:.0f}%** "
    f"({baseline[FAIRNESS_METRIC]:.3f} → {fair[FAIRNESS_METRIC]:.3f}) with an accuracy "
    f"change of {fair['Accuracy'] - baseline['Accuracy']:+.3f}. "
    "Hyperparameters were selected on the validation set only."
)

# --- Model comparison --------------------------------------------------------
st.header("Model comparison (test set)")

metric = st.selectbox("Metric", METRICS, index=METRICS.index(FAIRNESS_METRIC))
plot_df = comp_df.dropna(subset=[metric]).sort_values(metric, ascending=(metric == "Accuracy"))

fig = px.bar(plot_df, x=metric, y="Model", orientation="h", text=metric,
             color="Model", color_discrete_sequence=px.colors.qualitative.T10)
fig.update_traces(texttemplate="%{text:.3f}", textposition="outside", showlegend=False)
fig.update_layout(height=420, margin=dict(l=10, r=60, t=30, b=10),
                  xaxis_title=f"{metric} ({'higher = better' if metric == 'Accuracy' else 'lower = fairer'})",
                  yaxis_title=None)
st.plotly_chart(fig, width="stretch")

with st.expander("Full comparison table"):
    st.dataframe(comp_df.set_index("Model").style.format("{:.4f}", na_rep="—"),
                 width="stretch")

# --- Accuracy vs fairness tradeoff ------------------------------------------
st.header("Accuracy ↔ fairness tradeoff")
st.markdown(
    "Every point is one trained model from the α×β sweep. The ideal corner is "
    "**top-left** (high accuracy, low disparity). Stars are the models from "
    "the comparison table."
)

scatter = sweep_df.copy()
scatter["Architecture"] = scatter["Arch"].map(ARCH_LABELS)
fig = px.scatter(scatter, x=FAIRNESS_METRIC, y="Accuracy", color="Architecture",
                 hover_data=["Alpha", "Beta"],
                 color_discrete_sequence=["#4C78A8", "#F58518"])
fig.add_trace(go.Scatter(
    x=comp_df[FAIRNESS_METRIC], y=comp_df["Accuracy"], mode="markers+text",
    text=comp_df["Model"].str.replace(r" \(.*\)", "", regex=True),
    textposition="top center", textfont=dict(size=9),
    marker=dict(symbol="star", size=14, color="#E45756"),
    name="Comparison models"))
fig.update_layout(height=520, xaxis_title=f"{FAIRNESS_METRIC} (lower = fairer)",
                  margin=dict(l=10, r=10, t=30, b=10))
st.plotly_chart(fig, width="stretch")

# --- Sweep heatmaps ----------------------------------------------------------
st.header("α × β sweep heatmaps")
st.markdown(
    f"`{LOSS_CAPTION}` — **α** weights accuracy (α = 1 means no fairness term); "
    "**β** splits the fairness budget between the group term (SoftDP) and the "
    "individual term (SoftGE). With the fixed differentiable loss, the knobs "
    "have a visible effect (the original detached loss produced flat heatmaps)."
)

hc1, hc2 = st.columns(2)
arch_label = hc1.selectbox("Architecture", list(ARCH_LABELS.values()))
heat_metric = hc2.selectbox("Heatmap metric", METRICS,
                            index=METRICS.index(FAIRNESS_METRIC), key="heat_metric")

arch_key = [k for k, v in ARCH_LABELS.items() if v == arch_label][0]
pivot = (sweep_df[sweep_df["Arch"] == arch_key]
         .pivot_table(index="Alpha", columns="Beta", values=heat_metric))

fig = px.imshow(pivot, text_auto=".3f", aspect="auto",
                color_continuous_scale="Viridis" if heat_metric == "Accuracy" else "RdBu_r",
                labels=dict(x="β (1 = all group fairness)",
                            y="α (1 = no fairness term)",
                            color=heat_metric))
fig.update_layout(height=520, title=f"{heat_metric} — {arch_label} on test set",
                  margin=dict(l=10, r=10, t=50, b=10))
fig.update_xaxes(type="category")
fig.update_yaxes(type="category")
st.plotly_chart(fig, width="stretch")

# --- Methods -----------------------------------------------------------------
with st.expander("Methods & what was fixed"):
    st.markdown(f"""
**Setup.** 60/20/20 train/val/test split (seed 42), StandardScaler +
one-hot encoding. Models: logistic regression and a 64-64 MLP (PyTorch, Adam,
early stopping on validation loss). One model per (architecture, α, β) combination.

**The composite loss.** `{LOSS_CAPTION}` where *SoftDP* is the largest gap in
mean predicted probability between sensitive groups (differentiable
demographic parity) and *SoftGE* is the generalized entropy index (alpha = 2)
on the benefit vector *b = 1 + p − y* (Speicher et al., 2018).

**What was fixed vs. the original code.** The original loss converted the
fairness terms to numpy (`.round().detach().numpy()`) before backprop, cutting
them out of the gradient — only the accuracy term ever trained, which is why
earlier heatmaps were flat across β. It also applied sigmoid twice. Both fixed;
evaluation metrics are unchanged (fairlearn / AIF360).

**Metric caveat.** All-groups DPD/EOD take a max–min gap across every
sensitive group; on COMPAS the smallest race groups have 3–5 test samples, so
those metrics are pinned by noise. *DPD (Largest 2 Groups)* restricts to the
two largest groups and is stable across splits.

**Model selection.** Best (α, β) chosen on the **validation set only**:
the point closest to utopia (accuracy = 1, DPD = 0), excluding near-constant
predictors (predicting one class for everyone is trivially "fair"). Test
metrics are reported, never used for selection.

**Comparison baselines.** fairlearn `ExponentiatedGradient` (Demographic
Parity / Equalized Odds constraints), `ThresholdOptimizer` post-processing,
and an unconstrained Random Forest as an accuracy reference.
""")

st.caption("FairML research project — results regenerate via "
           "`python new_experiment/run_experiment.py --dataset both` in the main repo.")
