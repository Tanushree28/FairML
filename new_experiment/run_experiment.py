"""New experiment: fixed differentiable fairness loss + model comparison.

What it runs per dataset:
  1. Alpha x Beta sweep of the composite loss (now actually differentiable)
     for two architectures: logistic regression and a small MLP.
  2. Baselines / comparison models:
       - LogReg baseline (BCE only, alpha=1)
       - MLP baseline (BCE only, alpha=1)
       - Fair LogReg / Fair MLP (best alpha,beta picked on the validation set)
       - fairlearn ExponentiatedGradient (Demographic Parity / Equalized Odds)
       - fairlearn ThresholdOptimizer (post-processing, Demographic Parity)
       - Random Forest (accuracy reference, no fairness intervention)

Outputs (always rewritten, no stale-CSV caching), per dataset and per seed:
  new_experiment/RESULTS/<dataset>/seed<k>/sweep_results.csv
  new_experiment/RESULTS/<dataset>/seed<k>/model_comparison.csv
  new_experiment/RESULTS/<dataset>/seed<k>/PLOTS/heatmap_<arch>_<metric>.png   (single-seed runs only)
  new_experiment/RESULTS/<dataset>/seed<k>/PLOTS/model_comparison.png         (single-seed runs only)

Model checkpoints are cached in new_experiment/MODELS/<dataset>/seed<k>/ and
reused; pass --retrain to force retraining.

Usage:
  python new_experiment/run_experiment.py                  # COMPAS, seed 42
  python new_experiment/run_experiment.py --dataset german
  python new_experiment/run_experiment.py --dataset both --seeds "0-9"
  python new_experiment/run_experiment.py --dataset all --seeds "0-2,7"
"""

import argparse
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression as SkLogisticRegression
from sklearn.metrics import accuracy_score
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
from fairlearn.postprocessing import ThresholdOptimizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

# Reuse the original evaluation metrics so numbers stay comparable.
from GroupFairness import demographic_parity_difference, equalized_odds_difference
from IndividualFairness import theil_index, gini_coefficient

from data_loading import load_dataset
from models import LogisticRegression, MLP, train_model, predict_proba
from losses import soft_demographic_parity, soft_generalized_entropy
from baselines import reweighing_logreg, xgboost_reference

import warnings
warnings.filterwarnings("ignore")


def parse_seeds(spec):
    """'0-2,7' -> [0, 1, 2, 7]."""
    seeds = []
    for part in str(spec).split(","):
        if "-" in part:
            a, b = part.split("-")
            seeds.extend(range(int(a), int(b) + 1))
        else:
            seeds.append(int(part))
    return seeds


ALPHA_LIST = [0, 0.005, 0.05, 0.25, 0.5, 0.75, 1]
BETA_LIST = [0, 0.005, 0.05, 0.25, 0.5, 0.75, 1]

METRICS = [
    "Accuracy",
    "Demographic Parity Difference",
    "DPD (Largest 2 Groups)",
    "Equalized Odds Difference",
    "Theil Index",
    "Gini Coefficient",
]

ARCHITECTURES = {
    "logreg": ("Logistic Regression", LogisticRegression),
    "mlp": ("MLP (64-64)", MLP),
}

LOSS_CAPTION = "Loss = α·BCE + (1−α)·[β·SoftDP + (1−β)·SoftGE]"


def dpd_largest_two_groups(y_true, y_pred, sensitive_features):
    """Demographic parity difference restricted to the two largest sensitive
    groups (for COMPAS: African-American vs Caucasian, ProPublica's focal
    comparison). The all-groups max-min DPD is pinned by tiny groups (e.g.
    Asian n=5, Native American n=3 in the COMPAS test split), which makes it
    extremely noisy; this version is stable across splits. For German Credit
    (two sex groups) it equals the regular DPD."""
    s = pd.Series(np.asarray(sensitive_features)).reset_index(drop=True)
    top2 = s.value_counts().index[:2]
    mask = s.isin(top2).values
    return demographic_parity_difference(
        np.asarray(y_true)[mask], np.asarray(y_pred)[mask],
        sensitive_features=s[mask])


def evaluate_predictions(y_true, y_pred, sensitive_features):
    y_true_arr = np.asarray(y_true).astype(float)
    y_pred_arr = np.asarray(y_pred).astype(float)
    return {
        "Accuracy": accuracy_score(y_true_arr, y_pred_arr),
        "Demographic Parity Difference": demographic_parity_difference(
            y_true_arr, y_pred_arr, sensitive_features=sensitive_features),
        "DPD (Largest 2 Groups)": dpd_largest_two_groups(
            y_true_arr, y_pred_arr, sensitive_features),
        "Equalized Odds Difference": equalized_odds_difference(
            y_true_arr, y_pred_arr, sensitive_features=sensitive_features),
        "Theil Index": theil_index(y_true_arr, y_pred_arr),
        "Gini Coefficient": gini_coefficient(y_true_arr, y_pred_arr),
    }


def run_sweep(ds, arch_key, model_dir, args, seed):
    arch_label, arch_cls = ARCHITECTURES[arch_key]
    rows = []
    combos = list(product(ALPHA_LIST, BETA_LIST))
    sweep_start = time.perf_counter()

    for i, (alpha, beta) in enumerate(combos, 1):
        ckpt = model_dir / f"{arch_key}_alpha_{alpha}_beta_{beta}.pth"
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = arch_cls(ds["input_dim"])

        t0 = time.perf_counter()
        if ckpt.exists() and not args.retrain:
            model.load_state_dict(torch.load(ckpt, weights_only=True))
            trained = False
        else:
            train_model(
                model,
                ds["X_train_t"], ds["y_train_t"], ds["group_ids"]["train"],
                ds["X_val_t"], ds["y_val_t"], ds["group_ids"]["val"],
                alpha=alpha, beta=beta,
                num_epochs=args.epochs, patience=args.patience,
            )
            torch.save(model.state_dict(), ckpt)
            trained = True
        elapsed = time.perf_counter() - t0

        probs_test = predict_proba(model, ds["X_test_t"])
        y_pred_test = (probs_test >= 0.5).astype(int)
        y_pred_val = (predict_proba(model, ds["X_val_t"]) >= 0.5).astype(int)

        probs_t = torch.tensor(probs_test, dtype=torch.float32)
        y_true_t = torch.tensor(np.asarray(ds["y_test"], dtype=np.float32))
        soft_dp = float(soft_demographic_parity(probs_t, ds["group_ids"]["test"]))
        soft_ge = float(soft_generalized_entropy(probs_t, y_true_t))

        test_metrics = evaluate_predictions(ds["y_test"], y_pred_test, ds["sens"]["test"])
        val_acc = accuracy_score(ds["y_val"], y_pred_val)
        val_dpd2 = dpd_largest_two_groups(
            np.asarray(ds["y_val"], dtype=float), y_pred_val.astype(float),
            sensitive_features=ds["sens"]["val"])

        rows.append({
            "Model": arch_label,
            "Arch": arch_key,
            "Alpha": alpha,
            "Beta": beta,
            "Seed": seed,
            **{k: round(v, 4) for k, v in test_metrics.items()},
            "Soft DP": round(soft_dp, 4),
            "Soft GE": round(soft_ge, 4),
            "Val Accuracy": round(val_acc, 4),
            "Val DPD (Largest 2 Groups)": round(val_dpd2, 4),
            "Val Positive Rate": round(float(y_pred_val.mean()), 4),
            "Train Time (s)": round(elapsed, 3),
            "Trained This Run": trained,
        })
        print(f"  [{arch_key} {i}/{len(combos)}] alpha={alpha}, beta={beta} "
              f"-> {'trained' if trained else 'loaded'} in {elapsed:.2f}s | "
              f"test acc={test_metrics['Accuracy']:.3f}, DPD={test_metrics['Demographic Parity Difference']:.3f}")

    print(f"  {arch_key} sweep done in {time.perf_counter() - sweep_start:.1f}s")
    return pd.DataFrame(rows)


def pick_best_fair(sweep_df):
    """Best fairness-trained combo, chosen on the VALIDATION set only:
    among combos with a real fairness weight (alpha < 1), take the point
    closest to the utopia point (accuracy = 1, DPD = 0), i.e. minimize
    sqrt((1 - val_acc)^2 + val_dpd2^2). Uses the largest-two-groups DPD,
    which is stable across splits (the all-groups DPD is dominated by
    tiny-group noise). (Near-)constant predictors are excluded — predicting
    one class for everyone is trivially 'fair' but useless."""
    fair = sweep_df[(sweep_df["Alpha"] < 1)
                    & (sweep_df["Val Positive Rate"] >= 0.05)
                    & (sweep_df["Val Positive Rate"] <= 0.95)].copy()
    fair["dist"] = np.sqrt((1.0 - fair["Val Accuracy"]) ** 2
                           + fair["Val DPD (Largest 2 Groups)"] ** 2)
    return fair.sort_values("dist").iloc[0]


def run_fairlearn_baselines(ds, seed):
    """ExponentiatedGradient (DP / EO), ThresholdOptimizer, RandomForest."""
    rows = []
    X_tr, y_tr, s_tr = ds["X_train"], ds["y_train"], ds["sens"]["train"]
    X_te, s_te = ds["X_test"], ds["sens"]["test"]

    def add(name, y_pred):
        rows.append({"Model": name, "Family": name, "Seed": seed,
                     **{k: round(v, 4) for k, v in
                        evaluate_predictions(ds["y_test"], y_pred, s_te).items()}})
        print(f"  {name}: done")

    eg_dp = ExponentiatedGradient(SkLogisticRegression(max_iter=1000),
                                  constraints=DemographicParity())
    eg_dp.fit(X_tr, y_tr, sensitive_features=s_tr)
    add("ExpGrad LogReg (Demographic Parity)", eg_dp.predict(X_te, random_state=seed))

    eg_eo = ExponentiatedGradient(SkLogisticRegression(max_iter=1000),
                                  constraints=EqualizedOdds())
    eg_eo.fit(X_tr, y_tr, sensitive_features=s_tr)
    add("ExpGrad LogReg (Equalized Odds)", eg_eo.predict(X_te, random_state=seed))

    thr = ThresholdOptimizer(estimator=SkLogisticRegression(max_iter=1000),
                             constraints="demographic_parity",
                             predict_method="predict_proba", prefit=False)
    thr.fit(X_tr, y_tr, sensitive_features=s_tr)
    add("ThresholdOptimizer LogReg (Demographic Parity)",
        thr.predict(X_te, sensitive_features=s_te, random_state=seed))

    add("Reweighing LogReg (Kamiran-Calders)",
        reweighing_logreg(X_tr, y_tr, s_tr, X_te, seed))

    rf = RandomForestClassifier(n_estimators=300, random_state=seed)
    rf.fit(X_tr, y_tr)
    add("Random Forest (no fairness)", rf.predict(X_te))

    add("XGBoost (no fairness)", xgboost_reference(X_tr, y_tr, X_te, seed))

    return rows


def plot_heatmaps(sweep_df, ds, plots_dir):
    for arch_key, (arch_label, _) in ARCHITECTURES.items():
        arch_df = sweep_df[sweep_df["Arch"] == arch_key]
        for metric in METRICS:
            heatmap_data = arch_df.pivot_table(index="Alpha", columns="Beta", values=metric)
            plt.figure(figsize=(8.5, 6.5))
            sns.heatmap(heatmap_data, annot=True, fmt=".3f",
                        cmap="viridis" if metric == "Accuracy" else "coolwarm",
                        cbar_kws={"label": f"{metric} (test set)"})
            plt.title(f"{metric} — {arch_label} on {ds['label']}\n{LOSS_CAPTION}")
            plt.xlabel("β  (group ↔ individual fairness weight; 1 = all group)")
            plt.ylabel("α  (accuracy/BCE weight; 1 = no fairness term)")
            plt.tight_layout()
            plt.savefig(plots_dir / f"heatmap_{arch_key}_{metric.replace(' ', '_')}.png", dpi=150)
            plt.close()
    print(f"  Heatmaps written to {plots_dir}")


def plot_comparison(comparison_df, ds, plots_dir):
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    axes = axes.ravel()
    colors = plt.cm.tab10(np.linspace(0, 1, len(comparison_df)))

    for ax, metric in zip(axes, METRICS):
        values = comparison_df[metric].values
        ax.barh(comparison_df["Model"], values, color=colors)
        ax.set_title(f"{metric} (test set)", fontsize=12)
        ax.invert_yaxis()
        for j, v in enumerate(values):
            ax.text(v, j, f" {v:.3f}", va="center", fontsize=9)
        if metric != "Accuracy":
            ax.set_xlabel("lower = fairer")
        else:
            ax.set_xlabel("higher = better")
    for ax in axes[len(METRICS):]:
        ax.axis("off")

    fig.suptitle(f"Model comparison — {ds['label']} (test set)\n"
                 f"Fair models trained with {LOSS_CAPTION}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = plots_dir / "model_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Comparison chart written to {out}")


def run_dataset(name, args, seed):
    print(f"\n{'=' * 70}\nDataset: {name} | seed {seed}\n{'=' * 70}")
    ds = load_dataset(name, seed=seed)
    print(f"  {ds['label']} | input_dim={ds['input_dim']} | "
          f"train/val/test = {len(ds['y_train'])}/{len(ds['y_val'])}/{len(ds['y_test'])}")

    model_dir = HERE / "MODELS" / name / f"seed{seed}"
    results_dir = HERE / "RESULTS" / name / f"seed{seed}"
    plots_dir = results_dir / "PLOTS"
    model_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # --- Alpha x Beta sweep per architecture ---------------------------------
    sweep_df = pd.concat([run_sweep(ds, arch, model_dir, args, seed) for arch in ARCHITECTURES],
                         ignore_index=True)
    sweep_csv = results_dir / "sweep_results.csv"
    sweep_df.to_csv(sweep_csv, index=False)
    print(f"  Sweep results written to {sweep_csv}")

    # --- Build the comparison table ------------------------------------------
    comparison_rows = []
    for arch_key, (arch_label, _) in ARCHITECTURES.items():
        arch_df = sweep_df[sweep_df["Arch"] == arch_key]
        baseline = arch_df[(arch_df["Alpha"] == 1) & (arch_df["Beta"] == 0)].iloc[0]
        comparison_rows.append({"Model": f"{arch_label} baseline (BCE only)",
                                "Family": f"{arch_label} baseline",
                                "Seed": seed, "Alpha": 1, "Beta": 0,
                                **{m: baseline[m] for m in METRICS}})

        best = pick_best_fair(arch_df)
        comparison_rows.append({
            "Model": f"Fair {arch_label} (α={best['Alpha']}, β={best['Beta']})",
            "Family": f"Fair {arch_label}",
            "Seed": seed, "Alpha": best["Alpha"], "Beta": best["Beta"],
            **{m: best[m] for m in METRICS}})

    print("  Running baselines (ExpGrad, ThresholdOptimizer, RandomForest)...")
    comparison_rows.extend(run_fairlearn_baselines(ds, seed))

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_csv = results_dir / "model_comparison.csv"
    comparison_df.to_csv(comparison_csv, index=False)
    print(f"  Comparison table written to {comparison_csv}")
    print("\n" + comparison_df.to_string(index=False))

    # --- Plots (ad-hoc single-seed runs only; paper figures come from analysis/) ---
    if len(args.seed_list) == 1:
        plot_heatmaps(sweep_df, ds, plots_dir)
        plot_comparison(comparison_df, ds, plots_dir)


def main():
    parser = argparse.ArgumentParser(description="Fixed fairness-loss experiment + model comparison")
    parser.add_argument("--dataset", choices=["compas", "german", "adult", "both", "all"],
                        default="compas")
    parser.add_argument("--seeds", default="42",
                        help="comma list / ranges, e.g. '0-9' or '0-2,7'")
    parser.add_argument("--retrain", action="store_true",
                        help="retrain even if a checkpoint exists in MODELS/")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=20,
                        help="early-stopping patience on validation loss")
    args = parser.parse_args()
    args.seed_list = parse_seeds(args.seeds)

    dataset_map = {"both": ["compas", "german"],
                   "all": ["compas", "german", "adult"]}
    datasets = dataset_map.get(args.dataset, [args.dataset])
    start = time.perf_counter()
    for name in datasets:
        for seed in args.seed_list:
            run_dataset(name, args, seed)
    print(f"\nTotal time: {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()
