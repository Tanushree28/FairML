#!/usr/bin/env python
"""Adult demographic-parity replication for Agarwal et al. (2018).

All files produced by this script stay under reproduction/agarwal2018/.
The historical Fairlearn algorithm is vendored verbatim under third_party/ and
loaded with compatibility shims for modern NumPy/pandas APIs.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib
import json
import os
import platform
import subprocess
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


warnings.filterwarnings("ignore", category=FutureWarning, module=r"fairlearn\.classred")
warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"sklearn\.utils\.extmath")
np.seterr(all="ignore")


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1]
DATA_PATH = REPO_ROOT / "data" / "adult.csv"
HISTORICAL_COMMIT = "aeaa92d536406b54354a6c2db0d0ac5d14897782"
MODERN_FAIRLEARN_0_11_0_TAG_COMMIT = "ea33211bd9c0c8d2102bdd1e4f5cc17c37a97796"
HISTORICAL_DIR = BASE_DIR / "third_party" / "fairlearn_v0_2_0"
DEFAULT_SEEDS = tuple(range(10))
DEFAULT_EPSILONS = (0.001, 0.0031622776601683794, 0.01, 0.03162277660168379, 0.1)
PREPROCESSINGS = ("preserve_missing", "complete_case")
IMPLEMENTATIONS = ("authors-code", "modern-fairlearn")
MISSING_TOKEN = "__MISSING__"
PROTECTED_STATUS_PATHS = (
    "new_experiment",
    "data",
    "MODELS",
    "NEW_MODEL",
    "NEW_RESULTS",
    "GERMAN_MODEL",
    "GERMAN_RESULTS",
    "PLOTS",
    "results.csv",
    "requirements.txt",
)


@dataclass(frozen=True)
class PreparedSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    a_train: pd.Series
    a_test: pd.Series
    feature_names: list[str]
    raw_feature_columns: list[str]
    categorical_columns: list[str]
    numeric_columns: list[str]
    preprocessing: str
    seed: int


class WeightedLogisticRegression:
    """Pickle-friendly sklearn wrapper with the 2018 learner interface."""

    def __init__(self, random_state: int, max_iter: int = 1000):
        self.random_state = random_state
        self.max_iter = max_iter
        self.model = LogisticRegression(
            solver="liblinear",
            max_iter=max_iter,
            random_state=random_state,
        )

    def fit(self, X, y, sample_weight):
        self.model.fit(X, y, sample_weight=np.asarray(sample_weight))
        return self

    def predict(self, X):
        return self.model.predict(X)


def adult_sha256(path: Path = DATA_PATH) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_output(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, cwd=REPO_ROOT, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:  # pragma: no cover - defensive documentation path
        return f"unavailable: {exc}"


def protected_status() -> str:
    return git_output(["git", "status", "--short", "--", *PROTECTED_STATUS_PATHS])


def read_adult(preprocessing: str) -> pd.DataFrame:
    if preprocessing not in PREPROCESSINGS:
        raise ValueError(f"Unknown preprocessing variant: {preprocessing}")
    df = pd.read_csv(DATA_PATH)
    if preprocessing == "complete_case":
        df = df.dropna(axis=0).reset_index(drop=True)
    else:
        object_cols = df.select_dtypes(include=["object", "category"]).columns
        df[object_cols] = df[object_cols].fillna(MISSING_TOKEN)
    return df


def make_target(df: pd.DataFrame) -> pd.Series:
    y = (df["class"].astype(str) == ">50K").astype(int)
    y.name = "income_gt_50k"
    return y


def make_sensitive(df: pd.DataFrame) -> pd.Series:
    a = df["sex"].astype(str)
    a.name = "sex"
    return a


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    # Paper assumption: use all non-target Adult columns, including sex and fnlwgt.
    return df.drop(columns=["class"]).copy()


def build_preprocessor(X_train_raw: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    categorical_columns = X_train_raw.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_columns = [c for c in X_train_raw.columns if c not in categorical_columns]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_columns),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop=None),
                categorical_columns,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    return preprocessor, numeric_columns, categorical_columns


def prepare_split(preprocessing: str, seed: int) -> PreparedSplit:
    df = read_adult(preprocessing)
    X_raw = feature_frame(df)
    y = make_target(df)
    a = make_sensitive(df)
    X_train_raw, X_test_raw, y_train, y_test, a_train, a_test = train_test_split(
        X_raw,
        y,
        a,
        train_size=0.75,
        test_size=0.25,
        random_state=seed,
        shuffle=True,
        stratify=None,
    )
    preprocessor, numeric_columns, categorical_columns = build_preprocessor(X_train_raw)
    X_train_arr = preprocessor.fit_transform(X_train_raw)
    X_test_arr = preprocessor.transform(X_test_raw)
    if sparse.issparse(X_train_arr):
        X_train_arr = X_train_arr.toarray()
    if sparse.issparse(X_test_arr):
        X_test_arr = X_test_arr.toarray()
    feature_names = preprocessor.get_feature_names_out().tolist()
    X_train = pd.DataFrame(X_train_arr, columns=feature_names, index=X_train_raw.index)
    X_test = pd.DataFrame(X_test_arr, columns=feature_names, index=X_test_raw.index)
    return PreparedSplit(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train.astype(int),
        y_test=y_test.astype(int),
        a_train=a_train.astype(str),
        a_test=a_test.astype(str),
        feature_names=feature_names,
        raw_feature_columns=X_raw.columns.tolist(),
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
        preprocessing=preprocessing,
        seed=seed,
    )


def paper_dp_violation(y_pred_positive: Iterable[float], sensitive: pd.Series) -> float:
    pred = pd.Series(np.asarray(y_pred_positive, dtype=float), index=sensitive.index)
    sensitive = sensitive.astype(str)
    overall = float(pred.mean())
    diffs = pred.groupby(sensitive).mean() - overall
    signed = pd.concat([diffs, -diffs])
    return float(signed.max())


def expected_error(y_true: pd.Series, y_pred_positive: Iterable[float]) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(y_pred_positive, dtype=float), 0.0, 1.0)
    return float(np.mean(y * (1.0 - p) + (1.0 - y) * p))


def hard_predictions(y_pred_positive: Iterable[float]) -> np.ndarray:
    return (np.asarray(y_pred_positive, dtype=float) >= 0.5).astype(int)


def fairlearn_dpd(y_true: pd.Series, y_pred_hard: np.ndarray, sensitive: pd.Series) -> float:
    from fairlearn.metrics import demographic_parity_difference

    return float(
        demographic_parity_difference(
            y_true=np.asarray(y_true),
            y_pred=np.asarray(y_pred_hard),
            sensitive_features=np.asarray(sensitive.astype(str)),
        )
    )


def install_historical_compatibility_shims() -> dict[str, object]:
    import pandas.core.groupby.generic as pd_groupby_generic

    originals: dict[str, object] = {}
    if not hasattr(np, "PINF"):
        np.PINF = np.inf  # type: ignore[attr-defined]
        originals["np.PINF"] = None
    if not hasattr(pd.Index, "contains"):
        originals["pd.Index.contains"] = None
        pd.Index.contains = lambda self, key: key in self  # type: ignore[attr-defined]

    original_series_sum = pd.Series.sum
    originals["pd.Series.sum"] = original_series_sum

    def series_sum_compat(self, axis=None, skipna=True, numeric_only=False, min_count=0, **kwargs):
        level = kwargs.pop("level", None)
        if level is not None:
            return self.groupby(level=level).sum()
        return original_series_sum(
            self,
            axis=axis,
            skipna=skipna,
            numeric_only=numeric_only,
            min_count=min_count,
            **kwargs,
        )

    pd.Series.sum = series_sum_compat  # type: ignore[assignment]

    original_df_groupby_mean = pd_groupby_generic.DataFrameGroupBy.mean
    originals["DataFrameGroupBy.mean"] = original_df_groupby_mean

    def df_groupby_mean_compat(self, numeric_only=True, *args, **kwargs):
        return original_df_groupby_mean(self, numeric_only=numeric_only, *args, **kwargs)

    pd_groupby_generic.DataFrameGroupBy.mean = df_groupby_mean_compat  # type: ignore[assignment]
    return originals


def restore_historical_compatibility_shims(originals: dict[str, object]) -> None:
    import pandas.core.groupby.generic as pd_groupby_generic

    if "pd.Series.sum" in originals:
        pd.Series.sum = originals["pd.Series.sum"]  # type: ignore[assignment]
    if "DataFrameGroupBy.mean" in originals:
        pd_groupby_generic.DataFrameGroupBy.mean = originals["DataFrameGroupBy.mean"]  # type: ignore[assignment]


@contextlib.contextmanager
def historical_fairlearn_modules():
    originals = install_historical_compatibility_shims()
    old_path = list(sys.path)
    old_modules = {k: v for k, v in sys.modules.items() if k == "fairlearn" or k.startswith("fairlearn.")}
    for key in list(old_modules):
        sys.modules.pop(key, None)
    sys.path.insert(0, str(HISTORICAL_DIR))
    try:
        classred = importlib.import_module("fairlearn.classred")
        moments = importlib.import_module("fairlearn.moments")
        yield classred, moments
    finally:
        for key in [k for k in sys.modules if k == "fairlearn" or k.startswith("fairlearn.")]:
            sys.modules.pop(key, None)
        sys.modules.update(old_modules)
        sys.path[:] = old_path
        restore_historical_compatibility_shims(originals)


def verify_historical_synthetic() -> dict[str, object]:
    rng = np.random.default_rng(123)
    n = 240
    a = pd.Series(rng.choice(["Female", "Male"], size=n), name="sex")
    x0 = rng.normal(size=n)
    x1 = (a == "Male").astype(float).to_numpy()
    logits = 1.2 * x0 + 0.8 * x1 + rng.normal(scale=0.5, size=n)
    y = pd.Series((logits > np.median(logits)).astype(int), name="y")
    X = pd.DataFrame({"x0": x0, "sex_is_male": x1})
    with historical_fairlearn_modules() as (classred, moments):
        learner = WeightedLogisticRegression(random_state=123, max_iter=500)
        result = classred.expgrad(X, a, y, learner, cons=moments.DP(), eps=0.05, T=20)
        pred = result.best_classifier(X)
    return {
        "passed": bool(len(result.weights) > 0 and np.all(np.isfinite(pred))),
        "n_oracle_calls": int(result.n_oracle_calls),
        "n_predictors": int((result.weights > 1e-12).sum()),
        "dp_violation": paper_dp_violation(pred, a),
        "error": expected_error(y, pred),
    }


def run_historical(prepared: PreparedSplit, epsilon: float) -> dict[str, object]:
    with historical_fairlearn_modules() as (classred, moments):
        learner = WeightedLogisticRegression(random_state=prepared.seed, max_iter=1000)
        result = classred.expgrad(
            prepared.X_train,
            prepared.a_train,
            prepared.y_train,
            learner,
            cons=moments.DP(),
            eps=epsilon,
            T=50,
        )
        p_test = np.asarray(result.best_classifier(prepared.X_test), dtype=float)
        n_predictors = int((result.weights > 1e-12).sum())
        n_oracle_calls = int(result.n_oracle_calls)
        best_gap = float(result.best_gap)
    hard = hard_predictions(p_test)
    err = expected_error(prepared.y_test, p_test)
    return {
        "implementation": "authors-code",
        "epsilon": epsilon,
        "error": err,
        "accuracy": 1.0 - err,
        "hard_accuracy": float(accuracy_score(prepared.y_test, hard)),
        "paper_dp_violation": paper_dp_violation(p_test, prepared.a_test),
        "fairlearn_dpd": fairlearn_dpd(prepared.y_test, hard, prepared.a_test),
        "n_predictors": n_predictors,
        "n_oracle_calls": n_oracle_calls,
        "best_gap": best_gap,
        "status": "ok",
        "warning": "",
    }


def run_modern(prepared: PreparedSplit, epsilon: float) -> dict[str, object]:
    from fairlearn.reductions import DemographicParity, ExponentiatedGradient

    base = LogisticRegression(solver="liblinear", max_iter=1000, random_state=prepared.seed)
    constraint = DemographicParity(difference_bound=epsilon)
    estimator = ExponentiatedGradient(
        base,
        constraint,
        eps=0.01,
        max_iter=50,
        eta0=2.0,
        run_linprog_step=True,
    )
    estimator.fit(
        prepared.X_train,
        prepared.y_train,
        sensitive_features=prepared.a_train,
    )
    p_test = estimator._pmf_predict(prepared.X_test)[:, 1]
    hard = hard_predictions(p_test)
    err = expected_error(prepared.y_test, p_test)
    weights = getattr(estimator, "weights_", pd.Series(dtype=float))
    return {
        "implementation": "modern-fairlearn",
        "epsilon": epsilon,
        "error": err,
        "accuracy": 1.0 - err,
        "hard_accuracy": float(accuracy_score(prepared.y_test, hard)),
        "paper_dp_violation": paper_dp_violation(p_test, prepared.a_test),
        "fairlearn_dpd": fairlearn_dpd(prepared.y_test, hard, prepared.a_test),
        "n_predictors": int((weights > 1e-12).sum()) if len(weights) else 0,
        "n_oracle_calls": int(getattr(estimator, "n_oracle_calls_", -1)),
        "best_gap": float(getattr(estimator, "best_gap_", np.nan)),
        "status": "ok",
        "warning": "",
    }


def base_row(prepared: PreparedSplit, epsilon: float) -> dict[str, object]:
    return {
        "preprocessing": prepared.preprocessing,
        "seed": prepared.seed,
        "epsilon": epsilon,
        "n_train": int(len(prepared.y_train)),
        "n_test": int(len(prepared.y_test)),
        "n_features": int(prepared.X_train.shape[1]),
        "adult_rows": int(len(prepared.y_train) + len(prepared.y_test)),
        "protected_attribute_in_X": "sex" in prepared.raw_feature_columns,
        "estimator": "LogisticRegression(solver='liblinear', max_iter=1000)",
        "constraint": "DemographicParity",
        "randomized_prediction_metric": "expected_error_from_positive_probability",
    }


def run_one(implementation: str, prepared: PreparedSplit, epsilon: float) -> dict[str, object]:
    row = base_row(prepared, epsilon)
    try:
        if implementation == "authors-code":
            row.update(run_historical(prepared, epsilon))
        elif implementation == "modern-fairlearn":
            row.update(run_modern(prepared, epsilon))
        else:
            raise ValueError(f"Unknown implementation: {implementation}")
    except Exception as exc:
        row.update(
            {
                "implementation": implementation,
                "error": np.nan,
                "accuracy": np.nan,
                "hard_accuracy": np.nan,
                "paper_dp_violation": np.nan,
                "fairlearn_dpd": np.nan,
                "n_predictors": 0,
                "n_oracle_calls": -1,
                "best_gap": np.nan,
                "status": "failed",
                "warning": repr(exc),
            }
        )
    return row


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    ok = runs[runs["status"] == "ok"].copy()
    group_cols = ["implementation", "preprocessing", "epsilon"]
    metric_cols = ["error", "accuracy", "hard_accuracy", "paper_dp_violation", "fairlearn_dpd", "n_predictors", "n_oracle_calls"]
    summary = ok.groupby(group_cols)[metric_cols].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join([str(c) for c in col if c != ""]) for col in summary.columns.to_flat_index()]
    return summary


def validate_pre_run(epsilons: tuple[float, ...], seeds: tuple[int, ...]) -> list[str]:
    failures: list[str] = []
    preserve = read_adult("preserve_missing")
    complete = read_adult("complete_case")
    if len(preserve) != 48842:
        failures.append(f"preserve_missing row count {len(preserve)} != 48842")
    if len(complete) != 45222:
        failures.append(f"complete_case row count {len(complete)} != 45222")
    for name, df in [("preserve_missing", preserve), ("complete_case", complete)]:
        y = make_target(df)
        a = make_sensitive(df)
        if set(y.unique()) != {0, 1}:
            failures.append(f"{name}: target is not binary 0/1")
        if set(a.unique()) != {"Female", "Male"}:
            failures.append(f"{name}: sensitive groups are {sorted(a.unique())}")
        if "sex" not in feature_frame(df).columns:
            failures.append(f"{name}: sex missing from X")
        split = prepare_split(name, seeds[0])
        if len(split.y_train) + len(split.y_test) != len(df):
            failures.append(f"{name}: split loses rows")
        train_fraction = len(split.y_train) / len(df)
        if abs(train_fraction - 0.75) > 1e-4:
            failures.append(f"{name}: train fraction {train_fraction:.6f} != 0.75")
        if not any(col.startswith("cat__sex_") for col in split.feature_names):
            failures.append(f"{name}: encoded sex columns missing")
        train_indices = set(split.X_train.index)
        test_indices = set(split.X_test.index)
        if train_indices & test_indices:
            failures.append(f"{name}: train/test leakage in indices")
    if min(epsilons) > 0.001 or max(epsilons) < 0.1:
        failures.append("epsilon grid does not cover 0.001 through 0.1")
    return failures


def validate_results(runs: pd.DataFrame, epsilons: tuple[float, ...], seeds: tuple[int, ...]) -> list[str]:
    failures = []
    expected = len(IMPLEMENTATIONS) * len(PREPROCESSINGS) * len(seeds) * len(epsilons)
    if len(runs) != expected:
        failures.append(f"runs.csv has {len(runs)} rows, expected {expected}")
    required = {
        "implementation",
        "preprocessing",
        "seed",
        "epsilon",
        "error",
        "accuracy",
        "paper_dp_violation",
        "fairlearn_dpd",
        "n_train",
        "n_test",
        "n_predictors",
    }
    missing = required - set(runs.columns)
    if missing:
        failures.append(f"runs.csv missing columns: {sorted(missing)}")
    ok = runs[runs["status"] == "ok"]
    if len(ok) != len(runs):
        failures.append(f"{len(runs) - len(ok)} runs failed")
    for impl in IMPLEMENTATIONS:
        for prep in PREPROCESSINGS:
            subset = runs[(runs["implementation"] == impl) & (runs["preprocessing"] == prep)]
            if set(np.round(subset["epsilon"].astype(float), 12)) != set(np.round(epsilons, 12)):
                failures.append(f"{impl}/{prep}: epsilon grid mismatch")
            if set(subset["seed"].astype(int)) != set(seeds):
                failures.append(f"{impl}/{prep}: seed set mismatch")
            if subset["paper_dp_violation"].nunique(dropna=True) <= 1:
                failures.append(f"{impl}/{prep}: epsilon did not change paper_dp_violation")
    return failures


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to write")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_figure(summary: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6))
    markers = {"authors-code": "o", "modern-fairlearn": "s"}
    colors = {"preserve_missing": "#1f77b4", "complete_case": "#d62728"}
    for (impl, prep), df in summary.groupby(["implementation", "preprocessing"]):
        df = df.sort_values("epsilon")
        ax.errorbar(
            df["paper_dp_violation_mean"],
            df["error_mean"],
            xerr=df["paper_dp_violation_std"].fillna(0),
            yerr=df["error_std"].fillna(0),
            marker=markers[impl],
            color=colors[prep],
            linestyle="-",
            capsize=3,
            label=f"{impl}, {prep}",
        )
        for _, row in df.iterrows():
            ax.annotate(f"{row['epsilon']:.3g}", (row["paper_dp_violation_mean"], row["error_mean"]), fontsize=7)
    ax.set_xlabel("Paper-style demographic-parity violation")
    ax.set_ylabel("Test classification error")
    ax.set_title("Agarwal et al. (2018) Adult DP LR replication")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def dependency_versions() -> dict[str, str]:
    modules = ["numpy", "pandas", "scipy", "sklearn", "fairlearn", "matplotlib"]
    versions = {"python": platform.python_version()}
    for module_name in modules:
        try:
            module = importlib.import_module(module_name)
            versions[module_name] = getattr(module, "__version__", "unknown")
        except Exception as exc:
            versions[module_name] = f"unavailable: {exc}"
    return versions


def write_docs(
    runs: pd.DataFrame,
    summary: pd.DataFrame,
    synthetic: dict[str, object],
    pre_failures: list[str],
    result_failures: list[str],
    before_status: str,
    after_status: str,
    epsilons: tuple[float, ...],
    seeds: tuple[int, ...],
) -> None:
    versions = dependency_versions()
    env_lines = [
        "# Environment",
        "",
        f"- Git commit: `{git_output(['git', 'rev-parse', 'HEAD'])}`",
        f"- Git dirty status before protected-path run: `{before_status or 'clean for tracked protected paths'}`",
        f"- Git dirty status after protected-path run: `{after_status or 'clean for tracked protected paths'}`",
        f"- Historical Fairlearn commit: `{HISTORICAL_COMMIT}`",
        f"- Modern Fairlearn package: `fairlearn==0.11.0`; upstream tag target recorded during planning: `{MODERN_FAIRLEARN_0_11_0_TAG_COMMIT}`",
        f"- Historical code path: `{HISTORICAL_DIR.relative_to(REPO_ROOT)}`",
        f"- Adult data path: `{DATA_PATH.relative_to(REPO_ROOT)}`",
        f"- Adult data SHA256: `{adult_sha256()}`",
        "",
        "## Actual run environment",
        "",
    ]
    env_lines.extend([f"- {k}: `{v}`" for k, v in versions.items()])
    env_lines.extend(
        [
            "",
            "## Historical compatibility finding",
            "",
            "The public Fairlearn v0.2.0 setup.py lists unversioned `numpy`, `scipy`, and `pandas` dependencies and no scikit-learn dependency, and the paper does not publish a lockfile. The current host uses Python 3.13.9, where historically plausible 2018 packages such as pandas 0.23.x and scikit-learn 0.19.x are not installable. The historical algorithm was therefore run from the pinned source with compatibility shims outside the algorithm for removed NumPy/pandas APIs (`np.PINF`, `Index.contains`, `Series.sum(level=...)`, and old `DataFrameGroupBy.mean` behavior).",
            "",
            "See `requirements-historical-approx.txt` for a documented approximate 2018-era environment. It is not claimed as an established authors' environment.",
        ]
    )
    (BASE_DIR / "ENVIRONMENT.md").write_text("\n".join(env_lines) + "\n")

    assumptions = [
        "# Assumptions",
        "",
        "- The paper states Adult has 48,842 examples but does not specify missing-value handling. This replication runs both `preserve_missing`, which fills missing categorical values with `__MISSING__` and retains 48,842 rows, and `complete_case`, which drops rows with missing values and retains 45,222 rows.",
        "- The paper does not specify the Adult categorical encoding. This replication uses train-fitted one-hot encoding with `handle_unknown='ignore'` and no dropped category.",
        "- The paper does not specify numeric scaling. This replication uses train-fitted `StandardScaler` for numeric columns.",
        "- The paper says the protected attribute is included in X. This replication includes `sex` and all other non-target Adult columns, including `fnlwgt`, `education`, and `education-num`.",
        "- The original train/test random seed is unavailable. This replication runs seeds 0 through 9.",
        f"- The paper says epsilon in `{{0.001, ..., 0.1}}` but does not specify intermediate spacing. This replication uses the documented log-spaced grid: `{list(epsilons)}`.",
        "- Test error is computed as randomized expected classification error from the positive-class probability/mixed classifier, with `accuracy = 1 - error`. `hard_accuracy` is also reported for thresholded predictions.",
        "- `paper_dp_violation` is the maximum signed Adult demographic-parity moment violation, `max(max_a E[h|A=a]-E[h], max_a E[h]-E[h|A=a])`, computed on positive-class probabilities. `fairlearn_dpd` is Fairlearn's `demographic_parity_difference` on thresholded hard predictions.",
        "- Modern Fairlearn is a reimplementation, not exact authors' code. It uses `DemographicParity(difference_bound=epsilon)` and `ExponentiatedGradient(..., eps=0.01, max_iter=50)`.",
    ]
    (BASE_DIR / "ASSUMPTIONS.md").write_text("\n".join(assumptions) + "\n")

    compact = summary[
        [
            "implementation",
            "preprocessing",
            "epsilon",
            "error_mean",
            "error_std",
            "paper_dp_violation_mean",
            "paper_dp_violation_std",
            "fairlearn_dpd_mean",
            "n_predictors_mean",
        ]
    ].copy()
    table = compact.to_markdown(index=False, floatfmt=".6f")
    failed = runs[runs["status"] != "ok"]
    results_lines = [
        "# Results",
        "",
        "## What Was Reproduced Directly From Historical Public Code",
        "",
        f"- The historical `fairlearn.classred.expgrad` algorithm and `moments.DP` constraint were run from Fairlearn v0.2.0 commit `{HISTORICAL_COMMIT}`.",
        f"- Synthetic historical ExpGrad verification: `{json.dumps(synthetic, sort_keys=True)}`.",
        "",
        "## What Required Assumptions",
        "",
        "- Adult preprocessing, encoding, scaling, split seed, logistic-regression solver, and exact epsilon-grid spacing are not available in the paper/public code and are documented in `ASSUMPTIONS.md`.",
        "",
        "## Historical Implementation Results",
        "",
        "Historical rows are labeled `authors-code` in `results/runs.csv` and `results/summary.csv`.",
        "",
        "## Modern Fairlearn Reimplementation Results",
        "",
        "Modern rows are labeled `modern-fairlearn`. They are not exact reproductions of the authors' experiment.",
        "",
        "## Key Result Table",
        "",
        table,
        "",
        "## Comparison With Agarwal et al. (2018)",
        "",
        "The paper reports Adult results graphically as error-vs-constraint-violation frontiers, not exact numeric coordinates. This replication therefore compares qualitatively: lower epsilon values generally trace lower demographic-parity violation at some error cost, but no claim of numerical reproduction is made.",
        "",
        "## Differences From Existing FairML ExpGrad-DP Experiment",
        "",
        "- Existing `new_experiment/` uses a 60/20/20 split and validation workflow; this replication uses paper-style 75/25 train/test.",
        "- Existing `new_experiment/` drops Adult missing rows; this replication runs both full-row missing-as-category and complete-case variants.",
        "- Existing FairML reports accuracy and several additional fairness metrics; this replication emphasizes test error and paper-style DP violation.",
        "- Existing FairML uses modern Fairlearn as a baseline row; this replication separates pinned historical authors-code from modern Fairlearn reimplementation.",
        "",
        "## Validation",
        "",
        f"- Pre-run sanity failures: `{pre_failures}`",
        f"- Result sanity failures: `{result_failures}`",
        f"- Failed experiment rows: `{len(failed)}`",
    ]
    if len(failed):
        results_lines.extend(["", failed[["implementation", "preprocessing", "seed", "epsilon", "warning"]].to_markdown(index=False)])
    (BASE_DIR / "RESULTS.md").write_text("\n".join(results_lines) + "\n")

    readme = [
        "# Agarwal et al. (2018) Adult DP LR Replication",
        "",
        "This directory contains an isolated replication/reimplementation study for the Adult + demographic-parity + logistic-regression experiment associated with Agarwal et al. (2018).",
        "",
        "## Rerun From Scratch",
        "",
        "From the repository root:",
        "",
        "```bash",
        ".venv/bin/python -m pytest reproduction/agarwal2018/tests",
        "MPLCONFIGDIR=/private/tmp .venv/bin/python reproduction/agarwal2018/run_replication.py",
        ".venv/bin/python -m pytest reproduction/agarwal2018/tests",
        "```",
        "",
        "Outputs:",
        "",
        "- `results/runs.csv`",
        "- `results/summary.csv`",
        "- `figures/error_vs_dp_violation.png`",
        "- `ASSUMPTIONS.md`",
        "- `ENVIRONMENT.md`",
        "- `RESULTS.md`",
        "",
        "The script writes only under `reproduction/agarwal2018/`.",
    ]
    (BASE_DIR / "README.md").write_text("\n".join(readme) + "\n")


def parse_float_list(value: str) -> tuple[float, ...]:
    return tuple(float(x.strip()) for x in value.split(",") if x.strip())


def parse_int_list(value: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in value.split(",") if x.strip())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    parser.add_argument("--epsilons", default=",".join(str(e) for e in DEFAULT_EPSILONS))
    parser.add_argument("--implementations", default=",".join(IMPLEMENTATIONS))
    parser.add_argument("--preprocessings", default=",".join(PREPROCESSINGS))
    args = parser.parse_args()

    seeds = parse_int_list(args.seeds)
    epsilons = parse_float_list(args.epsilons)
    implementations = tuple(x.strip() for x in args.implementations.split(",") if x.strip())
    preprocessings = tuple(x.strip() for x in args.preprocessings.split(",") if x.strip())

    before_status = protected_status()
    (BASE_DIR / "results" / "protected_status_before.txt").write_text(before_status + "\n")

    pre_failures = validate_pre_run(epsilons, seeds)
    synthetic = verify_historical_synthetic()
    (BASE_DIR / "results" / "synthetic_check.json").write_text(json.dumps(synthetic, indent=2, sort_keys=True) + "\n")

    rows: list[dict[str, object]] = []
    for preprocessing in preprocessings:
        for seed in seeds:
            prepared = prepare_split(preprocessing, seed)
            for epsilon in epsilons:
                for implementation in implementations:
                    row = run_one(implementation, prepared, epsilon)
                    rows.append(row)
                    print(
                        f"{implementation} {preprocessing} seed={seed} eps={epsilon:g} "
                        f"status={row['status']} error={row['error']} dp={row['paper_dp_violation']}",
                        flush=True,
                    )

    runs_path = BASE_DIR / "results" / "runs.csv"
    write_csv(runs_path, rows)
    runs = pd.read_csv(runs_path)
    summary = summarize(runs)
    summary_path = BASE_DIR / "results" / "summary.csv"
    summary.to_csv(summary_path, index=False)
    write_figure(summary, BASE_DIR / "figures" / "error_vs_dp_violation.png")

    result_failures = validate_results(runs, epsilons, seeds)
    after_status = protected_status()
    (BASE_DIR / "results" / "protected_status_after.txt").write_text(after_status + "\n")
    write_docs(runs, summary, synthetic, pre_failures, result_failures, before_status, after_status, epsilons, seeds)

    if pre_failures or result_failures or not synthetic.get("passed", False):
        print("Validation warnings/failures detected. See RESULTS.md.", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
