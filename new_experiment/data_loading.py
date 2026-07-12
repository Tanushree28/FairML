"""Dataset loading for the new experiment.

Loads COMPAS and German Credit datasets with 60/20/20 train/val/test split
(seed-parameterized; default 42) and StandardScaler + OneHotEncoder(drop='first')
preprocessing. COMPAS now uses ProPublica's standard row filters and feature set
(excluding the label-leaking duration feature), so it no longer mirrors the
original Compas.py exactly.

The COMPAS CSV is cached to data/compas-scores-two-years.csv after the first
download so re-runs work offline.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent

COMPAS_URL = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
COMPAS_CACHE = ROOT / "data" / "compas-scores-two-years.csv"


def _load_compas_frame():
    if COMPAS_CACHE.exists():
        data = pd.read_csv(COMPAS_CACHE)
    else:
        data = pd.read_csv(COMPAS_URL)
        COMPAS_CACHE.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(COMPAS_CACHE, index=False)

    # ProPublica's standard row filters (github.com/propublica/compas-analysis).
    # NOTE: the previous version added duration = end - start, which correlates
    # -0.78 with the label for mechanical reasons (re-offending truncates the
    # observation window) — label leakage, removed for the paper.
    data = data[
        (data["days_b_screening_arrest"] <= 30)
        & (data["days_b_screening_arrest"] >= -30)
        & (data["is_recid"] != -1)
        & (data["c_charge_degree"] != "O")
        & (data["score_text"] != "N/A")
    ]

    features = ["age", "sex", "juv_fel_count", "juv_misd_count",
                "juv_other_count", "priors_count", "c_charge_degree", "race"]
    return data[features], data["two_year_recid"], "race"


def _load_german_frame():
    data = pd.read_csv(ROOT / "data" / "german_credit_data.csv", index_col=0)

    statlog_cols = [
        "existing_checking", "duration", "credit_history", "purpose", "credit_amount",
        "savings", "employment", "installment_rate", "personal_status_sex", "other_debtors",
        "residence_since", "property", "age", "other_installment", "housing",
        "existing_credits", "job", "num_dependents", "telephone", "foreign_worker", "target",
    ]
    statlog = pd.read_csv(ROOT / "statlog+german+credit+data" / "german.data",
                          sep=" ", header=None, names=statlog_cols)
    # Statlog target: 1 = good credit, 2 = bad. Encode 1 = bad credit risk so the
    # positive class is the adverse outcome, mirroring COMPAS.
    data["risk"] = (statlog["target"].values == 2).astype(int)
    data = data.fillna("unknown")

    features = ["Age", "Sex", "Job", "Housing", "Saving accounts",
                "Checking account", "Credit amount", "Duration", "Purpose"]
    return data[features], data["risk"], "Sex"


def load_dataset(name, seed=42):
    """Returns a dict with torch tensors, numpy arrays, sensitive-feature
    series, and integer group ids for train/val/test. `seed` controls the
    train/val/test split."""
    if name == "compas":
        X, y, sensitive_col = _load_compas_frame()
        label = "COMPAS (sensitive attribute: race)"
    elif name == "german":
        X, y, sensitive_col = _load_german_frame()
        label = "German Credit (sensitive attribute: sex)"
    else:
        raise ValueError(f"Unknown dataset: {name}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=seed)

    sens = {
        "train": X_train[sensitive_col].reset_index(drop=True),
        "val": X_val[sensitive_col].reset_index(drop=True),
        "test": X_test[sensitive_col].reset_index(drop=True),
    }
    # Integer codes for the torch loss; one shared category mapping across splits.
    categories = pd.Categorical(X[sensitive_col]).categories
    group_ids = {
        split: torch.tensor(pd.Categorical(s, categories=categories).codes, dtype=torch.long)
        for split, s in sens.items()
    }

    numeric = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(drop="first"), categorical),
    ])

    def densify(M):
        return np.asarray(M.todense()) if hasattr(M, "todense") else np.asarray(M)

    X_train_np = densify(preprocessor.fit_transform(X_train))
    X_val_np = densify(preprocessor.transform(X_val))
    X_test_np = densify(preprocessor.transform(X_test))

    def to_x(M):
        return torch.tensor(M, dtype=torch.float32)

    def to_y(s):
        return torch.tensor(s.values, dtype=torch.float32).view(-1, 1)

    return {
        "name": name,
        "label": label,
        "sensitive_col": sensitive_col,
        "input_dim": X_train_np.shape[1],
        "X_train": X_train_np, "X_val": X_val_np, "X_test": X_test_np,
        "y_train": y_train.reset_index(drop=True),
        "y_val": y_val.reset_index(drop=True),
        "y_test": y_test.reset_index(drop=True),
        "X_train_t": to_x(X_train_np), "X_val_t": to_x(X_val_np), "X_test_t": to_x(X_test_np),
        "y_train_t": to_y(y_train), "y_val_t": to_y(y_val), "y_test_t": to_y(y_test),
        "sens": sens,
        "group_ids": group_ids,
    }
