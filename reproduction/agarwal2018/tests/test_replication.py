from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import run_replication as rr


def test_adult_row_counts_and_labels():
    preserve = rr.read_adult("preserve_missing")
    complete = rr.read_adult("complete_case")
    assert len(preserve) == 48842
    assert len(complete) == 45222
    for df in [preserve, complete]:
        assert set(rr.make_target(df).unique()) == {0, 1}
        assert set(rr.make_sensitive(df).unique()) == {"Female", "Male"}
        assert "sex" in rr.feature_frame(df).columns


def test_split_preprocessing_and_no_leakage():
    split = rr.prepare_split("preserve_missing", seed=0)
    assert len(split.y_train) == 36631
    assert len(split.y_test) == 12211
    assert set(split.X_train.index).isdisjoint(set(split.X_test.index))
    assert any(name.startswith("cat__sex_") for name in split.feature_names)
    assert "sex" in split.raw_feature_columns
    assert np.isfinite(split.X_train.to_numpy()).all()
    assert np.isfinite(split.X_test.to_numpy()).all()


def test_complete_case_split_fraction():
    split = rr.prepare_split("complete_case", seed=0)
    assert len(split.y_train) == 33916
    assert len(split.y_test) == 11306
    assert abs(len(split.y_train) / (len(split.y_train) + len(split.y_test)) - 0.75) < 1e-4


def test_historical_synthetic_expgrad_runs():
    result = rr.verify_historical_synthetic()
    assert result["passed"] is True
    assert result["n_oracle_calls"] > 0
    assert result["n_predictors"] > 0
    assert 0 <= result["error"] <= 1
    assert result["dp_violation"] >= 0


def test_result_files_when_present():
    runs_path = BASE_DIR / "results" / "runs.csv"
    if not runs_path.exists():
        return
    runs = pd.read_csv(runs_path)
    failures = rr.validate_results(runs, rr.DEFAULT_EPSILONS, rr.DEFAULT_SEEDS)
    assert failures == []
    assert (BASE_DIR / "results" / "summary.csv").exists()
    assert (BASE_DIR / "figures" / "error_vs_dp_violation.png").exists()
