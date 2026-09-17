import numpy as np
import pytest
from baselines import kamiran_calders_weights


def test_weights_equalize_weighted_base_rates():
    # group a: 3/4 positive, group b: 1/4 positive, overall 1/2.
    y = np.array([1, 1, 1, 0, 1, 0, 0, 0])
    s = np.array(["a", "a", "a", "a", "b", "b", "b", "b"])
    w = kamiran_calders_weights(y, s)
    for g in ("a", "b"):
        m = s == g
        weighted_rate = (w[m] * y[m]).sum() / w[m].sum()
        assert abs(weighted_rate - 0.5) < 1e-9


def test_weights_are_uniform_when_independent():
    y = np.array([1, 0, 1, 0])
    s = np.array(["a", "a", "b", "b"])
    w = kamiran_calders_weights(y, s)
    assert np.allclose(w, 1.0)


def test_xgboost_reference_predicts_binary():
    import numpy as np
    from baselines import xgboost_reference
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5))
    y = (X[:, 0] > 0).astype(int)
    pred = xgboost_reference(X[:150], y[:150], X[150:], seed=0)
    assert set(np.unique(pred)) <= {0, 1}
    assert (pred == y[150:]).mean() > 0.8


def test_matches_aif360_reference_implementation():
    """Our four-line Reweighing must agree with AIF360's reference version.

    AIF360's Reweighing only accepts a binary privileged/unprivileged
    partition, so this checks the binary-sensitive-attribute case; our
    version additionally handles the >2-group case (COMPAS race) natively.
    """
    aif360_datasets = pytest.importorskip("aif360.datasets")
    aif360_pre = pytest.importorskip("aif360.algorithms.preprocessing")
    import pandas as pd

    rng = np.random.default_rng(0)
    s = rng.integers(0, 2, size=2000).astype(float)
    # label correlated with the sensitive attribute, so weights are non-trivial
    y = (rng.random(2000) < np.where(s == 1, 0.7, 0.3)).astype(float)

    ours = kamiran_calders_weights(y, s)

    bld = aif360_datasets.BinaryLabelDataset(
        df=pd.DataFrame({"sens": s, "label": y}),
        label_names=["label"], protected_attribute_names=["sens"])
    theirs = aif360_pre.Reweighing(
        unprivileged_groups=[{"sens": 0.0}],
        privileged_groups=[{"sens": 1.0}]).fit_transform(bld).instance_weights

    assert not np.allclose(ours, 1.0)          # guard: the test data is skewed
    assert np.allclose(ours, theirs)
