import numpy as np
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
