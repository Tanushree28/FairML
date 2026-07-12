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
