from data_loading import load_dataset


def test_same_seed_is_reproducible():
    a = load_dataset("german", seed=0)
    b = load_dataset("german", seed=0)
    assert a["y_test"].equals(b["y_test"])
    assert (a["X_test"] == b["X_test"]).all()


def test_different_seeds_give_different_splits():
    a = load_dataset("german", seed=0)
    b = load_dataset("german", seed=1)
    assert not a["y_test"].equals(b["y_test"])


def test_default_seed_matches_seed_42():
    a = load_dataset("german")
    b = load_dataset("german", seed=42)
    assert a["y_test"].equals(b["y_test"])


def test_compas_has_no_duration_feature():
    ds = load_dataset("compas", seed=0)
    # duration = end - start leaks the label (corr -0.78, mechanical); the
    # paper uses ProPublica's standard feature set instead.
    total = len(ds["y_train"]) + len(ds["y_val"]) + len(ds["y_test"])
    assert 6000 <= total <= 6300          # ProPublica filters => ~6172 rows
    from data_loading import _load_compas_frame
    X, y, sens = _load_compas_frame()
    assert "duration" not in X.columns
    assert sens == "race"
