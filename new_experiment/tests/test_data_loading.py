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
