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


def test_adult_loads_with_sex_sensitive():
    ds = load_dataset("adult", seed=0)
    assert ds["sensitive_col"] == "sex"
    total = len(ds["y_train"]) + len(ds["y_val"]) + len(ds["y_test"])
    assert 40000 < total < 49000          # 48842 rows minus missing-value rows
    assert set(ds["y_test"].unique()) <= {0, 1}
    from data_loading import _load_adult_frame
    X, y, sens = _load_adult_frame()
    assert "fnlwgt" not in X.columns      # sampling weight, standard exclusion


def test_taiwan_loads_with_sex_sensitive():
    ds = load_dataset("taiwan", seed=0)
    assert ds["sensitive_col"] == "SEX"
    total = len(ds["y_train"]) + len(ds["y_val"]) + len(ds["y_test"])
    assert total == 30000
    assert set(ds["sens"]["test"].unique()) == {"male", "female"}
    from data_loading import _load_taiwan_frame
    X, y, sens = _load_taiwan_frame()
    assert abs(y.mean() - 0.2212) < 0.001    # UCI-documented default rate
    assert "default" not in X.columns


def test_acs_employment_loads_with_sex_sensitive():
    ds = load_dataset("acs_employment", seed=0)
    assert ds["sensitive_col"] == "SEX"
    assert len(ds["y_train"]) + len(ds["y_val"]) + len(ds["y_test"]) == 50000
    assert set(ds["sens"]["test"].unique()) == {"male", "female"}


def test_acs_pubcov_restricted_to_white_and_black():
    ds = load_dataset("acs_pubcov", seed=0)
    assert ds["sensitive_col"] == "RAC1P"
    assert len(ds["y_train"]) + len(ds["y_val"]) + len(ds["y_test"]) == 50000
    for split in ("train", "val", "test"):
        assert set(ds["sens"][split].unique()) == {"White", "Black"}


def test_adult_seed7_rare_category_does_not_crash():
    # native-country has a single Holand-Netherlands row; for seed 7 it falls
    # outside the training split and used to crash the OneHotEncoder.
    ds = load_dataset("adult", seed=7)
    assert ds["X_test"].shape[0] == len(ds["y_test"])
