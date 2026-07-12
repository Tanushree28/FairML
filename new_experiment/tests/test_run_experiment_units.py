from run_experiment import parse_seeds


def test_parse_seeds_single():
    assert parse_seeds("42") == [42]


def test_parse_seeds_range_and_list():
    assert parse_seeds("0-2,7") == [0, 1, 2, 7]
    assert parse_seeds("0-9") == list(range(10))
