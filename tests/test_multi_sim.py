"""Two-suspect simulator on the synthetic grid (no city map), and its results table."""
import pytest

from planner.multi_sim import METHODS, run_multi_episode
from scripts.multi_suspect_table import mcnemar, summarise


@pytest.fixture(scope="module")
def episodes(grid_world):
    return [run_multi_episode(seed, 0.3, m, k=3, W=grid_world) for seed in range(4) for m in METHODS]


def test_episode_fields_and_score(episodes):
    for r in episodes:
        assert r["severe"] in (0, 1) and len(r["caught"]) == 2
        assert r["caught_severe"] == r["caught"][r["severe"]]
        expected = (3.0 * r["caught_severe"] + 1.0 * r["caught_minor"]) / 4.0
        assert r["score"] == pytest.approx(expected)


def test_same_scenario_for_every_method(episodes):
    by_seed = {}
    for r in episodes:
        by_seed.setdefault(r["seed"], set()).add(r["severe"])
    assert all(len(s) == 1 for s in by_seed.values())


def test_locked_methods_never_pull_a_unit_between_cases(episodes):
    assert all(r["pulls"] == 0 for r in episodes if r["method"] != "no_lock")


def test_table_lists_every_method_and_pairs(episodes):
    md = summarise(episodes)
    assert sum(line.startswith("| ") for line in md.splitlines()) == 1 + len(METHODS)  # header + one row each
    assert "4 paired scenarios per method" in md
    assert "vs Units split per case" in md


def test_mcnemar():
    assert mcnemar(0, 0) == 1.0
    assert mcnemar(5, 5) == 1.0
    assert mcnemar(10, 0) == pytest.approx(2 / 2 ** 10)
