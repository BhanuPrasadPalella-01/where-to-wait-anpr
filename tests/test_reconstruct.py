"""Route reconstruction between sightings on the synthetic grid (side streets 30 s, main roads 20 s)."""
import pytest

from planner.reconstruct import reconstruct
from tests.synthetic import node

COLS = 9


def test_same_junction_or_no_time_gives_nothing(grid):
    a = node(4, 2, COLS)
    assert reconstruct(grid, a, a, 60) == []
    assert reconstruct(grid, a, node(4, 4, COLS), 0) == []


def test_impossible_gap_gives_no_path(grid):
    # two main-road blocks = 40 s; even 30% speeding needs >= 28 s
    assert reconstruct(grid, node(4, 2, COLS), node(4, 4, COLS), 20) == []


def test_main_road_path_wins_when_time_matches(grid):
    a, b = node(4, 2, COLS), node(4, 4, COLS)
    paths = reconstruct(grid, a, b, 40)
    assert paths[0]["rank"] == 1 and paths[0]["travel_s"] == pytest.approx(40)
    assert paths[0]["nodes"] == [a, node(4, 3, COLS), b]
    assert sum(p["share"] for p in paths) == pytest.approx(1.0)


def test_slower_observed_gap_prefers_a_longer_path(grid):
    a, b = node(4, 2, COLS), node(4, 4, COLS)
    quick = reconstruct(grid, a, b, 40)[0]["travel_s"]
    slow = reconstruct(grid, a, b, 100)[0]["travel_s"]
    assert slow > quick  # a 100 s gap fits a detour better than the 40 s main road


def test_paths_are_ranked_by_share(grid):
    paths = reconstruct(grid, node(0, 0, COLS), node(2, 2, COLS), 120, k=3)
    shares = [p["share"] for p in paths]
    assert shares == sorted(shares, reverse=True)
    assert [p["rank"] for p in paths] == list(range(1, len(paths) + 1))
