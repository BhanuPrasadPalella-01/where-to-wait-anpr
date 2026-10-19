"""Shared pytest fixtures."""
from pathlib import Path

import pytest

from tests.synthetic import grid_graph

CITY_GRAPH = Path(__file__).resolve().parents[1] / "data" / "city_graph.json"


@pytest.fixture(scope="session")
def grid():
    return grid_graph()


@pytest.fixture(scope="session")
def grid_world(grid):
    # building a World samples a few hundred routes; share one across the test session
    from planner.simulate import World
    return World(g=grid)


@pytest.fixture(scope="session")
def city_graph():
    # the downloaded OSM map is not in git (python -m planner.roadgraph builds it); skip without it
    if not CITY_GRAPH.exists():
        pytest.skip(f"{CITY_GRAPH.relative_to(CITY_GRAPH.parents[1])} not found; run python -m planner.roadgraph")
    from planner.roadgraph import RoadGraph
    return RoadGraph.load(CITY_GRAPH)
