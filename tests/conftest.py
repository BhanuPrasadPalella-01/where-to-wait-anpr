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


EDGE_KEY = "smoke-test-key"


@pytest.fixture(scope="session")
def server_app(grid, tmp_path_factory):
    """The real FastAPI app (server/app.py) on the synthetic grid, an in-memory store and a known edge key.

    server.app builds its control room when it is imported, so the environment is set first; the
    module is reloaded if something imported it already. The clock loop only starts inside a
    `with TestClient(...)` block, so tests that use a plain client step the room themselves.
    """
    import importlib
    import os
    import sys
    path = tmp_path_factory.mktemp("graph") / "grid.json"
    grid.save(path)
    env = {"WTW_GRAPH": str(path), "WTW_DB": ":memory:", "WTW_EDGE_KEY": EDGE_KEY}
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        mod = importlib.reload(sys.modules["server.app"]) if "server.app" in sys.modules \
            else importlib.import_module("server.app")
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return mod


@pytest.fixture(scope="session")
def city_graph():
    # the downloaded OSM map is not in git (python -m planner.roadgraph builds it); skip without it
    if not CITY_GRAPH.exists():
        pytest.skip(f"{CITY_GRAPH.relative_to(CITY_GRAPH.parents[1])} not found; run python -m planner.roadgraph")
    from planner.roadgraph import RoadGraph
    return RoadGraph.load(CITY_GRAPH)
