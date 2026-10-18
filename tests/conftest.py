"""Shared pytest fixtures."""
import pytest

from tests.synthetic import grid_graph


@pytest.fixture(scope="session")
def grid():
    return grid_graph()


@pytest.fixture(scope="session")
def grid_world(grid):
    # building a World samples a few hundred routes; share one across the test session
    from planner.simulate import World
    return World(g=grid)
