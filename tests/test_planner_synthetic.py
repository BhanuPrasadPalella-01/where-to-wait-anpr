"""Greedy vs exact planning on the synthetic grid: always runs, no city map needed."""
import random

import pytest

from planner.belief import ParticleBelief
from planner.fugitive import RouteSampler, pick_exits
from planner.plan import coverage_sets, exact_plan, greedy_plan, objective, unit_etas
from tests.synthetic import node

COLS = 9


@pytest.fixture(scope="module")
def sampler(grid):
    return RouteSampler(grid, pick_exits(grid, n=8))


def case(grid, sampler, seed, n_units=3):
    """Belief seeded at a random junction, units at random junctions -> (cov, weights, units)."""
    rng = random.Random(seed)
    b = ParticleBelief(sampler, n=120, horizon_s=600, seed=seed)
    b.reseed(rng.randrange(grid.n), 0.0)
    units = {f"u{i}": rng.randrange(grid.n) for i in range(n_units)}
    etas = unit_etas(grid, units, 0.0, delta=10)
    return coverage_sets(b.future(0.0), b.w, etas, margin=10), b.w, units


def test_exits_lie_on_the_grid_boundary(grid, sampler):
    edge = {0, COLS - 1}
    assert sampler.exits and all(v // COLS in edge or v % COLS in edge for v in sampler.exits)


def test_greedy_is_within_the_half_bound_of_exact(grid, sampler):
    ratios = []
    for seed in range(12):
        cov, w, units = case(grid, sampler, seed)
        if not cov:
            continue
        gv = objective(greedy_plan(cov, w, units), cov, w)
        _, ev = exact_plan(cov, w, units, max_nodes=15)
        ev = max(ev, gv)  # exact only searches the top candidate nodes
        if ev > 0:
            ratios.append(gv / ev)
    assert len(ratios) >= 8
    assert min(ratios) >= 0.5          # Fisher-Nemhauser-Wolsey bound for a partition matroid
    assert sum(ratios) / len(ratios) >= 0.9


def test_greedy_plan_respects_the_constraints(grid, sampler):
    cov, w, units = case(grid, sampler, seed=4)
    plan = greedy_plan(cov, w, units)
    assert plan and all(a in cov for a in plan)
    assert len({u for u, _ in plan}) == len(plan) <= len(units)   # each unit at most one junction
    assert len({v for _, v in plan}) == len(plan)                 # each junction at most one unit


def test_forbidden_junctions_are_never_chosen(grid, sampler):
    cov, w, units = case(grid, sampler, seed=5)
    best = {v for _, v in greedy_plan(cov, w, units)}
    rng = random.Random(5)
    b = ParticleBelief(sampler, n=120, horizon_s=600, seed=5)
    b.reseed(rng.randrange(grid.n), 0.0)
    etas = unit_etas(grid, units, 0.0, delta=10)
    cov2 = coverage_sets(b.future(0.0), b.w, etas, margin=10, forbidden=frozenset(best))
    assert not best & {v for _, v in greedy_plan(cov2, b.w, units)}


def test_coverage_needs_the_unit_there_before_the_suspect(grid):
    # suspect drives the main road and reaches junction (4, 4) at t = 40 s; units start after a 30 s delay
    target = node(4, 4, COLS)
    route = [(node(4, 2, COLS), 0.0), (node(4, 3, COLS), 20.0), (target, 40.0)]
    etas = unit_etas(grid, {"far": node(0, 0, COLS), "near": target}, 0.0, delta=30)
    cov = coverage_sets([route], [1.0], etas, margin=5)
    assert cov[("near", target)] == {0}        # 30 s + 5 s margin <= 40 s
    assert ("far", target) not in cov          # 30 s + 200 s drive: far too late
    assert ("near", target) not in coverage_sets([route], [1.0], etas, margin=15)  # 45 s > 40 s
