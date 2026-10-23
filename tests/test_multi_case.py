"""Several suspects at once: severity-weighted coverage and unit locks (planner level, no city map)."""
import random

import pytest

from planner.belief import ParticleBelief
from planner.fugitive import RouteSampler, pick_exits
from planner.plan import (coverage_sets, exact_plan, greedy_plan, merge_cases, objective, serving_case,
                          unit_etas, useful_locks)


def test_merge_weights_elements_by_severity():
    cov_a = {("u1", 5): {0, 1}}
    cov_b = {("u1", 5): {0}, ("u2", 7): {1}}
    cov, w = merge_cases({"A": (cov_a, [0.5, 0.5], 3.0), "B": (cov_b, [0.25, 0.75], 1.0)})
    assert cov[("u1", 5)] == {("A", 0), ("A", 1), ("B", 0)}   # one junction can stop either suspect
    assert w[("A", 1)] == pytest.approx(1.5) and w[("B", 1)] == pytest.approx(0.75)
    assert objective([("u1", 5)], cov, w) == pytest.approx(3.25)


def test_greedy_prefers_the_more_severe_case():
    # one unit; it can cover all of case A or all of case B, not both
    cov_a = {("u1", 1): {0}}
    cov_b = {("u1", 2): {0}}
    cases = {"A": (cov_a, [1.0], 1.0), "B": (cov_b, [1.0], 4.0)}
    cov, w = merge_cases(cases)
    assert greedy_plan(cov, w, ["u1"]) == [("u1", 2)]
    cases["A"] = (cov_a, [1.0], 5.0)
    cov, w = merge_cases(cases)
    assert greedy_plan(cov, w, ["u1"]) == [("u1", 1)]


def test_locked_unit_only_serves_its_own_case():
    cov_a = {("u1", 1): {0}, ("u2", 3): {1}}
    cov_b = {("u1", 2): {0}}
    cases = {"A": (cov_a, [0.5, 0.5], 1.0), "B": (cov_b, [1.0], 5.0)}
    cov, w = merge_cases(cases, locks={"u1": "A"})
    assert ("u1", 2) not in cov                      # u1 cannot be pulled to the severe case B
    assert dict(greedy_plan(cov, w, ["u1", "u2"])) == {"u1": 1, "u2": 3}
    cov, w = merge_cases(cases)                      # without the lock the severe case wins u1
    assert dict(greedy_plan(cov, w, ["u1", "u2"]))["u1"] == 2
    assert serving_case(cov[("u1", 1)], w, lock="A") == "A"


def test_lock_to_a_closed_case_frees_the_unit():
    cov_b = {("u1", 2): {0}}
    cov, w = merge_cases({"B": (cov_b, [1.0], 2.0)}, locks={"u1": "A"})
    assert greedy_plan(cov, w, ["u1"]) == [("u1", 2)]


def test_lock_is_released_when_the_unit_no_longer_helps_its_case():
    cov_a = {("u1", 1): {0}, ("u1", 4): {1}}
    cases = {"A": (cov_a, [0.97, 0.03], 1.0), "B": ({("u1", 2): {0}}, [1.0], 1.0)}
    assert useful_locks({"u1": "A"}, cases, {"u1": 1}) == {"u1": "A"}
    assert useful_locks({"u1": "A"}, cases, {"u1": 4}) == {}          # covers 3% < 5% of case A
    assert useful_locks({"u1": "A"}, cases, {"u1": 4}, min_share=0.02) == {"u1": "A"}
    assert useful_locks({"u1": "A"}, cases, {}) == {}                 # no junction yet
    assert useful_locks({"u1": "C"}, cases, {"u1": 1}) == {}          # closed case


def test_serving_case_is_the_case_with_most_covered_weight():
    w = {("A", 0): 0.4, ("B", 0): 0.3, ("B", 1): 0.3}
    assert serving_case({("A", 0), ("B", 0), ("B", 1)}, w) == "B"
    assert serving_case(set(), w) is None


def test_merged_greedy_keeps_the_half_bound(grid):
    sampler = RouteSampler(grid, pick_exits(grid, n=8))
    ratios = []
    for seed in range(8):
        rng = random.Random(seed)
        units = {f"u{i}": rng.randrange(grid.n) for i in range(3)}
        etas = unit_etas(grid, units, 0.0, delta=10)
        cases = {}
        for c, sev in (("A", 3.0), ("B", 1.0)):
            b = ParticleBelief(sampler, n=60, horizon_s=600, seed=seed * 10 + len(cases))
            b.reseed(rng.randrange(grid.n), 0.0)
            cases[c] = (coverage_sets(b.future(0.0), b.w, etas, margin=10), b.w, sev)
        cov, w = merge_cases(cases)
        if not cov:
            continue
        gv = objective(greedy_plan(cov, w, units), cov, w)
        ev = max(exact_plan(cov, w, units, max_nodes=12)[1], gv)
        ratios.append(gv / ev)
    assert len(ratios) >= 5 and min(ratios) >= 0.5
