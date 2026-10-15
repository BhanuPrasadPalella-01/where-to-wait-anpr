"""Interception planner (PDF Phase 10 steps 12–13, Phase 11).

Particle k is covered by (u, v) if v is on particle k's route and E_u(v) + m <= T_k(v),
with E_u(v) = dist(x_u, v) + delta. Objective: maximise sum of covered particle weights,
each unit at most one node, each node at most one unit, forbidden nodes skipped.
F is monotone submodular and the constraint is a partition matroid, so greedy >= 1/2 OPT
(Fisher, Nemhauser & Wolsey 1978). `exact_plan` brute-forces small cases to measure the real gap.
"""
from __future__ import annotations

import itertools
from collections import defaultdict

from .roadgraph import INF, RoadGraph, dijkstra


def unit_etas(g: RoadGraph, unit_nodes: dict[str, int], t_now: float, delta: float) -> dict[str, list[float]]:
    return {u: [t_now + delta + d for d in dijkstra(g, x)[0]] for u, x in unit_nodes.items()}


def coverage_sets(futures, weights, etas, margin, forbidden=frozenset(), min_lead=0.0):
    """(unit, node) -> set of particle ids it covers. Only nodes some unit can reach in time are kept."""
    first_time: list[dict[int, float]] = []
    for route in futures:
        d: dict[int, float] = {}
        for v, t in route:
            d.setdefault(v, t)
        first_time.append(d)
    by_node: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for k, d in enumerate(first_time):
        if weights[k] <= 0:
            continue
        for v, t in d.items():
            if v not in forbidden:
                by_node[v].append((k, t))
    cov: dict[tuple[str, int], set[int]] = {}
    for u, eta in etas.items():
        for v, lst in by_node.items():
            e = eta[v]
            if e == INF:
                continue
            s = {k for k, t in lst if e + margin <= t and t - e >= min_lead}
            if s:
                cov[(u, v)] = s
    return cov


def objective(assign, cov, weights) -> float:
    covered = set().union(*(cov.get(a, set()) for a in assign)) if assign else set()
    return sum(weights[k] for k in covered)


def greedy_plan(cov, weights, units, eps: float = 1e-6, current: dict | None = None, sticky=0.0):
    """Repeatedly take the (unit, node) with the largest marginal covered weight.

    current/sticky: a unit's present target gets its gain multiplied by (1 + sticky), so units are
    only moved when the new junction is clearly better (avoids thrashing between re-plans).
    sticky may be a number or a dict unit -> number (e.g. larger for units already waiting).
    """
    current = current or {}
    covered: set[int] = set()
    free_units, used_nodes, assign = set(units), set(), []
    while free_units:
        best, gain = None, eps
        for (u, v), s in cov.items():
            if u in free_units and v not in used_nodes:
                g = sum(weights[k] for k in s - covered)
                if current.get(u) == v:
                    g *= 1.0 + (sticky.get(u, 0.0) if isinstance(sticky, dict) else sticky)
                if g > gain:
                    best, gain = (u, v), g
        if best is None:
            break
        assign.append(best)
        covered |= cov[best]
        free_units.discard(best[0])
        used_nodes.add(best[1])
    return assign


def exact_plan(cov, weights, units, max_nodes: int = 40):
    """Brute force over the most useful candidate nodes (small instances only)."""
    nodes = sorted({v for _, v in cov}, key=lambda v: -max(sum(weights[k] for k in cov.get((u, v), ())) for u in units))[:max_nodes]
    best, best_val = [], 0.0
    units = list(units)
    for r in range(1, len(units) + 1):
        for us in itertools.combinations(units, r):
            for vs in itertools.permutations(nodes, r):
                a = [(u, v) for u, v in zip(us, vs) if (u, v) in cov]
                if len(a) < r:
                    continue
                val = objective(a, cov, weights)
                if val > best_val:
                    best, best_val = a, val
    return best, best_val
