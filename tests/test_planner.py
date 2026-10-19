"""Unit tests: feasibility gate, negative evidence, greedy vs exact (PDF Phase 15 rows 3 and 5)."""
import random
import time
from pathlib import Path

from planner.belief import FeasibilityGate, ParticleBelief, Sighting
from planner.fugitive import RouteSampler, pick_exits
from planner.plan import coverage_sets, exact_plan, greedy_plan, objective, unit_etas
from planner.roadgraph import RoadGraph

ROOT = Path(__file__).resolve().parents[1]


def tiny_graph():
    # 0 - 1 - 2 - 3 line, 10 s per edge, both directions; 4 hangs off 1
    adj = [[] for _ in range(5)]
    for a, b in [(0, 1), (1, 2), (2, 3), (1, 4)]:
        adj[a].append((b, 10.0))
        adj[b].append((a, 10.0))
    return RoadGraph([(i, 0) for i in range(5)], adj)


def test_gate():
    gate = FeasibilityGate(tiny_graph(), sigma=0.3)
    a = Sighting(0, 0.0)
    assert gate.feasible(a, Sighting(3, 30.0))        # exactly the min travel time
    assert gate.feasible(a, Sighting(3, 21.5))        # 28% faster than possible: inside slack
    assert not gate.feasible(a, Sighting(3, 15.0))    # 50% faster: impossible -> clone/misread
    assert not gate.feasible(Sighting(3, 15.0), a)    # out-of-order arrival handled the same way


def test_negative_evidence():
    g = tiny_graph()
    s = RouteSampler(g, [3, 4])
    b = ParticleBelief(s, n=200, horizon_s=100, seed=1)
    b.reseed(0, 0.0)
    b.negative_evidence(25.0, {2: 0.9})  # camera at node 2 saw nothing by t=25
    via2 = sum(w for r, w in zip(b.routes, b.w) if any(v == 2 and t <= 25 for v, t in r))
    via4 = sum(w for r, w in zip(b.routes, b.w) if any(v == 4 for v, _ in r))
    assert via4 > via2, (via4, via2)


def test_greedy_vs_exact(city_graph, n_cases=20):
    g = city_graph
    exits = pick_exits(g)
    s = RouteSampler(g, exits)
    rng = random.Random(0)
    ratios, t_greedy = [], []
    for case in range(n_cases):
        start = rng.randrange(g.n)
        b = ParticleBelief(s, n=200, horizon_s=900, seed=case)
        b.reseed(start, 0.0)
        units = {f"u{i}": rng.randrange(g.n) for i in range(3)}
        etas = unit_etas(g, units, 0.0, delta=30)
        cov = coverage_sets(b.future(0.0), b.w, etas, margin=30)
        if not cov:
            continue
        a = time.perf_counter()
        gp = greedy_plan(cov, b.w, units)
        t_greedy.append(time.perf_counter() - a)
        gv = objective(gp, cov, b.w)
        _, ev = exact_plan(cov, b.w, units, max_nodes=25)
        ev = max(ev, gv)
        if ev > 0:
            ratios.append(gv / ev)
    assert min(ratios) >= 0.5
    print(f"greedy/exact over {len(ratios)} cases: min {min(ratios):.3f}, mean {sum(ratios) / len(ratios):.3f}; "
          f"greedy {1000 * sum(t_greedy) / len(t_greedy):.0f} ms per plan")


if __name__ == "__main__":
    test_gate(); print("gate: ok")
    test_negative_evidence(); print("negative evidence: ok")
    test_greedy_vs_exact(RoadGraph.load(ROOT / "data" / "city_graph.json"))
