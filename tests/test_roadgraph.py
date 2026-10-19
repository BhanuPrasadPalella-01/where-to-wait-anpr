"""Our Dijkstra and Yen vs NetworkX on the city graph (PDF Phase 15, row 3)."""
import random
import time
from pathlib import Path

import networkx as nx

from planner.roadgraph import RoadGraph, dijkstra, shortest_path, yen_k_shortest, path_cost

ROOT = Path(__file__).resolve().parents[1]


def to_nx(g: RoadGraph) -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_weighted_edges_from(g.edges())
    return G


def run(g: RoadGraph, n_pairs=1000, n_yen=30, k=5):
    G = to_nx(g)
    rng = random.Random(0)
    pairs = [(rng.randrange(g.n), rng.randrange(g.n)) for _ in range(n_pairs)]

    t_ours = t_nx = 0.0
    worst = 0.0
    for s, t in pairs:
        a = time.perf_counter(); d, _ = shortest_path(g, s, t); t_ours += time.perf_counter() - a
        a = time.perf_counter(); e = nx.dijkstra_path_length(G, s, t); t_nx += time.perf_counter() - a
        worst = max(worst, abs(d - e))
    assert worst < 1e-6, worst
    print(f"Dijkstra: {n_pairs} pairs identical (max diff {worst:.1e} s); "
          f"ours {1000 * t_ours / n_pairs:.1f} ms/pair, networkx {1000 * t_nx / n_pairs:.1f} ms/pair")

    a = time.perf_counter(); full, _ = dijkstra(g, 0); t_full = time.perf_counter() - a
    print(f"Dijkstra one-to-all: {1000 * t_full:.0f} ms for {g.n} nodes")

    mism = 0
    t_ours = t_nx = 0.0
    for s, t in pairs[:n_yen]:
        a = time.perf_counter(); ours = [c for c, _ in yen_k_shortest(g, s, t, k)]; t_ours += time.perf_counter() - a
        a = time.perf_counter()
        ref = []
        for p in nx.shortest_simple_paths(G, s, t, weight="weight"):
            ref.append(path_cost(g, p))
            if len(ref) == k:
                break
        t_nx += time.perf_counter() - a
        if any(abs(x - y) > 1e-6 for x, y in zip(ours, ref)) or len(ours) != len(ref):
            mism += 1
    assert mism == 0, f"{mism} Yen mismatches"
    print(f"Yen K={k}: {n_yen} pairs, path costs identical to networkx; "
          f"ours {1000 * t_ours / n_yen:.0f} ms/pair, networkx {1000 * t_nx / n_yen:.0f} ms/pair")


def test_roadgraph(city_graph):
    run(city_graph, 200, 10)


if __name__ == "__main__":
    run(RoadGraph.load(ROOT / "data" / "city_graph.json"))
