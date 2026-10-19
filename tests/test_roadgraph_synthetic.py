"""Dijkstra and Yen vs NetworkX on synthetic graphs: always runs, no city map needed."""
import random

import networkx as nx
import pytest

from planner.roadgraph import INF, dijkstra, path_cost, shortest_path, yen_k_shortest
from tests.synthetic import grid_graph, node, random_graph

GRAPHS = {"grid": lambda: grid_graph(), "random": lambda: random_graph(seed=3)}


def to_nx(g):
    G = nx.DiGraph()
    G.add_nodes_from(range(g.n))
    G.add_weighted_edges_from(g.edges())
    return G


@pytest.fixture(scope="module", params=sorted(GRAPHS))
def graphs(request):
    g = GRAPHS[request.param]()
    return g, to_nx(g)


def test_dijkstra_matches_networkx(graphs):
    g, G = graphs
    for s in random.Random(0).sample(range(g.n), 10):
        dist, parent = dijkstra(g, s)
        ref = nx.single_source_dijkstra_path_length(G, s, weight="weight")
        assert all(dist[v] == pytest.approx(ref[v]) for v in range(g.n))
        for v in range(g.n):
            if v != s:  # following parents back gives a path that really costs dist[v]
                p = [v]
                while p[-1] != s:
                    p.append(parent[p[-1]])
                assert path_cost(g, p[::-1]) == pytest.approx(dist[v])


def test_grid_distances_are_what_the_layout_says():
    g = grid_graph()  # 9 x 9, side streets 30 s, main row/column 20 s
    assert shortest_path(g, node(4, 0, 9), node(4, 8, 9))[0] == pytest.approx(8 * 20)
    assert shortest_path(g, node(0, 0, 9), node(0, 1, 9))[0] == pytest.approx(30)
    # corner to corner: 4 side blocks to the main road, 8 main blocks, 4 side blocks (not 16 side blocks)
    assert shortest_path(g, node(0, 0, 9), node(8, 8, 9))[0] == pytest.approx(8 * 30 + 8 * 20)


def test_banned_nodes_and_edges_force_a_detour():
    g = grid_graph()
    a, b, mid = node(4, 2, 9), node(4, 4, 9), node(4, 3, 9)
    assert shortest_path(g, a, b)[1] == [a, mid, b]
    d, p = shortest_path(g, a, b, banned_nodes={mid})
    assert mid not in p and d > 40
    d2, p2 = shortest_path(g, a, b, banned_edges={(a, mid)})
    assert p2[:2] != [a, mid] and d2 > 40


def test_unreachable_target():
    g = random_graph(n=10, extra=0, seed=1)
    g.adj.append([])  # node 10 has no roads in or out
    g.xy.append(g.xy[0])
    assert dijkstra(g, 0)[0][10] == INF
    assert shortest_path(g, 0, 10) == (INF, [])
    assert yen_k_shortest(g, 0, 10, 3) == []


def test_yen_costs_match_networkx(graphs):
    g, G = graphs
    rng = random.Random(1)
    for _ in range(8):
        s, t = rng.sample(range(g.n), 2)
        ours = yen_k_shortest(g, s, t, 5)
        ref = []
        for p in nx.shortest_simple_paths(G, s, t, weight="weight"):
            ref.append(path_cost(g, p))
            if len(ref) == 5:
                break
        assert [c for c, _ in ours] == pytest.approx(ref)


def test_yen_paths_are_loopless_distinct_and_sorted(graphs):
    g, _ = graphs
    rng = random.Random(2)
    for _ in range(8):
        s, t = rng.sample(range(g.n), 2)
        found = yen_k_shortest(g, s, t, 6)
        costs = [c for c, _ in found]
        assert costs == sorted(costs)
        assert len({tuple(p) for _, p in found}) == len(found)
        for c, p in found:
            assert p[0] == s and p[-1] == t and len(set(p)) == len(p)
            assert path_cost(g, p) == pytest.approx(c)
