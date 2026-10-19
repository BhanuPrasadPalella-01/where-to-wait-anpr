"""Small synthetic road graphs for tests that must run without the downloaded city map."""
from __future__ import annotations

from planner.roadgraph import RoadGraph

ORIGIN = (78.4867, 17.3850)  # (lon, lat): same area as the real study map, so map code sees sane values
STEP_DEG = 0.003             # ~330 m between neighbouring junctions


def node(r: int, c: int, cols: int) -> int:
    return r * cols + c


def grid_graph(rows: int = 9, cols: int = 9, side_s: float = 30.0, main_s: float = 20.0) -> RoadGraph:
    """rows x cols grid of two-way streets; the middle row and middle column are a faster main road.

    Every edge costs side_s seconds (main_s on the main roads), so the shortest time between two
    junctions is easy to reason about in tests: e.g. opposite corners are (rows + cols - 2) blocks apart.
    """
    xy = [(ORIGIN[0] + c * STEP_DEG, ORIGIN[1] + r * STEP_DEG) for r in range(rows) for c in range(cols)]
    adj: list[list[tuple[int, float]]] = [[] for _ in range(rows * cols)]
    main: set[tuple[int, int]] = set()
    mr, mc = rows // 2, cols // 2
    for r in range(rows):
        for c in range(cols):
            u = node(r, c, cols)
            for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                rr, cc = r + dr, c + dc
                if not (0 <= rr < rows and 0 <= cc < cols):
                    continue
                v = node(rr, cc, cols)
                on_main = (r == rr == mr) or (c == cc == mc)
                adj[u].append((v, main_s if on_main else side_s))
                if on_main:
                    main.add((u, v))
    return RoadGraph(xy, [sorted(a) for a in adj], main, list(range(rows * cols)))


def random_graph(n: int = 60, extra: int = 120, seed: int = 0) -> RoadGraph:
    """Strongly connected random road graph with uneven travel times (5-60 s per edge).

    A random two-way spanning tree keeps every node reachable; `extra` one-way shortcuts on top give
    many alternative routes, so Dijkstra and Yen are exercised on something less regular than a grid.
    """
    import random
    rng = random.Random(seed)
    xy = [(ORIGIN[0] + rng.random() * 0.03, ORIGIN[1] + rng.random() * 0.03) for _ in range(n)]
    w: dict[tuple[int, int], float] = {}
    for v in range(1, n):
        u = rng.randrange(v)
        w[(u, v)] = round(rng.uniform(5, 60), 1)
        w[(v, u)] = round(rng.uniform(5, 60), 1)
    while len(w) < 2 * (n - 1) + extra:
        u, v = rng.randrange(n), rng.randrange(n)
        if u != v and (u, v) not in w:
            w[(u, v)] = round(rng.uniform(5, 60), 1)
    adj: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for (u, v), c in sorted(w.items()):
        adj[u].append((v, c))
    return RoadGraph(xy, adj, set(), list(range(n)))
