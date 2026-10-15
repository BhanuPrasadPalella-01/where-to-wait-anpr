"""Road graph G = (V, E, w) and the shortest-path algorithms the planner uses, written from scratch.

V = intersections, E = directed road segments, w = free-flow travel time in seconds.
Dijkstra  : binary heap (heapq), O((V + E) log V)
Yen       : K loopless shortest paths, O(K * V * Dijkstra)
"""
from __future__ import annotations

import heapq
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INF = math.inf

# Default study area; change to your own city (lat, lon) and radius in metres.
DEFAULT_CENTER = (17.3850, 78.4867)  # Hyderabad, Abids / Koti area
DEFAULT_RADIUS_M = 3000
MAIN_ROADS = {"motorway", "trunk", "primary", "secondary", "motorway_link", "trunk_link", "primary_link"}


@dataclass
class RoadGraph:
    xy: list[tuple[float, float]]                    # (lon, lat) per node
    adj: list[list[tuple[int, float]]]               # u -> [(v, seconds)]
    main: set[tuple[int, int]] = field(default_factory=set)  # edges on main roads
    osm_ids: list[int] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.adj)

    def edges(self):
        for u, nbrs in enumerate(self.adj):
            for v, w in nbrs:
                yield u, v, w

    def reverse(self) -> "RoadGraph":
        radj = [[] for _ in range(self.n)]
        for u, v, w in self.edges():
            radj[v].append((u, w))
        return RoadGraph(self.xy, radj, {(v, u) for u, v in self.main}, self.osm_ids)

    # ---------- persistence ----------
    def save(self, path: Path):
        path.write_text(json.dumps({"xy": self.xy, "adj": self.adj, "main": sorted(self.main), "osm_ids": self.osm_ids}))

    @classmethod
    def load(cls, path: Path) -> "RoadGraph":
        d = json.loads(path.read_text())
        return cls([tuple(p) for p in d["xy"]], [[(v, w) for v, w in nb] for nb in d["adj"]],
                   {tuple(e) for e in d["main"]}, d["osm_ids"])


def dijkstra(g: RoadGraph, source: int, banned_nodes: set | None = None, banned_edges: set | None = None,
             target: int | None = None) -> tuple[list[float], list[int]]:
    """Single-source shortest travel times. Returns (dist, parent). Stops early if target is settled."""
    dist = [INF] * g.n
    parent = [-1] * g.n
    dist[source] = 0.0
    heap = [(0.0, source)]
    banned_nodes = banned_nodes or set()
    banned_edges = banned_edges or set()
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue  # stale heap entry
        if u == target:
            break
        for v, w in g.adj[u]:
            if v in banned_nodes or (u, v) in banned_edges:
                continue
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                parent[v] = u
                heapq.heappush(heap, (nd, v))
    return dist, parent


def path_to(parent: list[int], target: int) -> list[int]:
    if parent[target] == -1:
        return []
    out = [target]
    while parent[out[-1]] != -1:
        out.append(parent[out[-1]])
    return out[::-1]


def shortest_path(g: RoadGraph, s: int, t: int, **kw) -> tuple[float, list[int]]:
    if s == t:
        return 0.0, [s]
    dist, parent = dijkstra(g, s, target=t, **kw)
    return dist[t], (path_to(parent, t) if dist[t] < INF else [])


def path_cost(g: RoadGraph, path: list[int]) -> float:
    w = {(u, v): c for u in set(path) for v, c in g.adj[u]}
    return sum(w[(a, b)] for a, b in zip(path, path[1:]))


def yen_k_shortest(g: RoadGraph, s: int, t: int, k: int) -> list[tuple[float, list[int]]]:
    """Yen's algorithm: the k cheapest loopless s-t paths, cheapest first."""
    c, p = shortest_path(g, s, t)
    if not p:
        return []
    found = [(c, p)]
    candidates: list[tuple[float, list[int]]] = []
    seen = {tuple(p)}
    for _ in range(1, k):
        last = found[-1][1]
        for i in range(len(last) - 1):
            spur, root = last[i], last[: i + 1]
            banned_edges = {(q[i], q[i + 1]) for _, q in found if len(q) > i + 1 and q[: i + 1] == root}
            banned_nodes = set(root[:-1])
            sc, sp = shortest_path(g, spur, t, banned_nodes=banned_nodes, banned_edges=banned_edges)
            if not sp:
                continue
            total = root[:-1] + sp
            if tuple(total) not in seen:
                seen.add(tuple(total))
                heapq.heappush(candidates, (path_cost(g, total), total))
        if not candidates:
            break
        found.append(heapq.heappop(candidates))
    return found


def build_from_osm(center=DEFAULT_CENTER, radius_m=DEFAULT_RADIUS_M) -> RoadGraph:
    """Download the drivable network and keep its largest strongly connected component."""
    import networkx as nx
    import osmnx as ox

    G = ox.graph_from_point(center, dist=radius_m, network_type="drive", simplify=True)
    G = ox.add_edge_speeds(G)          # km/h from maxspeed, else highway-type defaults
    G = ox.add_edge_travel_times(G)    # seconds
    G = G.subgraph(max(nx.strongly_connected_components(G), key=len)).copy()
    ids = list(G.nodes)
    idx = {o: i for i, o in enumerate(ids)}
    adj: list[dict[int, float]] = [dict() for _ in ids]
    main = set()
    for u, v, data in G.edges(data=True):
        a, b = idx[u], idx[v]
        if a == b:
            continue
        t = float(data["travel_time"])
        if b not in adj[a] or t < adj[a][b]:  # keep the fastest of parallel edges
            adj[a][b] = t
        hw = data.get("highway")
        hw = hw if isinstance(hw, list) else [hw]
        if MAIN_ROADS & set(hw):
            main.add((a, b))
    xy = [(float(G.nodes[o]["x"]), float(G.nodes[o]["y"])) for o in ids]
    return RoadGraph(xy, [sorted(d.items()) for d in adj], main, [int(o) for o in ids])


if __name__ == "__main__":
    import sys
    out = ROOT / "data" / "city_graph.json"
    g = build_from_osm()
    g.save(out)
    print(f"nodes={g.n} edges={sum(len(a) for a in g.adj)} main_road_edges={len(g.main)} -> {out}", file=sys.stderr)
