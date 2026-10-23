"""Road graph G = (V, E, w) and the shortest-path algorithms the planner uses, written from scratch.

V = intersections, E = directed road segments, w = free-flow travel time in seconds.
Dijkstra  : binary heap (heapq), O((V + E) log V)
Yen       : K loopless shortest paths, O(K * V * Dijkstra)
"""
from __future__ import annotations

import heapq
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INF = math.inf

# Default study area; pass --center LAT,LON --radius M for your own city (see README "Your own city").
DEFAULT_CENTER = (17.3850, 78.4867)  # Hyderabad, Abids / Koti area
DEFAULT_RADIUS_M = 3000
DEFAULT_GRAPH = ROOT / "data" / "city_graph.json"
GRAPH_ENV = "WTW_GRAPH"  # graph file the simulator and control room load instead of DEFAULT_GRAPH
MAIN_ROADS = {"motorway", "trunk", "primary", "secondary", "motorway_link", "trunk_link", "primary_link"}


@dataclass
class RoadGraph:
    xy: list[tuple[float, float]]                    # (lon, lat) per node
    adj: list[list[tuple[int, float]]]               # u -> [(v, seconds)]
    main: set[tuple[int, int]] = field(default_factory=set)  # edges on main roads
    osm_ids: list[int] = field(default_factory=list)
    meta: dict = field(default_factory=dict)         # study area it was built from (center, radius_m)

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
        path.write_text(json.dumps({"xy": self.xy, "adj": self.adj, "main": sorted(self.main), "osm_ids": self.osm_ids,
                                    "meta": self.meta}))

    @classmethod
    def load(cls, path: Path) -> "RoadGraph":
        d = json.loads(path.read_text())
        return cls([tuple(p) for p in d["xy"]], [[(v, w) for v, w in nb] for nb in d["adj"]],
                   {tuple(e) for e in d["main"]}, d["osm_ids"], d.get("meta", {}))


def parse_center(text: str) -> tuple[float, float]:
    """'LAT,LON' in decimal degrees -> (lat, lon)."""
    parts = text.replace(" ", "").split(",")
    if len(parts) != 2:
        raise ValueError(f"center must be LAT,LON, got {text!r}")
    lat, lon = float(parts[0]), float(parts[1])
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError(f"center out of range: {lat}, {lon}")
    return lat, lon


def area_graph_path(center: tuple[float, float], radius_m: int) -> Path:
    """Where the graph of one study area is kept, so several cities can sit side by side in data/."""
    return ROOT / "data" / f"city_{center[0]:.4f}_{center[1]:.4f}_{int(radius_m)}m.json"


def default_graph_path() -> Path:
    """The map everything loads: $WTW_GRAPH if set, else data/city_graph.json."""
    env = os.environ.get(GRAPH_ENV)
    return Path(env) if env else DEFAULT_GRAPH


def ensure_graph(center: tuple[float, float], radius_m: int) -> Path:
    """Graph file for a study area, downloading it from OSM the first time."""
    path = area_graph_path(center, radius_m)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        build_from_osm(center, radius_m).save(path)
    return path


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
    return RoadGraph(xy, [sorted(d.items()) for d in adj], main, [int(o) for o in ids],
                     {"center": list(center), "radius_m": radius_m})


if __name__ == "__main__":
    import argparse
    import sys
    ap = argparse.ArgumentParser(description="Download the drivable OSM road graph of a study area.")
    ap.add_argument("--center", type=parse_center, default=DEFAULT_CENTER, help="LAT,LON (default: Hyderabad)")
    ap.add_argument("--radius", type=int, default=DEFAULT_RADIUS_M, help="metres around the center")
    ap.add_argument("--out", type=Path, default=DEFAULT_GRAPH, help="graph file (default data/city_graph.json)")
    a = ap.parse_args()
    if a.radius <= 0:
        ap.error("--radius must be positive")
    out = a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    g = build_from_osm(a.center, a.radius)
    g.save(out)
    print(f"nodes={g.n} edges={sum(len(a) for a in g.adj)} main_road_edges={len(g.main)} -> {out}", file=sys.stderr)
