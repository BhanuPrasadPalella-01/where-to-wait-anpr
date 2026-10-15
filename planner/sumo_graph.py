"""Planner RoadGraph built from the SUMO network, so planner and simulator share one map.

Node = SUMO junction, edge = SUMO edge (fastest of parallel edges), weight = free-flow time
length / speed limit. The planner only knows free-flow times; SUMO adds real congestion.
"""
from __future__ import annotations

import json
from pathlib import Path

from .roadgraph import MAIN_ROADS, RoadGraph

ROOT = Path(__file__).resolve().parents[1]
NET = ROOT / "sumo" / "city.net.xml"
OUT = ROOT / "data" / "sumo_graph.json"
EDGE_MAP = ROOT / "data" / "sumo_edges.json"


def build():
    import sumolib

    net = sumolib.net.readNet(str(NET))
    edges = [e for e in net.getEdges() if not e.isSpecial() and e.allows("passenger")]
    jids = sorted({e.getFromNode().getID() for e in edges} | {e.getToNode().getID() for e in edges})
    idx = {j: i for i, j in enumerate(jids)}
    best: dict[tuple[int, int], tuple[float, str, str]] = {}
    for e in edges:
        a, b = idx[e.getFromNode().getID()], idx[e.getToNode().getID()]
        if a == b:
            continue
        t = e.getLength() / max(e.getSpeed(), 1.0)
        if (a, b) not in best or t < best[(a, b)][0]:
            best[(a, b)] = (t, e.getID(), e.getType())
    adj = [[] for _ in jids]
    main = set()
    for (a, b), (t, eid, typ) in best.items():
        adj[a].append((b, t))
        if any(f"highway.{m}" in typ for m in MAIN_ROADS):
            main.add((a, b))
    xy = []
    for j in jids:
        x, y = net.getNode(j).getCoord()
        lon, lat = net.convertXY2LonLat(x, y)
        xy.append((lon, lat))
    g = RoadGraph(xy, adj, main, list(range(len(jids))))
    # keep the largest strongly connected component (Kosaraju with our own adjacency)
    keep = _largest_scc(g)
    remap = {old: new for new, old in enumerate(keep)}
    adj2 = [[(remap[v], w) for v, w in g.adj[u] if v in remap] for u in keep]
    main2 = {(remap[a], remap[b]) for a, b in main if a in remap and b in remap}
    g2 = RoadGraph([xy[u] for u in keep], adj2, main2, [int(u) for u in keep])
    g2.save(OUT)
    emap = {f"{remap[a]},{remap[b]}": eid for (a, b), (_, eid, _) in best.items() if a in remap and b in remap}
    EDGE_MAP.write_text(json.dumps({"edges": emap, "junction_ids": [jids[u] for u in keep]}))
    print(f"SUMO graph: {g2.n} junctions, {sum(len(a) for a in adj2)} edges, {len(main2)} main-road edges -> {OUT}")


def _largest_scc(g: RoadGraph) -> list[int]:
    n, order, seen = g.n, [], [False] * g.n
    for s in range(n):  # iterative DFS for finishing order
        if seen[s]:
            continue
        stack = [(s, iter(g.adj[s]))]
        seen[s] = True
        while stack:
            u, it = stack[-1]
            for v, _ in it:
                if not seen[v]:
                    seen[v] = True
                    stack.append((v, iter(g.adj[v])))
                    break
            else:
                stack.pop()
                order.append(u)
    r = g.reverse()
    comp = [-1] * n
    best, best_c, c = 0, 0, 0
    for s in reversed(order):
        if comp[s] != -1:
            continue
        stack, size = [s], 0
        comp[s] = c
        while stack:
            u = stack.pop()
            size += 1
            for v, _ in r.adj[u]:
                if comp[v] == -1:
                    comp[v] = c
                    stack.append(v)
        if size > best:
            best, best_c = size, c
        c += 1
    return [u for u in range(n) if comp[u] == best_c]


if __name__ == "__main__":
    build()
