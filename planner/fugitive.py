"""Fugitive route models (PDF Layer 6) and exit nodes.

A fleeing driver heads for an exit of the study area. Route models:
  FASTEST     shortest travel time to the exit, with 2% random wrong turns (van Droffelaar et al.)
  MAIN_ROAD   shortest time where main roads count as 40% faster (prefers arterials)
  RANDOM_WALK no-U-turn random walk, biased slightly toward the exit
  BACKSTREET  held out from the planner: avoids main roads (they count 60% slower). Used only as
              the "true" driver in half the experiments, to test behaviour the planner did not assume.
Next-hop trees (one reverse Dijkstra per exit per model) make sampling a route O(route length).
"""
from __future__ import annotations

import math
import random

from .roadgraph import INF, RoadGraph, dijkstra

FASTEST, MAIN_ROAD, RANDOM_WALK, BACKSTREET = "fastest", "main_road", "random_walk", "backstreet"
PLANNER_MODELS = (FASTEST, MAIN_ROAD, RANDOM_WALK)
WRONG_TURN = 0.02


def pick_exits(g: RoadGraph, n: int = 16) -> list[int]:
    """Spread n exits around the boundary: the farthest node from the centre in each angular sector."""
    cx = sum(x for x, _ in g.xy) / g.n
    cy = sum(y for _, y in g.xy) / g.n
    best: dict[int, tuple[float, int]] = {}
    for i, (x, y) in enumerate(g.xy):
        if len(g.adj[i]) == 0:
            continue
        sector = int((math.atan2(y - cy, x - cx) + math.pi) / (2 * math.pi) * n) % n
        r = math.hypot(x - cx, y - cy)
        if sector not in best or r > best[sector][0]:
            best[sector] = (r, i)
    return [i for _, i in best.values()]


def _weighted(g: RoadGraph, factor_main: float) -> RoadGraph:
    adj = [[(v, w * (factor_main if (u, v) in g.main else 1.0)) for v, w in nb] for u, nb in enumerate(g.adj)]
    return RoadGraph(g.xy, adj, g.main, g.osm_ids)


class RouteSampler:
    def __init__(self, g: RoadGraph, exits: list[int]):
        self.g = g
        self.exits = exits
        self.w = {(u, v): w for u, v, w in g.edges()}
        self.next_hop: dict[tuple[str, int], list[int]] = {}
        self.to_exit: dict[tuple[str, int], list[float]] = {}
        for model, factor in ((FASTEST, 1.0), (MAIN_ROAD, 0.6), (BACKSTREET, 1.6)):
            rg = _weighted(g, factor).reverse()
            for e in exits:
                dist, parent = dijkstra(rg, e)  # parent in reverse graph = next hop toward e
                self.next_hop[(model, e)] = parent
                self.to_exit[(model, e)] = dist

    def sample(self, model: str, start: int, t0: float, rng: random.Random, horizon_s: float,
               exit_node: int | None = None, prev: int | None = None) -> list[tuple[int, float]]:
        """Route as [(node, arrival_time)], from start at t0 until the exit or the time horizon."""
        g = self.g
        e = exit_node if exit_node is not None else rng.choice(self.exits)
        route = [(start, t0)]
        u, t, last = start, t0, prev
        tree_model = FASTEST if model == RANDOM_WALK else model
        hop, dist = self.next_hop[(tree_model, e)], self.to_exit[(tree_model, e)]
        steps = 0
        while u != e and t - t0 < horizon_s and steps < 2000:
            nbrs = [v for v, _ in g.adj[u] if v != last] or [v for v, _ in g.adj[u]]
            if model == RANDOM_WALK:
                # no U-turn walk; mild pull toward the exit so walkers do leave eventually
                closer = [v for v in nbrs if dist[v] < dist[u]]
                v = rng.choice(closer) if closer and rng.random() < 0.5 else rng.choice(nbrs)
            else:
                v = hop[u]
                if v == -1 or dist[u] == INF:
                    v = rng.choice(nbrs)
                elif rng.random() < WRONG_TURN and len(nbrs) > 1:
                    v = rng.choice([x for x in nbrs if x != hop[u]] or nbrs)
            t += self.w[(u, v)]
            last, u = u, v
            route.append((u, t))
            steps += 1
        return route
