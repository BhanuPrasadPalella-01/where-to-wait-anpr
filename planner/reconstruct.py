"""Route reconstruction between two sightings (the travel history between cameras).

Cameras only see the car at a few junctions; the road it took in between is unobserved. Between
consecutive accepted sightings (a, t_a) and (b, t_b) we take Yen's K loopless shortest paths from a to
b and keep the ones the car could really have driven in the observed gap (space-time prism idea,
Qi et al. 2021):

    feasible:   (1 - sigma) * T(path) <= t_b - t_a        (allowing for speeding)
    plausible:  weight = exp(-|T(path) - (t_b - t_a)| / scale)

so a path whose free-flow time matches the observed gap is preferred, and a path that would need
impossible speed is dropped. Shares are the normalised weights.
Complexity: Yen with K paths costs O(K * V * (E log V)) in the worst case; on the city graph K = 3
takes a few hundred milliseconds, once per accepted sighting.
"""
from __future__ import annotations

import math

from .roadgraph import RoadGraph, yen_k_shortest


def reconstruct(g: RoadGraph, a: int, b: int, dt: float, k: int = 3, sigma: float = 0.3) -> list[dict]:
    """Ranked candidate paths a -> b for an observed gap of dt seconds.

    Returns [{"rank", "share", "travel_s", "nodes"}], best first; [] if a == b or nothing is feasible.
    """
    if a == b or dt <= 0:
        return []
    cands = [(cost, path) for cost, path in yen_k_shortest(g, a, b, k) if (1 - sigma) * cost <= dt]
    if not cands:
        return []
    scale = max(30.0, 0.3 * dt)
    weights = [math.exp(-abs(cost - dt) / scale) for cost, _ in cands]
    total = sum(weights)
    order = sorted(range(len(cands)), key=lambda i: -weights[i])
    return [{"rank": r + 1, "share": weights[i] / total, "travel_s": cands[i][0], "nodes": cands[i][1]}
            for r, i in enumerate(order)]
