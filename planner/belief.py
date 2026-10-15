"""Feasibility gate and particle belief over the suspect's position (PDF Phase 10 steps 9–10, Phase 11).

Gate:   two sightings (a, t_a), (b, t_b) can be the same car only if t_b - t_a >= (1 - sigma) * tau_min(a, b)
Belief: N particles, each a sampled future route [(node, time)] from one of the planner's fugitive
        models; weight pi_k. A live camera that a particle passed without a sighting multiplies its
        weight by (1 - r_c) (negative evidence).
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from .fugitive import PLANNER_MODELS, RouteSampler
from .roadgraph import RoadGraph, shortest_path


@dataclass
class Sighting:
    camera: int          # graph node of the camera
    t: float             # capture time, seconds
    score: float = 1.0   # hotlist match score S(s)
    seq: int = 0
    true: bool = True    # ground truth for evaluation only; the planner never reads it
    prev: int | None = None  # node the car came from (camera's facing direction gives this)


class FeasibilityGate:
    def __init__(self, g: RoadGraph, sigma: float = 0.3):
        self.g, self.sigma = g, sigma
        self._cache: dict[tuple[int, int], float] = {}

    def tau_min(self, a: int, b: int) -> float:
        if (a, b) not in self._cache:
            self._cache[(a, b)] = shortest_path(self.g, a, b)[0]
        return self._cache[(a, b)]

    def feasible(self, last: Sighting, new: Sighting) -> bool:
        dt = abs(new.t - last.t)
        a, b = (last, new) if new.t >= last.t else (new, last)
        return dt >= (1 - self.sigma) * self.tau_min(a.camera, b.camera)


class ParticleBelief:
    def __init__(self, sampler: RouteSampler, n: int = 500, horizon_s: float = 900, seed: int = 0,
                 models=PLANNER_MODELS):
        self.sampler, self.n, self.horizon = sampler, n, horizon_s
        self.rng = random.Random(seed)
        self.models = models
        self.routes: list[list[tuple[int, float]]] = []
        self.meta: list[tuple[str, int]] = []   # (fugitive model, exit) per particle
        self.w: list[float] = []
        self.t_seed = 0.0
        self.checked_upto = 0.0

    def reseed(self, node: int, t: float, prev: int | None = None, meta=None):
        """Spread fresh particles from a node. meta: optional list of (model, exit) to draw from."""
        self.routes, self.meta = [], []
        for i in range(self.n):
            m, e = meta[i] if meta else (self.rng.choice(self.models), self.rng.choice(self.sampler.exits))
            self.routes.append(self.sampler.sample(m, node, t, self.rng, self.horizon, exit_node=e, prev=prev))
            self.meta.append((m, e))
        self.w = [1.0 / self.n] * self.n
        self.t_seed = self.checked_upto = t

    def condition(self, s: "Sighting", tol: float = 0.35, rejuvenate: float = 0.2):
        """Particle-filter update on a sighting at camera s.camera, time s.t.

        Particles whose route passes the camera near that time (relative tolerance on elapsed time)
        are kept, with weight; their (model, exit) are reused for the continuation from the camera.
        If none are consistent, fall back to a fresh spread from the camera.
        """
        elapsed = max(60.0, s.t - self.t_seed)
        keep = []
        for k, r in enumerate(self.routes):
            for v, t in r:
                if v == s.camera and abs(t - s.t) <= tol * elapsed + 30:
                    keep.append((self.meta[k], self.w[k]))
                    break
        if keep:
            total = sum(w for _, w in keep)
            pool = [m for m, w in keep]
            weights = [w / total for _, w in keep]
            # rejuvenation: a share of particles restarts from all models/exits, so a wrong
            # fugitive model can never take the whole belief with it
            n_keep = int(round((1 - rejuvenate) * self.n))
            meta = self.rng.choices(pool, weights=weights, k=n_keep)
            meta += [(self.rng.choice(self.models), self.rng.choice(self.sampler.exits))
                     for _ in range(self.n - n_keep)]
        else:
            meta = None
        self.reseed(s.camera, s.t, prev=s.prev, meta=meta)
        return bool(keep)

    def negative_evidence(self, t_now: float, live_cameras: dict[int, float], temper: float = 0.5):
        """Down-weight particles that passed a live camera in (checked_upto, t_now] without being seen."""
        lo = self.checked_upto
        for k, route in enumerate(self.routes):
            for node, t in route:
                if t > t_now:
                    break
                if t > lo and node in live_cameras:
                    self.w[k] *= 1.0 - temper * live_cameras[node]  # tempered: models are imperfect
        self.checked_upto = t_now
        s = sum(self.w)
        if s <= 1e-12:  # every particle contradicted: belief failed, fall back to uniform
            self.w = [1.0 / self.n] * self.n
        else:
            self.w = [x / s for x in self.w]

    def future(self, t_now: float):
        """For the planner: per particle, the not-yet-reached part of its route."""
        return [[(v, t) for v, t in r if t >= t_now] for r in self.routes]

    def reachable_set_size(self, t_now: float) -> int:
        return len({v for r in self.routes for v, t in r if t <= t_now})
