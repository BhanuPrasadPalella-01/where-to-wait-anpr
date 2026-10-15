"""Chase simulator for the interception experiments (PDF Phases 12–13), before moving to SUMO.

One episode: a crime at a central junction, the suspect drives a route drawn from a fugitive model
toward an exit; cameras at a share of junctions detect it with recall r and report after a delay;
k police units start at random junctions. A method decides where units go at the report time,
at every delivered sighting and every TICK seconds. The suspect is intercepted if some unit is
already waiting at a junction (arrived >= MARGIN s earlier) when the suspect reaches it.

Methods
  proposed   particle belief + negative evidence + feasibility gate + greedy re-plan on each accepted sighting
  timer      proposed, also re-planning every TICK s while unseen (the review's design; ablation)
  no_neg     proposed without negative evidence          (ablation)
  no_gate    proposed without the feasibility gate        (ablation)
  b2_once    proposed belief, planned once at report time, never re-planned   (van Droffelaar-style)
  b1_static  highest-betweenness junctions, units assigned by Hungarian algorithm (Samanta-style)
  b0_chase   nearest unit drives to the latest sighting / incident (original design)
"""
from __future__ import annotations

import os

# one math thread per worker process: 8 workers x many BLAS threads oversubscribed the CPU 4x
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import math
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from .belief import FeasibilityGate, ParticleBelief, Sighting
from .fugitive import BACKSTREET, PLANNER_MODELS, RouteSampler, pick_exits
from .plan import coverage_sets, greedy_plan, unit_etas
from .roadgraph import INF, RoadGraph, dijkstra, path_to

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("proposed", "timer", "no_neg", "no_gate", "b2_once", "b1_static", "b0_chase")

TICK = 30.0          # re-plan timer while the suspect is unseen (s)
MARGIN = 30.0        # unit must be in place this long before the suspect (s)
DISPATCH = 60.0      # delay before an idle unit starts moving (s)
REPORT = 60.0        # crime reported this long after it happens (s)
HORIZON = 1200.0     # suspect escapes if not stopped within 20 min, or at the exit
RECALL = 0.9         # chance a live camera detects and reads the plate
N_PARTICLES = 300
STICKY = 0.5         # unit still driving: move only if a new junction covers 50% more weight
STICKY_WAITING = 2.0 # unit already waiting at its junction: move only for 3x the coverage


@dataclass
class Unit:
    node: int
    schedule: list = field(default_factory=list)  # [(node, time)] of the current trip
    target: int | None = None
    waits: list = field(default_factory=list)     # [(node, arrive, leave)]
    started: bool = False
    moves: int = 0

    def position(self, t: float) -> int:
        pos = self.node
        for v, tv in self.schedule:
            if tv <= t:
                pos = v
            else:
                break
        return pos

    def send(self, g: RoadGraph, target: int, t: float):
        if target == self.target:
            return
        self._close_wait(t)
        self.moves += 1
        start = self.position(t)
        t0 = t + (0.0 if self.started else DISPATCH)
        self.started = True
        dist, parent = dijkstra(g, start, target=target)
        path = path_to(parent, target) if start != target else [start]
        w = {(u, v): c for u in path for v, c in g.adj[u]}
        sched, tt = [(start, t0)], t0
        for a, b in zip(path, path[1:]):
            tt += w[(a, b)]
            sched.append((b, tt))
        self.node, self.schedule, self.target = start, sched, target

    def _close_wait(self, t: float):
        if self.target is not None and self.schedule:
            arrive = self.schedule[-1][1]
            if arrive < t:
                self.waits.append((self.target, arrive, t))

    def finish(self, t: float):
        self._close_wait(t)
        self.target = None


class World:
    """Everything that depends only on the map: built once per worker process."""

    def __init__(self, graph_file: str = "city_graph.json"):
        self.g = RoadGraph.load(ROOT / "data" / graph_file)
        self.exits = pick_exits(self.g)
        self.sampler = RouteSampler(self.g, self.exits)
        self.gate = FeasibilityGate(self.g)
        self.junctions = [u for u in range(self.g.n) if len(self.g.adj[u]) >= 3]
        xs = np.array(self.g.xy)
        c = xs.mean(axis=0)
        r = np.hypot(*(xs - c).T)
        self.central = [u for u in self.junctions if r[u] < 0.5 * r.max()]
        self.betweenness = self._route_betweenness()

    def _route_betweenness(self, samples: int = 300) -> list[int]:
        """Junctions most used by shortest escape routes (sources: central junctions, sinks: exits)."""
        rng = random.Random(123)
        cnt = Counter()
        for _ in range(samples):
            route = self.sampler.sample("fastest", rng.choice(self.central), 0.0, random.Random(rng.random()), 1e9)
            cnt.update({v for v, _ in route[1:-1]})
        return [v for v, _ in cnt.most_common(200)]


_W: World | None = None


def world() -> World:
    global _W
    if _W is None:
        _W = World()
    return _W


def run_episode(seed: int, coverage: float, method: str, k: int = 4, held_out: bool = False,
                clone: bool = False) -> dict:
    W = world()
    g = W.g
    rng = random.Random(seed)            # scenario randomness: identical across methods for one seed
    cams = set(rng.sample(W.junctions, int(coverage * len(W.junctions))))
    incident = rng.choice(W.central)
    true_model = BACKSTREET if held_out else rng.choice(PLANNER_MODELS)
    route = W.sampler.sample(true_model, incident, 0.0, rng, HORIZON)
    first_t: dict[int, float] = {}
    for v, t in route:
        first_t.setdefault(v, t)
    end_t = route[-1][1]
    units = {f"u{i}": Unit(rng.choice(W.junctions)) for i in range(k)}

    # sightings: each pass of a camera is read with prob RECALL and arrives 2–10 s later
    sightings = []
    for i, (v, t) in enumerate(route):
        if v in cams and rng.random() < RECALL:
            sightings.append((t + rng.uniform(2, 10), Sighting(v, t, 1.0, i, True, route[i - 1][0] if i else None)))
    if clone:  # same plate seen at a far camera: impossible journey from the suspect's position
        tc = rng.uniform(120, min(600, end_t))
        here = max((v for v, t in route if t <= tc), key=lambda v: first_t[v], default=incident)
        dist = dijkstra(g, here)[0]
        far = [c for c in cams if dist[c] != INF and dist[c] > 400]
        if far:
            sightings.append((tc + rng.uniform(2, 10), Sighting(rng.choice(far), tc, 1.0, -1, False)))
    sightings.sort(key=lambda x: x[0])

    events = sorted({REPORT, *[a for a, _ in sightings], *np.arange(REPORT + TICK, min(end_t, HORIZON), TICK)})
    belief = ParticleBelief(W.sampler, n=N_PARTICLES, horizon_s=900, seed=seed + 7)
    live = {c: RECALL for c in cams}
    last_ok: Sighting | None = None
    clone_accepted = False
    plan_ms = []
    delivered = 0
    planned_once = False

    for t_now in events:
        if t_now >= end_t:
            break
        new = []
        while delivered < len(sightings) and sightings[delivered][0] <= t_now:
            new.append(sightings[delivered][1])
            delivered += 1

        if method == "b1_static":
            if t_now == REPORT:
                cps = W.betweenness[:k]
                ids = list(units)
                cost = np.array([[dijkstra(g, units[u].node, target=c)[0][c] for c in cps] for u in ids])
                rows, cols = linear_sum_assignment(np.minimum(cost, 1e6))
                for r_, c_ in zip(rows, cols):
                    units[ids[r_]].send(g, cps[c_], t_now)
            continue

        if method == "b0_chase":
            target = new[-1].camera if new else (incident if t_now == REPORT else None)
            if target is not None:
                best = min(units, key=lambda u: dijkstra(g, units[u].position(t_now), target=target)[0][target])
                units[best].send(g, target, t_now)
            continue

        # ---- belief-based methods ----
        if t_now == REPORT:
            belief.reseed(incident, 0.0)
            last_ok = Sighting(incident, 0.0)  # the reported crime scene anchors the feasibility gate
        accepted_now = False
        for s in sorted(new, key=lambda s: s.t):
            if last_ok is not None and method != "no_gate" and not W.gate.feasible(last_ok, s):
                continue  # flagged as possible clone / misread for the officer; belief untouched
            if last_ok is not None and s.t < last_ok.t:
                continue  # stale (older than the current belief)
            if not s.true:
                clone_accepted = True
            last_ok = s
            accepted_now = True
            belief.condition(s)
        if method != "no_neg":
            belief.negative_evidence(t_now, {c: r for c, r in live.items()
                                             if last_ok is None or c != last_ok.camera})
        if method == "b2_once" and planned_once:
            continue
        if method != "timer" and planned_once and not accepted_now:
            continue  # re-plan only when a new sighting is accepted (timer re-plans caused thrashing)
        a = time.perf_counter()
        pos = {u: x.position(t_now) for u, x in units.items()}
        etas = unit_etas(g, pos, t_now, 0.0)
        for u, x in units.items():  # idle units still need the dispatch delay
            if not x.started:
                etas[u] = [e + DISPATCH for e in etas[u]]
        cov = coverage_sets(belief.future(t_now), belief.w, etas, MARGIN)
        sticky = {u: (STICKY_WAITING if x.schedule and x.schedule[-1][1] <= t_now else STICKY) for u, x in units.items()}
        plan = greedy_plan(cov, belief.w, list(units), current={u: x.target for u, x in units.items()}, sticky=sticky)
        plan_ms.append(1000 * (time.perf_counter() - a))
        for u, v in plan:
            units[u].send(g, v, t_now)
        planned_once = True

    # ---- outcome ----
    stop_t = min(end_t, HORIZON)
    for x in units.values():
        x.finish(stop_t)
    caught = INF
    left_early = sum(1 for x in units.values() for v, arrive, leave in x.waits
                     if v in first_t and arrive + MARGIN <= first_t[v] and leave <= first_t[v])
    moves = sum(x.moves for x in units.values())
    for x in units.values():
        for v, arrive, leave in x.waits:
            tv = first_t.get(v)
            if tv is not None and arrive + MARGIN <= tv < leave and tv <= stop_t:
                caught = min(caught, tv)
    return {"seed": seed, "coverage": coverage, "method": method, "k": k, "held_out": held_out, "clone": clone,
            "true_model": true_model, "intercepted": caught < INF, "t_intercept": None if caught == INF else caught,
            "escape_t": end_t, "n_sightings": len(sightings), "clone_accepted": clone_accepted,
            "plan_ms_mean": float(np.mean(plan_ms)) if plan_ms else 0.0, "n_plans": len(plan_ms),
            "left_early": left_early, "moves": moves}


def _job(args):
    return run_episode(*args)


def sweep(out_name: str, coverages, methods, runs: int, k: int = 4, clone: bool = False, workers: int = 8,
          seed0: int = 0):
    jobs = [(seed, c, m, k, seed % 2 == 1, clone) for c in coverages for seed in range(seed0, seed0 + runs)
            for m in methods]
    t0 = time.time()
    rows = []
    with Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_job, jobs, chunksize=2)):
            rows.append(r)
            if (i + 1) % max(1, len(jobs) // 20) == 0:
                print(f"  {i + 1}/{len(jobs)} episodes, {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    path = ROOT / "reports" / f"{out_name}.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return rows


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="sim_pilot")
    ap.add_argument("--coverages", default="0.1,0.3")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--clone", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=0, help="tuning used seeds 0-99; final results use 10000+")
    a = ap.parse_args()
    rows = sweep(a.name, [float(c) for c in a.coverages.split(",")], a.methods.split(","), a.runs, a.k, a.clone,
                 a.workers, a.seed0)
    import pandas as pd
    df = pd.DataFrame(rows)
    print(df.groupby(["coverage", "method"]).agg(rate=("intercepted", "mean"), n=("intercepted", "size"),
                                                plan_ms=("plan_ms_mean", "mean")).round(3).to_string())
