"""Two suspects at once in the chase simulator (same world, units and rules as planner/simulate.py).

Crime A happens at t = 0, crime B at t = STAGGER at another central junction; each is reported REPORT s
later. One crime is severe (severity SEVERE), the other minor (1); which one is drawn per seed. Each case
has its own particle belief and feasibility gate; one plan covers every active case. A case ends when
its suspect is intercepted or leaves; its units are then free.

Methods
  weighted   severity-weighted merged plan; a unit stays locked to the case it was sent for while it
             still covers >= 5% of that suspect's belief (plan.useful_locks), then is released (proposed)
  equal      same, but both cases weigh 1 (severity ignored)
  hard_lock  severity-weighted, a unit stays locked to its case until that case ends
  no_lock    severity-weighted, a unit may be pulled from one case to the other at any time
  split      units divided between the cases up front; each case planned with its own units only

Score: severity-weighted catch = sum(severity of caught cases) / sum(severity).
"""
from __future__ import annotations

import random
import time

import numpy as np

from .belief import ParticleBelief, Sighting
from .fugitive import BACKSTREET, PLANNER_MODELS
from .plan import coverage_sets, greedy_plan, merge_cases, serving_case, unit_etas, useful_locks
from .roadgraph import INF
from .simulate import (DISPATCH, HORIZON, MARGIN, N_PARTICLES, RECALL, REPORT, STICKY, STICKY_WAITING, Unit,
                       world)

METHODS = ("weighted", "equal", "hard_lock", "no_lock", "split")
STAGGER = 120.0   # second crime this long after the first (s): units are already committed to the first
SEVERE = 3.0


def run_multi_episode(seed: int, coverage: float, method: str, k: int = 4, held_out: bool = False,
                      W=None) -> dict:
    W = W or world()
    g = W.g
    rng = random.Random(seed)            # scenario randomness: identical across methods for one seed
    cams = set(rng.sample(W.junctions, int(coverage * len(W.junctions))))
    severe = rng.randrange(2)
    cases = {}
    for c, t0 in ((0, 0.0), (1, STAGGER)):
        incident = rng.choice(W.central)
        model = BACKSTREET if held_out else rng.choice(PLANNER_MODELS)
        route = W.sampler.sample(model, incident, t0, rng, HORIZON)
        first_t: dict[int, float] = {}
        for v, t in route:
            first_t.setdefault(v, t)
        sightings = [(t + rng.uniform(2, 10), Sighting(v, t, 1.0, i, True, route[i - 1][0] if i else None))
                     for i, (v, t) in enumerate(route) if v in cams and rng.random() < RECALL]
        cases[c] = {"incident": incident, "t0": t0, "route": route, "first_t": first_t, "end_t": route[-1][1],
                    "sightings": sightings, "severity": SEVERE if c == severe else 1.0,
                    "belief": ParticleBelief(W.sampler, n=N_PARTICLES, horizon_s=900, seed=seed + 7 + 101 * c),
                    "last_ok": None, "caught": INF}
    units = {f"u{i}": Unit(rng.choice(W.junctions)) for i in range(k)}
    lock: dict[str, int] = {}
    if method == "split":  # half the units per case, severe case gets the odd one
        ids = list(units)
        lock = {u: (severe if i % 2 == 0 else 1 - severe) for i, u in enumerate(ids)}
    pulls = handoffs = 0
    plan_ms = []

    events = sorted({c["t0"] + REPORT for c in cases.values()}
                    | {a for c in cases.values() for a, _ in c["sightings"]})
    delivered = {c: 0 for c in cases}
    for t_now in events:
        active = {}
        replan = False
        for ci, c in cases.items():
            if t_now < c["t0"] + REPORT or t_now >= min(c["end_t"], c["caught"]):
                continue
            b = c["belief"]
            if c["last_ok"] is None:
                b.reseed(c["incident"], c["t0"])
                c["last_ok"] = Sighting(c["incident"], c["t0"])
                replan = True
            new = []
            while delivered[ci] < len(c["sightings"]) and c["sightings"][delivered[ci]][0] <= t_now:
                new.append(c["sightings"][delivered[ci]][1])
                delivered[ci] += 1
            for s in sorted(new, key=lambda s: s.t):
                if not W.gate.feasible(c["last_ok"], s) or s.t < c["last_ok"].t:
                    continue
                c["last_ok"] = s
                b.condition(s)
                replan = True
            b.negative_evidence(t_now, {n: RECALL for n in cams if n != c["last_ok"].camera})
            active[ci] = c
        _record_catches(cases, units, t_now)
        if not replan or not active:
            continue
        a = time.perf_counter()
        pos = {u: x.position(t_now) for u, x in units.items()}
        etas = unit_etas(g, pos, t_now, 0.0)
        for u, x in units.items():
            if not x.started:
                etas[u] = [e + DISPATCH for e in etas[u]]
        sev = {ci: (1.0 if method == "equal" else c["severity"]) for ci, c in active.items()}
        per_case = {ci: (coverage_sets(c["belief"].future(t_now), c["belief"].w, etas, MARGIN), c["belief"].w, sev[ci])
                    for ci, c in active.items()}
        current = {u: x.target for u, x in units.items()}
        locks = {u: ci for u, ci in lock.items() if ci in active}
        useful = useful_locks(locks, per_case, current)  # units still covering the suspect they were sent for
        if method == "no_lock":
            locks = {}
        elif method in ("weighted", "equal"):
            locks = useful
        cov, w = merge_cases(per_case, locks)
        if method == "split":  # a unit whose case has not started (or has ended) stays put
            cov = {(u, v): s for (u, v), s in cov.items() if lock[u] in active}
        sticky = {u: (STICKY_WAITING if x.schedule and x.schedule[-1][1] <= t_now else STICKY) for u, x in units.items()}
        plan = greedy_plan(cov, w, list(units), current=current, sticky=sticky)
        plan_ms.append(1000 * (time.perf_counter() - a))
        for u, v in plan:
            ci = serving_case(cov[(u, v)], w, locks.get(u))
            if units[u].target != v:
                if lock.get(u) in active and lock[u] != ci:
                    if u in useful:
                        pulls += 1     # taken off a suspect it was still covering
                    else:
                        handoffs += 1  # its own case no longer needed it there, so it was reused
                units[u].send(g, v, t_now)
                if method != "split":
                    lock[u] = ci
    stop_t = max(min(c["end_t"], HORIZON + c["t0"]) for c in cases.values())
    _record_catches(cases, units, stop_t)
    caught = [cases[c]["caught"] < INF for c in (0, 1)]
    sev = [cases[c]["severity"] for c in (0, 1)]
    return {"seed": seed, "coverage": coverage, "method": method, "k": k, "held_out": held_out,
            "severe": severe, "caught": caught, "caught_severe": caught[severe], "caught_minor": caught[1 - severe],
            "score": sum(s for s, c in zip(sev, caught) if c) / sum(sev), "pulls": pulls, "handoffs": handoffs,
            "moves": sum(x.moves for x in units.values()),
            "plan_ms_mean": float(np.mean(plan_ms)) if plan_ms else 0.0}


def _record_catches(cases, units, t_now):
    """A unit that has waited >= MARGIN s at a junction stops whichever suspect reaches it by t_now."""
    for c in cases.values():
        for x in units.values():
            spells = list(x.waits)
            if x.target is not None and x.schedule:
                spells.append((x.target, x.schedule[-1][1], INF))  # still waiting there
            for v, arrive, leave in spells:
                tv = c["first_t"].get(v)
                if tv is not None and arrive + MARGIN <= tv < leave and tv <= t_now and tv <= c["t0"] + HORIZON:
                    c["caught"] = min(c["caught"], tv)


def _job(args):
    return run_multi_episode(*args)


if __name__ == "__main__":
    import argparse
    import json
    import sys
    from multiprocessing import Pool

    from .simulate import ROOT, add_map_args, use_map
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="multi_pilot")
    ap.add_argument("--coverage", type=float, default=0.3)
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=40000)
    add_map_args(ap)
    a = ap.parse_args()
    use_map(a)
    jobs = [(s, a.coverage, m, a.k, s % 2 == 1) for s in range(a.seed0, a.seed0 + a.runs) for m in a.methods.split(",")]
    t0 = time.time()
    with Pool(a.workers) as pool:
        rows = list(pool.imap_unordered(_job, jobs, chunksize=2))
    print(f"{len(rows)} episodes in {time.time() - t0:.0f} s", file=sys.stderr)
    (ROOT / "reports" / f"{a.name}.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
