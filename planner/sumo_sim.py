"""Interception experiments in SUMO (PDF Phases 12–14): same methods as simulate.py, real traffic.

The suspect and police units are SUMO vehicles in background traffic (normal or peak demand).
The planner knows only free-flow travel times; SUMO supplies congestion, signals and queues.
Cameras fire when the suspect crosses a camera junction (recall 0.9, 2–10 s delivery delay).
Units are dispatched with TraCI route changes and stop at the end of the edge entering their junction.
Intercept = the suspect crosses a junction where a unit has been stopped for >= MARGIN s.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from .belief import ParticleBelief, Sighting
from .fugitive import BACKSTREET, PLANNER_MODELS
from .plan import coverage_sets, greedy_plan, unit_etas
from .roadgraph import INF, dijkstra, path_to
from .simulate import DISPATCH, HORIZON, MARGIN, N_PARTICLES, RECALL, REPORT, STICKY, STICKY_WAITING, World

ROOT = Path(__file__).resolve().parents[1]
SUMO_DIR = ROOT / "sumo"
WARMUP = 300  # s of background traffic before the crime
METHODS = ("proposed", "b2_once", "b1_static", "b0_chase")


def sumo_bin():
    import sumo
    os.environ["SUMO_HOME"] = sumo.SUMO_HOME
    return os.path.join(sumo.SUMO_HOME, "bin", "sumo")


def make_warm_state(demand: str):
    """Run the warm-up once and save it, so every episode starts from the same traffic state."""
    import traci
    out = SUMO_DIR / f"warm_{demand}.xml.gz"
    traci.start([sumo_bin(), "-n", str(SUMO_DIR / "city.net.xml"), "-r", str(SUMO_DIR / f"bg_{demand}.rou.xml"),
                 "--no-step-log", "--no-warnings", "--ignore-route-errors", "--time-to-teleport", "120",
                 "--seed", "1"], label=f"warm_{demand}")
    traci.simulationStep(WARMUP)
    traci.simulation.saveState(str(out))
    n = traci.vehicle.getIDCount()
    traci.close()
    return n


class SW(World):
    def __init__(self):
        super().__init__("sumo_graph.json")
        m = json.loads((ROOT / "data" / "sumo_edges.json").read_text())
        self.edge_of = {tuple(map(int, k.split(","))): v for k, v in m["edges"].items()}
        self.node_of_junction = {j: i for i, j in enumerate(m["junction_ids"])}


_SW: SW | None = None


def sworld() -> SW:
    global _SW
    if _SW is None:
        _SW = SW()
    return _SW


def _stop_pos(traci, edge):
    L = traci.lane.getLength(edge + "_0")
    return max(0.5 * L, L - 1.0)  # near the end of the edge; safe on very short edges


def run_episode(seed: int, coverage: float, method: str, demand: str = "normal", k: int = 4,
                held_out: bool = False) -> dict:
    import traci

    W = sworld()
    g = W.g
    rng = random.Random(seed)
    cams = set(rng.sample(W.junctions, int(coverage * len(W.junctions))))
    incident = rng.choice(W.central)
    true_model = BACKSTREET if held_out else rng.choice(PLANNER_MODELS)
    plan_route = W.sampler.sample(true_model, incident, 0.0, rng, HORIZON)
    nodes = [v for v, _ in plan_route]
    sus_edges = [W.edge_of[(a, b)] for a, b in zip(nodes, nodes[1:]) if (a, b) in W.edge_of]
    unit_start = {f"u{i}": rng.choice(W.junctions) for i in range(k)}
    det_rng = random.Random(seed + 1)

    label = f"ep{seed}_{method}_{coverage}_{demand}_{os.getpid()}"
    traci.start([sumo_bin(), "-n", str(SUMO_DIR / "city.net.xml"), "-r", str(SUMO_DIR / f"bg_{demand}.rou.xml"),
                 "--load-state", str(SUMO_DIR / f"warm_{demand}.xml.gz"), "--no-step-log", "--no-warnings",
                 "--ignore-route-errors", "--time-to-teleport", "120", "--seed", str(seed % 10000)], label=label)
    T0 = traci.simulation.getTime()
    traci.vehicletype.copy("DEFAULT_VEHTYPE", "fugitive")
    traci.vehicletype.setSpeedFactor("fugitive", 1.3)
    traci.vehicletype.copy("DEFAULT_VEHTYPE", "police")
    traci.vehicletype.setSpeedFactor("police", 1.2)
    traci.route.add("sus_r", sus_edges)
    traci.vehicle.add("suspect", "sus_r", typeID="fugitive", depart="now", departSpeed="max", departLane="best")

    units = {u: {"node": x, "veh": None, "target": None, "stopped_at": None, "moves": 0, "started": False}
             for u, x in unit_start.items()}

    def unit_position(u, t):
        x = units[u]
        if x["veh"] is None or x["veh"] not in traci.vehicle.getIDList():
            return x["node"], 0.0
        e = traci.vehicle.getRoadID(x["veh"])
        if e.startswith(":"):  # inside a junction: use the next edge
            r = traci.vehicle.getRoute(x["veh"])
            e = r[min(traci.vehicle.getRouteIndex(x["veh"]) + 1, len(r) - 1)]
        to = W.node_of_junction.get(traci.edge.getToJunction(e))
        rem = max(0.0, traci.lane.getLength(e + "_0") - traci.vehicle.getLanePosition(x["veh"])) / max(
            traci.lane.getMaxSpeed(e + "_0"), 1.0)
        return (to if to is not None else x["node"]), rem

    def send(u, target, t):
        x = units[u]
        if x["target"] == target:
            return
        pos, _ = unit_position(u, t)
        path = path_to(dijkstra(g, pos, target=target)[1], target) if pos != target else [pos]
        edges = [W.edge_of[(a, b)] for a, b in zip(path, path[1:])]
        x["moves"] += 1
        x["target"], x["stopped_at"] = target, None
        vid = x["veh"]
        alive = vid is not None and vid in traci.vehicle.getIDList()
        if not alive:
            if not edges:  # already at the junction: wait there
                x["stopped_at"] = t + (0 if x["started"] else DISPATCH)
                x["started"] = True
                return
            vid = f"{u}_{x['moves']}"
            traci.route.add(vid + "_r", edges)
            depart = T0 + t + (0 if x["started"] else DISPATCH)
            traci.vehicle.add(vid, vid + "_r", typeID="police", depart=str(depart), departLane="best")
            x["veh"], x["started"] = vid, True
            traci.vehicle.setStop(vid, edges[-1], pos=_stop_pos(traci, edges[-1]), duration=1e5)
            return
        try:
            if traci.vehicle.getStops(vid):
                traci.vehicle.replaceStop(vid, 0, "")
            cur = traci.vehicle.getRoadID(vid)
            if cur.startswith(":"):
                r = traci.vehicle.getRoute(vid)
                cur = r[min(traci.vehicle.getRouteIndex(vid) + 1, len(r) - 1)]
            new = [cur] + edges if edges else [cur]
            traci.vehicle.setRoute(vid, new)
            traci.vehicle.setStop(vid, new[-1], pos=_stop_pos(traci, new[-1]), duration=1e5)
        except traci.TraCIException:
            x["veh"] = None  # vehicle lost (teleported/removed): respawn on next send
            x["target"] = None

    belief = ParticleBelief(W.sampler, n=N_PARTICLES, horizon_s=900, seed=seed + 7)
    pending, sightings_n, last_ok, planned_once = [], 0, None, False
    caught, last_edge, passes = None, None, {}
    plan_ms = []
    t = 0.0
    while t < HORIZON:
        traci.simulationStep()
        t = traci.simulation.getTime() - T0
        ids = set(traci.vehicle.getIDList())
        if "suspect" not in ids:
            if t > 5:
                break
            continue
        e = traci.vehicle.getRoadID("suspect")
        if e != last_edge and not e.startswith(":") and e:
            j = W.node_of_junction.get(traci.edge.getFromJunction(e))
            if j is not None and last_edge is not None:
                passes.setdefault(j, t)
                # interception: a unit stopped at this junction long enough
                for x in units.values():
                    if x["target"] == j and x["stopped_at"] is not None and x["stopped_at"] + MARGIN <= t:
                        caught = t
                if j in cams and det_rng.random() < RECALL:
                    pending.append((t + det_rng.uniform(2, 10), Sighting(j, t, 1.0, sightings_n, True,
                                                                         prev=None)))
                    sightings_n += 1
            last_edge = e
        if caught is not None:
            break
        for u, x in units.items():  # record when units come to rest at their junction
            if x["stopped_at"] is None and x["veh"] in ids:
                try:
                    if traci.vehicle.isStopped(x["veh"]):
                        x["stopped_at"] = t
                except traci.TraCIException:
                    pass

        due = [s for a, s in pending if a <= t]
        pending = [(a, s) for a, s in pending if a > t]
        decide = (abs(t - REPORT) < 0.5) or bool(due)
        if not decide:
            continue
        if method == "b1_static":
            if abs(t - REPORT) < 0.5:
                cps = W.betweenness[:k]
                ids_u = list(units)
                cost = np.array([[dijkstra(g, units[u]["node"], target=c)[0][c] for c in cps] for u in ids_u])
                rr, cc = linear_sum_assignment(np.minimum(cost, 1e6))
                for a_, b_ in zip(rr, cc):
                    send(ids_u[a_], cps[b_], t)
            continue
        if method == "b0_chase":
            target = due[-1].camera if due else incident
            best = min(units, key=lambda u: dijkstra(g, unit_position(u, t)[0], target=target)[0][target])
            send(best, target, t)
            continue
        if abs(t - REPORT) < 0.5:
            belief.reseed(incident, 0.0)
            last_ok = Sighting(incident, 0.0)
        accepted = False
        for s in sorted(due, key=lambda s: s.t):
            if s.t < last_ok.t or not W.gate.feasible(last_ok, s):
                continue
            last_ok, accepted = s, True
            belief.condition(s)
        belief.negative_evidence(t, {c: RECALL for c in cams if c != last_ok.camera})
        if (method == "b2_once" and planned_once) or (planned_once and not accepted):
            continue
        a = time.perf_counter()
        pos = {u: unit_position(u, t) for u in units}
        etas = unit_etas(g, {u: p for u, (p, _) in pos.items()}, t, 0.0)
        for u, (_, rem) in pos.items():
            add = rem + (0 if units[u]["started"] else DISPATCH)
            etas[u] = [e_ + add for e_ in etas[u]]
        cov = coverage_sets(belief.future(t), belief.w, etas, MARGIN)
        sticky = {u: STICKY_WAITING if x["stopped_at"] is not None else STICKY for u, x in units.items()}
        plan = greedy_plan(cov, belief.w, list(units), current={u: x["target"] for u, x in units.items()},
                           sticky=sticky)
        plan_ms.append(1000 * (time.perf_counter() - a))
        for u, v in plan:
            send(u, v, t)
        planned_once = True

    escape_t = t
    traci.close()
    return {"seed": seed, "coverage": coverage, "method": method, "demand": demand, "k": k, "held_out": held_out,
            "true_model": true_model, "intercepted": caught is not None, "t_intercept": caught,
            "escape_t": escape_t, "n_sightings": sightings_n, "moves": sum(x["moves"] for x in units.values()),
            "plan_ms_mean": float(np.mean(plan_ms)) if plan_ms else 0.0, "n_plans": len(plan_ms),
            "free_flow_s": plan_route[-1][1]}


def _job(a):
    try:
        return run_episode(*a)
    except Exception as e:  # keep the sweep alive; record the failure
        return {"seed": a[0], "coverage": a[1], "method": a[2], "demand": a[3], "error": repr(e)[:300]}


def sweep(name, coverages, methods, demands, runs, seed0, workers=8, k=4):
    jobs = [(s, c, m, d, k, s % 2 == 1) for d in demands for c in coverages for s in range(seed0, seed0 + runs)
            for m in methods]
    rows, t0 = [], time.time()
    with Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_job, jobs)):
            rows.append(r)
            if (i + 1) % max(1, len(jobs) // 25) == 0:
                print(f"  {i + 1}/{len(jobs)} episodes, {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    (ROOT / "reports" / f"{name}.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    return rows


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="sumo_pilot")
    ap.add_argument("--coverages", default="0.3")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--demands", default="normal")
    ap.add_argument("--runs", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--warm", action="store_true", help="(re)build the warm-up traffic states first")
    a = ap.parse_args()
    if a.warm:
        for d in a.demands.split(","):
            print(d, "vehicles on the road after warm-up:", make_warm_state(d), file=sys.stderr)
    rows = sweep(a.name, [float(c) for c in a.coverages.split(",")], a.methods.split(","), a.demands.split(","),
                 a.runs, a.seed0, a.workers)
    import pandas as pd
    df = pd.DataFrame(rows)
    if "error" in df:
        print("errors:", df.error.notna().sum(), df.error.dropna().head(3).tolist())
        df = df[df.error.isna()]
    print(df.groupby(["demand", "coverage", "method"]).agg(rate=("intercepted", "mean"), n=("intercepted", "size"),
                                                          esc=("escape_t", "median")).round(3).to_string())
