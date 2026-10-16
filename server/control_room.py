"""Control-room state for the dashboard: one active case, cameras, units, belief and plan.

Sightings come from the demo chase (simulated) or from real edge cameras via POST /api/sighting.
Every accepted sighting updates the belief and produces a *proposed* plan; units only move after
the officer approves it (PDF Phase 10 step 14). Rejected or flagged sightings never move units.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

from planner.belief import Sighting
from planner.fugitive import BACKSTREET, PLANNER_MODELS
from planner.belief import ParticleBelief
from planner.plan import coverage_sets, greedy_plan, unit_etas
from planner.roadgraph import INF, dijkstra
from planner.simulate import (DISPATCH, MARGIN, N_PARTICLES, RECALL, STICKY, STICKY_WAITING, Unit, World)


@dataclass
class Camera:
    node: int
    live: bool = True
    last_heartbeat: float = 0.0


@dataclass
class Event:
    t: float
    kind: str      # info | sighting | flag | plan | approve | reject | intercept | escape | camera
    text: str
    data: dict = field(default_factory=dict)


class ControlRoom:
    def __init__(self, coverage: float = 0.25, n_units: int = 4, seed: int = 7):
        self.W = World()
        self.g = self.W.g
        self.coverage, self.n_units = coverage, n_units
        self.reset(seed)

    # ------------------------------------------------------------------ setup
    def reset(self, seed: int | None = None):
        self.seed = seed if seed is not None else random.randrange(10**6)
        rng = random.Random(self.seed)
        self.rng = rng
        self.cameras = {c: Camera(c) for c in rng.sample(self.W.junctions, int(self.coverage * len(self.W.junctions)))}
        self.units = {f"Unit {i + 1}": Unit(rng.choice(self.W.junctions)) for i in range(self.n_units)}
        self.t = 0.0
        self.case = None          # {"plate", "incident", "reported_at"}
        self.belief = None
        self.last_ok: Sighting | None = None
        self.sightings: list[dict] = []
        self.pending_plan: list[tuple[str, int]] | None = None
        self.events: list[Event] = []
        self.chase = None         # demo ground truth: {"route", "first_t", "next_i", "model"}
        self.outcome = None
        self.seq = 0
        self.edges: dict[str, dict] = getattr(self, "edges", {})   # survive reset: cameras stay registered
        for e in self.edges.values():
            self.cameras[e["junction"]] = Camera(e["junction"])
        self.reads: list[dict] = []
        self._edge_route = None
        self.log("info", f"Control room ready: {len(self.cameras)} cameras, {len(self.units)} units "
                         f"(scenario seed {self.seed}).")

    def log(self, kind, text, **data):
        self.events.append(Event(round(self.t, 1), kind, text, data))
        self.events = self.events[-200:]

    # ------------------------------------------------------------------ demo chase
    def start_demo(self, plate: str = "TS09EK4417", held_out: bool = False):
        rng = self.rng
        incident = rng.choice(self.W.central)
        model = BACKSTREET if held_out else rng.choice(PLANNER_MODELS)
        route = self.W.sampler.sample(model, incident, 0.0, rng, 1200)
        first_t = {}
        for v, t in route:
            first_t.setdefault(v, t)
        self.chase = {"route": route, "first_t": first_t, "next_i": 1, "model": model, "pending": []}
        self.open_case(plate, incident)

    def open_case(self, plate: str, incident: int | None = None):
        plate = "".join(c for c in plate.upper() if c.isalnum())
        self.case = {"plate": plate, "incident": incident, "reported_at": self.t, "live": incident is None}
        self.belief = None
        self.last_ok = None
        if incident is not None:
            self.belief = ParticleBelief(self.W.sampler, n=N_PARTICLES, horizon_s=900, seed=self.seed + 7)
            self.belief.reseed(incident, self.t)
            self.last_ok = Sighting(incident, self.t)
            self.log("info", f"Hotlist entry {plate}: incident reported at junction {incident}.")
            self.propose_plan("incident report")
        else:
            # live mode: only the real edge cameras exist, so "not seen" evidence comes only from them
            self.cameras = {e["junction"]: Camera(e["junction"]) for e in self.edges.values()}
            self.log("info", f"Hotlist entry {plate} opened (live mode, {len(self.cameras)} real camera(s)). "
                             f"Waiting for a camera to read it.")

    # ------------------------------------------------------------------ sightings
    def add_sighting(self, camera: int, t: float | None = None, plate: str | None = None,
                     score: float = 1.0, true: bool = True, source: str = "camera", thumb: str | None = None) -> dict:
        t = self.t if t is None else t
        self.seq += 1
        s = Sighting(camera, t, score, self.seq, true)
        rec = {"id": self.seq, "camera": camera, "t": round(t, 1), "plate": plate or (self.case or {}).get("plate"),
               "score": round(score, 3), "status": "accepted", "source": source,
               "lonlat": self.g.xy[camera], "thumb": thumb}
        if self.case is None:
            rec["status"] = "no active case"
        elif self.last_ok is None:  # live case: the first camera read starts the belief
            self.belief = ParticleBelief(self.W.sampler, n=N_PARTICLES, horizon_s=900, seed=self.seed + 7)
            self.belief.reseed(camera, t)
            self.last_ok = s
            self.log("sighting", f"Sighting #{self.seq}: {rec['plate']} read by {source} at junction {camera} "
                                 f"(score {score:.2f}) — first sighting, belief started.", id=self.seq)
            self.propose_plan(f"sighting #{self.seq}")
        elif t < self.last_ok.t:
            rec["status"] = "stale (older than current belief)"
        elif not self.W.gate.feasible(self.last_ok, s):
            need = self.W.gate.tau_min(self.last_ok.camera, camera)
            rec["status"] = "flagged: impossible journey"
            rec["why"] = (f"needs ≥ {0.7 * need:.0f} s from camera {self.last_ok.camera}, "
                          f"only {t - self.last_ok.t:.0f} s passed — possible clone or misread")
            self.log("flag", f"Sighting #{self.seq} at camera {camera} FLAGGED: {rec['why']}. Not used.", id=self.seq)
        else:
            self.last_ok = s
            self.belief.condition(s)
            self.log("sighting", f"Sighting #{self.seq}: {rec['plate']} read by {source} at junction {camera} "
                                 f"(score {score:.2f}) — accepted, belief updated.", id=self.seq)
            self.propose_plan(f"sighting #{self.seq}")
        self.sightings.append(rec)
        return rec

    # ------------------------------------------------------------------ planning
    def propose_plan(self, reason: str):
        if self.belief is None:
            return
        a = time.perf_counter()
        pos = {u: x.position(self.t) for u, x in self.units.items()}
        etas = unit_etas(self.g, pos, self.t, 0.0)
        for u, x in self.units.items():
            if not x.started:
                etas[u] = [e + DISPATCH for e in etas[u]]
        cov = coverage_sets(self.belief.future(self.t), self.belief.w, etas, MARGIN)
        sticky = {u: STICKY_WAITING if x.schedule and x.schedule[-1][1] <= self.t else STICKY
                  for u, x in self.units.items()}
        plan = greedy_plan(cov, self.belief.w, list(self.units),
                           current={u: x.target for u, x in self.units.items()}, sticky=sticky)
        covered = set().union(*(cov[p] for p in plan)) if plan else set()
        prob = sum(self.belief.w[k] for k in covered)
        ms = 1000 * (time.perf_counter() - a)
        changed = [(u, v) for u, v in plan if self.units[u].target != v]
        self.pending_plan = plan if changed else None
        self.plan_prob = prob
        if changed:
            self.log("plan", f"Plan after {reason}: {len(changed)} unit move(s), predicted interception "
                             f"chance {100 * prob:.0f}% ({ms:.0f} ms). Awaiting officer approval.",
                     moves=[(u, v) for u, v in changed])
        else:
            self.log("info", f"Plan after {reason}: no change ({100 * prob:.0f}% predicted).")

    def approve(self):
        if not self.pending_plan:
            return
        for u, v in self.pending_plan:
            self.units[u].send(self.g, v, self.t)
        self.log("approve", f"Officer approved: {', '.join(f'{u} → junction {v}' for u, v in self.pending_plan)}.")
        self.pending_plan = None

    def reject(self):
        if self.pending_plan:
            self.log("reject", "Officer rejected the plan; units keep their positions.")
        self.pending_plan = None

    # ------------------------------------------------------------------ edge cameras
    def edge_junctions(self):
        """Junctions for live cameras: consecutive spots ~45 s apart along a real main road through the
        centre (so a car passing phone A then phone B is a plausible journey), plus far-away spots."""
        if self._edge_route is None:
            rng = random.Random(99)
            start = min(self.W.central, key=lambda u: sum(abs(a - b) for a, b in zip(self.g.xy[u], self._centre())))
            route = self.W.sampler.sample("main_road", start, 0.0, rng, 1e9, exit_node=self.W.exits[0])
            spots, last_t = [route[0][0]], 0.0
            for v, t in route:
                if t - last_t >= 45 and len(self.g.adj[v]) >= 3:
                    spots.append(v)
                    last_t = t
            dist = dijkstra(self.g, start)[0]
            far = sorted(self.W.junctions, key=lambda u: -dist[u] if dist[u] < INF else 0)[:5]
            self._edge_route = {"route": spots, "far": far}
        return self._edge_route

    def _centre(self):
        xs = [x for x, _ in self.g.xy]
        ys = [y for _, y in self.g.xy]
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    def register_edge(self, name: str, role: str = "route", junction: int | None = None) -> dict:
        if name in self.edges:  # idempotent: nodes re-register periodically to survive server restarts
            j = self.edges[name]["junction"]
            return {"junction": j, "lonlat": self.g.xy[j]}
        if junction is not None:
            j = junction
        else:
            spots = self.edge_junctions()
            used = {e["junction"] for e in self.edges.values()}
            pool = [v for v in spots["far" if role == "far" else "route"][1:] if v not in used]
            j = pool[0] if pool else self.rng.choice(self.W.junctions)
        self.edges[name] = {"name": name, "junction": j, "role": role, "last_frame": 0.0, "reads": 0,
                            "matches": 0, "online": True}
        self.cameras[j] = Camera(j)
        self.log("camera", f"Edge camera '{name}' online at junction {j} ({role}).")
        return {"junction": j, "lonlat": self.g.xy[j]}

    def edge_read(self, r: dict, theta: float = 0.05) -> dict:
        e = self.edges.get(r["name"])
        if e is None:
            return {"ok": False, "error": "unregistered camera"}
        e["reads"] += 1
        self.read_seq = getattr(self, "read_seq", 0) + 1
        plate = r["plate"]
        target = (self.case or {}).get("plate")
        hit = bool(target and r.get("matched") and r.get("hotlist_plate") == target and r.get("score", 0) >= theta)
        rec = {"rid": self.read_seq, "t": round(self.t, 1), "camera": r["name"], "junction": e["junction"], "plate": plate,
               "conf": r.get("conf"), "score": r.get("score"), "hit": hit, "thumb": r.get("thumb")}
        self.reads.append(rec)
        self.reads = self.reads[-80:]
        if hit:
            e["matches"] += 1
            s = self.add_sighting(e["junction"], None, plate, float(r.get("score", 1.0)), source=r["name"],
                                  thumb=r.get("thumb"))
            rec["status"] = s["status"]
        return rec

    def edge_heartbeat(self, name: str, now: float):
        e = self.edges.get(name)
        if e:
            e["last_frame"] = now
            if not e["online"]:
                e["online"] = True
                self.set_camera(e["junction"], True)

    def check_edges(self, now: float, timeout: float = 5.0):
        for e in self.edges.values():
            if e["online"] and e["last_frame"] and now - e["last_frame"] > timeout:
                e["online"] = False
                self.set_camera(e["junction"], False)

    # ------------------------------------------------------------------ failures
    def set_camera(self, node: int, live: bool):
        if node in self.cameras:
            self.cameras[node].live = live
            self.log("camera", f"Camera {node} {'back online' if live else 'OFFLINE — marked blind; not-seen no longer counts as evidence'}.")

    def inject_clone(self):
        """Same plate read at a far live camera, now: the gate should flag it."""
        if self.case is None:
            return None
        here = self.suspect_node() or self.case["incident"] or (self.last_ok.camera if self.last_ok else None)
        if here is None:
            return None
        dist = dijkstra(self.g, here)[0]
        far = [c for c, cam in self.cameras.items() if cam.live and dist[c] != INF and dist[c] > 400]
        if not far:
            return None
        cam = self.rng.choice(far)
        self.log("info", f"[demo] A second car with plate {self.case['plate']} passes camera {cam}.")
        return self.add_sighting(cam, self.t, score=1.0, true=False, source="clone (demo)")

    # ------------------------------------------------------------------ clock
    def suspect_node(self):
        if not self.chase:
            return None
        node = None
        for v, t in self.chase["route"]:
            if t <= self.t:
                node = v
            else:
                break
        return node

    def step(self, dt: float):
        """Advance the demo clock: deliver camera reads, check interception/escape, decay belief."""
        if self.outcome:
            return
        self.t += dt
        if self.chase:
            ch = self.chase
            route = ch["route"]
            while ch["next_i"] < len(route) and route[ch["next_i"]][1] <= self.t:
                v, tv = route[ch["next_i"]]
                ch["next_i"] += 1
                cam = self.cameras.get(v)
                if cam and cam.live and self.rng.random() < RECALL:
                    ch["pending"].append((tv + self.rng.uniform(2, 6), v, tv))
                for u, x in self.units.items():  # interception: a unit waiting here long enough
                    if x.target == v and x.schedule and x.schedule[-1][1] + MARGIN <= tv:
                        self.outcome = {"kind": "intercept", "unit": u, "node": v, "t": tv}
                        self.log("intercept", f"INTERCEPTED by {u} at junction {v} (t = {tv:.0f} s). "
                                              f"Officer-led stop.")
                        return
            due = [p for p in ch["pending"] if p[0] <= self.t]
            ch["pending"] = [p for p in ch["pending"] if p[0] > self.t]
            for _, v, tv in due:
                self.add_sighting(v, tv, score=round(self.rng.uniform(0.85, 0.99), 2))
            if ch["next_i"] >= len(route) and not self.outcome:
                self.outcome = {"kind": "escape", "t": route[-1][1]}
                self.log("escape", f"Suspect left the area at t = {route[-1][1]:.0f} s.")
        if self.belief and self.last_ok:
            live = {c: RECALL for c, cam in self.cameras.items() if cam.live and c != self.last_ok.camera}
            self.belief.negative_evidence(self.t, live)

    # ------------------------------------------------------------------ view
    def snapshot(self, show_truth: bool = True) -> dict:
        g = self.g
        particles = []
        if self.belief:
            agg: dict[int, float] = {}
            for r, w in zip(self.belief.routes, self.belief.w):
                node = r[0][0]
                for v, tv in r:
                    if tv <= self.t:
                        node = v
                    else:
                        break
                agg[node] = agg.get(node, 0.0) + w
            top = sorted(agg.items(), key=lambda x: -x[1])[:250]
            mx = max((w for _, w in top), default=1.0)
            particles = [[*g.xy[v], round(w / mx, 3)] for v, w in top]
        units = []
        for u, x in self.units.items():
            pos = x.position(self.t)
            waiting = bool(x.schedule) and x.schedule[-1][1] <= self.t and x.target is not None
            path = [g.xy[v] for v, tv in x.schedule if tv >= self.t] if x.schedule else []
            units.append({"id": u, "lonlat": g.xy[pos], "target": x.target,
                          "target_lonlat": g.xy[x.target] if x.target is not None else None,
                          "state": "waiting" if waiting else ("driving" if x.target is not None else "idle"),
                          "path": [g.xy[pos]] + path})
        pend = [{"unit": u, "from": g.xy[self.units[u].position(self.t)], "to": g.xy[v], "node": v}
                for u, v in (self.pending_plan or [])]
        snap = {
            "t": round(self.t, 1), "seed": self.seed,
            "case": self.case and {**self.case, "lonlat": g.xy[self.case["incident"]] if self.case["incident"] is not None else None},
            "cameras": [{"node": c, "lonlat": g.xy[c], "live": cam.live} for c, cam in self.cameras.items()],
            "units": units, "pending_plan": pend, "particles": particles,
            "plan_prob": round(getattr(self, "plan_prob", 0.0), 3),
            "sightings": [{k: v for k, v in x.items() if k != "thumb"} | {"thumb": bool(x.get("thumb"))}
                          for x in self.sightings[-30:]],
            "events": [e.__dict__ for e in self.events[-60:]],
            "outcome": self.outcome,
            "reachable": self.belief.reachable_set_size(self.t) if self.belief else 0,
            "edges": [{**{k: v for k, v in e.items()}, "lonlat": g.xy[e["junction"]]} for e in self.edges.values()],
            "reads": [{k: v for k, v in r.items() if k != "thumb"} | {"has_thumb": bool(r.get("thumb"))}
                      for r in self.reads[-25:]],
        }
        if show_truth and self.chase:
            n = self.suspect_node()
            snap["truth"] = {"lonlat": g.xy[n] if n is not None else None, "model": self.chase["model"]}
        return snap
