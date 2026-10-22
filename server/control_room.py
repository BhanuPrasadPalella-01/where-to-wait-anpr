"""Control-room state for the dashboard: one active case, cameras, units, belief and plan.

Sightings come from the demo chase (simulated) or from real edge cameras via POST /api/sighting.
Every accepted sighting updates the belief and produces a *proposed* plan; units only move after
the officer approves it (PDF Phase 10 step 14). Rejected or flagged sightings never move units.

Persistent records (registered vehicles, sightings, plate reads, reconstructed routes, alerts and the
audit log) live in the SQLite store in server/db.py; an in-memory store is used when none is given.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

from planner.belief import ParticleBelief, Sighting
from planner.fugitive import BACKSTREET, PLANNER_MODELS
from planner.plan import coverage_sets, greedy_plan, unit_etas
from planner.reconstruct import reconstruct
from planner.roadgraph import INF, dijkstra
from planner.simulate import (DISPATCH, MARGIN, N_PARTICLES, RECALL, STICKY, STICKY_WAITING, Unit, World)

from .db import Database
from .messages import Deduper, verify

ACK_TIMEOUT = 30.0  # wall-clock seconds before an unacknowledged assignment is flagged to the officer


@dataclass
class Camera:
    node: int
    live: bool = True
    last_heartbeat: float = 0.0


@dataclass
class Event:
    t: float
    kind: str      # info | sighting | flag | replay | plan | approve | reject | intercept | escape | camera | alert
    text: str
    data: dict = field(default_factory=dict)


class ControlRoom:
    def __init__(self, coverage: float = 0.25, n_units: int = 4, seed: int = 7, world: World | None = None,
                 db: Database | None = None, edge_key: bytes | None = None):
        self.W = world or World()
        self.g = self.W.g
        self.db = db or Database(":memory:")
        self.edge_key = edge_key  # shared HMAC key; when set, unsigned or altered edge reads are dropped
        self.inbox = Deduper()    # (camera, boot, seq) already processed; survives reset like the cameras
        self.clock_rate = 1.0     # simulated seconds per wall-clock second (the dashboard's speed setting)
        self.coverage, self.n_units = coverage, n_units
        self._registry, self._registry_t = [], 0.0
        self.reset(seed)

    # ------------------------------------------------------------------ setup
    def reset(self, seed: int | None = None):
        self.seed = seed if seed is not None else random.randrange(10**6)
        rng = random.Random(self.seed)
        self.rng = rng
        self.cameras = {c: Camera(c) for c in rng.sample(self.W.junctions, int(self.coverage * len(self.W.junctions)))}
        self.units = {f"Unit {i + 1}": Unit(rng.choice(self.W.junctions)) for i in range(self.n_units)}
        self.t = 0.0
        self.case = None          # {"plate", "incident", "reported_at", "live", "vehicle"}
        self.belief = None
        self.last_ok: Sighting | None = None
        self.last_ok_sid: int | None = None  # database id of the last accepted sighting
        self.sightings: list[dict] = []
        self.timeline: list[dict] = []       # judged sightings in capture order, with the state before each
        self.track: list[dict] = []          # reconstructed route segments between accepted sightings
        self.pending_plan: list[tuple[str, int]] | None = None
        self.unit_alert: dict[str, dict] = {}  # unit -> {"id", "sent_wall", "acked"}
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
        if kind != "info":  # decisions and evidence go to the audit log
            self.db.audit(kind, text, t_sim=round(self.t, 1))

    # ------------------------------------------------------------------ hotlist registry
    def register_vehicle(self, **fields) -> dict:
        v = self.db.register_vehicle(**fields)
        self._registry_t = 0.0
        self.log("register", f"Vehicle {v['plate']} registered on the hotlist (case {v['case_no']}, "
                             f"{v['offence']}, severity {v['severity']}).")
        return v

    def close_vehicle(self, vid: int, actor: str = "officer"):
        self.db.close_vehicle(vid, actor)
        self._registry_t = 0.0

    def registry(self) -> list[dict]:
        if time.time() - self._registry_t > 5:
            self._registry, self._registry_t = self.db.vehicles(), time.time()
        return self._registry

    def hotlist(self) -> list[str]:
        plates = list(self.db.hotlist())
        if self.case and self.case["plate"] not in plates:
            plates.append(self.case["plate"])
        return plates

    def describe(self, plate: str | None = None) -> str:
        v = (self.case or {}).get("vehicle") if plate is None else self.db.active_vehicle(plate)
        plate = plate or (self.case or {}).get("plate", "")
        if not v:
            return plate
        bits = " ".join(b for b in (v.get("colour"), v.get("make_model") or v.get("vehicle_type")) if b)
        return f"{plate}{' — ' + bits if bits else ''} (case {v['case_no']}, {v['offence']})"

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

    def open_case(self, plate: str, incident: int | None = None, vehicle: dict | None = None):
        plate = "".join(c for c in plate.upper() if c.isalnum())
        vehicle = vehicle or self.db.active_vehicle(plate)
        self.case = {"plate": plate, "incident": incident, "reported_at": self.t, "live": incident is None,
                     "vehicle": vehicle}
        self.belief = None
        self.last_ok = None
        self.last_ok_sid = None
        self.timeline = []
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
        self.db.audit("case", f"Tracking opened for {self.describe()}", t_sim=self.t)

    # ------------------------------------------------------------------ sightings
    def add_sighting(self, camera: int, t: float | None = None, plate: str | None = None,
                     score: float = 1.0, true: bool = True, source: str = "camera", thumb: str | None = None) -> dict:
        """Judge a sighting by its capture time t. One that arrives after a later sighting was already
        judged (slow link, edge queue replayed) rolls the belief back to before that sighting and
        replays everything from there in capture-time order, so arrival order never changes the result."""
        t = self.t if t is None else t
        self.seq += 1
        s = Sighting(camera, t, score, self.seq, true)
        rec = {"id": self.seq, "camera": camera, "t": round(t, 1), "plate": plate or (self.case or {}).get("plate"),
               "score": round(score, 3), "status": "accepted", "source": source,
               "lonlat": self.g.xy[camera], "thumb": thumb}
        vid = ((self.case or {}).get("vehicle") or {}).get("id")
        if self.case is None:
            rec["status"] = "no active case"
        elif self.case["incident"] is not None and t < self.case["reported_at"]:
            rec["status"] = "stale (before the incident report)"
        if rec["status"] != "accepted":
            rec["db_id"] = self.db.add_sighting(rec, vid)
            self.sightings.append(rec)
            return rec

        later = self._rollback(t) if self.timeline and t < self.timeline[-1]["s"].t else []
        accepted = self._judge(s, rec, vid)
        if accepted:
            self.db.add_alert("all stations", "sighting",
                              f"Wanted {self.describe()} sighted at junction {camera} by {source}.",
                              vid, camera, self.g.xy[camera])
        for e in later:
            self._judge(e["s"], e["rec"], vid, replay=True)
        if later:
            self.log("replay", f"Sighting #{self.seq} arrived late (captured t = {t:.0f} s): belief rolled back "
                               f"and {len(later)} later sighting(s) replayed in capture order.", id=self.seq)
        self.sightings.append(rec)
        if accepted or later:
            self.propose_plan(f"{'late ' if later else ''}sighting #{self.seq}")
        return rec

    def _judge(self, s: Sighting, rec: dict, vid: int | None, replay: bool = False) -> bool:
        """Gate one sighting against the last accepted one and update the belief; store the verdict."""
        before = (self.belief.snapshot() if self.belief else None, self.last_ok, self.last_ok_sid)
        prev, prev_sid = self.last_ok, self.last_ok_sid
        old = rec["status"]
        rec.pop("why", None)
        what = f"Sighting #{s.seq}: {rec['plate']} read by {rec['source']} at junction {s.camera} (score {s.score:.2f})"
        if self.last_ok is None:  # live case: the first camera read starts the belief
            self.belief = ParticleBelief(self.W.sampler, n=N_PARTICLES, horizon_s=900, seed=self.seed + 7)
            self.belief.reseed(s.camera, s.t)
            self.last_ok, accepted = s, True
            rec["status"] = "accepted"
            kind, text = "sighting", f"{what} — first sighting, belief started."
        elif not self.W.gate.feasible(self.last_ok, s):
            need = self.W.gate.tau_min(self.last_ok.camera, s.camera)
            rec["status"], accepted = "flagged: impossible journey", False
            rec["why"] = (f"needs ≥ {0.7 * need:.0f} s from camera {self.last_ok.camera}, "
                          f"only {s.t - self.last_ok.t:.0f} s passed — possible clone or misread")
            kind, text = "flag", f"Sighting #{s.seq} at camera {s.camera} FLAGGED: {rec['why']}. Not used."
        else:
            self.last_ok = s
            self.belief.condition(s)
            rec["status"], accepted = "accepted", True
            kind, text = "sighting", f"{what} — accepted, belief updated."
        if not replay:
            rec["db_id"] = self.db.add_sighting(rec, vid)
            self.log(kind, text, id=s.seq)
        else:
            self.db.delete_routes_to(rec["db_id"])
            if rec["status"] != old:
                self.db.set_sighting_status(rec["db_id"], rec["status"], rec.get("why"))
                self.log(kind, f"Replay: {text} (was {old}).", id=s.seq)
        if accepted:
            self.last_ok_sid = rec["db_id"]
            if prev is not None:
                self._reconstruct(prev, s, prev_sid, rec["db_id"], vid)
        self.timeline.append({"s": s, "rec": rec, "before": before})
        return accepted

    def _rollback(self, t: float) -> list[dict]:
        """Undo every sighting captured after t; returns them (in capture order) for replay."""
        i = next(k for k, e in enumerate(self.timeline) if e["s"].t > t)
        later, self.timeline = self.timeline[i:], self.timeline[:i]
        snap, self.last_ok, self.last_ok_sid = later[0]["before"]
        if snap is None:
            self.belief = None
        else:
            self.belief.restore(snap)
        self.track = [seg for seg in self.track if seg["t1"] <= t]
        return later

    def _reconstruct(self, a: Sighting, b: Sighting, a_sid, b_sid, vehicle_id):
        """Rebuild the road the car most likely took between two accepted sightings (Yen's K paths)."""
        paths = reconstruct(self.g, a.camera, b.camera, b.t - a.t)
        for p in paths:
            p["lonlat"] = [self.g.xy[n] for n in p["nodes"]]
        seg = {"from": a.camera, "to": b.camera, "t0": a.t, "t1": b.t,
               "paths": [{k: v for k, v in p.items() if k != "nodes"} for p in paths]}
        self.track.append(seg)
        if paths:
            self.db.add_routes(vehicle_id, a_sid, b_sid, seg["paths"])
            best = paths[0]
            self.log("route", f"Route {a.camera} → {b.camera} reconstructed: {len(paths)} feasible path(s); "
                              f"best {best['travel_s']:.0f} s free-flow vs {b.t - a.t:.0f} s observed "
                              f"({100 * best['share']:.0f}% share).")
        else:
            self.log("route", f"No feasible road path {a.camera} → {b.camera} in {b.t - a.t:.0f} s.")

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
        vid = ((self.case or {}).get("vehicle") or {}).get("id")
        for u, v in self.pending_plan:
            if self.units[u].target == v:
                continue
            self.units[u].send(self.g, v, self.t)
            eta = max(0.0, self.units[u].schedule[-1][1] - self.t) if self.units[u].schedule else 0.0
            a = self.db.add_alert(u, "assign", f"Proceed to junction {v} and wait (ETA {eta:.0f} s). "
                                               f"Wanted: {self.describe()}. Officer-led stop only.",
                                  vid, v, self.g.xy[v])
            self.unit_alert[u] = {"id": a["id"], "sent_wall": time.time(), "acked": False, "flagged": False}
        self.log("approve", f"Officer approved: {', '.join(f'{u} → junction {v}' for u, v in self.pending_plan)}. "
                            f"Alerts sent to units.")
        self.pending_plan = None

    def reject(self):
        if self.pending_plan:
            self.log("reject", "Officer rejected the plan; units keep their positions.")
        self.pending_plan = None

    def ack_alert(self, aid: int) -> dict | None:
        a = self.db.ack_alert(aid)
        if a:
            st = self.unit_alert.get(a["recipient"])
            if st and st["id"] == aid:
                st["acked"] = True
            self.log("alert", f"{a['recipient']} acknowledged: {a['message'][:60]}…")
        return a

    def check_acks(self, now: float):
        """PDF Layer 5: an assignment not acknowledged in time is flagged so the officer can re-plan."""
        for u, st in self.unit_alert.items():
            if not st["acked"] and not st["flagged"] and now - st["sent_wall"] > ACK_TIMEOUT:
                st["flagged"] = True
                self.log("alert", f"{u} has not acknowledged its assignment for {ACK_TIMEOUT:.0f} s — "
                                  f"check the radio or re-plan without it.")

    # ------------------------------------------------------------------ edge cameras
    def edge_junctions(self):
        """Junctions for live cameras: consecutive spots ~45 s apart along a real main road through the
        centre (so a car passing phone A then phone B is a plausible journey), plus far-away spots."""
        if self._edge_route is None:
            start = min(self.W.central, key=lambda u: sum(abs(a - b) for a, b in zip(self.g.xy[u], self._centre())))
            from_start = dijkstra(self.g, start)[0]
            junctions = set(self.W.junctions)
            # chain of spots, each 40–60 s' drive from the previous one, moving outward from the centre
            spots = [start]
            for _ in range(8):
                dist = dijkstra(self.g, spots[-1])[0]
                cands = [v for v in junctions if 40 <= dist[v] <= 60 and v not in spots]
                if not cands:
                    break
                spots.append(max(cands, key=lambda v: from_start[v] if from_start[v] < INF else -1))
            far = sorted(self.W.junctions, key=lambda u: -from_start[u] if from_start[u] < INF else 0)[:5]
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

    def edge_read(self, r: dict, theta: float = 0.05, now: float | None = None) -> dict:
        """One plate read from an edge node (see server/messages.py for the signed, numbered format).

        The sighting time is the capture time, mapped from the node's wall clock onto the simulated
        clock, so a read that sat in the node's queue lands where it happened, not when it arrived."""
        if self.edge_key is not None and not verify(r, self.edge_key):
            self.log("flag", f"Edge read claiming to be from {r.get('name')!r} dropped: bad or missing signature.")
            return {"ok": False, "error": "bad signature"}
        e = self.edges.get(r["name"])
        if e is None:
            return {"ok": False, "error": "unregistered camera"}
        if r.get("seq") is not None and not self.inbox.add(r["name"], str(r.get("boot", "")), int(r["seq"])):
            return {"ok": True, "duplicate": True}
        now = time.time() if now is None else now
        delay = max(0.0, now - r["captured_at"]) if r.get("captured_at") else 0.0
        t = max(0.0, self.t - delay * self.clock_rate)
        e["reads"] += 1
        self.read_seq = getattr(self, "read_seq", 0) + 1
        plate = r["plate"]
        matched = r.get("hotlist_plate") if r.get("matched") and r.get("score", 0) >= theta else None
        # a registered vehicle seen while nothing is being tracked: start tracking it automatically
        if matched and self.case is None:
            v = self.db.active_vehicle(matched)
            if v:
                self.log("alert", f"Hotlist hit: {self.describe(matched)} read by {r['name']} — tracking started.")
                self.open_case(matched, vehicle=v)
        target = (self.case or {}).get("plate")
        hit = bool(matched and matched == target)
        rec = {"rid": self.read_seq, "t": round(t, 1), "camera": r["name"], "junction": e["junction"],
               "plate": plate, "conf": r.get("conf"), "score": r.get("score"), "hit": hit,
               "hotlist": bool(matched), "thumb": r.get("thumb")}
        self.reads.append(rec)
        self.reads = self.reads[-80:]
        self.db.add_read(rec)
        if hit:
            e["matches"] += 1
            s = self.add_sighting(e["junction"], t, plate, float(r.get("score", 1.0)), source=r["name"],
                                  thumb=r.get("thumb"))
            rec["status"] = s["status"]
        elif matched:  # another registered vehicle: tell the stations, do not divert the active case
            v = self.db.active_vehicle(matched)
            self.db.add_alert("all stations", "sighting",
                              f"Registered vehicle {self.describe(matched)} seen by {r['name']} at junction "
                              f"{e['junction']} (not the active case).", v and v["id"], e["junction"],
                              self.g.xy[e["junction"]])
            self.log("alert", f"Registered vehicle {matched} seen by {r['name']} (not the active case); stations told.")
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

    def _finish(self, outcome: dict, text: str):
        self.outcome = outcome
        self.log(outcome["kind"], text)
        vid = ((self.case or {}).get("vehicle") or {}).get("id")
        node = outcome.get("node")
        self.db.add_alert("all units", outcome["kind"], text, vid, node, self.g.xy[node] if node is not None else None)

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
                        self._finish({"kind": "intercept", "unit": u, "node": v, "t": tv},
                                     f"INTERCEPTED by {u} at junction {v} (t = {tv:.0f} s). Officer-led stop.")
                        return
            due = [p for p in ch["pending"] if p[0] <= self.t]
            ch["pending"] = [p for p in ch["pending"] if p[0] > self.t]
            for _, v, tv in due:
                self.add_sighting(v, tv, score=round(self.rng.uniform(0.85, 0.99), 2))
            if ch["next_i"] >= len(route) and not self.outcome:
                self._finish({"kind": "escape", "t": route[-1][1]}, f"Suspect left the area at t = {route[-1][1]:.0f} s.")
        if self.belief and self.last_ok:
            live = {c: RECALL for c, cam in self.cameras.items() if cam.live and c != self.last_ok.camera}
            self.belief.negative_evidence(self.t, live)

    # ------------------------------------------------------------------ view
    def unit_view(self, u: str) -> dict:
        x = self.units[u]
        g = self.g
        pos = x.position(self.t)
        waiting = bool(x.schedule) and x.schedule[-1][1] <= self.t and x.target is not None
        st = self.unit_alert.get(u)
        return {"id": u, "lonlat": g.xy[pos], "target": x.target,
                "target_lonlat": g.xy[x.target] if x.target is not None else None,
                "state": "waiting" if waiting else ("driving" if x.target is not None else "idle"),
                "path": [g.xy[pos]] + ([g.xy[v] for v, tv in x.schedule if tv >= self.t] if x.schedule else []),
                "eta": max(0.0, x.schedule[-1][1] - self.t) if x.schedule and not waiting else 0.0,
                "alert": st and {"id": st["id"], "acked": st["acked"], "late": st["flagged"] and not st["acked"]}}

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
        pend = [{"unit": u, "from": g.xy[self.units[u].position(self.t)], "to": g.xy[v], "node": v}
                for u, v in (self.pending_plan or [])]
        snap = {
            "t": round(self.t, 1), "seed": self.seed,
            "case": self.case and {**self.case, "lonlat": g.xy[self.case["incident"]] if self.case["incident"] is not None else None},
            "cameras": [{"node": c, "lonlat": g.xy[c], "live": cam.live} for c, cam in self.cameras.items()],
            "units": [self.unit_view(u) for u in self.units], "pending_plan": pend, "particles": particles,
            "plan_prob": round(getattr(self, "plan_prob", 0.0), 3),
            "sightings": [{k: v for k, v in x.items() if k != "thumb"} | {"thumb": bool(x.get("thumb"))}
                          for x in self.sightings[-30:]],
            "track": self.track[-20:],
            "events": [e.__dict__ for e in self.events[-60:]],
            "outcome": self.outcome,
            "reachable": self.belief.reachable_set_size(self.t) if self.belief else 0,
            "edges": [{**{k: v for k, v in e.items()}, "lonlat": g.xy[e["junction"]]} for e in self.edges.values()],
            "reads": [{k: v for k, v in r.items() if k != "thumb"} | {"has_thumb": bool(r.get("thumb"))}
                      for r in self.reads[-25:]],
            "registry": self.registry(),
            "alerts": self.db.alerts(limit=15),
        }
        if show_truth and self.chase:
            n = self.suspect_node()
            snap["truth"] = {"lonlat": g.xy[n] if n is not None else None, "model": self.chase["model"]}
        return snap
