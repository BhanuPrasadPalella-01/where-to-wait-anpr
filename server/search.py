"""Vehicle Search over the four RoundaboutHD cameras (the /search page).

Two ways to choose a vehicle:
  system search   start from one track (clicked in a camera, or its tracker number such as C1069);
                  the system finds the same vehicle in the other cameras by itself: candidates must arrive
                  in a time the road graph allows (0.4-1.3 x the minimum drive, Dijkstra on OpenStreetMap)
                  and move the way vehicles enter from that camera; the most similar fingerprint
                  (ResNet-50 embedding + colour histogram) wins. No ground truth is used.
  ground truth    look a dataset vehicle id up in the answer key (labelled as such on the page).
The dataset's ground-truth id of every track is returned only as a "check answer" flag.
Typical arrival ratios and entry/exit directions per camera pair are calibrated once from the
even-numbered ground-truth vehicles, as a deployment would calibrate its cameras.
"""
from __future__ import annotations

import json
import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from planner.reconstruct import reconstruct
from server.ground import Ground
from planner.roadgraph import RoadGraph, shortest_path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "roundabout"
CAMS = (1, 2, 3, 4)
NAMES = {1: "Roundabout 1 (west)", 2: "Roundabout 2", 3: "Roundabout 3", 4: "Roundabout 4 (east)"}
W_EMB, W_HIST = 0.7, 0.3
MIN_DIR = 0.43     # at least 3 of the 7 nearest calibration vehicles left that way
MIN_ARM = 0.29     # candidate must have come in from / left towards the previous camera
UNSURE_DIR = 0.14   # 1 of 7 neighbours: worth a look if the match is strong
UNSURE_SCORE = 0.85
MIN_SCORE = 0.5    # weaker best matches are reported as "no confident match"


def motion(t, first=True):
    b0, b1 = t["box"][0], t["box"][-1]
    d = np.array([(b1[0] + b1[2]) / 2 - (b0[0] + b0[2]) / 2, (b1[1] + b1[3]) / 2 - (b0[1] + b0[3]) / 2])
    return d / (np.linalg.norm(d) + 1e-9)


class SearchIndex:
    def __init__(self):
        self.ok = all((DATA / f"tracks_cam{c}.pkl").exists() for c in CAMS)
        if not self.ok:
            return
        self.tracks, self.fps = {}, 15.0
        for c in CAMS:
            d = pickle.load(open(DATA / f"tracks_cam{c}.pkl", "rb"))
            self.fps = d["fps"]
            for t in d["tracks"]:
                t["cam"] = c
            self.tracks[c] = {t["track"]: t for t in d["tracks"]}
        self.g = RoadGraph.load(ROOT / "data" / "carmel_graph.json")
        self.node = {int(k): v for k, v in json.loads((DATA / "camera_nodes.json").read_text()).items()}
        self.gps = {c: json.loads((DATA / f"cam0{c}_fitted_cam.json").read_text()) for c in CAMS}
        self.tau = {(a, b): shortest_path(self.g, self.node[a], self.node[b])[0] for a in CAMS for b in CAMS if a != b}
        self._calibrate()
        self.ground = Ground(self, DATA)

    # ------------------------------------------------------------------ calibration (even gt ids only)
    def _calibrate(self):
        seen = defaultdict(dict)
        for c in CAMS:
            for t in self.tracks[c].values():
                if t["gt"] is None or t["gt"] % 2:
                    continue
                cur = seen[t["gt"]].get(c)
                if cur is None or len(t["frames"]) > len(cur["frames"]):
                    seen[t["gt"]][c] = t
        enter, leave, ratio = defaultdict(list), defaultdict(list), defaultdict(list)
        for path in seen.values():
            path = sorted(path.values(), key=lambda t: t["t_first"])
            for a, b in zip(path, path[1:]):
                k = (a["cam"], b["cam"])
                enter[k].append(motion(b))
                leave[k].append(motion(a))
                ratio[k].append((b["t_first"] - a["t_last"]) / self.tau[k])
        unit = lambda v: (lambda m: m / (np.linalg.norm(m) + 1e-9))(np.mean(v, axis=0))  # noqa: E731
        self.cal = {k: {"enter": unit(enter[k]), "leave": unit(leave[k]), "mu": float(np.median(ratio[k])),
                        "sd": float(max(0.08, np.std(ratio[k]))), "n": len(ratio[k])}
                    for k in enter if len(ratio[k]) >= 5}

    # ------------------------------------------------------------------ helpers
    def _score(self, q_emb, q_hist, q_type, t):
        s = W_EMB * float(q_emb @ t["emb"]) + W_HIST * float(cv2.compareHist(q_hist, t["hist"], cv2.HISTCMP_CORREL))
        return s - (0.05 if t["type"] != q_type else 0.0)

    def _stop(self, t, score=None, cands=0, top=None, src_gt=None):
        return {"cam": t["cam"], "track": t["track"], "label": f"C{t['track']}", "type": t["type"],
                "t_first": round(t["t_first"], 2), "t_last": round(t["t_last"], 2),
                "score": None if score is None else round(score, 3), "candidates": cands,
                "alternatives": top or [], "gt": t["gt"],
                "correct": None if src_gt is None else (t["gt"] == src_gt)}

    def _step(self, q_emb, q_hist, q_type, prev, cam, forward, min_arm=0.0):
        """Best candidate in `cam`. forward: the vehicle goes prev -> cam; else it came cam -> prev.

        min_arm: the candidate's own learned entry (forward) or exit (backward) must point at prev's camera.
        """
        k = (prev["cam"], cam) if forward else (cam, prev["cam"])
        T, c = self.tau[k], self.cal.get(k)
        towards_prev = "W" if prev["cam"] < cam else "E"
        out = []
        for t in self.tracks[cam].values():
            if min_arm and self.ground.arms(cam, t["track"], 0 if forward else 1)[towards_prev] < min_arm:
                continue
            gap = (t["t_first"] - prev["t_last"]) if forward else (prev["t_first"] - t["t_last"])
            if not 0.4 * T <= gap <= 1.3 * T:
                continue
            if c is not None and float(motion(t) @ (c["enter"] if forward else c["leave"])) <= 0.5:
                continue
            s = self._score(q_emb, q_hist, q_type, t)
            if c is not None:
                z = (gap / T - c["mu"]) / c["sd"]
                s -= 0.05 * z * z
            out.append((s, t))
        out.sort(key=lambda x: -x[0])
        return out

    # ------------------------------------------------------------------ searches
    def search_from(self, cam: int, track: int, min_dir: float = MIN_DIR, min_score: float = MIN_SCORE,
                    min_arm: float = MIN_ARM, unsure: tuple = (UNSURE_DIR, UNSURE_SCORE)) -> dict:
        """Follow one track forwards and backwards in time, one roundabout at a time.

        At each roundabout the learned exit (where the vehicle leaves the view, which way it moves) decides
        whether it drove on towards a neighbouring camera or turned off; only then is the next camera
        searched, and its best candidate must clear min_score, otherwise the trail ends there.
        """
        unsure_dir, unsure_score = unsure
        src = self.tracks[cam][track]
        src_gt = src["gt"]
        stops = [self._stop(src, src_gt=src_gt)]
        ends = {}
        for when in ("after", "before"):
            forward = when == "after"
            q_emb, q_hist, prev = src["emb"], src["hist"], src
            while True:
                c = prev["cam"]
                arms = self.ground.arms(c, prev["track"], 1 if forward else 0)
                side = max(("E", "W"), key=arms.get)
                n = c + 1 if side == "E" else c - 1
                need = min_score if arms[side] >= min_dir else unsure_score  # unsure exit: only a strong match counts
                if arms[side] < unsure_dir or n not in CAMS:
                    why = ("turned off onto a side road" if forward else "came in from a side road") \
                        if arms["off"] >= arms[side] or n in CAMS else \
                        ("drove on past the last camera" if forward else "came from beyond the last camera")
                    ends[when] = {"cam": c, "why": why, "arms": arms}
                    break
                ranked = self._step(q_emb, q_hist, prev["type"], prev, n, forward, min_arm)
                if not ranked or ranked[0][0] < need:
                    ends[when] = {"cam": c, "why": f"no confident match at camera {n}"
                                  + (f" (best {ranked[0][0]:.2f} < {need})" if ranked else " (no candidate in the time window)"),
                                  "arms": arms, "towards": n}
                    break
                s, best = ranked[0]
                top = [{"track": t["track"], "label": f"C{t['track']}", "score": round(x, 3),
                        "t_first": round(t["t_first"], 2), "correct": t["gt"] == src_gt if src_gt else None}
                       for x, t in ranked[1:5]]
                stops.append(self._stop(best, s, len(ranked), top, src_gt))
                q_emb = (q_emb + best["emb"]) / np.linalg.norm(q_emb + best["emb"])
                q_hist = (q_hist + best["hist"]) / 2
                prev = best
        stops.sort(key=lambda s: s["t_first"])
        for st in stops:
            st["path"] = self.ground.lonlat(st["cam"], st["track"])
        return {"mode": "system", "source": {"cam": cam, "track": track}, "stops": stops, "ends": ends,
                "route": self.route(stops), "gt_of_source": src_gt,
                "truth_cams": [st["cam"] for st in self.ground_truth(src_gt)["stops"]] if src_gt is not None else None}

    def ground_truth(self, vid: int) -> dict:
        stops = []
        for c in CAMS:
            mine = [t for t in self.tracks[c].values() if t["gt"] == vid]
            if mine:
                stops.append(self._stop(max(mine, key=lambda t: len(t["frames"])), src_gt=vid))
        stops.sort(key=lambda s: s["t_first"])
        for st in stops:
            st["path"] = self.ground.lonlat(st["cam"], st["track"])
        return {"mode": "ground truth", "vehicle": vid, "stops": stops, "route": self.route(stops)}

    def overview(self) -> dict:
        """Every annotated vehicle: the cameras it passed (answer key), its paths, and what the system recovered."""
        if getattr(self, "_overview", None):
            return self._overview
        by_gt = defaultdict(dict)
        for c in CAMS:
            for t in self.tracks[c].values():
                if t["gt"] is not None:
                    cur = by_gt[t["gt"]].get(c)
                    if cur is None or len(t["frames"]) > len(cur["frames"]):
                        by_gt[t["gt"]][c] = t
        rows, flows = [], defaultdict(int)
        for vid, per_cam in sorted(by_gt.items()):
            ts = sorted(per_cam.values(), key=lambda t: t["t_first"])
            cams = [t["cam"] for t in ts]
            for a, b in zip(cams, cams[1:]):
                flows[f"{a}-{b}"] += 1
            row = {"vehicle": vid, "cams": cams, "type": Counter(t["type"] for t in ts).most_common(1)[0][0],
                   "t_first": round(ts[0]["t_first"], 1), "t_last": round(ts[-1]["t_last"], 1),
                   "direction": "east" if cams == sorted(cams) and len(cams) > 1 else
                                "west" if cams == sorted(cams, reverse=True) and len(cams) > 1 else
                                "one camera" if len(cams) == 1 else "mixed",
                   "tracks": [{"cam": t["cam"], "track": t["track"]} for t in ts],
                   "paths": [[[round(lon, 6), round(lat, 6)] for _, lon, lat in self.ground.lonlat(t["cam"], t["track"])[::2]]
                             for t in ts]}
            if len(cams) > 1:
                r = self.search_from(ts[0]["cam"], ts[0]["track"])
                found = [s["cam"] for s in r["stops"] if s["correct"]]
                wrong = sum(1 for s in r["stops"] if s["correct"] is False)
                row["system"] = {"found": found, "wrong": wrong, "full": set(found) == set(cams) and not wrong}
            rows.append(row)
        roads = {}
        for k in flows:
            a, b = map(int, k.split("-"))
            if abs(a - b) == 1:
                lo, hi = min(a, b), max(a, b)
                roads[f"{lo}-{hi}"] = [list(p) for p in (self.g.xy[n] for n in shortest_path(self.g, self.node[lo], self.node[hi])[1])]
        multi = [r for r in rows if len(r["cams"]) > 1]
        self._overview = {
            "vehicles": rows, "flows": dict(flows), "roads": roads,
            "summary": {"vehicles": len(rows), "tracks": sum(len(v) for v in self.tracks.values()),
                        "by_count": {n: sum(1 for r in rows if len(r["cams"]) == n) for n in (1, 2, 3, 4)},
                        "multi": len(multi), "system_full": sum(r["system"]["full"] for r in multi),
                        "system_wrong": sum(r["system"]["wrong"] > 0 for r in multi)}}
        return self._overview

    def by_label(self, number: int) -> list[dict]:
        return [{"cam": c, "track": number} for c in CAMS if number in self.tracks[c]]

    def route(self, stops):
        """The road the vehicle most likely took between consecutive cameras (Yen, time-feasible), as lon/lat."""
        segs = []
        for a, b in zip(stops, stops[1:]):
            u = self._nearest(a["path"][-1][1:]) if a.get("path") else self.node[a["cam"]]   # where it left
            v = self._nearest(b["path"][0][1:]) if b.get("path") else self.node[b["cam"]]    # where it came in
            dt = max(1.0, b["t_first"] - a["t_last"])
            paths = reconstruct(self.g, u, v, dt + 20.0)  # + time spent inside the camera views
            nodes = paths[0]["nodes"] if paths else shortest_path(self.g, u, v)[1]
            segs.append({"from": a["cam"], "to": b["cam"], "lonlat": [self.g.xy[n] for n in nodes],
                         "drive_s": round(dt, 1)})
        return segs

    def _nearest(self, lonlat):
        lon, lat = lonlat
        k = math.cos(math.radians(lat))
        return min(range(self.g.n), key=lambda n: ((self.g.xy[n][0] - lon) * k) ** 2 + (self.g.xy[n][1] - lat) ** 2)

    # ------------------------------------------------------------------ page data
    def meta(self) -> dict:
        four = defaultdict(set)
        for c in CAMS:
            for t in self.tracks[c].values():
                if t["gt"] is not None:
                    four[t["gt"]].add(c)
        return {"fps": self.fps, "duration": 600.0,
                "cameras": [{"cam": c, "name": NAMES[c], "lat": self.gps[c]["gps_lat"], "lon": self.gps[c]["gps_lon"],
                             "tracks": len(self.tracks[c])} for c in CAMS],
                "examples": sorted(v for v, cs in four.items() if len(cs) == 4)[:12]}

    def boxes(self, cam: int, t0: float, t1: float) -> list[dict]:
        f0, f1 = int(t0 * self.fps), int(t1 * self.fps)
        out = []
        for t in self.tracks[cam].values():
            if t["frames"][-1] < f0 or t["frames"][0] > f1:
                continue
            fb = [[f, b] for f, b in zip(t["frames"], t["box"]) if f0 <= f <= f1]
            if fb:
                out.append({"track": t["track"], "type": t["type"], "fb": fb})
        return out

    def pick(self, cam: int, t: float, x: float, y: float):
        f = int(round(t * self.fps))
        best = None
        for tr in self.tracks[cam].values():
            if tr["frames"][0] > f + 8 or tr["frames"][-1] < f - 8:
                continue
            for fr, b in zip(tr["frames"], tr["box"]):
                if abs(fr - f) <= 8 and b[0] - 6 <= x <= b[2] + 6 and b[1] - 6 <= y <= b[3] + 6:
                    area = (b[2] - b[0]) * (b[3] - b[1])
                    if best is None or abs(fr - f) < best[0] or (abs(fr - f) == best[0] and area < best[1]):
                        best = (abs(fr - f), area, tr["track"])
        return best[2] if best else None

    def crop(self, cam: int, track: int) -> bytes | None:
        t = self.tracks.get(cam, {}).get(track)
        return t["crop_jpg"] if t else None
