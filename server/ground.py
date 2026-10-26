"""Where each tracked vehicle was on the ground, and which way it left each roundabout.

Projection: every box's bottom-centre (where the tyres meet the road) goes through the camera's
fitted calibration that ships with RoundaboutHD (position, height, tilt, heading, lens distortion;
the cameratransform format) onto the road plane, giving metres east/north of the camera.

Checked against OpenStreetMap: the vehicles going round each roundabout land on the OSM roundabout to
within about 10 m, so the dataset's calibration is used as it is.

Exits: from where a vehicle leaves the view and which way it is moving, a k-nearest-neighbour vote over
the even-numbered ground-truth vehicles says whether it drove on towards the next camera east, the
next camera west, or turned off onto a side road ("off"); the same for where it came in from.
Results are cached in data/roundabout/ground.pkl.
"""
from __future__ import annotations

import math
import pickle
from collections import Counter, defaultdict

import numpy as np

K = 7
STEP = 3                 # project every 3rd frame (5 points a second)


def _metres(lon, lat, lat0, lon0):
    return (lon - lon0) * 111320 * math.cos(math.radians(lat0)), (lat - lat0) * 110540


class Ground:
    def __init__(self, idx, data_dir):
        self.idx, self.cache = idx, data_dir / "ground.pkl"
        if self.cache.exists():
            d = pickle.load(open(self.cache, "rb"))
        else:
            d = self._build(data_dir)
            pickle.dump(d, open(self.cache, "wb"))
        self.xy = d["xy"]
        self._train()

    # ------------------------------------------------------------------ projection
    def _build(self, data_dir):
        import cameratransform as ct
        raw = {}
        for c, tracks in self.idx.tracks.items():
            cam = ct.load_camera(str(data_dir / f"cam0{c}_fitted_cam.json"))
            sx = cam.image_width_px / 1280  # track boxes are in 1280-wide pixels
            for t in tracks.values():
                ii = list(range(0, len(t["box"]), STEP))
                if ii[-1] != len(t["box"]) - 1:
                    ii.append(len(t["box"]) - 1)
                pts = np.array([[(t["box"][i][0] + t["box"][i][2]) / 2 * sx, t["box"][i][3] * sx] for i in ii])
                xy = cam.spaceFromImage(pts)[:, :2]
                ok = np.isfinite(xy).all(1) & (np.linalg.norm(np.nan_to_num(xy, nan=1e9), axis=1) < 150)
                raw[(c, t["track"])] = ([t["frames"][i] for i, k in zip(ii, ok) if k], xy[ok])
        return {"xy": raw}

    # ------------------------------------------------------------------ exits
    def _feat(self, c, tr, end):
        fr, xy = self.xy.get((c, tr), ([], np.zeros((0, 2))))
        if len(xy) < 3:
            return None
        if end:
            p, d = xy[-1], xy[-1] - xy[max(0, len(xy) - 6)]
        else:
            p, d = xy[0], xy[min(len(xy) - 1, 5)] - xy[0]
        d = d / (np.linalg.norm(d) + 1e-6)
        return np.array([p[0] / 30, p[1] / 30, d[0], d[1]])

    def _train(self):
        J = defaultdict(list)
        for tracks in self.idx.tracks.values():
            for t in tracks.values():
                if t["gt"] is not None and t["gt"] % 2 == 0:   # calibration vehicles only
                    J[t["gt"]].append(t)
        self.train = defaultdict(list)
        for ts in J.values():
            ts.sort(key=lambda t: t["t_first"])
            for i, t in enumerate(ts):
                c = t["cam"]
                if not (40 < t["t_first"] and t["t_last"] < 560):
                    continue  # its neighbour sighting may fall outside the 10-minute recording
                nxt = ts[i + 1]["cam"] if i + 1 < len(ts) else None
                prv = ts[i - 1]["cam"] if i else None
                for end, other in ((1, nxt), (0, prv)):
                    lab = "E" if other == c + 1 else "W" if other == c - 1 else "off"
                    x = self._feat(c, t["track"], end)
                    if x is not None:
                        self.train[(c, end)].append((x, lab))
        self.X = {k: np.array([x for x, _ in v]) for k, v in self.train.items()}

    def arms(self, c, tr, end):
        """{'E', 'W', 'off'} -> share of the K nearest calibration vehicles. end=1: where it went; 0: came from."""
        x = self._feat(c, tr, end)
        if x is None or (c, end) not in self.X:
            return {"E": 1 / 3, "W": 1 / 3, "off": 1 / 3}
        nn = np.argsort(np.linalg.norm(self.X[(c, end)] - x, axis=1))[:K]
        votes = Counter(self.train[(c, end)][i][1] for i in nn)
        return {k: votes.get(k, 0) / K for k in ("E", "W", "off")}

    def lonlat(self, c, tr):
        """The track's path on the map, as (frame, lon, lat) points."""
        fr, xy = self.xy.get((c, tr), ([], np.zeros((0, 2))))
        lat0, lon0 = self.idx.gps[c]["gps_lat"], self.idx.gps[c]["gps_lon"]
        return [[int(f), lon0 + x / (111320 * math.cos(math.radians(lat0))), lat0 + y / 110540] for f, (x, y) in zip(fr, xy)]
