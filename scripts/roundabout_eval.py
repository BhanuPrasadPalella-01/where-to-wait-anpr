"""Score camera-to-camera vehicle hand-off on all of RoundaboutHD against its ground truth.

For every ground-truth vehicle seen by 2+ cameras: the officer marks it once in its first camera (its
longest track there), then at each next camera the system picks one track as the same vehicle.
Errors carry forward (the next search starts from whatever was picked), as they would live.

Methods
  appearance  most similar track (ResNet-50 embedding 0.7 + colour histogram 0.3) among all tracks of
              that camera that start after the last sighting
  + travel    the same, among tracks whose arrival fits the road graph: at least (1 - sigma) of the
              minimum driving time (Dijkstra on OpenStreetMap) and at most MAX_GAP_S after the last sighting
  + tight     arrival within [0.4, 1.3] x the road-graph minimum, ranked with a time prior; the candidate
              must move the way vehicles from that camera enter this one (entry direction)

Calibration (time prior, entry directions) is learned from even-numbered ground-truth vehicles only;
every method is reported on the odd-numbered vehicles, which calibration never sees.

Reports hand-off accuracy (top-1, top-5), full-journey success, and how many candidates each method
had to choose from. Writes reports/roundabout/eval.json and eval.md.
"""
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from planner.roadgraph import RoadGraph, shortest_path  # noqa: E402

DATA = ROOT / "data" / "roundabout"
OUT = ROOT / "reports" / "roundabout"
SIGMA, MAX_GAP_S, EARLY_S = 0.5, 240.0, 20.0
W_EMB, W_HIST = 0.7, 0.3


def score(q_emb, q_hist, q_type, t):
    s = W_EMB * float(q_emb @ t["emb"]) + W_HIST * float(cv2.compareHist(q_hist, t["hist"], cv2.HISTCMP_CORREL))
    return s - (0.05 if t["type"] != q_type else 0.0)


def motion(t):
    b0, b1 = t["box"][0], t["box"][-1]
    d = np.array([(b1[0] + b1[2]) / 2 - (b0[0] + b0[2]) / 2, (b1[1] + b1[3]) / 2 - (b0[1] + b0[3]) / 2])
    return d / (np.linalg.norm(d) + 1e-9)


def calibrate(J, tau):
    """Per camera pair: entry direction and arrival-time ratio (gap / road-graph minimum)."""
    enter, ratio = defaultdict(list), defaultdict(list)
    for v, path in J.items():
        if v % 2:
            continue
        for a, b in zip(path, path[1:]):
            enter[(a["cam"], b["cam"])].append(motion(b))
            ratio[(a["cam"], b["cam"])].append((b["t_first"] - a["t_last"]) / tau[(a["cam"], b["cam"])])
    cal = {}
    for k in enter:
        m = np.mean(enter[k], axis=0)
        r = np.array(ratio[k])
        cal[k] = {"dir": m / (np.linalg.norm(m) + 1e-9), "mu": float(np.median(r)), "sd": float(max(0.08, r.std())),
                  "n": len(r)}
    return cal


def journeys(pool):
    """Ground-truth journeys as seen by our tracks: vehicle -> cameras in time order."""
    seen = defaultdict(dict)
    for cam, tracks in pool.items():
        for t in tracks:
            if t["gt"] is None:
                continue
            cur = seen[t["gt"]].get(cam)
            if cur is None or len(t["frames"]) > len(cur["frames"]):
                seen[t["gt"]][cam] = t
    return {v: sorted(d.values(), key=lambda t: t["t_first"]) for v, d in seen.items() if len(d) >= 2}


def run(pool, tau, gated, cal=None, test_only=True):
    rows = []
    for vid, path in sorted(journeys(pool).items()):
        if test_only and vid % 2 == 0:
            continue
        q = path[0]
        q_emb, q_hist, prev = q["emb"], q["hist"], q
        steps = []
        for nxt in path[1:]:
            cam = nxt["cam"]
            T = tau[(prev["cam"], cam)]
            if gated == "tight":
                c = cal.get((prev["cam"], cam))
                cands = [t for t in pool[cam] if 0.4 * T <= t["t_first"] - prev["t_last"] <= 1.3 * T]
                if c is not None and c["n"] >= 5:
                    cands = [t for t in cands if float(motion(t) @ c["dir"]) > 0.5]

                def total(t, c=c):
                    s = score(q_emb, q_hist, prev["type"], t)
                    if c is not None and c["n"] >= 5:
                        z = ((t["t_first"] - prev["t_last"]) / T - c["mu"]) / c["sd"]
                        s -= 0.05 * z * z  # Gaussian time prior (log-likelihood, scaled)
                    return s
            elif gated:
                need = (1 - SIGMA) * T
                cands = [t for t in pool[cam] if need - EARLY_S <= t["t_first"] - prev["t_last"] <= MAX_GAP_S]
                total = lambda t: score(q_emb, q_hist, prev["type"], t)  # noqa: E731
            else:
                cands = [t for t in pool[cam] if t["t_first"] > prev["t_first"]]
                total = lambda t: score(q_emb, q_hist, prev["type"], t)  # noqa: E731
            ranked = sorted(cands, key=lambda t: -total(t))
            rank = next((i + 1 for i, t in enumerate(ranked) if t["gt"] == vid), None)
            pick = ranked[0] if ranked else None
            steps.append({"from": prev["cam"], "to": cam, "candidates": len(ranked), "rank": rank,
                          "correct": bool(pick and pick["gt"] == vid),
                          "pick_track": pick["track"] if pick else None,
                          "score": round(total(pick), 3) if pick else None,
                          "runner_up": round(total(ranked[1]), 3) if len(ranked) > 1 else None})
            if pick is None:
                break
            q_emb = (q_emb + pick["emb"]) / np.linalg.norm(q_emb + pick["emb"])
            q_hist = (q_hist + pick["hist"]) / 2
            prev = pick
        rows.append({"vehicle": vid, "cameras": len(path), "start": {"cam": q["cam"], "track": q["track"]}, "steps": steps,
                     "journey_ok": all(s["correct"] for s in steps) and len(steps) == len(path) - 1})
    return rows


def summary(rows):
    steps = [s for r in rows for s in r["steps"]]
    four = [r for r in rows if r["cameras"] == 4]
    return {"vehicles": len(rows), "handoffs": len(steps),
            "top1": round(sum(s["correct"] for s in steps) / len(steps), 3),
            "top5": round(sum(1 for s in steps if s["rank"] and s["rank"] <= 5) / len(steps), 3),
            "mean_candidates": round(sum(s["candidates"] for s in steps) / len(steps), 1),
            "journeys_all_correct": round(sum(r["journey_ok"] for r in rows) / len(rows), 3),
            "four_camera_vehicles": len(four),
            "four_camera_journeys_all_correct": round(sum(r["journey_ok"] for r in four) / max(1, len(four)), 3)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pool = {}
    for p in sorted(DATA.glob("tracks_cam*.pkl")):
        d = pickle.load(open(p, "rb"))
        pool[d["cam"]] = d["tracks"]
    if len(pool) < 2:
        sys.exit("need tracks from at least two cameras")
    g = RoadGraph.load(ROOT / "data" / "carmel_graph.json")
    node = {int(k): v for k, v in json.loads((DATA / "camera_nodes.json").read_text()).items()}
    tau = {(a, b): shortest_path(g, node[a], node[b])[0] for a in node for b in node if a != b}
    cal = calibrate(journeys(pool), tau)
    res = {}
    for name, gated in (("appearance only", False), ("appearance + travel time", True),
                        ("appearance + travel time + direction", "tight")):
        rows = run(pool, tau, gated, cal)
        res[name] = {"summary": summary(rows), "rows": rows}
        print(name, json.dumps(res[name]["summary"]))
    (OUT / "eval.json").write_text(json.dumps(res, indent=2))
    md = ["Test set: odd-numbered ground-truth vehicles (calibration used even-numbered ones only).", "",
          "| Method | Hand-offs | Top-1 correct | Top-5 | Candidates per hand-off | Full journeys right | 4-camera journeys right |",
          "|---|---|---|---|---|---|---|"]
    for name, r in res.items():
        s = r["summary"]
        md.append(f"| {name} | {s['handoffs']} | {100 * s['top1']:.1f}% | {100 * s['top5']:.1f}% | {s['mean_candidates']} | "
                  f"{100 * s['journeys_all_correct']:.1f}% of {s['vehicles']} | {100 * s['four_camera_journeys_all_correct']:.1f}% of {s['four_camera_vehicles']} |")
    (OUT / "eval.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
