"""Follow one vehicle across the four RoundaboutHD cameras with our own pipeline, then check it
against the dataset's ground truth.

Per camera clip: YOLO26s detects vehicles, ByteTrack keeps one id per vehicle pass, and each track gets
an appearance fingerprint (ResNet-50 image embedding + HSV colour histogram) from its largest crops.
The suspect is marked once in its first camera (the officer's click). In each next camera, only tracks
that arrive in a physically possible time after the last sighting are candidates (minimum driving time
from the OpenStreetMap road graph, Dijkstra), and the most similar one is taken. Plates are not
readable in this dataset (the cameras were set up so), so this is the appearance + travel-time
fallback of the design.

Usage: python scripts/roundabout_track.py VEHICLE_ID
Writes reports/roundabout/v<ID>_result.json, annotated clips and a journey reel.
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import models
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from planner.roadgraph import RoadGraph, shortest_path  # noqa: E402

DATA = ROOT / "data" / "roundabout"
CLIPS = DATA / "clips"
OUT = ROOT / "reports" / "roundabout"
GT_SCALE = 1280 / 3840            # ground-truth boxes are in 4K pixels, clips are 1280 wide
VEHICLE = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
SIGMA = 0.5                       # slack below the free-flow minimum driving time
W_EMB, W_HIST = 0.7, 0.3


def load_gt():
    gt = defaultdict(lambda: defaultdict(dict))  # cam -> frame -> obj -> box (1280 px)
    for line in open(DATA / "Multi_CAM_Ground_Turth.txt"):
        p = line.split()
        c, o, f = int(p[0]), int(p[1]), int(p[2])
        x, y, w, h = (float(v) * GT_SCALE for v in p[3:7])
        gt[c][f][o] = (x, y, x + w, y + h)
    return gt


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


class Embedder:
    def __init__(self):
        net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        net.fc = torch.nn.Identity()
        self.net = net.eval().to("mps" if torch.backends.mps.is_available() else "cpu")
        self.dev = next(self.net.parameters()).device
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    @torch.no_grad()
    def __call__(self, crops):
        x = np.stack([cv2.resize(cv2.cvtColor(c, cv2.COLOR_BGR2RGB), (224, 224)) for c in crops]).astype(np.float32) / 255
        x = (torch.from_numpy(x).permute(0, 3, 1, 2) - self.mean) / self.std
        e = self.net(x.to(self.dev)).cpu().numpy()
        e = e.mean(axis=0)
        return e / (np.linalg.norm(e) + 1e-9)


def colour_hist(crops):
    hs = []
    for c in crops:
        h, w = c.shape[:2]
        core = cv2.cvtColor(c[int(.2 * h):int(.8 * h), int(.15 * w):int(.85 * w)], cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([core], [0, 1], None, [18, 8], [0, 180, 0, 256])
        hs.append(cv2.normalize(hist, None).flatten())
    return np.mean(hs, axis=0).astype(np.float32)


def tracks_in_clip(det, clip_path, meta, cam, gt):
    """ByteTrack tracks of one clip with times, boxes, type, best crops and their ground-truth id."""
    cap = cv2.VideoCapture(str(clip_path))
    fps = meta["fps"]
    tr = defaultdict(lambda: {"t": [], "box": [], "cls": [], "crops": [], "gt": Counter(), "frames": []})
    det.predictor = None  # fresh tracker state per clip
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = meta["t0"] + i / fps
        gt_frame = gt[cam].get(int(round(t * 15)), {})
        r = det.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.3, classes=list(VEHICLE), verbose=False,
                      device="mps")[0]
        if r.boxes.id is not None:
            for b, tid, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.id.int().cpu().tolist(), r.boxes.cls.int().cpu().tolist()):
                x1, y1, x2, y2 = b
                if (x2 - x1) * (y2 - y1) < 900:
                    continue
                d = tr[tid]
                d["t"].append(t); d["box"].append(b.tolist()); d["cls"].append(c); d["frames"].append(i)
                crop = frame[int(y1):int(y2), int(x1):int(x2)].copy()
                d["crops"].append(((x2 - x1) * (y2 - y1), crop))
                d["crops"] = sorted(d["crops"], key=lambda z: -z[0])[:6]
                best = max(gt_frame.items(), key=lambda kv: iou(b, kv[1]), default=None)
                if best and iou(b, best[1]) > 0.3:
                    d["gt"][best[0]] += 1
        i += 1
    out = []
    for tid, d in tr.items():
        if len(d["t"]) < 5:
            continue
        out.append({"cam": cam, "clip": clip_path.name, "track": tid, "t_first": min(d["t"]), "t_last": max(d["t"]),
                    "type": VEHICLE[Counter(d["cls"]).most_common(1)[0][0]], "crops": [c for _, c in d["crops"]],
                    "gt": d["gt"].most_common(1)[0][0] if d["gt"] else None, "box": d["box"], "frames": d["frames"]})
    return out


def main():
    vid = int(sys.argv[1])
    OUT.mkdir(parents=True, exist_ok=True)
    gt = load_gt()
    g = RoadGraph.load(ROOT / "data" / "carmel_graph.json")
    node = {int(k): v for k, v in json.loads((DATA / "camera_nodes.json").read_text()).items()}
    tau = {(a, b): shortest_path(g, node[a], node[b])[0] for a in node for b in node if a != b}

    det, emb = YOLO(str(ROOT / "weights" / "yolo26s.pt")), Embedder()
    # every clip of every camera is searched: the suspect's own pass plus other vehicles' passes (distractors)
    pool = defaultdict(list)
    for meta_path in sorted(CLIPS.glob("v*_cam*.json")):
        meta = json.loads(meta_path.read_text())
        for t in tracks_in_clip(det, CLIPS / meta["file"], meta, meta["cam"], gt):
            t["emb"], t["hist"] = emb(t["crops"]), colour_hist(t["crops"])
            pool[t["cam"]].append(t)
    journey_cams = sorted({json.loads(p.read_text())["cam"] for p in CLIPS.glob(f"v{vid}_cam*.json")},
                          key=lambda c: json.loads((CLIPS / f"v{vid}_cam{c}.json").read_text())["t0"])
    print(f"vehicle {vid}: journey order {journey_cams}; tracks per camera "
          + ", ".join(f"cam{c}: {len(v)}" for c, v in sorted(pool.items())), flush=True)

    # 1. the officer marks the suspect in its first camera (we use the ground truth to place that click)
    first = journey_cams[0]
    marked = max((t for t in pool[first] if t["gt"] == vid and t["clip"] == f"v{vid}_cam{first}.mp4"),
                 key=lambda t: len(t["frames"]), default=None)
    if marked is None:
        sys.exit("suspect not found by the detector in its first camera")
    steps = [{"cam": first, "track": marked["track"], "clip": marked["clip"], "t": marked["t_first"],
              "type": marked["type"], "score": 1.0, "correct": True, "candidates": 1, "rank_of_truth": 1}]
    q_emb, q_hist, prev = marked["emb"], marked["hist"], marked
    # 2. hand-off camera to camera
    for cam in journey_cams[1:]:
        need = (1 - SIGMA) * tau[(prev["cam"], cam)]
        cands = [t for t in pool[cam] if t["t_first"] - prev["t_last"] >= need - 20 and t["t_first"] > prev["t_first"]]
        scored = []
        for t in cands:
            s = W_EMB * float(q_emb @ t["emb"]) + W_HIST * float(cv2.compareHist(q_hist, t["hist"], cv2.HISTCMP_CORREL))
            if t["type"] != prev["type"]:
                s -= 0.05
            scored.append((s, t))
        scored.sort(key=lambda x: -x[0])
        if not scored:
            steps.append({"cam": cam, "track": None, "correct": False, "candidates": 0})
            break
        s, best = scored[0]
        truth_rank = next((i + 1 for i, (_, t) in enumerate(scored) if t["gt"] == vid), None)
        steps.append({"cam": cam, "track": best["track"], "clip": best["clip"], "t": best["t_first"], "type": best["type"],
                      "score": round(s, 3), "correct": best["gt"] == vid, "candidates": len(scored),
                      "rank_of_truth": truth_rank, "min_drive_s": round(tau[(prev["cam"], cam)]),
                      "gap_s": round(best["t_first"] - prev["t_last"], 1),
                      "runner_up": round(scored[1][0], 3) if len(scored) > 1 else None})
        print(f"  cam{prev['cam']} -> cam{cam}: {len(scored)} candidates, picked track {best['track']} "
              f"(score {s:.3f}) -> {'CORRECT' if best['gt'] == vid else 'WRONG'} (truth ranked {truth_rank})", flush=True)
        # the newest view of the suspect joins the query, so the fingerprint follows changing angles
        q_emb = (q_emb + best["emb"]) / np.linalg.norm(q_emb + best["emb"])
        q_hist = (q_hist + best["hist"]) / 2
        prev = best

    res = {"vehicle": vid, "steps": steps, "camera_nodes": node,
           "min_drive_s": {f"{a}-{b}": round(v) for (a, b), v in tau.items()}}
    (OUT / f"v{vid}_result.json").write_text(json.dumps(res, indent=2))
    render(vid, steps, pool)


def render(vid, steps, pool):
    """Annotated clip per step (suspect in green, other vehicles grey) and one journey reel."""
    reel = None
    for k, st in enumerate(steps):
        if st.get("track") is None:
            continue
        meta = json.loads((CLIPS / st["clip"].replace(".mp4", ".json")).read_text())
        tracks = [t for t in pool[st["cam"]] if t["clip"] == st["clip"]]
        by_frame = defaultdict(list)
        for t in tracks:
            for f, b in zip(t["frames"], t["box"]):
                by_frame[f].append((t, b))
        cap = cv2.VideoCapture(str(CLIPS / st["clip"]))
        i = 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            for t, b in by_frame.get(i, []):
                x1, y1, x2, y2 = (int(v) for v in b)
                hit = t["track"] == st["track"]
                col = (60, 220, 60) if hit else (150, 150, 150)
                cv2.rectangle(img, (x1, y1), (x2, y2), col, 3 if hit else 1)
                if hit:
                    label = "SUSPECT (marked)" if k == 0 else f"SUSPECT match {st['score']:.2f}"
                    cv2.putText(img, label, (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
            t_now = meta["t0"] + i / meta["fps"]
            bar = f"Camera {st['cam']}  |  t = {t_now:5.1f} s  |  step {k + 1} of {len(steps)}"
            if k:
                bar += f"  |  {st['candidates']} candidates, travel-time check passed"
            cv2.rectangle(img, (0, 0), (img.shape[1], 36), (20, 20, 20), -1)
            cv2.putText(img, bar, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2)
            if reel is None:
                reel = cv2.VideoWriter(str(OUT / f"v{vid}_journey.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                       meta["fps"], (img.shape[1], img.shape[0]))
            reel.write(img)
            if i == len(by_frame) // 2 or (i == 0 and False):
                pass
            i += 1
        # a still of the best moment for slides
        best_f = max((f for t in tracks if t["track"] == st["track"] for f in t["frames"]), default=0)
        cap = cv2.VideoCapture(str(CLIPS / st["clip"]))
        tr = next(t for t in tracks if t["track"] == st["track"])
        mid = tr["frames"][len(tr["frames"]) // 2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
        ok, img = cap.read()
        if ok:
            b = tr["box"][len(tr["frames"]) // 2]
            cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (60, 220, 60), 4)
            cv2.imwrite(str(OUT / f"v{vid}_step{k + 1}_cam{st['cam']}.jpg"), img)
    if reel:
        reel.release()


if __name__ == "__main__":
    main()
