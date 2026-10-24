"""Journey reel: one vehicle followed camera to camera, as our system picked it.

Reads reports/roundabout/eval.json (method "appearance + travel time") and the per-camera tracks and
960x540 preview videos from the full pass. For each camera of the journey it plays the picked
vehicle's pass with that vehicle boxed in green (its match score shown) and other vehicles in grey,
then a strip of the four plate-free crops side by side.

Usage: python scripts/roundabout_reel.py VEHICLE_ID
Writes reports/roundabout/reel_v<ID>.mp4 and reel_v<ID>_strip.jpg.
"""
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "roundabout"
OUT = ROOT / "reports" / "roundabout"
S = 960 / 1280  # tracks are in 1280-wide pixels, previews are 960 wide
PAD_S = 1.5


def main():
    vid = int(sys.argv[1])
    ev = json.loads((OUT / "eval.json").read_text())[sys.argv[2] if len(sys.argv) > 2 else "appearance + travel time + direction"]["rows"]
    row = next(r for r in ev if r["vehicle"] == vid)
    pool = {}
    for p in DATA.glob("tracks_cam*.pkl"):
        d = pickle.load(open(p, "rb"))
        pool[d["cam"]] = {"fps": d["fps"], "tracks": {t["track"]: t for t in d["tracks"]}}
    picks = [(row["start"]["cam"], row["start"]["track"], None, True)]
    picks += [(s["to"], s["pick_track"], s["score"], s["correct"]) for s in row["steps"] if s["pick_track"] is not None]

    writer, crops = None, []
    for k, (cam, tid, sc, ok) in enumerate(picks):
        fps = pool[cam]["fps"]
        tr = pool[cam]["tracks"][tid]
        f0, f1 = max(0, tr["frames"][0] - int(PAD_S * fps)), tr["frames"][-1] + int(PAD_S * fps)
        others = defaultdict(list)
        for t in pool[cam]["tracks"].values():
            for f, b in zip(t["frames"], t["box"]):
                if f0 <= f <= f1:
                    others[f].append((t["track"], b))
        cap = cv2.VideoCapture(str(DATA / f"cam{cam}_preview.mp4"))
        cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
        for f in range(f0, f1 + 1):
            ok_read, img = cap.read()
            if not ok_read:
                break
            for t_id, b in others.get(f, []):
                x1, y1, x2, y2 = (int(v * S) for v in b)
                hit = t_id == tid
                col = (60, 220, 60) if hit else (160, 160, 160)
                cv2.rectangle(img, (x1, y1), (x2, y2), col, 3 if hit else 1)
                if hit:
                    label = "SUSPECT (marked by officer)" if k == 0 else f"SUSPECT  match {sc:.2f}"
                    cv2.putText(img, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
            cv2.rectangle(img, (0, 0), (960, 34), (20, 20, 20), -1)
            head = f"Camera {cam}  |  t = {f / fps:5.1f} s  |  camera {k + 1} of {len(picks)}"
            if k:
                head += "  |  " + ("matches ground truth" if ok else "WRONG vehicle")
            cv2.putText(img, head, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (240, 240, 240), 2)
            if writer is None:
                writer = cv2.VideoWriter(str(OUT / f"reel_v{vid}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (960, 540))
            writer.write(img)
        crop = cv2.imdecode(np.frombuffer(tr["crop_jpg"], np.uint8), cv2.IMREAD_COLOR)
        crop = cv2.resize(crop, (220, int(220 * crop.shape[0] / crop.shape[1])))
        crop = cv2.copyMakeBorder(crop, 30, max(0, 190 - crop.shape[0]), 4, 4, cv2.BORDER_CONSTANT, value=(255, 255, 255))[:220]
        cv2.putText(crop, f"Cam {cam}" + ("" if k == 0 else f"  {sc:.2f}"), (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        crops.append(crop)
    if writer:
        writer.release()
    cv2.imwrite(str(OUT / f"reel_v{vid}_strip.jpg"), np.hstack(crops))
    print("reel:", OUT / f"reel_v{vid}.mp4", "| cameras", [p[0] for p in picks], "| correct", [p[3] for p in picks])


if __name__ == "__main__":
    main()
