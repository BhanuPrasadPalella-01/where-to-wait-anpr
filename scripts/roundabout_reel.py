"""Journey reel: one vehicle followed camera to camera, as our system picked it.

Reads reports/roundabout/eval.json and the per-camera tracks and 960x540 preview videos from the full
pass. For each camera of the journey: a title card (camera, candidates checked, match score, ground-truth
check), then the vehicle's pass with the suspect in a thick green box and a zoomed inset, other vehicles in
thin grey. Ends with a summary card of the suspect as each camera saw it. Encoded as H.264 (yuv420p) so it
plays in browsers, QuickTime and GitHub.

Usage: python scripts/roundabout_reel.py VEHICLE_ID [method]
Writes reports/roundabout/reel_v<ID>.mp4 and reel_v<ID>_strip.jpg.
"""
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import av
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "roundabout"
OUT = ROOT / "reports" / "roundabout"
W, H = 960, 540
S = W / 1280  # tracks are in 1280-wide pixels, previews are 960 wide
PAD_S = 1.5
GREEN, GREY, DARK, WHITE = (60, 230, 60), (170, 170, 170), (24, 24, 24), (245, 245, 245)
FONT = cv2.FONT_HERSHEY_SIMPLEX


class H264:
    def __init__(self, path, fps):
        self.c = av.open(str(path), mode="w")
        codec = "libx264" if "libx264" in av.codecs_available else "h264_videotoolbox"
        self.s = self.c.add_stream(codec, rate=int(round(fps)))
        self.s.width, self.s.height, self.s.pix_fmt = W, H, "yuv420p"
        if codec == "libx264":
            self.s.options = {"crf": "20", "preset": "medium"}

    def write(self, bgr, times=1):
        frame = av.VideoFrame.from_ndarray(bgr, format="bgr24")
        for _ in range(times):
            for pkt in self.s.encode(frame):
                self.c.mux(pkt)

    def close(self):
        for pkt in self.s.encode():
            self.c.mux(pkt)
        self.c.close()


def label(img, text, x, y, col, scale=0.7, thick=2):
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thick)
    y = max(th + 8, y)
    cv2.rectangle(img, (x, y - th - 8), (x + tw + 10, y + 4), col, -1)
    cv2.putText(img, text, (x + 5, y - 2), FONT, scale, DARK, thick, cv2.LINE_AA)


def card(lines, sub=None):
    img = np.full((H, W, 3), 18, np.uint8)
    y = 200 if sub is None else 150
    for i, (t, sc, col) in enumerate(lines):
        (tw, _), _ = cv2.getTextSize(t, FONT, sc, 2)
        cv2.putText(img, t, ((W - tw) // 2, y + i * 56), FONT, sc, col, 2, cv2.LINE_AA)
    if sub is not None:
        img[H - sub.shape[0] - 30:H - 30, (W - sub.shape[1]) // 2:(W + sub.shape[1]) // 2] = sub
    return img


def main():
    vid = int(sys.argv[1])
    method = sys.argv[2] if len(sys.argv) > 2 else "appearance + travel time + direction"
    row = next(r for r in json.loads((OUT / "eval.json").read_text())[method]["rows"] if r["vehicle"] == vid)
    pool = {}
    for p in DATA.glob("tracks_cam*.pkl"):
        d = pickle.load(open(p, "rb"))
        pool[d["cam"]] = {"fps": d["fps"], "tracks": {t["track"]: t for t in d["tracks"]}}
    picks = [(row["start"]["cam"], row["start"]["track"], None, True, None)]
    picks += [(s["to"], s["pick_track"], s["score"], s["correct"], s["candidates"]) for s in row["steps"]
              if s["pick_track"] is not None]
    fps = pool[picks[0][0]]["fps"]
    out = H264(OUT / f"reel_v{vid}.mp4", fps)
    out.write(card([("Following one vehicle across 4 real traffic cameras", 0.85, WHITE),
                    ("RoundaboutHD, Carmel (Indiana)  |  plates unreadable by design", 0.6, GREY),
                    ("looks + road-graph travel time + direction", 0.6, GREY)]), int(2.5 * fps))
    crops = []
    for k, (cam, tid, sc, ok, ncand) in enumerate(picks):
        tr = pool[cam]["tracks"][tid]
        if k == 0:
            title = [(f"Camera {cam}: suspect marked by the officer", 0.85, GREEN)]
        else:
            verdict = "matches ground truth" if ok else "WRONG vehicle"
            title = [(f"Camera {cam}: {ncand} candidates checked", 0.85, WHITE),
                     (f"best match {sc:.2f}  ->  {verdict}", 0.8, GREEN if ok else (60, 60, 230))]
        out.write(card(title), int(1.6 * fps))
        f0, f1 = max(0, tr["frames"][0] - int(PAD_S * fps)), tr["frames"][-1] + int(PAD_S * fps)
        boxes = defaultdict(list)
        for t in pool[cam]["tracks"].values():
            for f, b in zip(t["frames"], t["box"]):
                if f0 <= f <= f1:
                    boxes[f].append((t["track"], b))
        cap = cv2.VideoCapture(str(DATA / f"cam{cam}_preview.mp4"))
        cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
        last_inset, corner = None, (W - 252, 50)
        for f in range(f0, f1 + 1):
            ok_read, img = cap.read()
            if not ok_read:
                break
            raw = img.copy()
            for t_id, b in boxes.get(f, []):
                if t_id != tid:
                    x1, y1, x2, y2 = (int(v * S) for v in b)
                    cv2.rectangle(img, (x1, y1), (x2, y2), GREY, 1)
            for t_id, b in boxes.get(f, []):
                if t_id == tid:
                    x1, y1, x2, y2 = (int(v * S) for v in b)
                    cv2.rectangle(img, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), GREEN, 4)
                    label(img, "SUSPECT" if k == 0 else f"SUSPECT  {sc:.2f}", x1 - 3, y1 - 8, GREEN)
                    crop = raw[max(0, y1):y2, max(0, x1):x2]
                    if crop.size:
                        last_inset = cv2.resize(crop, (240, int(240 * crop.shape[0] / max(1, crop.shape[1]))))[:180]
                        # put the zoomed inset in the corner farthest from the suspect, so it never hides it
                        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                        corners = [(12, 50), (W - 252, 50), (12, H - 192), (W - 252, H - 192)]
                        corner = max(corners, key=lambda c: (c[0] + 120 - cx) ** 2 + (c[1] + 90 - cy) ** 2)
            if last_inset is not None:
                ih, iw = last_inset.shape[:2]
                ix, iy = corner
                cv2.rectangle(img, (ix - 4, iy - 4), (ix + iw + 4, iy + ih + 4), GREEN, 3)
                img[iy:iy + ih, ix:ix + iw] = last_inset
                cv2.putText(img, "zoom", (ix + 4, iy + 18), FONT, 0.5, GREEN, 2, cv2.LINE_AA)
            cv2.rectangle(img, (0, 0), (W, 38), DARK, -1)
            head = f"Camera {cam}   t = {f / fps:5.1f} s   camera {k + 1} of {len(picks)}"
            cv2.putText(img, head, (12, 26), FONT, 0.7, WHITE, 2, cv2.LINE_AA)
            out.write(img)
        c = cv2.imdecode(np.frombuffer(tr["crop_jpg"], np.uint8), cv2.IMREAD_COLOR)
        c = cv2.resize(c, (200, int(200 * c.shape[0] / c.shape[1])))[:160]
        c = cv2.copyMakeBorder(c, 30, 160 - c.shape[0], 4, 4, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(c, f"Cam {cam}" + ("" if k == 0 else f"  {sc:.2f}"), (8, 21), FONT, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
        crops.append(c)
    strip = np.hstack(crops)
    cv2.imwrite(str(OUT / f"reel_v{vid}_strip.jpg"), strip)
    n_ok = sum(p[3] for p in picks[1:])
    sub = strip if strip.shape[1] <= W - 40 else cv2.resize(strip, (W - 40, int(strip.shape[0] * (W - 40) / strip.shape[1])))
    out.write(card([(f"Same vehicle, {len(picks)} cameras: {n_ok} of {len(picks) - 1} hand-offs correct", 0.8, GREEN),
                    ("checked against the dataset's ground-truth ids", 0.6, GREY)], sub), int(3 * fps))
    out.close()
    print("reel:", OUT / f"reel_v{vid}.mp4", "| cameras", [p[0] for p in picks], "| correct", [p[3] for p in picks])


if __name__ == "__main__":
    main()
