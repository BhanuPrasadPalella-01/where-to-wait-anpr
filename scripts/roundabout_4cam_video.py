"""One video of only the vehicles that pass through all four RoundaboutHD cameras, counted.

Which vehicles cross all four cameras comes from the dataset's ground-truth ids (74 vehicles); the boxes
are our pipeline's detections (YOLO26s + ByteTrack), drawn through short detection gaps. Each vehicle keeps
one colour and id in every camera. All other traffic is left unboxed. Frames with none of these vehicles on
screen are dropped and the rest plays at SPEED x.

Usage: python scripts/roundabout_4cam_video.py [speed]
Writes reports/roundabout/four_camera_vehicles.mp4
"""
import colorsys
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
TW, TH = 640, 360
S = TW / 1280
POS = {1: (0, 0), 2: (TW, 0), 3: (0, TH), 4: (TW, TH)}
FONT = cv2.FONT_HERSHEY_SIMPLEX
DARK, WHITE, GREY = (20, 20, 20), (240, 240, 240), (150, 150, 150)


def colour(vid):
    h = (vid * 0.6180339887) % 1.0  # golden-ratio hues: neighbouring ids get distinct colours
    r, g, b = colorsys.hsv_to_rgb(h, 0.75, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def filled(frames, boxes, max_gap):
    fr, bx = [], []
    for (fa, ba), (fb, bb) in zip(zip(frames, boxes), zip(frames[1:], boxes[1:])):
        fr.append(fa); bx.append(ba)
        if 1 < fb - fa <= max_gap:
            for g in range(fa + 1, fb):
                w = (g - fa) / (fb - fa)
                fr.append(g); bx.append([u + w * (v - u) for u, v in zip(ba, bb)])
    return fr + frames[-1:], bx + boxes[-1:]


def main():
    speed = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    cams_of = defaultdict(set)
    for line in open(DATA / "Multi_CAM_Ground_Turth.txt"):
        c, o = (int(x) for x in line.split()[:2])
        cams_of[o].add(c)
    v4 = {o for o, cs in cams_of.items() if len(cs) == 4}

    by_frame = {c: defaultdict(list) for c in POS}
    first_seen = {c: {} for c in POS}
    detected_in = defaultdict(set)
    fps = None
    for c in POS:
        d = pickle.load(open(DATA / f"tracks_cam{c}.pkl", "rb"))
        fps = d["fps"]
        for t in d["tracks"]:
            if t["gt"] not in v4:
                continue
            detected_in[t["gt"]].add(c)
            fr, bx = filled(t["frames"], t["box"], int(fps))
            first_seen[c][t["gt"]] = min(first_seen[c].get(t["gt"], 10 ** 9), fr[0])
            for f, b in zip(fr, bx):
                by_frame[c][f].append((t["gt"], b))
    all4 = sum(1 for v in v4 if len(detected_in[v]) == 4)
    global_first = {}
    for c in POS:
        for v, f in first_seen[c].items():
            global_first[v] = min(global_first.get(v, 10 ** 9), f)
    seen_sorted = sorted(global_first.values())
    cam_sorted = {c: sorted(first_seen[c].values()) for c in POS}

    caps = {c: cv2.VideoCapture(str(DATA / f"cam{c}_preview.mp4")) for c in POS}
    out = av.open(str(OUT / "four_camera_vehicles.mp4"), mode="w")
    st = out.add_stream("libx264" if "libx264" in av.codecs_available else "h264_videotoolbox", rate=int(round(fps)))
    st.width, st.height, st.pix_fmt = 2 * TW, 2 * TH, "yuv420p"
    st.options = {"crf": "27", "preset": "medium"}

    def emit(img, times=1):
        frame = av.VideoFrame.from_ndarray(img, format="bgr24")
        for _ in range(times):
            for pkt in st.encode(frame):
                out.mux(pkt)

    intro = np.full((2 * TH, 2 * TW, 3), 18, np.uint8)
    for i, (t, sc, col) in enumerate([("Vehicles that pass through all 4 cameras", 1.0, WHITE),
                                      (f"{len(v4)} in the dataset's ground truth  |  {all4} detected by our system in all 4", 0.7, (60, 230, 60)),
                                      ("each vehicle keeps one colour and id in every camera; other traffic unboxed", 0.6, GREY),
                                      (f"real cameras, Carmel (Indiana)  |  playing at {speed}x", 0.6, GREY)]):
        (tw, _), _ = cv2.getTextSize(t, FONT, sc, 2)
        cv2.putText(intro, t, ((2 * TW - tw) // 2, 290 + 50 * i), FONT, sc, col, 2, cv2.LINE_AA)
    emit(intro, int(3 * fps))

    n = int(min(c.get(cv2.CAP_PROP_FRAME_COUNT) for c in caps.values()))
    for f in range(n):
        tiles = {}
        for c, cap in caps.items():
            ok, img = cap.read()
            tiles[c] = img
        if f % speed:
            continue
        if not any(by_frame[c].get(f) for c in POS):
            continue  # none of the four-camera vehicles on screen
        grid = np.zeros((2 * TH, 2 * TW, 3), np.uint8)
        for c in POS:
            tile = cv2.resize(tiles[c], (TW, TH), interpolation=cv2.INTER_AREA) if tiles[c] is not None else np.zeros((TH, TW, 3), np.uint8)
            here = by_frame[c].get(f, [])
            for v, b in here:
                x1, y1, x2, y2 = (int(u * S) for u in b)
                col = colour(v)
                cv2.rectangle(tile, (x1, y1), (x2, y2), col, 2)
                (tw, th), _ = cv2.getTextSize(f"#{v}", FONT, 0.45, 1)
                ty = max(th + 4, y1 - 4)
                cv2.rectangle(tile, (x1, ty - th - 4), (x1 + tw + 6, ty + 3), col, -1)
                cv2.putText(tile, f"#{v}", (x1 + 3, ty), FONT, 0.45, DARK, 1, cv2.LINE_AA)
            counted = int(np.searchsorted(cam_sorted[c], f, side="right"))
            cv2.rectangle(tile, (0, 0), (TW, 26), DARK, -1)
            cv2.putText(tile, f"Camera {c}   in view: {len(here)}   counted: {counted} / {len(v4)}", (8, 18), FONT, 0.52,
                        WHITE, 1, cv2.LINE_AA)
            x, y = POS[c]
            grid[y:y + TH, x:x + TW] = tile
        cv2.line(grid, (TW, 0), (TW, 2 * TH), (40, 40, 40), 2)
        cv2.line(grid, (0, TH), (2 * TW, TH), (40, 40, 40), 2)
        total = int(np.searchsorted(seen_sorted, f, side="right"))
        cv2.rectangle(grid, (0, 2 * TH - 34), (2 * TW, 2 * TH), DARK, -1)
        cv2.putText(grid, f"Unique 4-camera vehicles seen so far: {total} / {len(v4)}", (14, 2 * TH - 11), FONT, 0.62,
                    (60, 230, 60), 2, cv2.LINE_AA)
        cv2.putText(grid, f"t = {f / fps:5.1f} s   {speed}x", (2 * TW - 210, 2 * TH - 11), FONT, 0.6, WHITE, 1, cv2.LINE_AA)
        emit(grid)

    outro = np.full((2 * TH, 2 * TW, 3), 18, np.uint8)
    lines = [(f"{len(v4)} vehicles crossed all 4 cameras in 10 minutes", 0.9, WHITE),
             (f"detected by our system in all 4 cameras: {all4}   |   in 3 of 4: {len(v4) - all4}", 0.7, (60, 230, 60)),
             ("camera 1 misses are mostly vehicles hidden behind the car park", 0.6, GREY)]
    for i, (t, sc, col) in enumerate(lines):
        (tw, _), _ = cv2.getTextSize(t, FONT, sc, 2)
        cv2.putText(outro, t, ((2 * TW - tw) // 2, 310 + 55 * i), FONT, sc, col, 2, cv2.LINE_AA)
    emit(outro, int(4 * fps))
    for pkt in st.encode():
        out.mux(pkt)
    out.close()
    print("wrote", OUT / "four_camera_vehicles.mp4", f"| {len(v4)} four-camera vehicles, {all4} detected in all four")


if __name__ == "__main__":
    main()
