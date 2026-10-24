"""Full 10-minute video of all four RoundaboutHD cameras with every tracked vehicle boxed.

A synchronized 2x2 grid (each camera 640x360 inside a 1280x720 frame). Every vehicle the pipeline tracked
gets a box coloured by type with its track id; each camera shows how many vehicles are in view now and
how many it has counted so far. Encoded as H.264 so it plays anywhere.

Usage: python scripts/roundabout_fullvideo.py [seconds]   (default: the full 600 s)
Writes reports/roundabout/all_cameras_tracked.mp4
"""
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
TW, TH = 640, 360            # one camera tile
S = TW / 1280                # track boxes are in 1280-wide pixels
COLOUR = {"car": (80, 200, 255), "truck": (60, 230, 60), "bus": (255, 120, 60), "motorcycle": (230, 90, 230)}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def main():
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 600.0
    cams, by_frame, first_frame = {}, {}, {}
    for cam in (1, 2, 3, 4):
        d = pickle.load(open(DATA / f"tracks_cam{cam}.pkl", "rb"))
        cams[cam] = d
        bf = defaultdict(list)
        for t in d["tracks"]:
            first_frame[(cam, t["track"])] = t["frames"][0]
            for f, b in zip(t["frames"], t["box"]):
                bf[f].append((t["track"], t["type"], b))
        by_frame[cam] = bf
    fps = cams[1]["fps"]
    caps = {c: cv2.VideoCapture(str(DATA / f"cam{c}_preview.mp4")) for c in cams}
    starts = {c: sorted(first_frame[(c, t["track"])] for t in cams[c]["tracks"]) for c in cams}

    out = av.open(str(OUT / "all_cameras_tracked.mp4"), mode="w")
    st = out.add_stream("libx264" if "libx264" in av.codecs_available else "h264_videotoolbox", rate=int(round(fps)))
    st.width, st.height, st.pix_fmt = 2 * TW, 2 * TH, "yuv420p"
    st.options = {"crf": "26", "preset": "veryfast"}
    pos = {1: (0, 0), 2: (TW, 0), 3: (0, TH), 4: (TW, TH)}
    n_frames = int(seconds * fps)
    for f in range(n_frames):
        grid = np.zeros((2 * TH, 2 * TW, 3), np.uint8)
        for cam, cap in caps.items():
            ok, img = cap.read()
            if not ok:
                continue
            tile = cv2.resize(img, (TW, TH), interpolation=cv2.INTER_AREA)
            boxes = by_frame[cam].get(f, [])
            for tid, typ, b in boxes:
                x1, y1, x2, y2 = (int(v * S) for v in b)
                col = COLOUR.get(typ, (200, 200, 200))
                cv2.rectangle(tile, (x1, y1), (x2, y2), col, 2)
                cv2.putText(tile, f"{typ[0].upper()}{tid}", (x1, max(12, y1 - 4)), FONT, 0.38, col, 1, cv2.LINE_AA)
            counted = int(np.searchsorted(starts[cam], f, side="right"))
            cv2.rectangle(tile, (0, 0), (TW, 26), (20, 20, 20), -1)
            cv2.putText(tile, f"Camera {cam}   in view: {len(boxes):2d}   counted so far: {counted}", (8, 18), FONT, 0.5,
                        (240, 240, 240), 1, cv2.LINE_AA)
            x, y = pos[cam]
            grid[y:y + TH, x:x + TW] = tile
        cv2.line(grid, (TW, 0), (TW, 2 * TH), (40, 40, 40), 2)
        cv2.line(grid, (0, TH), (2 * TW, TH), (40, 40, 40), 2)
        cv2.rectangle(grid, (2 * TW - 250, 2 * TH - 26), (2 * TW, 2 * TH), (20, 20, 20), -1)
        cv2.putText(grid, f"t = {f / fps:6.1f} s", (2 * TW - 240, 2 * TH - 8), FONT, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
        for pkt in st.encode(av.VideoFrame.from_ndarray(grid, format="bgr24")):
            out.mux(pkt)
        if f % 1500 == 0:
            print(f"frame {f}/{n_frames}", flush=True)
    for pkt in st.encode():
        out.mux(pkt)
    out.close()
    print("wrote", OUT / "all_cameras_tracked.mp4")


if __name__ == "__main__":
    main()
