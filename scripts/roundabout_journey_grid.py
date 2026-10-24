"""Four-camera grid of one vehicle's journey: the suspect highlighted as it passes camera to camera.

Cuts the synchronized 2x2 view to the window from the vehicle's first sighting to its last. The suspect
(as our system picked it in each camera) gets a thick green box and score, and that camera's tile a green
border; every other tracked vehicle keeps a thin box. A banner ticks off the cameras as the suspect
reaches them. Stretches where the suspect is in no camera play at FAST x speed (marked on screen).

Usage: python scripts/roundabout_journey_grid.py VEHICLE_ID
Writes reports/roundabout/journey_grid_v<ID>.mp4 (H.264)
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
TW, TH = 640, 360
S = TW / 1280
FAST = 4
PAD_S = 2.0
GREEN, GREY, DARK, WHITE = (60, 230, 60), (150, 150, 150), (20, 20, 20), (240, 240, 240)
COLOUR = {"car": (80, 200, 255), "truck": (60, 160, 230), "bus": (255, 120, 60), "motorcycle": (230, 90, 230)}
FONT = cv2.FONT_HERSHEY_SIMPLEX
POS = {1: (0, 0), 2: (TW, 0), 3: (0, TH), 4: (TW, TH)}


def main():
    vid = int(sys.argv[1])
    row = next(r for r in json.loads((OUT / "eval.json").read_text())["appearance + travel time + direction"]["rows"]
               if r["vehicle"] == vid)
    picks = [(row["start"]["cam"], row["start"]["track"], None)] + [(s["to"], s["pick_track"], s["score"]) for s in row["steps"]]
    cams = {c: pickle.load(open(DATA / f"tracks_cam{c}.pkl", "rb")) for c in (1, 2, 3, 4)}
    fps = cams[1]["fps"]
    tracks = {c: {t["track"]: t for t in d["tracks"]} for c, d in cams.items()}
    suspect = {c: (tid, sc) for c, tid, sc in picks}
    by_frame = {c: defaultdict(list) for c in cams}
    for c, d in cams.items():
        for t in d["tracks"]:
            fr, bx = t["frames"], t["box"]
            if suspect.get(c, (None,))[0] == t["track"]:
                # the suspect's box is drawn through short detection gaps (linear interpolation up to 1 s),
                # so a small, distant vehicle does not blink in and out
                fill_f, fill_b = [], []
                for (fa, ba), (fb, bb) in zip(zip(fr, bx), zip(fr[1:], bx[1:])):
                    fill_f.append(fa); fill_b.append(ba)
                    if 1 < fb - fa <= int(fps):
                        for g in range(fa + 1, fb):
                            w = (g - fa) / (fb - fa)
                            fill_f.append(g); fill_b.append([u + w * (v - u) for u, v in zip(ba, bb)])
                fr, bx = fill_f + [fr[-1]], fill_b + [bx[-1]]
            for f, b in zip(fr, bx):
                by_frame[c][f].append((t["track"], t["type"], b))
    spans = [(c, tracks[c][tid]["frames"][0], tracks[c][tid]["frames"][-1]) for c, tid, _ in picks]
    f0 = max(0, spans[0][1] - int(PAD_S * fps))
    f1 = spans[-1][2] + int(PAD_S * fps)
    live = [(a - int(PAD_S * fps), b + int(PAD_S * fps)) for _, a, b in spans]

    caps = {c: cv2.VideoCapture(str(DATA / f"cam{c}_preview.mp4")) for c in cams}
    for cap in caps.values():
        cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    out = av.open(str(OUT / f"journey_grid_v{vid}.mp4"), mode="w")
    st = out.add_stream("libx264" if "libx264" in av.codecs_available else "h264_videotoolbox", rate=int(round(fps)))
    st.width, st.height, st.pix_fmt = 2 * TW, 2 * TH, "yuv420p"
    st.options = {"crf": "24", "preset": "medium"}

    order = [c for c, _, _ in picks]
    for f in range(f0, f1 + 1):
        tiles = {}
        for c, cap in caps.items():
            ok, img = cap.read()
            tiles[c] = cv2.resize(img, (TW, TH), interpolation=cv2.INTER_AREA) if ok else np.zeros((TH, TW, 3), np.uint8)
        in_view = any(a <= f <= b for a, b in live)
        if not in_view and (f - f0) % FAST:
            continue  # fast-forward between cameras
        grid = np.zeros((2 * TH, 2 * TW, 3), np.uint8)
        for c, tile in tiles.items():
            sus_here = False
            for tid, typ, b in by_frame[c].get(f, []):
                x1, y1, x2, y2 = (int(v * S) for v in b)
                if suspect.get(c, (None,))[0] == tid:
                    continue
                cv2.rectangle(tile, (x1, y1), (x2, y2), COLOUR.get(typ, GREY), 1)
            for tid, typ, b in by_frame[c].get(f, []):
                if suspect.get(c, (None,))[0] == tid:
                    sus_here = True
                    x1, y1, x2, y2 = (int(v * S) for v in b)
                    cv2.rectangle(tile, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), GREEN, 3)
                    sc = suspect[c][1]
                    text = "SUSPECT" if sc is None else f"SUSPECT {sc:.2f}"
                    (tw, th), _ = cv2.getTextSize(text, FONT, 0.55, 2)
                    ty = max(th + 6, y1 - 8)
                    cv2.rectangle(tile, (x1 - 3, ty - th - 6), (x1 + tw + 5, ty + 4), GREEN, -1)
                    cv2.putText(tile, text, (x1 + 1, ty - 1), FONT, 0.55, DARK, 2, cv2.LINE_AA)
            cv2.rectangle(tile, (0, 0), (TW, 26), DARK, -1)
            cv2.putText(tile, f"Camera {c}", (8, 18), FONT, 0.55, GREEN if sus_here else WHITE, 2 if sus_here else 1, cv2.LINE_AA)
            if sus_here:
                cv2.rectangle(tile, (1, 1), (TW - 2, TH - 2), GREEN, 4)
            x, y = POS[c]
            grid[y:y + TH, x:x + TW] = tile
        # journey banner
        cv2.rectangle(grid, (0, 2 * TH - 40), (2 * TW, 2 * TH), DARK, -1)
        x = 14
        cv2.putText(grid, "Suspect path:", (x, 2 * TH - 14), FONT, 0.6, WHITE, 1, cv2.LINE_AA)
        x += 140
        for i, (c, a, b) in enumerate(spans):
            reached = f >= a
            text = f"Cam {c}" + (" OK" if reached else "")
            col = GREEN if reached else GREY
            cv2.putText(grid, text, (x, 2 * TH - 14), FONT, 0.6, col, 2 if reached else 1, cv2.LINE_AA)
            (tw, _), _ = cv2.getTextSize(text, FONT, 0.6, 2)
            x += tw + 12
            if i < len(spans) - 1:
                cv2.putText(grid, "->", (x, 2 * TH - 14), FONT, 0.6, GREY, 1, cv2.LINE_AA)
                x += 34
        tag = f"t = {f / fps:5.1f} s" + ("" if in_view else f"   >> {FAST}x")
        cv2.putText(grid, tag, (2 * TW - 230, 2 * TH - 14), FONT, 0.6, WHITE if in_view else (60, 200, 255), 1, cv2.LINE_AA)
        for pkt in st.encode(av.VideoFrame.from_ndarray(grid, format="bgr24")):
            out.mux(pkt)
    for pkt in st.encode():
        out.mux(pkt)
    out.close()
    print("wrote", OUT / f"journey_grid_v{vid}.mp4", "| window", f"{f0 / fps:.0f}-{f1 / fps:.0f}s", "| path", order)


if __name__ == "__main__":
    main()
