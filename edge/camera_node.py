"""Edge camera node (PDF Layer 2–3): video -> plate detection -> tracking -> OCR vote -> hotlist match
-> colour / type check.

Every plate the camera reads is reported to the control room (matches *and* non-matches), so the
dashboard shows the continuous checking. Only hotlist matches become sightings on the map.
The body colour and vehicle type are compared with the registered vehicle (edge/attributes.py); a
mismatch lowers the match score. Annotated frames are streamed to the dashboard so the viewer sees
exactly what the camera sees.

Sources:  --source 0               Mac webcam (or iPhone via Continuity Camera: try 1)
          --source http://PHONE_IP:8080/video   Android "IP Webcam" app
          --source clip.mp4 --loop  a recorded video
Run:      .venv/bin/python -m edge.camera_node --name cam-A --source 0
"""
from __future__ import annotations

import argparse
import base64
import json
import queue
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from hotlist import Hotlist  # noqa: E402
from edge.attributes import attribute_factor, body_region, dominant_colour, vehicle_type  # noqa: E402
from indian_plate_format import correct  # noqa: E402
from server.messages import KEY_ENV, Numberer, load_key, sign  # noqa: E402

DET = ROOT / "weights" / "final" / "plate_detector_yolo26n.pt"
OCR = ROOT / "weights" / "final" / "indian_plate_ocr.onnx"
OCR_CFG = ROOT / "weights" / "final" / "indian_plate_ocr_config.yaml"
VEH = ROOT / "weights" / "yolo26n.pt"  # stock COCO model for the vehicle type (optional)


class Server:
    """Small HTTP client; reads that cannot be delivered are queued and replayed (edge-only fallback).

    Each read is numbered and signed once, when it is queued, so a retry is byte-for-byte the same
    message and the control room can drop it as a duplicate (server/messages.py)."""

    def __init__(self, base: str, key: bytes | None = None, max_backlog: int = 2000):
        self.base = base.rstrip("/")
        self.key, self.numberer, self.max_backlog = key, Numberer(), max_backlog
        self.backlog: list[dict] = []
        self.refused = 0  # reads the server will never take (bad signature, malformed): dropped, not retried
        self.frames: queue.Queue = queue.Queue(maxsize=2)
        threading.Thread(target=self._frame_pusher, daemon=True).start()

    def call(self, path: str, body=None, raw: bytes | None = None, ctype="application/json", timeout=3):
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        req = urllib.request.Request(self.base + path, data=data, method="POST" if data is not None else "GET",
                                     headers={"Content-Type": ctype})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read() or b"{}")

    def send_read(self, read: dict) -> bool:
        msg = self.numberer.stamp(read)
        self.backlog.append(sign(msg, self.key) if self.key else msg)
        if len(self.backlog) > self.max_backlog:  # long outage: forget the oldest non-match first
            drop = next((i for i, m in enumerate(self.backlog) if not m.get("matched")), 0)
            self.backlog.pop(drop)
        while self.backlog:
            try:
                self.call("/api/edge/read", self.backlog[0])
            except urllib.error.HTTPError as e:
                if not 400 <= e.code < 500:
                    return False
                self.refused += 1  # retrying a refused message can never succeed and would block the queue
                print(f"[edge] read #{self.backlog[0].get('seq')} refused by the control room "
                      f"(HTTP {e.code}); check {KEY_ENV} matches on both sides", flush=True)
            except Exception:
                return False
            self.backlog.pop(0)
        return True

    def push_frame(self, name: str, jpg: bytes):
        try:
            self.frames.put_nowait((name, jpg))
        except queue.Full:
            pass

    def _frame_pusher(self):
        while True:
            name, jpg = self.frames.get()
            try:
                self.call(f"/api/edge/frame/{name}", raw=jpg, ctype="image/jpeg", timeout=2)
            except Exception:
                pass


def sharpness(img) -> float:
    return float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def pick_device(requested: str = "auto") -> str:
    """Detector device: Apple GPU (mps) or CUDA when available, otherwise CPU."""
    if requested != "auto":
        return requested
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def vote(reads: list[tuple[str, float]]) -> tuple[str, float]:
    """Combine OCR reads of the same plate: confidence-weighted vote over whole strings,
    then per-character vote among reads of the winning length."""
    by_len = defaultdict(list)
    for t, c in reads:
        by_len[len(t)].append((t, c))
    L = max(by_len, key=lambda k: sum(c for _, c in by_len[k]))
    same = by_len[L]
    chars = []
    for i in range(L):
        cnt = Counter()
        for t, c in same:
            cnt[t[i]] += c
        chars.append(cnt.most_common(1)[0][0])
    return "".join(chars), max(c for _, c in same)


def context_crop(frame, box, wide: float = 4.0, tall: float = 8.0):
    """Region around a plate big enough to hold the whole vehicle, and the plate box inside it."""
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    xa, xb = max(0, int(x1 - wide * w)), min(W, int(x2 + wide * w))
    ya, yb = max(0, int(y1 - tall * h)), min(H, int(y2 + 2 * h))
    return frame[ya:yb, xa:xb].copy(), (x1 - xa, y1 - ya, x2 - xa, y2 - ya)


def track_attributes(samples, vehicle_det=None) -> tuple[str | None, str | None]:
    """Colour and type of a tracked vehicle from its sharpest views.

    samples: [(context crop, plate box in it)]; colour is a share-weighted vote over the views,
    type comes from vehicle_det(image) -> [(x1, y1, x2, y2, coco_class)] on the sharpest view."""
    votes = Counter()
    for img, box in samples:
        colour, share = dominant_colour(body_region(img, box))
        if colour:
            votes[colour] += share
    colour = votes.most_common(1)[0][0] if votes else None
    vtype = vehicle_type(vehicle_det(samples[0][0]), samples[0][1]) if vehicle_det and samples else None
    return colour, vtype


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    ap.add_argument("--name", default="cam-A")
    ap.add_argument("--server", default="http://localhost:8000")
    ap.add_argument("--role", default="route", choices=["route", "far"],
                    help="route: next junction along the demo road; far: a junction far away (clone demo)")
    ap.add_argument("--junction", type=int, default=None, help="force a specific map junction id")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--theta", type=float, default=0.05, help="hotlist alert threshold (match score)")
    ap.add_argument("--loop", action="store_true", help="loop a video file")
    ap.add_argument("--show", action="store_true", help="also open a local preview window")
    ap.add_argument("--max-fps", type=float, default=12)
    ap.add_argument("--device", default="auto", help="detector device: auto, mps, cuda:0 or cpu")
    ap.add_argument("--vehicle-model", default=str(VEH),
                    help="COCO YOLO model for the vehicle type; 'none' checks the colour only")
    ap.add_argument("--key", default=None, help=f"shared HMAC key for signing reads (default: ${KEY_ENV})")
    a = ap.parse_args()

    from fast_plate_ocr import LicensePlateRecognizer
    from ultralytics import YOLO

    key = load_key(a.key)
    if key is None:
        print(f"[{a.name}] no {KEY_ENV} / --key: reads are sent unsigned", flush=True)
    srv = Server(a.server, key)
    reg = srv.call("/api/edge/register", {"name": a.name, "role": a.role, "junction": a.junction})
    junction = reg["junction"]
    print(f"[{a.name}] registered at map junction {junction}", flush=True)

    det, device = YOLO(str(DET)), pick_device(a.device)
    vehicle_det = None
    if a.vehicle_model != "none" and Path(a.vehicle_model).exists():
        veh = YOLO(a.vehicle_model)

        def vehicle_det(img):
            r = veh(img, classes=[2, 3, 5, 7], conf=0.3, verbose=False, device=device)[0]
            return [(*b, c) for b, c in zip(r.boxes.xyxy.cpu().numpy().tolist(), r.boxes.cls.int().cpu().tolist())]
    else:
        print(f"[{a.name}] no vehicle model ({a.vehicle_model}): checking colour only", flush=True)
    ocr = LicensePlateRecognizer(onnx_model_path=str(OCR), plate_config_path=str(OCR_CFG), device="cpu")
    src = int(a.source) if a.source.isdigit() else a.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        sys.exit(f"cannot open video source {a.source!r}")

    hot_plates, hot_attrs, hl, hot_t, reg_t = [], {}, None, 0.0, time.time()
    tracks: dict[int, dict] = {}
    frame_i, last_push = 0, 0.0
    banner = ("", 0.0)
    while True:
        t0 = time.time()
        ok, frame = cap.read()
        if not ok:
            if a.loop and isinstance(src, str):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                tracks.clear()
                continue
            break
        frame_i += 1
        if time.time() - reg_t > 5:  # re-register (idempotent) so a restarted control room knows this camera
            try:
                junction = srv.call("/api/edge/register", {"name": a.name, "role": a.role, "junction": junction})["junction"]
            except Exception:
                pass
            reg_t = time.time()
        if time.time() - hot_t > 2:  # refresh the hotlist from the control room
            try:
                got = srv.call("/api/hotlist")
                hot_plates, hot_attrs = got.get("plates", []), got.get("attributes", {})
                hl = Hotlist(hot_plates) if hot_plates else None
            except Exception:
                pass
            hot_t = time.time()

        res = det.track(frame, persist=True, tracker="bytetrack.yaml", conf=a.conf, verbose=False, device=device)[0]
        H, W = frame.shape[:2]
        seen = set()
        if res.boxes.id is not None:
            for box, tid, sc in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.id.int().cpu().tolist(),
                                    res.boxes.conf.cpu().numpy()):
                x1, y1, x2, y2 = box.astype(int)
                mx, my = int(0.04 * (x2 - x1)), int(0.08 * (y2 - y1))
                crop = frame[max(0, y1 - my):min(H, y2 + my), max(0, x1 - mx):min(W, x2 + mx)]
                if crop.size == 0:
                    continue
                tr = tracks.setdefault(tid, {"crops": [], "last": frame_i, "sent": None, "box": None})
                tr["last"], tr["box"] = frame_i, (x1, y1, x2, y2)
                tr["crops"].append((sharpness(crop), crop.copy(), context_crop(frame, (x1, y1, x2, y2))))
                tr["crops"] = sorted(tr["crops"], key=lambda c: -c[0])[:3]
                seen.add(tid)

        # finalise a track once it has 3 crops (fast) or when it leaves the frame
        for tid, tr in list(tracks.items()):
            gone = frame_i - tr["last"] > 15
            if tr["sent"] is None and tr["crops"] and (len(tr["crops"]) >= 3 or gone):
                crops = [cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for _, c, _ in tr["crops"]]
                preds = ocr.run(crops, return_confidence=True)
                reads = []
                for p in preds:
                    txt = correct("".join(ch for ch in p.plate.upper() if ch.isalnum()))
                    if len(txt) >= 6:
                        reads.append((txt, float(np.min(p.char_probs)) if p.char_probs is not None else 0.5))
                if reads:
                    plate, conf = vote(reads)
                    best, plate_score = hl.match(plate, conf) if hl else (None, 0.0)
                    colour, vtype = track_attributes([c for _, _, c in tr["crops"]], vehicle_det)
                    factor, attr = attribute_factor(colour, vtype, hot_attrs.get(best)) if best else (1.0, {})
                    score = plate_score * factor
                    matched = best is not None and score >= a.theta
                    tr["sent"] = (plate, conf, matched, best, score)
                    ok_jpg, buf = cv2.imencode(".jpg", tr["crops"][0][1], [cv2.IMWRITE_JPEG_QUALITY, 85])
                    delivered = srv.send_read({
                        "name": a.name, "junction": junction, "plate": plate, "conf": round(conf, 3),
                        "matched": matched, "hotlist_plate": best, "score": round(score, 3),
                        "plate_score": round(plate_score, 3), "colour": colour, "vehicle_type": vtype, "attr": attr,
                        "thumb": base64.b64encode(buf.tobytes()).decode() if ok_jpg else None,
                        "captured_at": time.time()})
                    status = ("HOTLIST MATCH" if matched else "not on hotlist" if best is None
                              else f"near {best}, score {score:.2f} too low"
                                   + (f" after {'/'.join(k for k, v in attr.items() if v != 'ok')} mismatch"
                                      if factor < 1 else ""))
                    looks = " ".join(x for x in (colour, vtype) if x) or "colour ?"
                    print(f"[{a.name}] read {plate} (conf {conf:.2f}, {looks}) -> {status}"
                          f"{'' if delivered else '  [server unreachable: LOCAL ALERT, queued]'}", flush=True)
                    if matched:
                        banner = (f"HOTLIST MATCH  {plate}  (score {score:.2f})", time.time())
            if gone and frame_i - tr["last"] > 60:
                del tracks[tid]

        # annotate and stream the frame
        vis = frame.copy()
        for tid, tr in tracks.items():
            if tid not in seen or tr["box"] is None:
                continue
            x1, y1, x2, y2 = tr["box"]
            sent = tr["sent"]
            col = (60, 200, 60) if sent and sent[2] else (0, 200, 255) if sent else (255, 200, 0)
            label = f"#{tid} {sent[0]} {sent[1]:.2f}" + (" MATCH" if sent[2] else "") if sent else f"#{tid} reading..."
            cv2.rectangle(vis, (x1, y1), (x2, y2), col, max(2, W // 300))
            cv2.putText(vis, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, max(0.6, W / 1400), col,
                        max(2, W // 500))
        hdr = f"{a.name} | junction {junction} | hotlist: {', '.join(hot_plates) or '(empty)'}"
        cv2.rectangle(vis, (0, 0), (W, 34), (20, 20, 20), -1)
        cv2.putText(vis, hdr, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2)
        if banner[0] and time.time() - banner[1] < 4:
            cv2.rectangle(vis, (0, H - 50), (W, H), (40, 140, 40), -1)
            cv2.putText(vis, banner[0], (12, H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        if time.time() - last_push > 0.12:
            small = cv2.resize(vis, (640, int(640 * H / W)))
            ok_jpg, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok_jpg:
                srv.push_frame(a.name, buf.tobytes())
            last_push = time.time()
        if a.show:
            cv2.imshow(a.name, vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        dt = time.time() - t0
        if isinstance(src, str) and not src.startswith(("http", "rtsp")):
            time.sleep(max(0.0, 1 / a.max_fps - dt))  # play files at roughly real time


if __name__ == "__main__":
    main()
