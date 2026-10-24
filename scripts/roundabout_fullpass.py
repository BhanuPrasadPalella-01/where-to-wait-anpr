"""One full pass over one RoundaboutHD camera: download, track every vehicle, keep fingerprints, delete.

1. Download the camera's 4K video out of the 8 GB dataset zip (only this member, deflate-streamed).
2. Decode all 9,000 frames; YOLO26s + ByteTrack give one track per vehicle pass; each track keeps its
   times, boxes, majority type, largest crops and the ground-truth id it overlaps (for scoring only).
3. Each track gets an appearance fingerprint: ResNet-50 embedding + HSV colour histogram.
4. Save tracks (data/roundabout/tracks_cam<N>.pkl) and a 960x540 preview video; delete the 4K file.

Usage: python scripts/roundabout_fullpass.py CAM
"""
import io
import os
import pickle
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import av
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_zip import HttpRangeFile  # noqa: E402
from roundabout_track import VEHICLE, Embedder, colour_hist, iou, load_gt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "roundabout"
URL = "https://huggingface.co/datasets/yl4300/RoundaboutHD/resolve/main/RoundaboutHD.zip"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def download(cam: int) -> Path:
    dst = DATA / f"cam{cam}_4k.mp4"
    z = zipfile.ZipFile(io.BufferedReader(HttpRangeFile(URL), buffer_size=32 << 20))
    info = z.getinfo(f"RoundaboutHD/imagesc00{cam}/video.mp4")
    if dst.exists() and dst.stat().st_size == info.file_size:
        log(f"cam{cam}: already downloaded")
        return dst
    t0, done = time.time(), 0
    with z.open(info) as src, open(dst, "wb") as out:
        while chunk := src.read(16 << 20):
            out.write(chunk)
            done += len(chunk)
            if done // (200 << 20) != (done - len(chunk)) // (200 << 20):
                log(f"cam{cam}: {done / 1e9:.2f}/{info.file_size / 1e9:.2f} GB, {done / 1e6 / (time.time() - t0):.1f} MB/s")
    log(f"cam{cam}: downloaded {info.file_size / 1e9:.2f} GB in {(time.time() - t0) / 60:.1f} min")
    return dst


def track_all(cam: int, video: Path):
    from ultralytics import YOLO
    det = YOLO(str(ROOT / "weights" / "yolo26s.pt"))
    gt = load_gt()
    tr = defaultdict(lambda: {"t": [], "box": [], "cls": [], "crops": [], "gt": Counter(), "frames": []})
    preview = None
    t0 = time.time()
    with av.open(str(video)) as ct:
        st = ct.streams.video[0]
        st.thread_type = "AUTO"
        fps = float(st.average_rate)
        for i, frame in enumerate(ct.decode(st)):
            img = cv2.resize(frame.to_ndarray(format="bgr24"), (1280, 720), interpolation=cv2.INTER_AREA)
            t = i / fps
            gt_frame = gt[cam].get(int(round(t * 15)), {})
            r = det.track(img, persist=True, tracker="bytetrack.yaml", conf=0.3, classes=list(VEHICLE),
                          verbose=False, device="mps")[0]
            if r.boxes.id is not None:
                for b, tid, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.id.int().cpu().tolist(),
                                     r.boxes.cls.int().cpu().tolist()):
                    x1, y1, x2, y2 = b
                    if (x2 - x1) * (y2 - y1) < 900:
                        continue
                    d = tr[tid]
                    d["t"].append(t); d["box"].append([round(float(v), 1) for v in b]); d["cls"].append(c); d["frames"].append(i)
                    d["crops"].append(((x2 - x1) * (y2 - y1), img[int(y1):int(y2), int(x1):int(x2)].copy()))
                    d["crops"] = sorted(d["crops"], key=lambda z: -z[0])[:6]
                    best = max(gt_frame.items(), key=lambda kv: iou(b, kv[1]), default=None)
                    if best and iou(b, best[1]) > 0.3:
                        d["gt"][best[0]] += 1
            small = cv2.resize(img, (960, 540), interpolation=cv2.INTER_AREA)
            if preview is None:
                preview = cv2.VideoWriter(str(DATA / f"cam{cam}_preview.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (960, 540))
            preview.write(small)
            if i % 1500 == 0:
                log(f"cam{cam}: frame {i}, {len(tr)} tracks so far, {i / max(1e-6, time.time() - t0):.1f} fps")
    preview.release()
    return tr, fps


def main():
    cam = int(sys.argv[1])
    video = download(cam)
    tr, fps = track_all(cam, video)
    emb = Embedder()
    tracks = []
    for tid, d in tr.items():
        if len(d["t"]) < 5:
            continue
        crops = [c for _, c in d["crops"]]
        ok, jpg = cv2.imencode(".jpg", crops[0], [cv2.IMWRITE_JPEG_QUALITY, 85])
        tracks.append({"cam": cam, "track": tid, "t_first": min(d["t"]), "t_last": max(d["t"]),
                       "type": VEHICLE[Counter(d["cls"]).most_common(1)[0][0]],
                       "gt": d["gt"].most_common(1)[0][0] if d["gt"] else None,
                       "frames": d["frames"], "box": d["box"], "emb": emb(crops), "hist": colour_hist(crops),
                       "crop_jpg": jpg.tobytes() if ok else None})
    with open(DATA / f"tracks_cam{cam}.pkl", "wb") as f:
        pickle.dump({"cam": cam, "fps": fps, "tracks": tracks}, f)
    n_gt = len({t["gt"] for t in tracks if t["gt"] is not None})
    log(f"cam{cam}: {len(tracks)} tracks saved, {n_gt} ground-truth vehicles covered")
    os.remove(video)
    log(f"cam{cam}: 4K file deleted")


if __name__ == "__main__":
    main()
