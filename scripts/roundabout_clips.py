"""Cut short clips from the RoundaboutHD 4K videos without storing the videos.

The dataset is one 8 GB zip on Hugging Face. Its videos are deflate-compressed inside the zip and keep
their MP4 index (moov) at the end, so each video must be streamed once from start to end. While it
streams, only three kinds of bytes are written to a *sparse* local file: the header, the stretches of
frame data around the wanted time windows (located by the near-constant bitrate of a fixed camera),
and the index at the end. Unwritten stretches are holes that take no disk space. PyAV then decodes
the windows from the sparse file and the file is deleted.

Usage: python scripts/roundabout_clips.py CAM VEHICLE_ID [VEHICLE_ID ...]
Writes data/roundabout/clips/v<ID>_cam<CAM>.mp4 (1280x720) and v<ID>_cam<CAM>.json.
"""
import io
import json
import os
import struct
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import av
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_zip import HttpRangeFile  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
URL = "https://huggingface.co/datasets/yl4300/RoundaboutHD/resolve/main/RoundaboutHD.zip"
DATA = ROOT / "data" / "roundabout"
OUT = DATA / "clips"
FPS_GT = 15          # ground-truth frame ids are at 15 fps
DURATION = 600.0     # every camera recorded 10 minutes
MARGIN_S = 3.0       # seconds kept before and after each pass in the output clip
BYTE_SLACK_S = 15.0  # extra seconds of frame data kept around a window (bitrate is only roughly constant)


def windows_for(cam: int, vids):
    span = defaultdict(lambda: [1e9, -1])
    for line in open(DATA / "Multi_CAM_Ground_Turth.txt"):
        c, o, f = (int(x) for x in line.split()[:3])
        if c == cam and o in vids:
            s = span[o]
            s[0], s[1] = min(s[0], f), max(s[1], f)
    return {o: (max(0.0, a / FPS_GT - MARGIN_S), min(DURATION, b / FPS_GT + MARGIN_S)) for o, (a, b) in span.items()}


def sparse_copy(member, size, keep, path):
    """Stream the member start to end; write only byte ranges in `keep` (sorted, merged)."""
    with open(path, "wb") as dst:
        dst.truncate(size)  # a sparse file of the full length
        pos, t0, chunk = 0, time.time(), 8 << 20
        ki = 0
        while True:
            buf = member.read(chunk)
            if not buf:
                break
            a, b = pos, pos + len(buf)
            while ki < len(keep) and keep[ki][1] <= a:
                ki += 1
            j = ki
            while j < len(keep) and keep[j][0] < b:
                s, e = max(a, keep[j][0]), min(b, keep[j][1])
                dst.seek(s)
                dst.write(buf[s - a:e - a])
                j += 1
            pos = b
            if pos // (256 << 20) != (pos - len(buf)) // (256 << 20):
                print(f"  {pos / 1e9:.2f}/{size / 1e9:.2f} GB, {pos / 1e6 / (time.time() - t0):.1f} MB/s", flush=True)


def decode_window(path, t0, t1, out_path, width=1280):
    with av.open(str(path)) as ct:
        st = ct.streams.video[0]
        st.thread_type = "AUTO"
        fps = float(st.average_rate)
        ct.seek(int(t0 / st.time_base), stream=st, backward=True)
        writer, n = None, 0
        for frame in ct.decode(st):
            t = float(frame.pts * st.time_base)
            if t < t0:
                continue
            if t > t1:
                break
            img = frame.to_ndarray(format="bgr24")
            img = cv2.resize(img, (width, int(img.shape[0] * width / img.shape[1])), interpolation=cv2.INTER_AREA)
            if writer is None:
                writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (img.shape[1], img.shape[0]))
            writer.write(img)
            n += 1
        if writer:
            writer.release()
        return n, fps


def main():
    cam = int(sys.argv[1])
    vids = {int(v) for v in sys.argv[2:]}
    OUT.mkdir(parents=True, exist_ok=True)
    wins = windows_for(cam, vids)
    z = zipfile.ZipFile(io.BufferedReader(HttpRangeFile(URL), buffer_size=32 << 20))
    info = z.getinfo(f"RoundaboutHD/imagesc00{cam}/video.mp4")
    size = info.file_size
    with z.open(info) as m:
        head = m.read(1 << 16)
    # box layout: ftyp, free, mdat(size), moov at the end
    i, mdat_start, mdat_end = 0, None, None
    while i + 8 <= len(head):
        bsize, typ = struct.unpack(">I4s", head[i:i + 8])
        if typ == b"mdat":
            mdat_start, mdat_end = i + 8, i + bsize
            break
        i += bsize
    rate = (mdat_end - mdat_start) / DURATION
    keep = [(0, mdat_start + (1 << 20)), (mdat_end, size)]
    for t0, t1 in wins.values():
        keep.append((int(mdat_start + max(0, t0 - BYTE_SLACK_S) * rate), int(mdat_start + min(DURATION, t1 + BYTE_SLACK_S) * rate)))
    keep.sort()
    merged = []
    for s, e in keep:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    kept = sum(e - s for s, e in merged)
    print(f"cam{cam}: streaming {size / 1e9:.2f} GB, keeping {kept / 1e6:.0f} MB for vehicles {sorted(wins)}", flush=True)
    sparse = DATA / f"cam{cam}_sparse.mp4"
    with z.open(info) as m:
        sparse_copy(m, size, merged, sparse)
    for vid, (t0, t1) in sorted(wins.items()):
        out = OUT / f"v{vid}_cam{cam}.mp4"
        n, fps = decode_window(sparse, t0, t1, out)
        (OUT / f"v{vid}_cam{cam}.json").write_text(json.dumps({"vehicle": vid, "cam": cam, "t0": t0, "t1": t1,
                                                             "frames": n, "fps": fps, "file": out.name}))
        print(f"cam{cam}: vehicle {vid} {t0:.1f}-{t1:.1f}s -> {n} frames -> {out.name}", flush=True)
    os.remove(sparse)


if __name__ == "__main__":
    main()
