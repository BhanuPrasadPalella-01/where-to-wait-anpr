"""Re-encode a camera's 960x540 preview as browser-friendly H.264 for the Vehicle Search page.

The full-pass previews use MPEG-4 Part 2, which browsers do not play. Output: H.264 (yuv420p), a key frame
every second so seeking to any moment is instant, and the MP4 index at the front (faststart) so playback
starts before the whole file has loaded.

Usage: python scripts/roundabout_web_video.py CAM
Writes data/roundabout/web/cam<CAM>.mp4
"""
import sys
from fractions import Fraction
from pathlib import Path

import av

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "roundabout"


def main():
    cam = int(sys.argv[1])
    out_dir = DATA / "web"
    out_dir.mkdir(parents=True, exist_ok=True)
    src = av.open(str(DATA / f"cam{cam}_preview.mp4"))
    ist = src.streams.video[0]
    fps = int(round(float(ist.average_rate)))
    dst = av.open(str(out_dir / f"cam{cam}.mp4"), mode="w", options={"movflags": "+faststart"})
    ost = dst.add_stream("libx264" if "libx264" in av.codecs_available else "h264_videotoolbox", rate=fps)
    ost.width, ost.height, ost.pix_fmt = ist.width, ist.height, "yuv420p"
    ost.options = {"crf": "26", "preset": "veryfast", "g": str(fps), "keyint_min": str(fps)}
    ost.time_base = Fraction(1, fps)
    n = 0
    for frame in src.decode(ist):
        frame = frame.reformat(format="yuv420p")
        frame.pts, frame.time_base = n, Fraction(1, fps)  # renumber: one tick per frame
        for pkt in ost.encode(frame):
            dst.mux(pkt)
        n += 1
    for pkt in ost.encode():
        dst.mux(pkt)
    dst.close()
    print(f"cam{cam}: {n} frames -> {out_dir / f'cam{cam}.mp4'}")


if __name__ == "__main__":
    main()
