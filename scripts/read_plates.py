"""End-to-end plate reading: image -> YOLO26n plate box -> crop -> fine-tuned Indian OCR.

Usage: python scripts/read_plates.py <image or folder> [--out reports/demo]
Writes annotated images and a CSV of reads (plate text, per-character confidence, box, detector score).
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from fast_plate_ocr import LicensePlateRecognizer
from ultralytics import YOLO

from indian_plate_format import correct

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--det", default=str(ROOT / "runs" / "plate_yolo26n" / "weights" / "best.pt"))
    ap.add_argument("--ocr", default=str(ROOT / "weights" / "final" / "indian_plate_ocr.onnx"))
    ap.add_argument("--ocr-config", default=str(ROOT / "weights" / "final" / "indian_plate_ocr_config.yaml"))
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--out", default=str(ROOT / "reports" / "demo"))
    args = ap.parse_args()

    det = YOLO(args.det)
    ocr = LicensePlateRecognizer(onnx_model_path=args.ocr, plate_config_path=args.ocr_config, device="cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    src = Path(args.source)
    images = sorted(src.glob("*.jpg")) + sorted(src.glob("*.png")) if src.is_dir() else [src]

    rows = []
    for path in images:
        img = cv2.imread(str(path))
        H, W = img.shape[:2]
        res = det.predict(img, conf=args.conf, device="mps", verbose=False)[0]
        for box, score in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = box.astype(int)
            # small margin so the OCR sees the plate border, as in its training crops
            mx, my = int(0.04 * (x2 - x1)), int(0.08 * (y2 - y1))
            crop = img[max(0, y1 - my):min(H, y2 + my), max(0, x1 - mx):min(W, x2 + mx)]
            if crop.size == 0:
                continue
            pred = ocr.run(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), return_confidence=True)[0]
            text = correct("".join(c for c in pred.plate.upper() if c.isalnum()))
            conf = np.asarray(pred.char_probs if pred.char_probs is not None else [1.0])
            rows.append({"image": path.name, "plate": text, "min_char_conf": round(float(conf.min()), 3),
                         "det_score": round(float(score), 3), "box": [int(v) for v in box]})
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), max(2, W // 300))
            cv2.putText(img, f"{text} {score:.2f}", (x1, max(30, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                        max(0.8, W / 1200), (0, 0, 255), max(2, W // 400))
        cv2.imwrite(str(out / path.name), img)

    with open(out / "reads.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "plate", "min_char_conf", "det_score", "box"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(images)} images, {len(rows)} plates read -> {out}")


if __name__ == "__main__":
    main()
