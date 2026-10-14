"""Build the datasets for the perception stages.

detector (YOLO format, one class 'plate'):
  generic   : keremberke/license-plate-object-detection (Roboflow, CC BY 4.0), COCO json -> YOLO
  indian_det: Dataclusterlabspvtltd/indian-number-plates-dataset (47 real Indian photos,
              CC BY-NC-ND 4.0) -> used ONLY as a held-out Indian test set, never for training
OCR (plate crop + text):
  indian_ocr: zenitsu09/indian-number-plate (1,709 crops, 965 unique plates). The source contains
              Roboflow augmentation duplicates, so the split is grouped by plate number to stop
              the same plate leaking from train into test.
"""
import io
import json
import random
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data"
SEED = 0
DC_URL = "https://huggingface.co/datasets/Dataclusterlabspvtltd/indian-number-plates-dataset/resolve/main"


def coco_to_yolo(zip_path: Path, split_dir: Path) -> int:
    img_dir, lbl_dir = split_dir / "images", split_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        coco = json.loads(z.read("_annotations.coco.json"))
        anns = defaultdict(list)
        for a in coco["annotations"]:
            anns[a["image_id"]].append(a["bbox"])
        for im in coco["images"]:
            name = im["file_name"]
            (img_dir / name).write_bytes(z.read(name))
            W, H = im["width"], im["height"]
            lines = [f"0 {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}"
                     for x, y, w, h in anns[im["id"]] if w > 1 and h > 1]
            (lbl_dir / (Path(name).stem + ".txt")).write_text("\n".join(lines))
    return len(coco["images"])


def _fetch(url: str, tries: int = 5):
    """GET with retries; returns None on 404 (end of the numbered files)."""
    for k in range(tries):
        try:
            return urllib.request.urlopen(url, timeout=60).read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(2 ** k)
        except Exception:
            time.sleep(2 ** k)
    raise RuntimeError(f"failed to download {url}")


def indian_detection_test(max_side: int = 1280) -> int:
    """Download the 47 Datacluster photos, shrink them (originals are ~8 MP) and write YOLO labels."""
    img_dir = OUT / "indian_det" / "test" / "images"
    lbl_dir = OUT / "indian_det" / "test" / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for i in range(1, 200):
        stem = f"image_{i:04d}"
        local = RAW / "datacluster"  # filled by huggingface_hub.snapshot_download if available
        if (local / "images" / f"{stem}.jpg").exists():
            xml = (local / "Annotations" / f"{stem}.xml").read_bytes()
            raw = (local / "images" / f"{stem}.jpg").read_bytes()
        else:
            xml = _fetch(f"{DC_URL}/Annotations/{stem}.xml")
            raw = _fetch(f"{DC_URL}/images/{stem}.jpg") if xml else None
        if not raw:
            continue
        img = Image.open(io.BytesIO(raw))
        img = ImageOps.exif_transpose(img).convert("RGB")
        W, H = img.size  # the XML size can disagree with EXIF rotation; trust the decoded pixels
        root = ET.fromstring(xml)
        xw, xh = float(root.find("size/width").text), float(root.find("size/height").text)
        sx, sy = W / xw, H / xh
        lines = []
        for obj in root.iter("object"):
            b = obj.find("bndbox")
            x1, y1, x2, y2 = (float(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
            x1, x2, y1, y2 = x1 * sx, x2 * sx, y1 * sy, y2 * sy
            lines.append(f"0 {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
        img.thumbnail((max_side, max_side))
        img.save(img_dir / f"{stem}.jpg", quality=92)
        (lbl_dir / f"{stem}.txt").write_text("\n".join(lines))
        n += 1
    return n


def indian_ocr_split() -> dict:
    df = pd.read_parquet(RAW / "indian.parquet")
    df["plate_text"] = df["plate_text"].str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)
    plates = sorted(df["plate_text"].unique())
    random.Random(SEED).shuffle(plates)
    n = len(plates)
    split_of = {p: ("train" if i < 0.7 * n else "val" if i < 0.85 * n else "test") for i, p in enumerate(plates)}
    rows, counts = [], defaultdict(int)
    for i, r in df.iterrows():
        split = split_of[r["plate_text"]]
        d = OUT / "indian_ocr" / split
        d.mkdir(parents=True, exist_ok=True)
        name = f"{split}/{i:05d}_{r['plate_text']}.png"
        Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB").save(OUT / "indian_ocr" / name)
        rows.append({"image_path": name, "plate_text": r["plate_text"], "split": split, "source": r["source"]})
        counts[split] += 1
    pd.DataFrame(rows).to_csv(OUT / "indian_ocr" / "labels.csv", index=False)
    for split in ("train", "val", "test"):
        sub = pd.DataFrame([x for x in rows if x["split"] == split])[["image_path", "plate_text"]]
        sub.to_csv(OUT / "indian_ocr" / f"{split}.csv", index=False)
    return dict(counts)


def write_yaml(name: str, train: str, val: str, test: str):
    (OUT / f"{name}.yaml").write_text(
        f"path: {OUT}\ntrain: {train}\nval: {val}\ntest: {test}\nnames:\n  0: plate\n")


if __name__ == "__main__":
    for split in ("train", "valid", "test"):
        print(f"generic/{split}: {coco_to_yolo(RAW / f'generic_{split}.zip', OUT / 'generic' / split)} images")
    print("indian_det/test:", indian_detection_test(), "images")
    print("indian_ocr:", indian_ocr_split())
    write_yaml("plates", "generic/train/images", "generic/valid/images", "generic/test/images")
    write_yaml("plates_indian_test", "generic/train/images", "indian_det/test/images", "indian_det/test/images")
