"""How often the 10-slot OCR cuts a plate, and what the truncation check (edge/plate_length.py) does about it.

Every crop of data/indian_ocr/{train,val,test}.csv is read by the fine-tuned model and format-corrected,
exactly as on the edge node. Per split:
  long plates     : crops whose label has more than 10 characters (the model cannot spell them)
  flagged         : reads that plate_length.check() calls likely truncated
  recovered       : flagged reads where the fallback (second read of the right half + stitch) gives the label
  false flags     : flagged reads of plates that do fit in 10 characters
  wrong prefix    : flagged reads of 10-character plates whose prefix lookup hits another plate on a
                    hotlist of 100,000 synthetic plates + every label (the cost of prefix matching)
The train split was seen in fine-tuning; its numbers are listed for the count of long plates only.

Run: .venv/bin/python scripts/eval_truncation.py   (CPU, about a minute)
"""
import random
import sys
from pathlib import Path

import cv2
import pandas as pd
from fast_plate_ocr import LicensePlateRecognizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from edge import plate_length  # noqa: E402
from eval_hotlist import synthetic_plates  # noqa: E402
from hotlist import Hotlist  # noqa: E402
from indian_plate_format import correct  # noqa: E402

OCR = ROOT / "data" / "indian_ocr"


def clean(t: str) -> str:
    return "".join(c for c in t.upper() if c.isalnum())


def main():
    ocr = LicensePlateRecognizer(onnx_model_path=str(ROOT / "weights/final/indian_plate_ocr.onnx"),
                                 plate_config_path=str(ROOT / "weights/final/indian_plate_ocr_config.yaml"), device="cpu")
    splits = {s: pd.read_csv(OCR / f"{s}.csv") for s in ("train", "val", "test")}
    labels = set().union(*(set(df.plate_text) for df in splits.values()))
    hl = Hotlist(synthetic_plates(100_000, random.Random(0)) | labels)

    rows, examples = [], []
    for split, df in splits.items():
        imgs = [cv2.cvtColor(cv2.imread(str(OCR / p)), cv2.COLOR_BGR2RGB) for p in df.image_path]
        reads = [correct(clean(p.plate)) for p in ocr.run(imgs)]
        n_long = flagged = flagged_long = recovered = false_flags = wrong_prefix = 0
        for img, read, label in zip(imgs, reads, df.plate_text):
            long_ = len(label) > plate_length.SLOTS
            n_long += long_
            cut, why = plate_length.check(read)
            if not cut:
                continue
            flagged += 1
            if long_:
                flagged_long += 1
                plate = read
                if plate_length.should_retry(img.shape[1], img.shape[0]):
                    tail = clean(ocr.run([plate_length.tail_crop(img)])[0].plate)
                    plate = plate_length.fallback(read, [tail]) or read
                recovered += plate == label
                examples.append((split, label, read, plate, why))
            else:
                false_flags += 1
                best, _ = hl.match(read, 1.0, truncated=True)
                wrong_prefix += best is not None and best != label and read in hl.prefix
        rows.append({"split": split, "crops": len(df), "long plates": n_long, "flagged": flagged,
                     "flagged long": flagged_long, "recovered": recovered, "false flags": false_flags,
                     "wrong prefix": wrong_prefix})

    cols = list(rows[0])
    print("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols))
    for r in rows:
        print("| " + " | ".join(str(r[c]) for c in cols) + " |")
    print("\nLong plates that were flagged (split, label, first read, after fallback, reason):")
    for e in examples:
        print("  ", *e)


if __name__ == "__main__":
    main()
