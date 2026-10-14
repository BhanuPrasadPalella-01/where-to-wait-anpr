"""Stage D baseline: how well do pretrained fast-plate-ocr models read Indian plates (no fine-tuning)?

Metrics on the held-out Indian OCR split (grouped by plate number, so no plate seen in train):
  exact      share of plates read with every character right (what a hotlist lookup needs)
  char_acc   1 - edit_distance / len(truth), averaged
  within_2   share with edit distance <= 2 (what a BK-tree radius-2 hotlist search would still catch)
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd
from fast_plate_ocr import LicensePlateRecognizer

from indian_plate_format import correct

ROOT = Path(__file__).resolve().parents[1]
OCR = ROOT / "data" / "indian_ocr"
MODELS = ["cct-s-v2-global-model", "cct-xs-v2-global-model", "global-plates-mobile-vit-v2-model"]


def levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def evaluate(model_name: str, split: str, onnx_path: str | None = None, plate_config: str | None = None,
             fix_format: bool = False) -> dict:
    df = pd.read_csv(OCR / f"{split}.csv")
    if onnx_path:
        rec = LicensePlateRecognizer(onnx_model_path=onnx_path, plate_config_path=plate_config, device="cpu")
    else:
        rec = LicensePlateRecognizer(model_name, device="cpu")
    paths = [str(OCR / p) for p in df.image_path]
    t0 = time.perf_counter()
    preds = rec.run(paths)
    ms = (time.perf_counter() - t0) * 1000 / len(paths)
    texts = [(p.plate if hasattr(p, "plate") else str(p)).upper().replace("_", "") for p in preds]
    texts = ["".join(c for c in t if c.isalnum()) for t in texts]
    if fix_format:
        texts = [correct(t) for t in texts]
    d = [levenshtein(t, g) for t, g in zip(texts, df.plate_text)]
    df = df.assign(pred=texts, dist=d)
    return {
        "model": model_name, "split": split, "n": len(df),
        "exact": round(float((df.dist == 0).mean()), 4),
        "char_acc": round(float((1 - df.dist / df.plate_text.str.len()).clip(lower=0).mean()), 4),
        "within_2": round(float((df.dist <= 2).mean()), 4),
        "ms_per_plate_cpu": round(ms, 2),
        "examples": df[df.dist > 0].head(8)[["plate_text", "pred"]].values.tolist(),
    }


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "test"
    if len(sys.argv) > 2:  # fine-tuned model: <split> <model.onnx> <plate_config.yaml>
        out = [dict(evaluate("fine-tuned", split, sys.argv[2], sys.argv[3], fix), model=f"fine-tuned{'+format' if fix else ''}")
               for fix in (False, True)]
        for r in out:
            print(json.dumps({k: v for k, v in r.items() if k != "examples"}))
            print("   errors e.g.", r["examples"][:5])
        (ROOT / "reports" / f"ocr_finetuned_{split}.json").write_text(json.dumps(out, indent=2))
        sys.exit()
    out = [evaluate(m, split) for m in MODELS]
    for r in out:
        print(json.dumps({k: v for k, v in r.items() if k != "examples"}))
        print("   errors e.g.", r["examples"][:5])
    (ROOT / "reports" / f"ocr_baseline_{split}.json").write_text(json.dumps(out, indent=2))
