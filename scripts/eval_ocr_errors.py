"""OCR error analysis and label audit on the held-out Indian crops (scripts/ocr_errors.py, scripts/label_audit.py).

Every crop of data/indian_ocr/{val,test}.csv is read by the fine-tuned model and format-corrected, as on
the edge node. Printed as markdown, per split:
  error types       exact / dropped last character / other drops / substitutions / insertions / mixed /
                    different plate
  where             which part of the plate the edits fall in
  confusion pairs   label character -> read character, most frequent first
  format correction reads it fixed, broke, or changed without fixing
  label audit       crops whose label the model and the format check both speak against, with the reason
Neither split was trained on; val chose the checkpoint (early stopping) and the hotlist threshold.

Run: .venv/bin/python scripts/eval_ocr_errors.py [--split val test] [--top 12]   (CPU, a few seconds)
Writes reports/ocr_errors.json (counts and the suspect list; plate strings only, no images).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import label_audit, ocr_errors  # noqa: E402
from scripts.indian_plate_format import correct  # noqa: E402

OCR = ROOT / "data" / "indian_ocr"


def read_split(ocr, split: str) -> list[dict]:
    """One row per crop: label, raw read, format-corrected read, lowest character confidence."""
    import numpy as np
    import pandas as pd
    df = pd.read_csv(OCR / f"{split}.csv")
    preds = ocr.run([str(OCR / p) for p in df.image_path], return_confidence=True)
    rows = []
    for p, label, path in zip(preds, df.plate_text, df.image_path):
        raw = "".join(c for c in p.plate.upper() if c.isalnum())
        conf = float(np.min(p.char_probs)) if p.char_probs is not None and len(p.char_probs) else 1.0
        rows.append({"split": split, "image": path, "label": label, "raw": raw, "read": correct(raw),
                     "conf": round(conf, 3)})
    return rows


def table(header: list[str], lines: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(out + ["| " + " | ".join(str(x) for x in line) + " |" for line in lines])


def pct(k: int, n: int) -> str:
    return f"{k} ({100 * k / n:.1f}%)" if n else "0"


def main(splits: list[str], top: int):
    from fast_plate_ocr import LicensePlateRecognizer
    ocr = LicensePlateRecognizer(onnx_model_path=str(ROOT / "weights/final/indian_plate_ocr.onnx"),
                                 plate_config_path=str(ROOT / "weights/final/indian_plate_ocr_config.yaml"), device="cpu")
    rows = {s: read_split(ocr, s) for s in splits}
    pairs = {s: [(r["label"], r["read"]) for r in rs] for s, rs in rows.items()}
    report = {}

    counts = {s: ocr_errors.breakdown(p) for s, p in pairs.items()}
    print("Error types (format-corrected read vs label)\n")
    print(table(["Error type"] + [f"{s} ({len(rows[s])} crops)" for s in splits],
                [[t] + [pct(counts[s][t], len(rows[s])) for s in splits] for t in ocr_errors.TYPES]))

    where = {s: ocr_errors.error_positions(p) for s, p in pairs.items()}
    print("\nWhere the edits fall (misreads of the same plate only)\n")
    print(table(["Part of the plate"] + splits,
                [[k] + [pct(where[s][k], sum(where[s].values())) for s in splits]
                 for k in ("state", "middle", "number", "last")]))

    conf = {s: ocr_errors.confusion_pairs(p) for s, p in pairs.items()}
    both = sum(conf.values(), start=type(conf[splits[0]])())
    print(f"\nConfusion pairs, label -> read (top {top} of {len(both)} pairs, {sum(both.values())} substitutions)\n")
    print(table(["Label", "Read as"] + splits + ["Total"],
                [[a, b] + [conf[s][(a, b)] for s in splits] + [k] for (a, b), k in both.most_common(top)]))

    fmt = {s: ocr_errors.format_effect([(r["label"], r["raw"], r["read"]) for r in rs]) for s, rs in rows.items()}
    print("\nFormat correction (raw read -> corrected read)\n")
    print(table(["Effect"] + splits, [[k] + [fmt[s][k] for s in splits] for k in fmt[splits[0]]]))

    suspects = label_audit.audit([r for s in splits for r in rows[s]])
    print(f"\nLabel audit: {len(suspects)} suspect crops, {len({s['label'] for s in suspects})} labels\n")
    print(table(["Reason", "Crops", "Labels"],
                [[why, v["crops"], v["labels"]] for why, v in label_audit.summary(suspects).items()]))
    print()
    print(table(["Split", "Label", "Read", "Conf", "Edits", "Other crops read as label", "Reason"],
                [[s["split"], s["label"], s["read"] if s["read"] == s["raw"] else f"{s['read']} (raw {s['raw']})",
                  s["conf"], s["edits"], s["agree"], s["reason"]] for s in suspects]))

    for s in splits:
        report[s] = {"crops": len(rows[s]), "error_types": counts[s], "edit_positions": dict(where[s]),
                     "format_correction": fmt[s],
                     "confusion_pairs": [[a, b, k] for (a, b), k in conf[s].most_common()]}
    report["label_audit"] = {"min_conf": label_audit.MIN_CONF, "summary": label_audit.summary(suspects),
                             "suspects": suspects}
    out = ROOT / "reports" / "ocr_errors.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", nargs="+", default=["val", "test"], choices=["val", "test"])
    ap.add_argument("--top", type=int, default=12, help="confusion pairs to print")
    a = ap.parse_args()
    main(a.split, a.top)
