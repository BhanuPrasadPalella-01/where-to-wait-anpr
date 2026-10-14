"""Hotlist matching with real OCR output from the fine-tuned Indian model.

Hotlist = 100,000 synthetic Indian plates + half of the held-out test plates ("wanted").
The other half of the test plates are "innocent" cars that must not trigger an alert.
Every test crop is read by the OCR, then matched three ways:
  exact   : hash lookup only (what plain ANPR hotlists do)
  fuzzy   : BK-tree radius 2, best candidate by confusion-weighted distance, alert if score >= theta
Reports wanted-vehicle recall, wrong-vehicle alert rate, and BK-tree comparisons vs a linear scan.
"""
import json
import random
import string
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from fast_plate_ocr import LicensePlateRecognizer

from hotlist import Hotlist

ROOT = Path(__file__).resolve().parents[1]
OCR = ROOT / "data" / "indian_ocr"
STATES = "AP AR AS BR CG GA GJ HR HP JH KA KL MP MH MN ML MZ NL OD PB RJ SK TN TS TR UP UK WB DL JK".split()


def synthetic_plates(n: int, rng: random.Random) -> set:
    out = set()
    while len(out) < n:
        series = "".join(rng.choices(string.ascii_uppercase, k=rng.choice([1, 2, 2, 2])))
        out.add(f"{rng.choice(STATES)}{rng.randint(1, 99):02d}{series}{rng.randint(1, 9999):04d}")
    return out


def main(n_synth: int = 100_000):
    rng = random.Random(0)
    df = pd.read_csv(OCR / "test.csv")
    ocr = LicensePlateRecognizer(onnx_model_path=str(ROOT / "weights/final/indian_plate_ocr.onnx"),
                                 plate_config_path=str(ROOT / "weights/final/indian_plate_ocr_config.yaml"), device="cpu")
    preds = ocr.run([str(OCR / p) for p in df.image_path], return_confidence=True)
    df["read"] = ["".join(c for c in p.plate.upper() if c.isalnum()) for p in preds]
    df["conf"] = [float(np.min(p.char_probs)) if p.char_probs is not None and len(p.char_probs) else 1.0 for p in preds]

    plates = sorted(df.plate_text.unique())
    rng.shuffle(plates)
    wanted = set(plates[: len(plates) // 2])
    synth = synthetic_plates(n_synth, rng) - set(plates)
    t0 = time.perf_counter()
    hl = Hotlist(list(synth | wanted))
    build_s = time.perf_counter() - t0

    rows, comps = [], []
    for r in df.itertuples():
        is_wanted = r.plate_text in wanted
        exact_hit = r.read in hl.exact
        best, score = hl.match(r.read, r.conf)
        _, c = hl.tree.search(r.read, hl.radius)
        comps.append(c)
        rows.append({"wanted": is_wanted, "truth": r.plate_text, "read": r.read, "exact_hit": exact_hit,
                     "fuzzy_plate": best, "score": score})
    res = pd.DataFrame(rows)
    w, inn = res[res.wanted], res[~res.wanted]

    out = {"hotlist_size": hl.tree.size, "bk_build_s": round(build_s, 1),
           "test_reads": len(res), "wanted_reads": len(w), "innocent_reads": len(inn),
           "bk_mean_comparisons": round(float(np.mean(comps))), "linear_scan_comparisons": hl.tree.size,
           "exact": {"wanted_recall": round(float((w.exact_hit & (w.read == w.truth)).mean()), 4),
                     "wrong_vehicle_alert_rate": round(float(res.exact_hit[res.read != res.truth].mean()), 4)},
           "fuzzy": {}}
    for theta in (0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7):
        alert = res.score >= theta
        correct = alert & (res.fuzzy_plate == res.truth)
        wrong = alert & (res.fuzzy_plate != res.truth)
        out["fuzzy"][f"theta={theta}"] = {
            "wanted_recall": round(float(correct[res.wanted].mean()), 4),
            "wrong_vehicle_alerts_per_1000_reads": round(1000 * float(wrong.mean()), 1),
            "innocent_cars_flagged": round(float(alert[~res.wanted].mean()), 4)}
    print(json.dumps(out, indent=2))
    (ROOT / "reports" / "hotlist_eval.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 100_000)
