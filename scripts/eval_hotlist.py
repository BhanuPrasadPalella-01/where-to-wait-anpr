"""Hotlist matching with real OCR output from the fine-tuned Indian model.

Hotlist = 100,000 synthetic Indian plates + half of the held-out test plates ("wanted").
The other half of the test plates are "innocent" cars that must not trigger an alert.
Every test crop is read by the OCR, then matched three ways:
  exact   : hash lookup only (what plain ANPR hotlists do)
  fuzzy   : BK-tree radius 2, best candidate by confusion-weighted distance, alert if score >= theta
Reports wanted-vehicle recall, wrong-vehicle alert rate, and BK-tree comparisons vs a linear scan.

theta is never read off the test set:
  --split val   run the same protocol on data/indian_ocr/val.csv, choose theta (highest wanted recall with
                0 innocent cars flagged) and freeze it in reports/hotlist_theta.json
  --split test  (default) measure once on test.csv at the frozen theta; the full sweep is kept for reference
"""
import argparse
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
from hotlist_threshold import THETAS, choose_theta, load_frozen, save_frozen, sweep

ROOT = Path(__file__).resolve().parents[1]
OCR = ROOT / "data" / "indian_ocr"
REPORTS = ROOT / "reports"
FROZEN = REPORTS / "hotlist_theta.json"
STATES = "AP AR AS BR CG GA GJ HR HP JH KA KL MP MH MN ML MZ NL OD PB RJ SK TN TS TR UP UK WB DL JK".split()


def synthetic_plates(n: int, rng: random.Random) -> set:
    out = set()
    while len(out) < n:
        series = "".join(rng.choices(string.ascii_uppercase, k=rng.choice([1, 2, 2, 2])))
        out.add(f"{rng.choice(STATES)}{rng.randint(1, 99):02d}{series}{rng.randint(1, 9999):04d}")
    return out


def main(split: str = "test", n_synth: int = 100_000, theta: float | None = None):
    if split == "test" and theta is None:
        if not FROZEN.exists():
            sys.exit(f"no frozen theta in {FROZEN.relative_to(ROOT)}: run with --split val first")
        theta = load_frozen(FROZEN)
    rng = random.Random(0)
    df = pd.read_csv(OCR / f"{split}.csv")
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

    out = {"split": split, "hotlist_size": hl.tree.size, "bk_build_s": round(build_s, 1),
           "reads": len(res), "wanted_reads": len(w), "innocent_reads": len(inn),
           "bk_mean_comparisons": round(float(np.mean(comps))), "linear_scan_comparisons": hl.tree.size,
           "exact": {"wanted_recall": round(float((w.exact_hit & (w.read == w.truth)).mean()), 4),
                     "wrong_vehicle_alert_rate": round(float(res.exact_hit[res.read != res.truth].mean()), 4)},
           "fuzzy": {f"theta={t}": m for t, m in sweep(rows, THETAS).items()}}
    if split == "val":
        theta = choose_theta(sweep(rows, THETAS))
        save_frozen(FROZEN, theta, split, sweep(rows, [theta])[theta])
        print(f"theta = {theta} chosen on val, frozen in {FROZEN.relative_to(ROOT)}")
    out["theta"] = theta
    out["at_theta"] = sweep(rows, [theta])[theta]
    print(json.dumps(out, indent=2))
    name = "hotlist_eval.json" if split == "test" else f"hotlist_eval_{split}.json"
    (REPORTS / name).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--n-synth", type=int, default=100_000, help="synthetic plates added to the hotlist")
    ap.add_argument("--theta", type=float, help="test split only: override the frozen theta")
    a = ap.parse_args()
    if a.theta is not None and a.split == "val":
        ap.error("--theta is chosen on val, not given")
    main(a.split, a.n_synth, a.theta)
