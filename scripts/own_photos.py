"""Turn a folder of your own labelled plate photos into a YOLO dataset for fine-tuning.

Input (see FINETUNE.md):
  <src>/images/*.jpg|png       your photos
  <src>/labels/<stem>.txt      YOLO labels, one line per plate: "0 cx cy w h" (normalised 0..1)
  <src>/conditions.csv         image,condition[,group]  condition in day / night / two_row (or your own);
                               group (optional) keeps photos of the same car or burst in one split

Output (data/own by default, never committed):
  train/ val/ test/ {images,labels}, own.yaml (train/val), own_test_<condition>.yaml per condition,
  manifest.json (which file went where and why any were dropped).

Splitting is by group, not by photo, so near-identical shots of one car cannot sit in both train and
test. Only the standard library is imported here, so the checks are unit-tested in CI.
"""
import argparse
import csv
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_CONDITIONS = ("day", "night", "two_row")


def check_label_file(text: str, n_classes: int = 1) -> list[str]:
    """Problems in one YOLO label file (empty list = fine). An empty file means 'no plate' and is allowed."""
    problems = []
    for i, line in enumerate(text.splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            problems.append(f"line {i}: expected 5 numbers, got {len(parts)}")
            continue
        try:
            cls = int(parts[0])
            cx, cy, w, h = map(float, parts[1:])
        except ValueError:
            problems.append(f"line {i}: not numeric")
            continue
        if not 0 <= cls < n_classes:
            problems.append(f"line {i}: class {cls} outside 0..{n_classes - 1}")
        if not (0 < w <= 1 and 0 < h <= 1):
            problems.append(f"line {i}: width/height must be in (0, 1]")
        elif not (0 <= cx - w / 2 + 1e-6 and cx + w / 2 <= 1 + 1e-6 and 0 <= cy - h / 2 + 1e-6 and cy + h / 2 <= 1 + 1e-6):
            problems.append(f"line {i}: box leaves the image (pixel coordinates instead of normalised?)")
    return problems


def read_conditions(path: Path) -> dict[str, tuple[str, str]]:
    """image file name -> (condition, group). The group defaults to the image stem."""
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            name = (row.get("image") or "").strip()
            if not name:
                continue
            cond = (row.get("condition") or "").strip().lower().replace("-", "_").replace(" ", "_")
            group = (row.get("group") or "").strip() or Path(name).stem
            out[name] = (cond, group)
    return out


def collect(src: Path, conditions: dict[str, tuple[str, str]]) -> tuple[list[dict], list[dict]]:
    """Pair images with labels and conditions; returns (usable items, dropped items with a reason)."""
    items, dropped = [], []
    for img in sorted((src / "images").iterdir()):
        if img.suffix.lower() not in IMAGE_EXT:
            continue
        lbl = src / "labels" / f"{img.stem}.txt"
        if img.name not in conditions:
            dropped.append({"image": img.name, "reason": "not in conditions.csv"})
            continue
        if not lbl.exists():
            dropped.append({"image": img.name, "reason": "no label file (unlabelled)"})
            continue
        problems = check_label_file(lbl.read_text())
        if problems:
            dropped.append({"image": img.name, "reason": "; ".join(problems)})
            continue
        cond, group = conditions[img.name]
        items.append({"image": img.name, "label": lbl.name, "condition": cond, "group": group,
                      "plates": sum(1 for ln in lbl.read_text().splitlines() if ln.strip())})
    return items, dropped


def split_groups(items: list[dict], val: float = 0.15, test: float = 0.25, seed: int = 0) -> dict[str, str]:
    """group -> split. Stratified by each group's main condition so every condition reaches the test set."""
    by_cond = defaultdict(list)
    group_cond = {}
    for it in items:
        group_cond.setdefault(it["group"], Counter())[it["condition"]] += 1
    for g, c in sorted(group_cond.items()):
        by_cond[c.most_common(1)[0][0]].append(g)
    rng = random.Random(seed)
    out = {}
    for cond in sorted(by_cond):
        groups = by_cond[cond]
        rng.shuffle(groups)
        n = len(groups)
        n_test = max(1, round(n * test)) if n >= 2 else 0
        n_val = max(1, round(n * val)) if n - n_test >= 2 else 0
        for i, g in enumerate(groups):
            out[g] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    return out


def write_yaml(path: Path, base: Path, train: str, val: str, test: str):
    path.write_text(f"path: {base}\ntrain: {train}\nval: {val}\ntest: {test}\nnames:\n  0: plate\n")


def build(src: Path, out: Path, val: float = 0.15, test: float = 0.25, seed: int = 0) -> dict:
    conditions = read_conditions(src / "conditions.csv")
    items, dropped = collect(src, conditions)
    if not items:
        raise SystemExit(f"no usable labelled photos in {src} (dropped {len(dropped)}; see FINETUNE.md)")
    split_of = split_groups(items, val, test, seed)
    if out.exists():
        shutil.rmtree(out)
    for it in items:
        it["split"] = split_of[it["group"]]
        for sub, name in (("images", it["image"]), ("labels", it["label"])):
            d = out / it["split"] / sub
            d.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src / sub / name, d / name)
        # per-condition test folders hold the same files, so ultralytics can score each condition alone
        if it["split"] == "test":
            for sub, name in (("images", it["image"]), ("labels", it["label"])):
                d = out / "test_by_condition" / it["condition"] / sub
                d.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src / sub / name, d / name)

    val_dir = "val/images" if (out / "val").exists() else "test/images"
    write_yaml(out / "own.yaml", out.resolve(), "train/images", val_dir, "test/images")
    test_conds = sorted({it["condition"] for it in items if it["split"] == "test"})
    for cond in test_conds:
        write_yaml(out / f"own_test_{cond}.yaml", out.resolve(), "train/images", val_dir,
                   f"test_by_condition/{cond}/images")

    counts = defaultdict(Counter)
    for it in items:
        counts[it["split"]][it["condition"]] += 1
    manifest = {"source": str(src), "seed": seed, "items": items, "dropped": dropped,
                "counts": {s: dict(c) for s, c in sorted(counts.items())}, "test_conditions": test_conds,
                "unknown_conditions": sorted({it["condition"] for it in items} - set(DEFAULT_CONDITIONS))}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path, help="folder with images/, labels/ and conditions.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "own")
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    m = build(args.src, args.out, args.val, args.test, args.seed)
    for split, c in m["counts"].items():
        print(f"{split:5s} " + ", ".join(f"{k} {v}" for k, v in sorted(c.items())))
    for d in m["dropped"]:
        print(f"dropped {d['image']}: {d['reason']}")
    if m["unknown_conditions"]:
        print("note: conditions outside day/night/two_row:", ", ".join(m["unknown_conditions"]))
    print(f"wrote {args.out / 'own.yaml'}")


if __name__ == "__main__":
    main()
