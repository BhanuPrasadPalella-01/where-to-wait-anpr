"""Fine-tune the plate detector on your own photos and compare it with the shipped one, per condition.

Starts from weights/final/plate_detector_yolo26n.pt (not from the COCO yolo26n.pt), uses a small
learning rate and frozen early layers, and by default mixes a random share of the generic training set
back in ("replay") so the detector does not forget the plates it already finds. After training, both
the shipped and the fine-tuned model are scored on:
  - each condition of your own test split (day / night / two_row / ...), built by scripts/own_photos.py
  - the generic test split and the 47 Indian photos, if data/ has them (to see any forgetting)
and the numbers are written to reports/<name>_eval.json and reports/<name>_eval.md.

  .venv/bin/python -m scripts.own_photos my_photos/
  .venv/bin/python -m scripts.finetune_own --epochs 30
  .venv/bin/python -m scripts.finetune_own --eval-only          # re-score an existing run
"""
import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "weights" / "final" / "plate_detector_yolo26n.pt"
METRICS = ("precision", "recall", "mAP50", "mAP50-95")


def replay_list(own: Path, generic_train: Path, share: float, seed: int = 0) -> Path:
    """Text file listing own training images plus `share` x as many random generic training images."""
    own_imgs = sorted(str(p) for p in (own / "train" / "images").iterdir())
    gen = sorted(str(p) for p in generic_train.iterdir()) if generic_train.exists() else []
    k = min(len(gen), round(share * len(own_imgs)))
    picked = random.Random(seed).sample(gen, k) if k else []
    path = own / "train_with_replay.txt"
    path.write_text("\n".join(own_imgs + picked) + "\n")
    return path


def train_yaml(own: Path, share: float, seed: int) -> Path:
    """own.yaml as is, or a copy whose train entry is the replay list."""
    if share <= 0:
        return own / "own.yaml"
    lst = replay_list(own, ROOT / "data" / "generic" / "train" / "images", share, seed)
    text = (own / "own.yaml").read_text().replace("train: train/images", f"train: {lst.name}")
    path = own / "own_replay.yaml"
    path.write_text(text)
    return path


def eval_sets(own: Path) -> list[tuple[str, Path]]:
    """(tag, yaml) for every set that exists: own conditions first, then the shipped test sets."""
    sets = [(f"own_{p.stem.removeprefix('own_test_')}", p) for p in sorted(own.glob("own_test_*.yaml"))]
    sets.append(("own_all", own / "own.yaml"))
    for tag, name in (("generic_test", "plates.yaml"), ("indian_test", "plates_indian_test.yaml")):
        if (ROOT / "data" / name).exists():
            sets.append((tag, ROOT / "data" / name))
    return sets


def render_table(results: dict, n_test: dict | None = None) -> str:
    """Markdown table: one row per test set, shipped vs fine-tuned mAP50 / recall and the change."""
    n_test = n_test or {}
    rows = ["| Test set | Photos | mAP50 shipped | mAP50 fine-tuned | Δ | Recall shipped | Recall fine-tuned | Δ |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for tag in results["shipped"]:
        a, b = results["shipped"][tag], results.get("finetuned", {}).get(tag)
        cells = [tag, str(n_test.get(tag, "")), f"{a['mAP50']:.3f}"]
        if b:
            cells += [f"{b['mAP50']:.3f}", f"{b['mAP50'] - a['mAP50']:+.3f}"]
        else:
            cells += ["–", "–"]
        cells.append(f"{a['recall']:.3f}")
        cells += [f"{b['recall']:.3f}", f"{b['recall'] - a['recall']:+.3f}"] if b else ["–", "–"]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows) + "\n"


def score(weights: Path, sets, imgsz: int, device: str, name: str) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    out = {}
    for tag, yaml in sets:
        m = model.val(data=str(yaml), split="test", imgsz=imgsz, device=device, plots=False, verbose=False,
                      project=str(ROOT / "runs"), name=f"{name}_{tag}", exist_ok=True)
        out[tag] = dict(zip(METRICS, (m.box.mp, m.box.mr, m.box.map50, m.box.map)))
        print(f"{weights.parent.parent.name if weights != BASE else 'shipped':>12s} {tag:14s} "
              f"mAP50 {out[tag]['mAP50']:.3f}  recall {out[tag]['recall']:.3f}")
    return out


def pick_device(arg: str) -> str:
    if arg != "auto":
        return arg
    import torch

    if torch.cuda.is_available():
        return "0"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--own", type=Path, default=ROOT / "data" / "own", help="output of scripts/own_photos.py")
    ap.add_argument("--base", type=Path, default=BASE)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr0", type=float, default=0.002, help="small: we adjust a trained detector")
    ap.add_argument("--freeze", type=int, default=10, help="freeze the first N layers (backbone); 0 = none")
    ap.add_argument("--replay", type=float, default=1.0,
                    help="generic training images mixed in, as a multiple of your own count (0 = off)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto", help="auto / mps / cpu / 0")
    ap.add_argument("--name", default="own_finetune")
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()

    if not (args.own / "own.yaml").exists():
        raise SystemExit(f"{args.own / 'own.yaml'} missing: run  python -m scripts.own_photos <folder>  first")
    device = pick_device(args.device)
    best = ROOT / "runs" / args.name / "weights" / "best.pt"
    if not args.eval_only:
        from ultralytics import YOLO

        YOLO(str(args.base)).train(
            data=str(train_yaml(args.own, args.replay, args.seed)),
            epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=device,
            lr0=args.lr0, lrf=0.1, warmup_epochs=1, freeze=args.freeze or None,
            patience=10, seed=args.seed, deterministic=False, workers=4,
            project=str(ROOT / "runs"), name=args.name, exist_ok=True, plots=True,
        )

    sets = eval_sets(args.own)
    results = {"base": str(args.base.relative_to(ROOT)) if args.base.is_relative_to(ROOT) else str(args.base),
               "settings": {k: getattr(args, k) for k in ("epochs", "imgsz", "lr0", "freeze", "replay", "seed")},
               "shipped": score(args.base, sets, args.imgsz, device, f"{args.name}_shipped")}
    if best.exists():
        results["finetuned"] = score(best, sets, args.imgsz, device, args.name)
    manifest = json.loads((args.own / "manifest.json").read_text())
    n_test = {f"own_{c}": n for c, n in manifest["counts"].get("test", {}).items()}
    n_test["own_all"] = sum(manifest["counts"].get("test", {}).values())
    n_test.update({"generic_test": "", "indian_test": 47})

    out = ROOT / "reports" / f"{args.name}_eval"
    out.with_suffix(".json").write_text(json.dumps(results, indent=2))
    out.with_suffix(".md").write_text(render_table(results, n_test))
    print(render_table(results, n_test))
    print(f"wrote {out.with_suffix('.json')} and .md")


if __name__ == "__main__":
    main()
