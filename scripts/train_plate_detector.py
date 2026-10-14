"""Stage C of the pipeline: YOLO26n licence-plate detector.

Trains on the generic plate set, selects the best epoch on its validation split, then reports
on two test sets the model never trained on: the generic test split and 47 real Indian photos.
"""
import argparse
import json
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "weights" / "yolo26n.pt"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--fraction", type=float, default=1.0, help="share of training images (for timing runs)")
    ap.add_argument("--name", default="plate_yolo26n")
    ap.add_argument("--resume", action="store_true", help="continue an interrupted run from its last.pt")
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()

    last = ROOT / "runs" / args.name / "weights" / "last.pt"
    if args.eval_only:
        pass
    elif args.resume:
        YOLO(str(last)).train(resume=True)
    else:
        YOLO(args.model).train(
            data=str(ROOT / "data" / "plates.yaml"),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device="mps",
            workers=4,
            patience=10,
            seed=0,
            deterministic=False,
            fraction=args.fraction,
            project=str(ROOT / "runs"),
            name=args.name,
            exist_ok=True,
            plots=True,
        )

    best = YOLO(str(ROOT / "runs" / args.name / "weights" / "best.pt"))
    results = {}
    for tag, yaml, split in [("generic_test", "plates.yaml", "test"),
                             ("indian_test", "plates_indian_test.yaml", "test")]:
        m = best.val(data=str(ROOT / "data" / yaml), split=split, imgsz=args.imgsz, device="mps",
                     project=str(ROOT / "runs"), name=f"{args.name}_{tag}", exist_ok=True, plots=True)
        results[tag] = {"precision": m.box.mp, "recall": m.box.mr, "mAP50": m.box.map50, "mAP50-95": m.box.map,
                        "ms_per_image": m.speed["inference"]}
        print(tag, json.dumps(results[tag], indent=2))
    (ROOT / "reports" / f"{args.name}_results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
