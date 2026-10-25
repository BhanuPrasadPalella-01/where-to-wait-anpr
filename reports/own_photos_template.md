# Plate detector fine-tuned on own photos — evaluation report

Template: copy to `reports/own_photos_<place>_<date>.md` and fill in. Produced by
`scripts/own_photos.py` and `scripts/finetune_own.py` (see [`FINETUNE.md`](../FINETUNE.md)).
Do not commit the photos themselves, or images derived from them.

## Photos

| | |
|---|---|
| Place / camera positions | _e.g. two junctions in Abids, pole at 3 m, facing traffic_ |
| Camera / phone, resolution | |
| Dates and times taken | |
| Labelling tool, who labelled, checked by | |
| Photos usable / dropped (reason) | _from `data/own/manifest.json`_ |

Split (groups kept together, seed 0) — from the printed counts:

| condition | train | val | test |
|---|---:|---:|---:|
| day | | | |
| night | | | |
| two_row | | | |

## Settings

| epochs | imgsz | lr0 | freeze | replay | seed | best epoch | training time / device |
|---:|---:|---:|---:|---:|---:|---:|---|
| | 640 | 0.002 | 10 | 1.0 | 0 | | |

## Results

Paste `reports/own_finetune_eval.md` here. For reference, the shipped detector on the two existing
test sets (measured, `reports/plate_yolo26n_results.json`):

| Test set | Photos | mAP50 shipped | Recall shipped |
|---|---:|---:|---:|
| generic_test | 882 | 0.993 | 0.979 |
| indian_test | 47 | 0.799 | 0.750 |

_Own-photo table here._

## Reading

* Which conditions improved, and by more than noise (~0.05 mAP50 with ~25 test photos per condition)?
* Did `generic_test` or `indian_test` drop (forgetting)? If so, the replay share used.
* Typical remaining misses (look at `runs/own_finetune_own_<condition>/`): blur, glare, angle,
  occlusion, two-row plates split into two boxes, ...
* Decision: keep the shipped weights, or replace `weights/final/plate_detector_yolo26n.pt`.
