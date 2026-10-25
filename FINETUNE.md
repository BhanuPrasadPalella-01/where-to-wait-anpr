# Fine-tuning the plate detector on your own photos

The shipped detector (`weights/final/plate_detector_yolo26n.pt`) reaches mAP50 0.993 on the generic
test split but 0.799 on 47 real Indian photos: it was trained on foreign plates only. The quickest way
to close that gap for your cameras is a few hundred labelled photos from where the cameras will stand.
This guide covers taking, labelling, splitting, fine-tuning and evaluating them per condition.

Your photos stay local: `data/` and `runs/` are git-ignored. Do not commit them, and do not publish
photos that show faces or readable plates of people who did not agree to it.

## 1. Take the photos

* Aim for at least 150 photos, ideally 300+, from the camera positions and heights you will use.
* Cover each condition you care about, roughly evenly. The scripts expect these names, though any
  other name also works and gets its own row in the report:

  | condition | what it means |
  |---|---|
  | `day` | daylight, single-row plate |
  | `night` | street light, headlights, IR or dusk |
  | `two_row` | two-line plates (autos, trucks, many two-wheelers) |

* Several shots of the same car in a row are near duplicates. Give them the same `group` (step 3) so
  they all land in the same split; otherwise the test score is inflated.

## 2. Label in YOLO format

Any tool that exports YOLO detection labels works, for example
[labelImg](https://github.com/HumanSignal/labelImg) (choose *YOLO* format) or CVAT / Label Studio
(*YOLO 1.1* export). One class only, `plate`, index 0. Draw the box tight around the plate including
its border. A photo without any plate can keep an empty `.txt` file (useful negatives).

Folder layout:

```
my_photos/
  images/  IMG_0001.jpg  IMG_0002.jpg ...
  labels/  IMG_0001.txt  IMG_0002.txt ...      0 0.512 0.633 0.081 0.034   (class cx cy w h, 0..1)
  conditions.csv
```

## 3. List the conditions

`conditions.csv`, one row per photo (`group` is optional, defaults to the photo itself):

```csv
image,condition,group
IMG_0001.jpg,day,car01
IMG_0002.jpg,day,car01
IMG_0003.jpg,night,car02
IMG_0004.jpg,two_row,auto07
```

## 4. Build the dataset

```bash
.venv/bin/python -m scripts.own_photos my_photos/              # writes data/own/
```

It checks every label (wrong number of values, class other than 0, pixel coordinates instead of
0..1 fractions), drops photos that are unlabelled, broken or missing from `conditions.csv` and prints
why, then splits by group into train / val / test (60 / 15 / 25 %, stratified by condition, seed 0)
and writes `data/own/own.yaml`, one `own_test_<condition>.yaml` per condition and `manifest.json`.

## 5. Fine-tune and evaluate

```bash
.venv/bin/python -m scripts.finetune_own --epochs 30           # fine-tune, then score both models
.venv/bin/python -m scripts.finetune_own --eval-only           # re-score an existing run
```

Defaults and why:

| option | default | reason |
|---|---|---|
| `--base` | `weights/final/plate_detector_yolo26n.pt` | adjust the trained detector, do not start over |
| `--lr0` | 0.002 | a tenth of a from-scratch rate, so few photos cannot wreck the weights |
| `--freeze` | 10 | backbone frozen; only neck and head adapt to the new plates |
| `--replay` | 1.0 | as many generic training images as own ones, mixed in, against forgetting |
| `--device` | auto | CUDA, then Apple `mps`, then CPU |

The best epoch (on your val split) is saved to `runs/own_finetune/weights/best.pt`. Both the shipped
and the fine-tuned model are then scored on every own condition, all own test photos, and, if `data/`
has them, the generic test split and the 47 Indian photos. Results go to
`reports/own_finetune_eval.json` and `reports/own_finetune_eval.md`; copy the table into
[`reports/own_photos_template.md`](reports/own_photos_template.md) and fill in the rest.

## 6. Use the new weights

If the fine-tuned model is better on your conditions and has not lost much on `generic_test`, copy it
over the shipped weights (keep a backup; `*.pt.bak` is ignored by git):

```bash
cp weights/final/plate_detector_yolo26n.pt weights/final/plate_detector_yolo26n.pt.bak
cp runs/own_finetune/weights/best.pt weights/final/plate_detector_yolo26n.pt
```

The edge node (`edge/camera_node.py`) loads that file at start, so restart the nodes.

## Reading the numbers

* With 25 % of 300 photos in test, each condition has only ~25 photos: a change under ~0.05 mAP50 is
  within noise. Report the photo counts with the scores (the table does).
* A drop on `generic_test` means forgetting: raise `--replay` or lower `--epochs`.
* No gain at night usually means too few night photos; add some before training longer.
