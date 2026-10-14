# Fact check of "Criminal Vehicle Tracking — Critical Review & Rebuild" (8 Oct 2026)

Checked against live sources on 8 Oct 2026.

## Claims checked

| Claim in the PDF | Result | Note |
|---|---|---|
| YOLO26 is real, Ultralytics, launched in v8.4.0 (Jan 2026) | ✅ Correct | `pip` lists ultralytics 8.4.0 → 8.4.174; docs confirm Jan 2026 release |
| YOLO26 COCO table (n 40.9 / s 48.6 / m 53.1 / l 55.0 / x 57.5 mAP, params, FLOPs, latencies) | ✅ Exact match | docs.ultralytics.com/models/yolo26 |
| NMS-free, DFL removed, MuSGD, STAL + ProgLoss, AGPL-3.0 | ✅ Correct | same page |
| YOLO26 does not read characters; OCR is separate | ✅ Correct | |
| Aurora (CO) ALPR stop, Aug 2020, $1.9M settlement | ✅ Correct | stop: 2 Aug 2020; settlement: Feb 2024 (Gilliam family; plate matched a stolen Montana motorcycle) |
| UK National ANPR Service: 12,076 fixed cameras, 80–100M reads/day | ⚠️ Outdated | Home Office DPIA v3.1 (Mar 2024): **18,000+ cameras, ~90M reads/day**; the 2021 DPIA said ~12,700 cameras / ~60M reads. Cite the 2024 figure |
| van Droffelaar et al., graph coarsening for fugitive interception, Applied Network Science 2025 | ✅ Exists | code + data on 4TU.ResearchData — reusable as a baseline |
| Samanta, Uniyal, Sen, Ghosh — vehicle interdiction in SUMO on IIT Kharagpur map (arXiv 2305.14372) | ✅ Exists | May 2023 |
| Indian-LPR dataset: 16,192 images, 21,683 plates (arXiv 2111.06054) | ⚠️ Exists but **not downloadable** | Repo README: "We can't make dataset public because of legalities". The PDF's Phase 14/15 plan depends on it — replaced below |
| fast-plate-ocr exists, MIT, ONNX | ✅ Correct | PyPI 1.1.0 |

## Feasibility on the user's machine (MacBook Air M4, 16 GB)

* PyTorch 2.14.1 with Apple MPS works; YOLO26n trains at ~1.3 it/s (batch 16, 640 px) → ~5.5 min/epoch on 6,176 images.
* YOLO26n inference measured at 1.3 ms/image on MPS (validation pass) — real-time for 4–6 phone cameras is plausible; measure end-to-end FPS in the demo.

## Data actually used (replacing Indian-LPR)

| Stage | Dataset | Licence | Use |
|---|---|---|---|
| Plate detector | keremberke/license-plate-object-detection (Roboflow) — 6,176 / 1,765 / 882 | CC BY 4.0 | train / val / test |
| Plate detector, Indian check | Dataclusterlabspvtltd/indian-number-plates-dataset — 47 real Indian photos | CC BY-NC-ND 4.0 | **test only** |
| OCR | zenitsu09/indian-number-plate — 1,709 crops, 965 unique plates | not stated (check before publishing) | split by plate number 70/15/15 |

Data pitfalls found and fixed:
* The Indian OCR set contains Roboflow augmentation copies (same plate ×3). A random image split would leak test plates into training; the split is grouped by plate number.
* Several Datacluster phone photos carry EXIF rotation; boxes were drawn on the rotated view. Images are now EXIF-transposed before labels are written (see `label_check_indian.jpg`).
