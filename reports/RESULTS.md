# Perception results — Criminal Vehicle Tracking (8–9 Oct 2026)

All numbers below were measured on this MacBook Air M4 with the scripts in `scripts/`. Seeds fixed at 0.
See `fact_check.md` for the fact check of the review PDF and the data sources.

## 1. Plate detector — YOLO26n (Stage C)

Trained 30 epochs from COCO-pretrained YOLO26n on 6,176 generic plate images (Roboflow, CC BY 4.0),
640 px, MPS. Best weights: `weights/final/plate_detector_yolo26n.pt`.

| Test set (never trained on) | Precision | Recall | mAP50 | mAP50-95 | ms/img (MPS) |
|---|---|---|---|---|---|
| Generic plates, 882 images | 0.993 | 0.979 | **0.993** | 0.716 | 1.4 |
| Real Indian photos, 47 images | 0.826 | 0.750 | **0.799** | 0.518 | 7.4* |

\* includes larger images; not a throughput benchmark.

The 0.99 → 0.80 drop is the domain gap the review predicted (Indian roads, two-row bike plates,
night). **Next step:** label 200–500 Indian frames (own photos) and fine-tune; report per condition.

## 2. Plate OCR — fast-plate-ocr cct-xs-v2, fine-tuned on Indian plates (Stage D)

1,709 Indian plate crops, split by plate number (no plate in both train and test).
Fine-tuned 85 epochs (early stop, best @55). Model: `weights/final/indian_plate_ocr.onnx`.

| Model (267 held-out Indian crops) | Exact plate | Char acc. | Within 2 edits | CPU ms/plate |
|---|---|---|---|---|
| Pretrained global (India not in its training regions) | 44.2% | 83.8% | 79.0% | 3.1 |
| Pretrained + Indian format correction | 47.6% | 84.3% | 79.0% | 3.1 |
| **Fine-tuned on Indian plates** | **75.7%** | **93.6%** | **92.1%** | ~5 |

Remaining errors: mostly a dropped last digit; ≥1 source label looks wrong (MH47AV6753).
Limitation: the model has 10 character slots; 0.5% of Indian plates have 11.

## 3. Hotlist matching with real OCR output (Layer 3)

Hotlist = 100,000 synthetic Indian plates + 97 reads of "wanted" test plates; 170 reads of other
test plates act as innocent cars. Hash map for exact hits, BK-tree (radius 2) for near misses,
re-scored with an OCR-confusion-weighted edit distance; alert if score ≥ θ.

| Matching | Wanted cars caught | Innocent cars flagged |
|---|---|---|
| Exact hash only (plain ANPR) | 70.1% | 0% |
| Fuzzy, θ = 0.05 | **87.6%** | **0%** |
| Fuzzy, θ = 0.02 | 89.7% | 25.3% |
| Fuzzy, no threshold | 90.7% | 100% |

* With a realistic hotlist, 29% of innocent plates lie within 2 edits of *some* wanted plate —
  near-match alone must never trigger a stop (the Aurora failure mode). The confidence-weighted
  score separates them.
* θ was read off the test set here; for the report choose θ on the validation split and re-measure.
* BK-tree at radius 2 on a dense 100k list checks ~45.6k plates per query (54% fewer than a linear
  scan) — a real but modest saving; say so honestly.

## 4. End-to-end pipeline on the 47 Indian photos

`python scripts/read_plates.py data/indian_det/test/images` → annotated images in `reports/demo/`.
38 plates read. Visual check of the first 24 photos:

* All reads with OCR min-character confidence ≥ 0.9 were correct (single-row car plates).
* Reads below ~0.6 confidence were mostly wrong.
* Two-row motorbike plates and night shots are the main failure (missed or misread).

## What this means for the project

* The perception front end is feasible on a laptop; the numbers above are your own, measured.
* Confidence must gate alerts, as the review's design says — now backed by measurements.
* Biggest gaps to close: Indian detector fine-tuning data, two-row plates, night images.
* Not built yet (the review's actual novelty): feasibility gate on a road graph, particle belief,
  greedy interception planner, SUMO experiments.

## Reproduce

```
python3 -m venv .venv && .venv/bin/pip install ultralytics pandas pyarrow "fast-plate-ocr[onnx]"
.venv/bin/python scripts/prepare_data.py
.venv/bin/python scripts/train_plate_detector.py            # ~2.5 h on M4
.venv/bin/python scripts/eval_ocr_baseline.py test
# OCR fine-tune uses a separate venv with fast-plate-ocr[train] (TensorFlow) — see reports/train_ocr_indian.log
.venv/bin/python scripts/eval_hotlist.py
.venv/bin/python scripts/read_plates.py data/indian_det/test/images
```

---

# Part 2 — Interception planner (9 Oct 2026)

Code: `planner/` (road graph, fugitive models, belief + gate, greedy planner, simulator), tests in `tests/`.
Map: OpenStreetMap drive network, 3 km around Hyderabad Abids/Koti (`planner/roadgraph.py` →
`DEFAULT_CENTER`; change to your own city) — 10,838 junctions, 26,598 road segments.

## 5. Own algorithms vs NetworkX

| Check | Result |
|---|---|
| Our Dijkstra (binary heap), 1,000 random pairs | identical to NetworkX (max diff 0 s); 4.8 ms vs 7.3 ms per pair |
| Our Yen K=5 shortest paths, 30 pairs | identical path costs; 615 ms vs 732 ms per pair |
| Greedy planner vs brute force (20 cases, 3 units, best 25 junctions) | worst 0.959, mean 0.998 of optimum (theory guarantees ≥ 0.5); 5 ms per plan |
| Feasibility gate unit tests | rejects a 50%-too-fast journey, accepts ≤ 30% speeding, handles out-of-order sightings |

## 6. Interception experiments (own chase simulator, before SUMO)

Setup: crime at a central junction, reported after 60 s; suspect drives toward one of 16 exits;
cameras at a share of junctions read the plate with recall 0.9, 2–10 s delay; 4 police units at
random junctions, 60 s dispatch delay; intercept = a unit already waiting ≥ 30 s at a junction when
the suspect reaches it. In half the runs the suspect drives a **back-streets model the planner does
not have**. Methods were tuned on seeds 0–99; **all numbers below are on unseen seeds 10,000+**,
200 chases per point. Plot: `reports/final_coverage.png`.

| Cameras | B0 chase | B1 checkpoints | B2 plan once | **Proposed** | + 30 s timer |
|---|---|---|---|---|---|
| 10% | 0% | 6.0% | 34.5% | **42.5%** | 39.5% |
| 20% | 0% | 5.0% | 37.5% | **47.5%** | 47.0% |
| 30% | 0% | 4.5% | 41.0% | **49.0%** | 45.5% |
| 40% | 0% | 6.5% | 39.5% | **45.0%** | 44.5% |
| 60% | 0% | 6.0% | 39.0% | **55.0%** | 50.0% |

* Proposed vs B2, paired over 1,000 chases: 140 wins / 45 losses, **+9.5 points, McNemar p = 1.5e-12**.
* Against the driver model the planner did not assume: 40.0% vs 34.4% (B2) — the gain holds.
* Re-planning on each sighting beats the review's 30 s timer re-plan (61 / 36, p = 0.014): the timer
  made units abandon good positions (thrashing). **Correction to the review's design.**
* Negative evidence (unseen camera passes) gave **no measurable gain** (45 / 41, p = 0.75). Do not
  claim it as a contribution; report it as a negative result.
* Planner latency: 139 ms mean, 185 ms p95 per plan on the M4 — far below the time between sightings.

## 7. Cloned plate experiment (30% cameras, 200 chases, seeds 20,000+)

A second car with the same plate is seen at a far camera mid-chase.

| | Clone fooled the belief | Interception |
|---|---|---|
| Without feasibility gate | 93.5% | 36.0% |
| **With gate** | **6.0%** | **46.5%** |

Paired: 25 wins / 4 losses, p = 1e-4. The remaining 6% are clones that *were* physically reachable
from the last real sighting (sightings sparse) — no travel-time test can catch those; the colour/type
check and the officer are the backstop.

## 8. Number of police units (30% cameras, 200 chases each, seeds 30,000+)

| Units | B1 checkpoints | B2 plan once | **Proposed** |
|---|---|---|---|
| 1 | 2.0% | 16.0% | **22.5%** |
| 2 | 3.0% | 25.0% | **35.0%** |
| 4 | 7.0% | 43.0% | **51.0%** |
| 6 | 8.0% | 50.0% | **57.5%** |
| 8 | 10.5% | 57.0% | **67.0%** |

Proposed leads at every fleet size (+6.5 to +10 points); the planner never assigns two units to one
junction by construction.

## 9. SUMO replication pilot (30% cameras, 4 units, 20 chases per cell, seeds 10,000+)

Same chase, methods and odd-seed held-out driver as section 6, but the suspect and the units are
SUMO vehicles in background traffic (normal or peak demand, after a 300 s warm-up). The planner still
knows only free-flow times; SUMO adds signals, queues and congestion. Run with
`scripts/sumo_pilot.py --runs 20 --demands normal,peak` (20 min on the M4, 8 workers); table in
`reports/sumo_pilot.md`, rows in `reports/sumo_pilot.jsonl`.

| Demand | B0 chase | B1 checkpoints | B2 plan once | **Proposed** |
|---|---|---|---|---|
| normal | 5% (1/20) | 15% (3/20) | 15% (3/20) | **35% (7/20)** |
| peak | 0% (0/20) | 0% (0/20) | 25% (5/20) | **30% (6/20)** |

* 0 of 160 episodes failed. Before the fixes, 2 of the first 6 trial episodes crashed (camera reads
  arriving before the crime was reported), and units whose stop SUMO refused ("too close to brake")
  drove on without a target. Now a refused stop parks the unit at its junction from its free-flow
  arrival time; this happened in 2 of 160 episodes.
* The ordering matches the own simulator (Proposed > B2 > B1 > B0), at lower rates than section 6
  (49% at 30% cameras): units are slowed by real traffic the planner does not model.
* 20 chases per cell gives wide intervals (Proposed normal: 18–57%), so this is a pilot, not a
  result: the gap to B2 is not significant yet. The full replication needs ~200 chases per cell.

## 10. Two suspects at once (30% cameras, 4 units, 200 paired scenarios, seeds 40,000+)

Two crimes 120 s apart at different central junctions; one suspect is severe (severity 3), the other
minor (1), drawn per seed. Each case has its own belief and gate; one plan covers both
(`planner/multi_sim.py`). Run with `python -m planner.multi_sim --name multi_suspect --runs 200`
(odd seeds use the held-out back-street driver); table from `scripts/multi_suspect_table.py`, rows in
`reports/multi_suspect.jsonl`. A *pull* is a unit re-sent from one suspect to the other while still
useful there; a *hand-off* is a unit released by its case (it no longer covers 5% of that belief) and
re-used for the other.

| Method | Severe caught | Minor caught | Severity-weighted catch | Pulls / ep | Hand-offs / ep | Plan ms |
|---|---|---|---|---|---|---|
| **Severity-weighted, useful locks** | 44.5% | 37.5% | 0.427 | 0.00 | 2.51 | 202 |
| Equal weights, useful locks | 47.0% | 39.5% | 0.451 | 0.00 | 2.98 | 202 |
| Severity-weighted, hard locks | 34.0% | 32.0% | 0.335 | 0.00 | 0.00 | 189 |
| Severity-weighted, no lock | 45.5% | 42.0% | 0.446 | 1.20 | 2.08 | 218 |
| Units split per case | 34.0% | 31.5% | 0.334 | 0.00 | 0.00 | 190 |

* **Locks must release.** Keeping a unit on its case until the case ends (hard locks) or splitting the
  fleet up front both lose about 10 points on the severe suspect (25 wins / 4 losses and 30 / 9 paired,
  McNemar p = 0.0001 and 0.001). Releasing a unit once it covers < 5% of its suspect's belief keeps the
  "never pulled while useful" rule (0 pulls) at no measurable cost against no locking at all (9 / 11,
  p = 0.82), which pulls a unit 1.2 times per episode.
* **Severity weights did not help here.** Equal weights caught the severe suspect slightly more often
  (3 / 8, p = 0.23, not significant). With 4 units and two suspects far apart, the plans rarely compete
  for the same unit, so the weight seldom changes a decision. The weight is kept because the officer
  sets it and it costs nothing; a scenario with fewer units or closer crimes is needed to test it.
* Planning two cases at once costs ~200 ms per plan, the same order as one case.

## Design changes found by the experiments (vs the review PDF)

1. Sightings must **condition** the particle belief (keep particles consistent with the camera and
   time, reuse their direction/exit), not re-seed it — re-seeding made re-planning worse than planning once.
2. **Rejuvenation** (20% fresh particles) and **tempered** negative evidence are needed, or a wrong
   fugitive model collapses the belief (interception fell to 15% against the unknown driver).
3. **Commitment**: a unit already waiting keeps its junction unless a new one covers 3× the weight.
4. Re-plan **on sightings**, not on a timer.
5. Anchor the feasibility gate at the reported crime location and time.

## Still to do

Full-size SUMO replication (200 chases per cell, all coverages; pilot in section 9); severity weights under unit
scarcity (section 10); dashboard + tabletop demo;
Indian detector fine-tuning on own photos; your own city's map.
