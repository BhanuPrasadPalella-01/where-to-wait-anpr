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

Remaining errors: mostly substitutions (68% of the 65 wrong test reads), not dropped digits; a few
source labels are wrong. Breakdown and label audit in §2b (the earlier note here, "mostly a dropped last
digit", came from the first eight examples, three of them crops of one two-row plate).
Limitation: the model has 10 character slots; 0.5% of Indian plates have 11.

### 2a. 11-character plates: truncation check (`edge/plate_length.py`, `scripts/eval_truncation.py`, 14 Nov 2026)

No retraining. A read that fills all 10 slots is flagged as likely cut when it is a legal plate with a
3-letter series and fewer than 4 digits (what DL14CTC0153 looks like without its last digit) or not a legal
plate at all. A flagged read from a crop at least 80 px wide is read again from its right half and joined
to the first read where they overlap (≥ 3 characters, exactly one character added, legal 11-character
result). A read still flagged after that is marked for review in the read log and matched against the
hotlist as a prefix (the missing tail costs nothing).

| Split | Crops | Labels > 10 chars | Flagged | Flagged, label > 10 | Recovered by fallback | Flagged, label ≤ 10 | Wrong prefix hit |
|---|---|---|---|---|---|---|---|
| val | 226 | 1 | 1 | 1 | 0 | 0 | 0 |
| test | 267 | 1 | 5 | 0 | 0 | 5 | 0 |
| train (seen in fine-tuning) | 1,216 | 8 | 12 | 5 | 1 | 7 | 0 |

Wrong prefix hit: a flagged read of a ≤ 10-character plate whose prefix lookup hits another plate on a
hotlist of 100,000 synthetic plates + all labels.

* Only two of the ten long labels are legal 11-character plates (DL14CTC0153, PB03AGT8979); the rest are
  trade plates (MH14TC206AN) or label noise (KL01KL0KL01, GJW115A1138, a 12-character KA42TC131011).
  Of their three crops, two are flagged. The fallback recovers DL14CTC0153 exactly (tail read `TC0153`); the
  read of PB03AGT8979 is wrong before the cut (`PB03AGT189`), so the join would be 12 characters and is
  refused. The third crop (DL14CTC0153, 66 px wide) reads `DL14CT0153`: a middle letter is lost, and the
  result is a legal 10-character plate, so it is not flagged.
* The trade plate MH14TC206AN reads as `MH14TC2061`, a legal 10-character plate: the format check cannot
  see it, and is not meant to.
* All 5 test-split flags are on ≤ 10-character plates: 3 crops labelled MH14TCF460 (a 3-letter series with
  3 digits; real or a label missing a digit, it fits in 10 slots) and 2 misreads that break the format. A false flag costs a review note, not an alert: no
  flagged read hit a wrong plate by prefix.
* Effect on the match: with plain fuzzy matching, `DL14CTC015` scores conf × e^(−1.5) = 0.22 × conf
  against DL14CTC0153 (one insertion); as a flagged prefix it scores conf. That keeps a low-confidence
  read above θ = 0.05.
* The numbers are tiny (2 legal long plates in 1,709 crops). They show the mechanism works on real crops,
  not a rate; §3's hotlist numbers are unchanged, because untruncated reads match exactly as before.

### 2b. Error analysis and label audit (`scripts/ocr_errors.py`, `scripts/label_audit.py`, `scripts/eval_ocr_errors.py`, 15 Nov 2026)

Every val and test crop read by the fine-tuned model, format-corrected, and aligned to its label with the
cheapest edit script (ties blame the rightmost character). Neither split was trained on; val chose the
checkpoint and θ. Counts in `reports/ocr_errors.json`.

| Error type | Val (226 crops) | Test (267 crops) |
|---|---|---|
| Exact | 160 (70.8%) | 202 (75.7%) |
| Dropped last character | 1 (0.4%) | 2 (0.7%) |
| Dropped other character(s) | 0 | 1 (0.4%) |
| One substitution | 22 (9.7%) | 31 (11.6%) |
| Several substitutions | 28 (12.4%) | 13 (4.9%) |
| Insertion(s) | 2 (0.9%) | 1 (0.4%) |
| Mixed (substitution + drop / insertion) | 9 (4.0%) | 8 (3.0%) |
| Different plate (more than half the label differs) | 4 (1.8%) | 9 (3.4%) |

| Where the edits fall (misreads of the same plate) | Val | Test |
|---|---|---|
| State code (first 2 characters) | 19 (15.2%) | 16 (16.3%) |
| Middle (district, series) | 47 (37.6%) | 29 (29.6%) |
| Number, except its last digit | 31 (24.8%) | 29 (29.6%) |
| Last character | 28 (22.4%) | 24 (24.5%) |

* Substitutions are the error, not drops: 44 of the 65 wrong test reads (68%) and 50 of 66 on val are
  substitutions only; 3 test reads drop a character, all three crops of one two-row plate (MH47AV6753 →
  MH47AV675, MH47AV673). An exact-match hotlist loses these reads; the radius-2 fuzzy match keeps the
  single substitutions (§3).
* The last character is the weakest single position: about a quarter of all edits on roughly a tenth of
  the characters. It is usually read as another digit, not lost.
* No confusion pair dominates: 192 substitutions spread over 112 pairs, the top ones E→C (6), 8→0 (5),
  U→D (5), 8→3, 4→0, 0→O, 1→7, 4→2, T→I (4 each). Crops of one plate repeat, so pairs cluster by plate (all
  six E→C are val crops of two plates); these are too few to re-weight the hotlist's confusion costs.
* Format correction changes 26 of 493 reads: it fixes 2, breaks 2 and leaves 22 wrong. Two of those are a
  Q read as 0 with confidence > 0.9 (TN19Q0835, UP45Q3028), which the correction turns into O: Q is not in
  its digit-to-letter table.

**Label audit.** A crop is listed when the model and the format check both speak against its label:
the label is not a legal plate, or it is legal but the model reads a different legal plate with every
character ≥ 0.9 confident. 18 of 493 crops (18 labels) are listed; each crop was then looked at by eye.

| Reason | Crops | By eye |
|---|---|---|
| Label not legal, read legal, look-alike apart | 2 | 2 labels wrong: JK1182360 is JK11B2360, WB02A03017 is WB02AD3017 |
| Label not a legal plate | 9 | 3 labels wrong (MH02B16890 has BI, BF06P00006 starts BR, AP310U7562 most likely 31BU); 2 trade plates labelled correctly (MH20TC189B, MH14TC206AN); 3 not Indian Latin-script plates (a dealer plate "TERRANO", a Devanagari plate labelled DEVANAGRI, a German plate); 1 too blurred to tell (PB2313752) |
| Legal label, confident read of another plate | 1 | label wrong: the fourth crop labelled MH47AV6753 shows MH12FT9458, which the model reads exactly |
| Legal label, confident near-miss read | 6 | 0 labels wrong: six confident misreads (MH01CY6333 → CV, MH02ER9194 → 9192, MH01DE2780 → DC2700, TN63BV7954 → 7952, and the two Q → 0 reads) |

* 6 of the 18 listed labels are wrong, and 5 of the 11 illegal ones are plates the 10-slot Indian format
  cannot describe. The "confident near-miss" rule found no wrong label: a confidence of 0.9 does not make a
  one-character disagreement the label's fault, and those six are real OCR errors at high confidence.
* MH47AV6753, the suspect named above: the label is right on three crops (a two-row plate, last digit
  dropped) and wrong on the fourth, which is another vehicle.
* Effect on §2's headline: with the 6 labels corrected, 3 reads become exact (JK11B2360 on val; WB02AD3017
  and MH12FT9458 on test) — test 204/267 = 76.4% instead of 75.7%, val 161/226 = 71.2%. The other three
  reads are still wrong against the corrected label. The tables above and §2 keep the dataset's labels.
* Limits: the audit only sees labels the model disagrees with, so a wrong label the model happens to
  reproduce (it reads DN0909704 as labelled) is not found; the by-eye check is one person on small crops,
  and the train split (1,216 crops, seen in fine-tuning) was not audited.

## 3. Hotlist matching with real OCR output (Layer 3)

Hotlist = 100,000 synthetic Indian plates + half of the split's plates ("wanted"); reads of the other
half act as innocent cars. Hash map for exact hits, BK-tree (radius 2) for near misses, re-scored with an
OCR-confusion-weighted edit distance; alert if score ≥ θ. Val and test share no plates.

θ is chosen on the validation split only — highest wanted-car recall with 0% innocent cars flagged
(ties → larger θ) — frozen in `reports/hotlist_theta.json`, then measured once on test:
**θ = 0.05**.

| Matching | Val: wanted caught | Val: innocent flagged | Test: wanted caught | Test: innocent flagged |
|---|---|---|---|---|
| Exact hash only (plain ANPR) | 69.9% | 0% | 70.1% | 0% |
| **Fuzzy, frozen θ = 0.05** | **80.6%** | **0%** | **87.6%** | **0%** |
| Fuzzy, θ = 0.04 (next lower) | 84.5% | 5.7% | 87.6% | 21.2% |
| Fuzzy, θ = 0.02 | 89.3% | 7.3% | 89.7% | 25.3% |
| Fuzzy, no threshold | 92.2% | 8.1% | 90.7% | 28.8% |

Val: 226 reads (103 wanted, 123 innocent). Test: 267 reads (97 wanted, 170 innocent).

* The frozen θ gives 0 innocent cars flagged on test too, so the earlier test-set choice was not an
  overfit; the margin is thin, though: one grid step lower (0.04) flags 21% of innocent test cars.
* With a realistic hotlist, 29% of innocent test plates lie within 2 edits of *some* wanted plate —
  near-match alone must never trigger a stop (the Aurora failure mode). The confidence-weighted
  score separates them.
* "No threshold" now counts only reads that have a candidate within radius 2; the earlier table
  counted every innocent read as flagged at θ = 0 (100%).
* BK-tree at radius 2 on a dense 100k list checks ~43–46k plates per query (about 55% fewer than a
  linear scan; the count shifts a little between runs with the tree's insertion order) — a real but
  modest saving; say so honestly.

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
.venv/bin/python scripts/eval_ocr_errors.py             # error types, confusion pairs, label audit (§2b)
.venv/bin/python scripts/eval_hotlist.py --split val    # choose and freeze θ
.venv/bin/python scripts/eval_hotlist.py --split test   # measure once at the frozen θ
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

---

## 11. Camera-to-camera tracking on real footage (RoundaboutHD)

Data: [RoundaboutHD](https://arxiv.org/abs/2507.08729) — four real 4K traffic cameras at roundabouts along
~2.3 km in Carmel, Indiana, 10 minutes each, with ground-truth vehicle ids shared across cameras. Plates
are deliberately unreadable in this dataset, so this measures the system's plate-unreadable fallback.

Pipeline (`scripts/roundabout_fullpass.py`, `scripts/roundabout_eval.py`): each camera's video is streamed
out of the 8 GB dataset zip (`scripts/remote_zip.py`), YOLO26s + ByteTrack track every vehicle (1,631
tracks), each track gets a ResNet-50 + colour-histogram fingerprint, the suspect is marked once in its
first camera, and each next camera picks the most similar candidate. Minimum driving times between
cameras come from Dijkstra on the OpenStreetMap road graph. Calibration (typical arrival ratio, entry
direction) uses even-numbered vehicles only; results are on the odd-numbered ones.

| Method | Hand-offs | Top-1 correct | Top-5 | Candidates per hand-off | 4-camera journeys fully right |
|---|---|---|---|---|---|
| Appearance only | 245 | 21.2% | 33.9% | 195.9 | 3.3% of 30 |
| + travel time (loose window) | 244 | 23.0% | 37.7% | 123.8 | 3.3% of 30 |
| **+ road-graph travel time + entry direction** | 245 | **46.1%** | **75.9%** | **17.7** | **26.7% of 30** |

The road-graph time window and direction cut the candidates from ~124 to ~18 and double top-1 accuracy.
Direction and travel-time masks are established practice in multi-camera tracking (AI City Challenge);
the contribution here is measuring them inside this system on real footage. Journey reels are generated
locally (`scripts/roundabout_reel.py`) and not committed.
