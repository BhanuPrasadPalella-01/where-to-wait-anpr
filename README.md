# Where to Wait — re-planning police interception from sparse, uncertain ANPR sightings

A student research prototype. Cameras read number plates; this project asks the harder question that
ANPR systems leave open: **given a few uncertain sightings of a wanted car, where should patrol units
wait so they get there first?**

Each sighting is checked for physical plausibility (catches misreads and cloned plates), updates a
particle belief over where the car can be on the road graph, and the units are re-assigned by a
greedy coverage planner. An officer confirms every step; nothing here issues stop orders.

## Headline results (all measured, see [`reports/RESULTS.md`](reports/RESULTS.md))

| | Result |
|---|---|
| Interception, proposed vs plan-once baseline (1,000 paired simulated chases, unseen seeds) | **+9.5 points**, 140 wins / 45 losses, McNemar p = 1.5e-12 |
| Against a fugitive model the planner did not assume | 40.0% vs 34.4% |
| Cloned plate injected mid-chase | belief fooled 93.5% → **6%** with the feasibility gate |
| Greedy planner vs brute-force optimum | ≥ 0.959 (theory: ≥ 0.5), ~5 ms per plan |
| Own Dijkstra / Yen vs NetworkX | identical on 1,000 / 30 random pairs |
| Plate OCR on held-out Indian plates | 44.2% → **75.7%** exact after fine-tuning |
| Fuzzy hotlist match (BK-tree + confusion-weighted distance) | wanted cars caught 70.1% → **87.6%**, 0 innocent cars flagged |
| YOLO26n plate detector | mAP50 0.993 generic test, 0.799 on 47 real Indian photos |

Negative results are reported too: evidence from cameras that did *not* see the car gave no
measurable gain, and the 30 s re-plan timer from the original design made units thrash.

## Layout

```
planner/   roadgraph.py   OSM road graph, own Dijkstra (binary heap) and Yen's K-shortest paths
           fugitive.py    fugitive route models (fastest+2% wrong turns, main-road, random walk, held-out back-streets)
           belief.py      feasibility gate + particle belief (conditioning, rejuvenation, negative evidence)
           plan.py        greedy submodular interception planner + brute-force check
           simulate.py    chase simulator, baselines B0/B1/B2, ablations, parallel sweeps
           sumo_graph.py  planner graph from a SUMO network
           sumo_sim.py    SUMO/TraCI replication (work in progress, see below)
scripts/   prepare_data.py, train_plate_detector.py, eval_ocr_baseline.py, indian_plate_format.py,
           hotlist.py, eval_hotlist.py, read_plates.py, plot_results.py
tests/     test_roadgraph.py (vs NetworkX), test_planner.py (gate, belief, greedy vs exact)
weights/final/  plate_detector_yolo26n.pt, indian_plate_ocr.onnx (+ config)
reports/   RESULTS.md, fact_check.md, result tables and raw episode logs (.jsonl)
```

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m planner.roadgraph            # download the OSM road graph (edit DEFAULT_CENTER for your city)
.venv/bin/python -m tests.test_roadgraph
.venv/bin/python -m tests.test_planner
.venv/bin/python -m planner.simulate --name demo --coverages 0.1,0.3 --runs 50
.venv/bin/python scripts/plot_results.py demo
```

Perception: `scripts/prepare_data.py` downloads the public datasets; `scripts/read_plates.py <images>`
runs detector + OCR end to end with the weights in `weights/final/`.

## Status

This repository is being published in two parts: perception first, the interception planner next.

* Done: perception (detector, Indian OCR, hotlist), road graph + own algorithms, gate, belief,
  planner, chase simulator with baselines, coverage / clone / fleet-size experiments.
* In progress: SUMO replication (`planner/sumo_sim.py` — known bug: TraCI stop position on very short
  edges), MQTT messaging with loss/replay, multiple suspects, dashboard, tabletop demo.

## Data and ethics

Datasets are not redistributed: Roboflow licence-plate set (CC BY 4.0), Datacluster Indian plates
(CC BY-NC-ND 4.0, test only), zenitsu09 Indian plate crops (licence not stated). The design keeps a
human officer in the loop, alerts only above a confidence threshold, and keeps only hotlist hits —
see the Aurora (2020) wrongful-stop case in `reports/fact_check.md`.

## Licence

AGPL-3.0 (required by Ultralytics YOLO, which this project uses).
