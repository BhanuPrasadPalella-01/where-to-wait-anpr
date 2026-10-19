# Where to Wait — re-planning police interception from sparse, uncertain ANPR sightings

A student research prototype. Cameras read number plates; this project asks the harder question that
ANPR systems leave open: **given a few uncertain sightings of a wanted car, where should patrol units
wait so they get there first?**

Each sighting is checked for physical plausibility (catches misreads and cloned plates), updates a
particle belief over where the car can be on the road graph, and the units are re-assigned by a
greedy coverage planner. An officer confirms every step; nothing here issues stop orders.

![Interception rate vs camera coverage](reports/final_coverage.png)

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
           reconstruct.py route reconstruction between sightings (Yen's K paths, space-time feasibility)
           sumo_graph.py  planner graph from a SUMO network
           sumo_sim.py    SUMO/TraCI replication (work in progress, see below)
server/    control_room.py (case state, gate, belief, plan, officer approval, alerts), db.py (SQLite store),
           app.py (FastAPI + WebSocket), static/index.html (Leaflet dashboard), static/unit.html (patrol phone)
edge/      camera_node.py (webcam / phone / video -> YOLO26n -> ByteTrack -> OCR vote -> hotlist -> control room)
scripts/   prepare_data.py, train_plate_detector.py, eval_ocr_baseline.py, indian_plate_format.py,
           hotlist.py, eval_hotlist.py, read_plates.py, plot_results.py
tests/     test_roadgraph.py (vs NetworkX), test_planner.py (gate, belief, greedy vs exact) on the
           city map (skipped if it is missing); test_roadgraph_synthetic.py, test_planner_synthetic.py
           (the same checks on synthetic graphs, always run), test_camera_node.py (OCR vote, read backlog), test_control_room.py (cases, gate, approval,
           camera heartbeat, edge reads; runs on the synthetic grid in synthetic.py, no city map needed),
           test_db.py, test_reconstruct.py, test_control_room_records.py (registry, routes, alerts)
weights/final/  plate_detector_yolo26n.pt, indian_plate_ocr.onnx (+ config)
reports/   RESULTS.md, fact_check.md, result tables and raw episode logs (.jsonl)
```

## System overview

How each part of the original design maps to the code:

| Original design | Implementation |
|---|---|
| Traffic cameras | `edge/camera_node.py` — any webcam, phone (IP Webcam / Continuity Camera) or video file |
| YOLO + ANPR (vehicle/plate detection) | YOLO26n plate detector + ByteTrack + fine-tuned Indian OCR, one read per vehicle pass |
| Detection processing server | `server/app.py` (FastAPI): camera registration, reads, video streams, WebSocket to the dashboard |
| Database | `server/db.py` (SQLite, `data/control_room.db`): registry, sightings, plate reads, reconstructed routes, alerts, audit log |
| Graph tracking engine | `planner/` — road graph from OpenStreetMap, feasibility gate, route reconstruction, particle belief |
| Police web dashboard | `server/static/index.html` — live map, camera feeds, plate-read log, route history, plan approval |
| Criminal (vehicle) registration | Registry tab: plate, case/FIR number, offence, colour, type, make, severity, officer; 28-day expiry |
| Route prediction | particle belief over three fugitive models → predicted positions; greedy plan of where units wait |
| Live vehicle tracking | sightings from real camera reads; route between sightings rebuilt with Yen's K shortest paths |
| Police alerts & notifications | station alert per sighting; unit assignment alerts on approval; patrol phone page `/unit/<name>` with Acknowledge |

### Database

SQLite file `data/control_room.db` (created on first start, not committed). Registered vehicles need a case
number, offence and officer, and expire after 28 days. Only hotlist hits are kept long term; plate reads
that matched nothing are deleted after a day. Every registration, decision, approval and acknowledgement
is written to the audit log (`GET /api/audit`).

### Route reconstruction

Between two accepted sightings the planner takes Yen's three shortest road paths between the cameras and
keeps those the car could have driven in the observed time (`(1 − σ)·T ≤ Δt`), ranked by how well their
travel time matches the gap. The best path is drawn solid on the map, alternatives dashed; segments are
stored per vehicle and can be replayed later from the Registry tab (**History**).

### Police alerts

Approving a plan sends each moved unit an assignment alert; each accepted sighting alerts all stations;
interception or escape is broadcast to all units. Patrol units open `http://<server>:8000/unit/Unit%201`
on a phone: assignment, ETA, wanted-vehicle details with the plate photo, map, and **Acknowledge**.
An assignment not acknowledged within 30 s is flagged on the dashboard.

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q tests                # all unit tests; city-map tests skip until the map exists
.venv/bin/python -m planner.roadgraph            # download the OSM road graph (edit DEFAULT_CENTER for your city)
.venv/bin/python -m tests.test_roadgraph         # full-size runs on the city map, with timings
.venv/bin/python -m tests.test_planner
.venv/bin/python -m planner.simulate --name demo --coverages 0.1,0.3 --runs 50
.venv/bin/python scripts/plot_results.py demo
```

Control-room dashboard (live demo): `.venv/bin/python -m uvicorn server.app:app --port 8000`, open
http://localhost:8000, press **Start demo chase**. Sightings appear, the predicted area spreads, the
officer approves each plan, units drive to their junctions; **Inject cloned plate** shows the
feasibility gate flagging an impossible journey, **Camera offline** marks a camera blind. Real edge
cameras post to `POST /api/sighting` (`{"camera": <junction>, "plate": "...", "score": 0.93}`).

## Live demo

Real cameras reading real plates, no simulated chase: start the control room, then one edge node per
camera (Mac webcam, iPhone via Continuity Camera, an Android phone running "IP Webcam", or a video file):

```bash
.venv/bin/python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
.venv/bin/python -m edge.camera_node --name cam-A --source 0
.venv/bin/python -m edge.camera_node --name cam-FAR --role far --source http://PHONE_IP:8080/video
```

Each node tracks plates with ByteTrack, reads the three sharpest crops of every track, votes over the
OCR strings and fuzzy-matches the hotlist served by the control room. Every read appears in the
dashboard's read log; only hotlist matches become sightings on the map. A `--role far` camera shows the
cloned-plate check, and a camera whose stream stops for 5 s is marked blind. Reads made while the
control room is down are queued on the node and replayed. Full script and tips: [`DEMO.md`](DEMO.md).

Perception: `scripts/prepare_data.py` downloads the public datasets; `scripts/read_plates.py <images>`
runs detector + OCR end to end with the weights in `weights/final/`.

## Status

* Done: perception (detector, Indian OCR, hotlist), road graph + own algorithms, gate, belief,
  planner, chase simulator with baselines, coverage / clone / fleet-size experiments.
* Done: control-room dashboard (FastAPI + WebSocket + Leaflet) with officer approval, clone flagging,
  camera-offline handling and a demo chase.
* Done: live mode with real edge cameras (`edge/camera_node.py`, [`DEMO.md`](DEMO.md)).
* Done: pytest tests for the control room on a synthetic grid graph.
* Done: graph-free Dijkstra/Yen and greedy-vs-exact tests; city-map tests skip when the map is missing.
* Next: see [`ROADMAP.md`](ROADMAP.md) — CI, SUMO replication fix (TraCI stop
  position on very short edges), signed messaging with replay, multiple suspects.

## Data and ethics

Datasets are not redistributed: Roboflow licence-plate set (CC BY 4.0), Datacluster Indian plates
(CC BY-NC-ND 4.0, test only), zenitsu09 Indian plate crops (licence not stated). The design keeps a
human officer in the loop, alerts only above a confidence threshold, and keeps only hotlist hits —
see the Aurora (2020) wrongful-stop case in `reports/fact_check.md`.

## Licence

AGPL-3.0 (required by Ultralytics YOLO, which this project uses).
