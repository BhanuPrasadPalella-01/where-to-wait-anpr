# Roadmap

Work queue for the project, top to bottom. Each item is sized for about five commits.
Mark an item `[x]` when it is done and add the date.

## Live demo
- [x] (2026-10-17) **Live-mode release** — commit the finished live-camera work: `edge/camera_node.py`
      (video → YOLO26n plates → ByteTrack → OCR vote → hotlist → control room), `DEMO.md`,
      a README "Live demo" section, this roadmap, and `.claude/` in `.gitignore`.
- [x] (2026-10-26) **Vehicle Search release** — commit the finished (uncommitted) Vehicle Search work: `/search` page
      (four camera players, follow-cam, real-map route, check answer) and `/vehicles` page (every vehicle,
      cameras passed, traffic flow, routes); `server/search.py`, `server/ground.py` (camera calibration →
      ground paths, learned roundabout exits), endpoints in `server/app.py`, dashboard link,
      `scripts/roundabout_web_video.py`, `cameratransform` in requirements, `tests/test_search.py`, and the
      results section in `reports/roundabout/README.md`.
- [x] (2026-10-27) **Guided demo** — commit the finished (uncommitted) "? Guided demo" pop-up tour in `server/static/index.html`
      (simulated-chase and live-camera tracks that run the dashboard and wait for real events) and the
      control-room clock fix in `server/app.py` (a failing tick is logged instead of silently stopping every
      dashboard update — the cause of the black map). Split into five commits: clock fix; tour styles and
      markup; simulated-chase steps; live-camera steps; README/DEMO.md mention + this roadmap entry.

## Reliability
- [x] (2026-10-18) **Control-room tests** — pytest tests for `server/control_room.py`: live case opening, first
      sighting starts the belief, feasibility gate flags an impossible journey, officer approval
      moves units, camera heartbeat timeout marks a camera blind, edge read with/without a hotlist
      match. Build a tiny synthetic graph so tests do not need the downloaded city map.
- [x] (2026-10-19) **Graph-free test fixtures** — make `tests/test_roadgraph.py` and `tests/test_planner.py` skip
      cleanly (pytest.skip) when `data/city_graph.json` is missing, and add synthetic-graph versions
      of the Dijkstra/Yen/greedy checks that always run.
- [x] (2026-10-20) **Continuous integration** — GitHub Actions workflow running the graph-free tests on push
      (Python 3.12, CPU only, no model downloads); README badge.

## Experiments
- [x] (2026-10-21) **SUMO replication fix** — fix `planner/sumo_sim.py` (TraCI "End position on lane must be after
      start position" when setting a stop on very short edges / when the vehicle is already past the
      stop position); add a pilot run script; record results in `reports/RESULTS.md`.
- [x] (2026-10-22) **Messaging layer** — signed (HMAC), numbered sighting messages between edge and control room;
      duplicate drop by (camera, seq); out-of-order sightings replayed by capture time (belief
      snapshot + rollback); queue on the edge while the server is down; tests.
- [x] (2026-10-24) **Multiple suspects** — several active cases with officer-set severity weights; greedy plan
      maximises severity-weighted coverage; one unit never pulled between cases; simulator
      experiment and dashboard support.
- [x] (2026-10-24) **Your own city** — `--center LAT,LON --radius M` options for `planner.roadgraph` and the
      simulator; document how to switch city; regenerate nothing large into git.

## Perception
- [x] (2026-10-25) **Fine-tune on own photos** — script and guide to label own plate photos (YOLO format) and
      fine-tune the plate detector from `weights/final/plate_detector_yolo26n.pt`; per-condition
      evaluation (day / night / two-row) report template.
- [ ] **Colour and type check** — dominant-colour (HSV histogram) and vehicle-type attributes on the
      edge node; mismatch lowers the match score; shown in the read log.
