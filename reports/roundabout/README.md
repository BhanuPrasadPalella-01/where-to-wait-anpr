# Camera-to-camera tracking on real footage

Journey reels: one vehicle followed through four real traffic cameras by this project's pipeline
(YOLO26s + ByteTrack + appearance fingerprint + road-graph travel time + entry direction). The suspect is
boxed in green with its match score; other vehicles are grey. Every hand-off shown matches the dataset's
ground truth. Accuracy over all held-out vehicles is in [eval.md](eval.md).

| Reel | Vehicle | Cameras |
|---|---|---|
| [reel_v699.mp4](reel_v699.mp4) | dark-blue dump truck | 1 → 2 → 3 → 4 |
| [reel_v663.mp4](reel_v663.mp4) | white utility van | 1 → 2 → 3 → 4 |
| [reel_v705.mp4](reel_v705.mp4) | black SUV | 4 → 3 → 2 → 1 |

`reel_v*_strip.jpg` shows the same vehicle as each camera saw it.

## All vehicles that cross the four cameras

[four_camera_vehicles.mp4](four_camera_vehicles.mp4): only the 74 vehicles that pass through all four cameras
(per the dataset's ground truth), boxed by our pipeline, each with one colour and id in every camera; other
traffic is left unboxed. Per-camera and overall counters; 2x speed, frames without these vehicles dropped.
Our system detected 61 of the 74 in all four cameras and the other 13 in three of four
(`scripts/roundabout_4cam_video.py`).

## One vehicle's journey on all four cameras

[journey_grid_v699.mp4](journey_grid_v699.mp4): the four cameras side by side, trimmed to the 135 s in which the
dump truck (vehicle 699) drives camera 1 → 2 → 3 → 4. Whichever camera it is in gets a green border and the
truck a green SUSPECT box with its match score; a banner ticks off each camera; stretches where it is in no
camera play at 4× speed (`scripts/roundabout_journey_grid.py`).

## Full video and detection counts

[all_cameras_tracked.mp4](all_cameras_tracked.mp4): all four cameras for the full 10 minutes with every
tracked vehicle boxed. Our pipeline found 795 of the 833 annotated vehicle passes (95.4%); details in
[detections.md](detections.md).

## Footage credit

Video is from **RoundaboutHD: High-Resolution Real-World Urban Environment Benchmark for Multi-Camera
Vehicle Tracking** (University of Bath and Starwit Technologies), [arXiv:2507.08729](https://arxiv.org/abs/2507.08729),
data at [researchdata.bath.ac.uk/1574](https://researchdata.bath.ac.uk/1574/). The cameras were set up by the
dataset authors so that no readable licence plate or face is recorded. These short clips are shown for
academic demonstration with credit to the authors; they will be removed on request.

## Vehicle Search and All vehicles pages

With the control room running (`uvicorn server.app:app`), two pages work on this footage:

- **`/search`**: click a vehicle in any of the four camera players (or type its track number, e.g. `C434`) and
  the system follows it on its own: which way it left each roundabout, road travel time and appearance; no
  plates, no answer key. It shows a zoomed follow-cam, the vehicle's real path through each roundabout on the
  map, the road between cameras, and why the trail ends ("turned off onto a side road", "no confident
  match"). *Check answer* compares every hand-off with the dataset's ground truth.
- **`/vehicles`**: every annotated vehicle: how many cameras it passed, traffic between the cameras in each
  direction, all routes on the map, and whether the system recovered each route by itself.

Paths on the map: each box's bottom-centre goes through the camera calibration that ships with RoundaboutHD
(`cameratransform` format) onto the road plane (`server/ground.py`). Whether a vehicle drove on east, west or
turned off is a 7-nearest-neighbour vote over the even-numbered ground-truth vehicles (exit position and
heading on the ground): 71–96 % correct on the odd-numbered vehicles, depending on the roundabout.

Starting from any annotated vehicle (odd ids, held out), the hand-offs the page shows:

| Search | Right | Wrong | Precision | Recall |
|---|---|---|---|---|
| always search the next camera (old) | 324 | 2,080 | 13 % | 44 % |
| only where the vehicle drove on, match ≥ 0.5, candidate came from the right side | 281 | 337 | 45 % | 38 % |

Most vehicles leave the four-camera corridor at a roundabout, so searching the next camera regardless picked
some other vehicle. Match scores of 0.9 or more were right 71 % of the time, 0.8–0.9 55 %, below 0.8 about
40 %; the page labels matches strong / likely / weak accordingly. Out of 287 vehicles seen by two or more
cameras, the system recovered the whole route by itself for 71 (25 %).

The players need browser-playable copies of the camera previews: `python scripts/roundabout_web_video.py N`
for N = 1..4 (writes `data/roundabout/web/camN.mp4`).
