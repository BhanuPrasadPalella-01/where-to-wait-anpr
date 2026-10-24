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
