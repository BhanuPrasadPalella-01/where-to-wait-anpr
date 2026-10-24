# Vehicle detection over the full 10 minutes (all four cameras)

YOLO26s + ByteTrack on every frame (9,000 per camera). "Annotated vehicles" = vehicles in the dataset's
multi-camera ground truth for that camera; a vehicle counts as found when one of our tracks overlaps its
ground-truth box (IoU > 0.3). Extra tracks are mostly parked cars, side-road traffic, and one vehicle split
into two tracks after an occlusion.

| Camera | Vehicles tracked | Annotated vehicles | Found | Found by more than one track |
|---|---|---|---|---|
| 1 | 444 | 179 | 147 (82.1%) | 35 |
| 2 | 370 | 222 | 220 (99.1%) | 60 |
| 3 | 447 | 243 | 239 (98.4%) | 99 |
| 4 | 370 | 189 | 189 (100.0%) | 52 |
| **All** | **1,631** | **833** | **795 (95.4%)** | **246** |

Types: 1,370 cars, 250 trucks and vans, 11 buses. Camera 1 is lowest because it overlooks a busy car park
with many partly hidden vehicles.

Video: [all_cameras_tracked.mp4](all_cameras_tracked.mp4) — the four cameras in sync for the full 10 minutes,
every tracked vehicle boxed by type with its track id, and per-camera counts of vehicles in view and counted
so far (`scripts/roundabout_fullvideo.py`).
