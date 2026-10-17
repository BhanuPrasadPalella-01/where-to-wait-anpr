# Live demo — real cameras reading real plates

What the viewer sees: each camera's video with YOLO26 boxes and the OCR text on every plate, a list of
**every** plate read (match or not), and the map reacting only when a camera reads the wanted plate.
No simulated chase in live mode.

## 1. Start the control room
```bash
.venv/bin/python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
```
Open http://localhost:8000 (others on the same Wi-Fi: http://<your-ip>:8000, `ipconfig getifaddr en0`).

## 2. Start one node per camera (each in its own terminal)
```bash
# Mac webcam
.venv/bin/python -m edge.camera_node --name cam-A --source 0
# iPhone via Continuity Camera (usually index 1)
.venv/bin/python -m edge.camera_node --name cam-B --source 1
# Android phone running the "IP Webcam" app (same Wi-Fi)
.venv/bin/python -m edge.camera_node --name cam-C --source http://PHONE_IP:8080/video
# camera far across the city, for the cloned-plate moment
.venv/bin/python -m edge.camera_node --name cam-FAR --role far --source http://PHONE2_IP:8080/video
# a recorded clip instead of a live camera
.venv/bin/python -m edge.camera_node --name cam-A --source clip.mp4 --loop
```
`--role route` cameras sit on consecutive junctions ~45 s apart along one real road; `--role far` sits
~10 min away. macOS asks once for camera permission for the terminal.

## 3. Script (5 minutes)
1. Let the viewer pick a plate. Type it in **WANTED PLATE**, press **Open live case**.
2. Pass a few *other* toy cars past cam-A: they appear in **Camera reads** as "not wanted" — continuous checking.
3. Pass the wanted car past cam-A: green **HOTLIST MATCH** on the video, **MATCH ✔** in the list, sighting on
   the map with the plate crop, predicted area spreads, a plan appears. Click **Approve**.
4. Move the car to cam-B: second sighting, belief narrows, plan updates.
5. Clone: a second toy car with the **same plate** in front of cam-FAR a few seconds later → red flagged
   sighting: "impossible journey — possible clone or misread. Not used." Units are not diverted.
6. Unplug / cover a phone's stream: after 5 s that camera turns OFFLINE (blind) on the map.

Tabletop time scale: 1 real second = 10 city seconds (shown on screen), because toy cars cross the table
far faster than a real car crosses real junctions.

## Tips
* Print plates large (≥ 8 cm wide), matte paper, single-row first; two-row plates read less reliably.
* Good light, camera ~30–50 cm from the cars, slight downward angle.
* Indian format plates (e.g. `KL02BM4659`) read best — the OCR was fine-tuned on them.
