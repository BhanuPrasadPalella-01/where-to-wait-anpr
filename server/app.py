"""Dashboard server: FastAPI + WebSocket around the ControlRoom.

Run:  .venv/bin/python -m uvicorn server.app:app --port 8000
Edge cameras post signed, numbered plate reads to POST /api/edge/read (server/messages.py);
POST /api/sighting is an unsigned manual entry, refused when WTW_EDGE_KEY is set.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import time

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ValidationError

from .control_room import ControlRoom
from .db import Database
from .messages import KEY_ENV, load_key

STATIC = Path(__file__).parent / "static"
app = FastAPI(title="Where to Wait — control room")
db = Database()  # data/control_room.db: registry, sightings, reads, routes, alerts, audit log
room = ControlRoom(db=db, edge_key=load_key())  # set WTW_EDGE_KEY to require signed edge reads
state = {"speed": 5.0, "running": False, "show_truth": True}
clients: set[WebSocket] = set()


class SightingIn(BaseModel):
    camera: int
    plate: str | None = None
    score: float = 1.0
    t: float | None = None


class CaseIn(BaseModel):
    plate: str


class VehicleIn(BaseModel):
    plate: str
    case_no: str
    offence: str
    officer: str
    colour: str = ""
    vehicle_type: str = ""
    make_model: str = ""
    severity: int = 3
    notes: str = ""


class EdgeRegister(BaseModel):
    name: str
    role: str = "route"
    junction: int | None = None


class EdgeRead(BaseModel):
    name: str
    junction: int
    plate: str
    conf: float | None = None
    matched: bool = False
    hotlist_plate: str | None = None
    score: float = 0.0
    thumb: str | None = None
    captured_at: float | None = None
    boot: str | None = None   # signed, numbered messages: see server/messages.py
    seq: int | None = None
    sig: str | None = None


frames: dict[str, bytes] = {}


class Settings(BaseModel):
    speed: float | None = None
    show_truth: bool | None = None


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.post("/api/demo/start")
def demo_start(held_out: bool = False, seed: int | None = None):
    room.reset(seed)
    room.start_demo(held_out=held_out)
    state["running"] = True
    return {"ok": True, "seed": room.seed}


@app.post("/api/case")
def open_live_case(c: CaseIn):
    """Live mode: the officer enters the wanted plate; only real camera reads drive the map."""
    room.reset(room.seed)
    room.open_case(c.plate)
    state["running"] = True
    state["speed"] = 10.0  # tabletop scale: 1 real second = 10 city seconds (toy car vs real junction distances)
    return {"ok": True, "plate": room.case["plate"]}


@app.get("/api/hotlist")
def hotlist():
    """Plates every edge camera checks against: all active registered vehicles (+ the open case)."""
    return {"plates": room.hotlist()}


# ---------------------------------------------------------------- criminal / vehicle registration
@app.get("/api/vehicles")
def vehicles(all: bool = False):
    return db.vehicles(include_inactive=all)


@app.post("/api/vehicles")
def register_vehicle(v: VehicleIn):
    try:
        return room.register_vehicle(**(v.dict() if not hasattr(v, "model_dump") else v.model_dump()))
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@app.post("/api/vehicles/{vid}/track")
def track_vehicle(vid: int):
    """Open a live case for a registered vehicle: the cameras' reads now drive the map for it."""
    v = db.vehicle(vid)
    if not v or v["status"] != "active":
        return JSONResponse({"error": "no such active vehicle"}, status_code=404)
    room.reset(room.seed)
    room.open_case(v["plate"], vehicle=v)
    state["running"] = True
    state["speed"] = 10.0
    return {"ok": True, "plate": v["plate"]}


@app.post("/api/vehicles/{vid}/close")
def close_vehicle(vid: int):
    room.close_vehicle(vid)
    return {"ok": True}


@app.get("/api/vehicles/{vid}/history")
def vehicle_history(vid: int):
    """Stored travel history: every sighting and every reconstructed route segment for this vehicle."""
    return {"vehicle": db.vehicle(vid), "sightings": db.sightings(vid), "routes": db.routes(vid)}


@app.get("/api/history_thumb/{sid}")
def history_thumb(sid: int):
    import base64
    t = db.sighting_thumb(sid)
    return Response(base64.b64decode(t), media_type="image/jpeg") if t else Response(status_code=404)


# ---------------------------------------------------------------- police alerts
@app.get("/api/alerts")
def alerts(recipient: str | None = None):
    return db.alerts(recipient)


@app.post("/api/alerts/{aid}/ack")
def ack_alert(aid: int):
    return room.ack_alert(aid) or JSONResponse({"error": "no such alert"}, status_code=404)


@app.get("/api/units/{name}")
def unit(name: str):
    if name not in room.units:
        return JSONResponse({"error": "no such unit"}, status_code=404)
    last = next((x for x in reversed(room.sightings) if x["status"] == "accepted"), None)
    return {"unit": room.unit_view(name), "alerts": db.alerts(name, limit=20), "case": room.case,
            "last_sighting": last and {k: v for k, v in last.items() if k != "thumb"} | {"thumb": bool(last.get("thumb"))},
            "track": room.track[-10:], "t": room.t}


@app.get("/unit/{name}")
def unit_page(name: str):
    return FileResponse(STATIC / "unit.html")


@app.get("/api/audit")
def audit(limit: int = 100):
    return db.audit_log(limit)


@app.post("/api/edge/register")
def edge_register(r: EdgeRegister):
    return room.register_edge(r.name, r.role, r.junction)


@app.post("/api/edge/read")
async def edge_read(request: Request):
    # the signature covers the body exactly as the node sent it, so verify the raw JSON, not the model
    body = await request.json()
    try:
        EdgeRead(**body)
    except (TypeError, ValidationError) as e:
        return JSONResponse({"error": str(e)}, status_code=422)
    res = room.edge_read(body)
    return JSONResponse(res, status_code=401) if res.get("error") == "bad signature" else res


@app.post("/api/edge/frame/{name}")
async def edge_frame(name: str, request: Request):
    frames[name] = await request.body()
    room.edge_heartbeat(name, time.time())
    return {"ok": True}


@app.get("/api/edge/stream/{name}")
async def edge_stream(name: str):
    async def gen():
        last = None
        while True:
            f = frames.get(name)
            if f is not None and f is not last:
                last = f
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + f + b"\r\n"
            await asyncio.sleep(0.08)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/thumb/{kind}/{idx}")
def thumb(kind: str, idx: int):
    """Plate crop for a sighting (kind=sighting, idx=sighting id) or a camera read (kind=read, idx=read id)."""
    import base64
    src, key = (room.sightings, "id") if kind == "sighting" else (room.reads, "rid")
    for r in src:
        if r.get(key) == idx and r.get("thumb"):
            return Response(base64.b64decode(r["thumb"]), media_type="image/jpeg")
    return Response(status_code=404)


@app.post("/api/demo/pause")
def demo_pause():
    state["running"] = not state["running"]
    return {"running": state["running"]}


@app.post("/api/demo/clone")
def demo_clone():
    return room.inject_clone() or {"ok": False}


@app.post("/api/approve")
def approve():
    room.approve()
    return {"ok": True}


@app.post("/api/reject")
def reject():
    room.reject()
    return {"ok": True}


@app.post("/api/camera/{node}/{status}")
def camera(node: int, status: str):
    room.set_camera(node, status == "online")
    return {"ok": True}


@app.post("/api/camera/random_offline")
def camera_random_offline():
    live = [c for c, cam in room.cameras.items() if cam.live]
    if not live:
        return {"ok": False}
    # prefer a camera inside the current predicted area so the effect is visible
    snap_nodes = set()
    if room.belief:
        for r in room.belief.routes:
            snap_nodes.update(v for v, t in r if t >= room.t)
    pick = [c for c in live if c in snap_nodes] or live
    node = room.rng.choice(pick)
    room.set_camera(node, False)
    return {"ok": True, "node": node}


@app.post("/api/sighting")
def sighting(s: SightingIn):
    if room.edge_key is not None:  # this manual endpoint is unsigned; with a key only edge reads count
        return JSONResponse({"error": f"{KEY_ENV} is set: send signed reads to /api/edge/read"}, status_code=403)
    return room.add_sighting(s.camera, s.t, s.plate, s.score, source="edge camera")


@app.post("/api/settings")
def settings(s: Settings):
    if s.speed is not None:
        state["speed"] = max(0.5, min(30.0, s.speed))
    if s.show_truth is not None:
        state["show_truth"] = s.show_truth
    return state


@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept()
    clients.add(sock)
    try:
        while True:
            await sock.receive_text()  # keep-alive; commands go through the REST API
    except WebSocketDisconnect:
        clients.discard(sock)


@app.on_event("startup")
async def clock():
    async def loop():
        tick = 0.25
        while True:
            await asyncio.sleep(tick)
            room.check_edges(time.time())
            room.check_acks(time.time())
            room.clock_rate = state["speed"] if state["running"] and room.case else 0.0
            if state["running"] and room.case:
                room.step(tick * state["speed"])
            if clients:
                snap = room.snapshot(state["show_truth"])
                snap["running"], snap["speed"] = state["running"], state["speed"]
                for c in list(clients):
                    try:
                        await c.send_json(snap)
                    except Exception:
                        clients.discard(c)
    asyncio.create_task(loop())
