"""End-to-end smoke test of the dashboard server through its HTTP API (FastAPI TestClient).

The real server/app.py runs on the synthetic grid with an in-memory store (fixture server_app in
tests/conftest.py): an officer opens a case, edge cameras send signed, numbered reads, the officer
approves the plan and a unit acknowledges its alert; the Vehicle Search API answers with a clean 404 for
anything it does not have. No city map, models or network are needed.
"""
import secrets

import pytest

from server.messages import Numberer, sign
from tests.conftest import EDGE_KEY
from tests.synthetic import node

pytest.importorskip("httpx")  # fastapi.testclient needs it (requirements-test.txt)
from fastapi.testclient import TestClient  # noqa: E402

COLS = 9
A, B, FAR = node(4, 2, COLS), node(4, 4, COLS), node(8, 8, COLS)  # A -> B: two 20 s main-road blocks
PLATE = "TS09EK4417"
KEY = EDGE_KEY.encode()


@pytest.fixture
def client(server_app):
    c = TestClient(server_app.app)
    for name, role, j in (("cam-A", "route", A), ("cam-B", "route", B), ("cam-FAR", "far", FAR)):
        assert c.post("/api/edge/register", json={"name": name, "role": role, "junction": j}).json()["junction"] == j
    return c


def read(nb: Numberer, name: str, junction: int, plate: str = PLATE, score: float = 0.93) -> dict:
    """A plate read as edge/camera_node.py sends it: matched on the node, numbered, then signed."""
    msg = {"name": name, "junction": junction, "plate": plate, "conf": 0.9, "matched": True,
           "hotlist_plate": plate, "score": score}
    return sign(nb.stamp(msg), KEY)


def test_pages_are_served(client):
    for path in ("/", "/search", "/vehicles", "/unit/Unit%201"):
        r = client.get(path)
        assert r.status_code == 200 and "<html" in r.text.lower(), path


def test_live_case_signed_reads_and_approved_plan(server_app, client):
    room = server_app.room
    assert client.post("/api/case", json={"plate": "ts09 ek-4417"}).json() == {"ok": True, "plate": PLATE}
    assert PLATE in client.get("/api/hotlist").json()["plates"]
    nb = Numberer(secrets.token_hex(4))

    # first signed read at A starts the belief and proposes a plan
    first = read(nb, "cam-A", A)
    r = client.post("/api/edge/read", json=first).json()
    assert r["hit"] and r["status"] == "accepted" and r["junction"] == A
    assert room.belief is not None and room.last_ok.camera == A
    # the node retries after a lost reply: same (camera, boot, seq), dropped
    assert client.post("/api/edge/read", json=first).json() == {"ok": True, "duplicate": True}

    # the officer approves; every moved unit gets an assignment alert and can acknowledge it
    moves = list(room.pending_plan or [])
    assert moves, "units start at random junctions, so the first plan moves someone"
    assert client.post("/api/approve").json() == {"ok": True}
    assert room.pending_plan is None
    unit, target = moves[0]
    view = client.get(f"/api/units/{unit}").json()
    assert view["case"]["plate"] == PLATE
    alert = next(a for a in view["alerts"] if a["kind"] == "assign")
    assert f"junction {target}" in alert["message"]
    assert client.post(f"/api/alerts/{alert['id']}/ack").json()["id"] == alert["id"]

    # 60 simulated seconds later the car is read at B (40 s away by road): accepted, route rebuilt
    room.step(60.0)
    r = client.post("/api/edge/read", json=read(nb, "cam-B", B)).json()
    assert r["status"] == "accepted" and room.last_ok.camera == B
    assert room.track, "a route between the two sightings is reconstructed"

    # 5 s after B, the same plate at the far camera: impossible journey, flagged, belief unchanged
    room.step(5.0)
    r = client.post("/api/edge/read", json=read(nb, "cam-FAR", FAR)).json()
    assert r["status"] == "flagged: impossible journey"
    assert room.last_ok.camera == B

    kinds = {e["kind"] for e in client.get("/api/audit").json()}
    assert {"sighting", "plan", "approve", "flag"} <= kinds


def test_forged_and_unsigned_reads_are_refused(client):
    client.post("/api/case", json={"plate": PLATE})
    nb = Numberer(secrets.token_hex(4))
    forged = read(nb, "cam-A", A)
    forged["junction"] = B  # altered on the way: the signature no longer matches
    assert client.post("/api/edge/read", json=forged).status_code == 401
    unsigned = {k: v for k, v in read(nb, "cam-A", A).items() if k != "sig"}
    assert client.post("/api/edge/read", json=unsigned).status_code == 401
    # with WTW_EDGE_KEY set, the unsigned manual endpoint is closed too
    assert client.post("/api/sighting", json={"camera": A, "plate": PLATE}).status_code == 403
    assert client.post("/api/edge/read", json={"name": "cam-A"}).status_code == 422  # malformed


def test_registered_vehicle_tracked_from_the_registry(server_app, client):
    v = client.post("/api/vehicles", json={"plate": "AP28CX1234", "case_no": "CR-7/2026", "offence": "theft",
                                           "officer": "SI Rao", "colour": "white", "vehicle_type": "car"}).json()
    assert any(x["id"] == v["id"] for x in client.get("/api/vehicles").json())
    assert client.post(f"/api/vehicles/{v['id']}/track").json() == {"ok": True, "plate": "AP28CX1234"}
    assert server_app.room.case["vehicle"]["id"] == v["id"]
    nb = Numberer(secrets.token_hex(4))
    assert client.post("/api/edge/read", json=read(nb, "cam-A", A, "AP28CX1234")).json()["status"] == "accepted"
    hist = client.get(f"/api/vehicles/{v['id']}/history").json()
    assert [(s["junction"], s["source"], s["status"]) for s in hist["sightings"]] == [(A, "cam-A", "accepted")]
    assert client.post(f"/api/vehicles/{v['id']}/close").json() == {"ok": True}
    assert "AP28CX1234" not in client.get("/api/hotlist").json()["plates"]


def test_websocket_streams_snapshots(server_app):
    # inside `with`, the startup clock loop runs and pushes a snapshot to every dashboard 4 times a second
    with TestClient(server_app.app) as c:
        c.post("/api/case", json={"plate": PLATE})
        with c.websocket_connect("/ws") as ws:
            snap = ws.receive_json()
    assert snap["case"]["plate"] == PLATE and snap["running"] is True
    assert {"cameras", "units", "events", "edges", "reads"} <= set(snap)


def test_search_api_answers_404_not_500(server_app, client):
    # with the RoundaboutHD tracks missing (CI) every endpoint says what to run; with them built
    # (a full checkout), unknown cameras, tracks and vehicles are still 404s, never server errors
    built = server_app.search.ok
    r = client.get("/api/search/meta")
    assert r.status_code == (200 if built else 404)
    if not built:
        assert "roundabout_fullpass" in r.json()["error"]
    for path in ("/api/search/boxes/9?t0=0&t1=5", "/api/search/pick/9?t=1&x=0.5&y=0.5",
                 "/api/search/track/9/1", "/api/search/truth/999999", "/api/search/crop/9/1"):
        assert client.get(path).status_code == 404, path
