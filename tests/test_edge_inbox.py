"""Control-room side of the edge messaging layer: signatures, duplicates, capture time (synthetic grid)."""
import pytest

from server.control_room import ControlRoom
from server.messages import Numberer, sign
from tests.synthetic import node

COLS = 9
A, B = node(4, 2, COLS), node(4, 4, COLS)
KEY = b"test-key"
PLATE = "TS09EK4417"


@pytest.fixture
def room(grid_world):
    room = ControlRoom(coverage=0.25, n_units=3, seed=1, world=grid_world, edge_key=KEY)
    room.register_edge("cam-A", junction=A)
    room.register_edge("cam-B", junction=B)
    room.open_case(PLATE)
    return room


def read(name="cam-A", captured_at=None, **extra):
    return {"name": name, "plate": PLATE, "conf": 0.8, "score": 0.9, "matched": True, "hotlist_plate": PLATE,
            "captured_at": captured_at, **extra}


def test_signed_read_becomes_sighting(room):
    rec = room.edge_read(sign(Numberer("b1").stamp(read()), KEY))
    assert rec["hit"] and rec["status"] == "accepted"


@pytest.mark.parametrize("tamper", ["unsigned", "altered", "wrong key"])
def test_bad_signature_is_dropped(room, tamper):
    msg = sign(Numberer("b1").stamp(read()), KEY if tamper != "wrong key" else b"guess")
    if tamper == "unsigned":
        del msg["sig"]
    elif tamper == "altered":
        msg["name"] = "cam-B"
    assert room.edge_read(msg) == {"ok": False, "error": "bad signature"}
    assert room.reads == [] and room.sightings == []
    assert room.events[-1].kind == "flag" and "bad or missing signature" in room.events[-1].text


def test_duplicate_is_dropped(room):
    msg = sign(Numberer("b1").stamp(read()), KEY)
    room.edge_read(msg)
    assert room.edge_read(msg) == {"ok": True, "duplicate": True}  # a retry after the reply was lost
    assert room.edges["cam-A"]["reads"] == 1 and len(room.sightings) == 1


def test_restarted_node_starts_a_new_numbering(room):
    room.edge_read(sign(Numberer("b1").stamp(read()), KEY))
    rec = room.edge_read(sign(Numberer("b2").stamp(read()), KEY))
    assert "duplicate" not in rec and room.edges["cam-A"]["reads"] == 2


def test_duplicates_are_remembered_across_case_resets(room):
    msg = sign(Numberer("b1").stamp(read()), KEY)
    room.edge_read(msg)
    room.reset(room.seed)
    assert room.edge_read(msg) == {"ok": True, "duplicate": True}


def test_without_a_key_unsigned_reads_are_accepted(grid_world):
    room = ControlRoom(coverage=0.25, n_units=3, seed=1, world=grid_world)
    room.register_edge("cam-A", junction=A)
    room.open_case(PLATE)
    assert room.edge_read(read())["hit"]


def test_capture_time_is_mapped_to_the_sim_clock(room):
    room.t, room.clock_rate = 300.0, 2.0
    rec = room.edge_read(sign(read(captured_at=1000.0), KEY), now=1050.0)  # queued 50 s on the node
    assert rec["t"] == 200.0 and room.sightings[-1]["t"] == 200.0
    rec = room.edge_read(sign(read(captured_at=1060.0), KEY), now=1050.0)  # node clock ahead: never future
    assert rec["t"] == 300.0


def test_queued_reads_delivered_backwards_are_replayed(room):
    room.t, room.clock_rate = 200.0, 1.0
    n = Numberer("b1")
    first = sign(n.stamp(read("cam-A", captured_at=1000.0)), KEY)
    second = sign(n.stamp(read("cam-B", captured_at=1060.0)), KEY)
    room.edge_read(second, now=1100.0)  # B: t = 160
    room.edge_read(first, now=1100.0)   # A: t = 100, arrives after B
    assert [e["s"].camera for e in room.timeline] == [A, B]
    assert all(s["status"] == "accepted" for s in room.sightings)
    assert any(e.kind == "replay" for e in room.events)
