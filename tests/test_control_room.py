"""Control-room state machine on the synthetic grid: cases, sightings and the feasibility gate.

Runs without the downloaded city map (see tests/synthetic.py).
"""
import pytest

from server.control_room import ControlRoom
from tests.synthetic import node

COLS = 9
A, B, FAR = node(4, 2, COLS), node(4, 4, COLS), node(8, 8, COLS)  # A -> B: two main-road blocks


@pytest.fixture
def room(grid_world):
    return ControlRoom(coverage=0.25, n_units=3, seed=1, world=grid_world)


def live_case(room, plate="TS09EK4417"):
    room.register_edge("cam-A", junction=A)
    room.register_edge("cam-B", junction=B)
    room.register_edge("cam-FAR", role="far", junction=FAR)
    room.open_case(plate)


def test_sighting_without_case_is_ignored(room):
    rec = room.add_sighting(A, 0.0, plate="TS09EK4417")
    assert rec["status"] == "no active case"
    assert room.belief is None


def test_open_case_normalises_plate(room):
    room.open_case("ts09 ek-4417")
    assert room.case["plate"] == "TS09EK4417"


def test_demo_case_with_incident_starts_belief_and_proposes_plan(room):
    room.open_case("TS09EK4417", incident=B)
    assert room.belief is not None and room.last_ok.camera == B
    assert room.pending_plan, "units start at random junctions, so the first plan moves someone"
    assert room.events[-1].kind == "plan"


def test_live_case_uses_only_real_cameras(room):
    live_case(room)
    assert set(room.cameras) == {A, B, FAR}
    assert room.case["live"] and room.case["incident"] is None
    assert room.belief is None and room.last_ok is None


def test_first_live_sighting_starts_belief(room):
    live_case(room)
    rec = room.add_sighting(A, 0.0, score=0.93)
    assert rec["status"] == "accepted"
    assert room.belief is not None and room.last_ok.camera == A
    assert any("first sighting" in e.text for e in room.events)


def test_plausible_second_sighting_is_accepted(room):
    live_case(room)
    room.add_sighting(A, 0.0)
    rec = room.add_sighting(B, 60.0)  # two 20 s blocks in 60 s: possible
    assert rec["status"] == "accepted"
    assert room.last_ok.camera == B


def test_gate_flags_impossible_journey(room):
    live_case(room)
    room.add_sighting(A, 0.0)
    rec = room.add_sighting(FAR, 10.0)  # ten blocks away, 10 s later: a clone or a misread
    assert rec["status"] == "flagged: impossible journey"
    assert "possible clone or misread" in rec["why"]
    assert room.last_ok.camera == A, "a flagged sighting must not move the belief"
    assert room.events[-1].kind == "flag"


def test_gate_accepts_the_same_far_camera_given_enough_time(room):
    live_case(room)
    room.add_sighting(A, 0.0)
    assert room.add_sighting(FAR, 600.0)["status"] == "accepted"


def test_sighting_older_than_belief_is_stale(room):
    live_case(room)
    room.add_sighting(B, 100.0)
    rec = room.add_sighting(A, 50.0)
    assert rec["status"].startswith("stale")
    assert room.last_ok.camera == B
