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


# ---------------------------------------------------------------- officer approval

def test_units_wait_for_officer_approval(room):
    room.open_case("TS09EK4417", incident=B)
    plan = list(room.pending_plan)
    before = {u: x.target for u, x in room.units.items()}
    assert all(t is None for t in before.values()), "nothing moves before approval"
    room.approve()
    assert room.pending_plan is None
    for u, v in plan:
        assert room.units[u].target == v and room.units[u].schedule[-1][0] == v
    assert room.events[-1].kind == "approve"


def test_rejected_plan_moves_nobody(room):
    room.open_case("TS09EK4417", incident=B)
    room.reject()
    assert room.pending_plan is None
    assert all(x.target is None for x in room.units.values())
    assert room.events[-1].kind == "reject"


def test_approve_without_plan_is_a_no_op(room):
    n = len(room.events)
    room.approve()
    assert len(room.events) == n


# ---------------------------------------------------------------- edge cameras

def test_register_edge_is_idempotent(room):
    first = room.register_edge("cam-A", junction=A)
    again = room.register_edge("cam-A", junction=FAR)  # re-registration keeps the original spot
    assert first == again and first["junction"] == A
    assert len(room.edges) == 1


def test_heartbeat_timeout_marks_camera_blind_and_recovers(room):
    room.register_edge("cam-A", junction=A)
    room.edge_heartbeat("cam-A", 100.0)
    room.check_edges(104.0)
    assert room.edges["cam-A"]["online"] and room.cameras[A].live
    room.check_edges(106.0)  # no frame for more than 5 s
    assert not room.edges["cam-A"]["online"] and not room.cameras[A].live
    assert "OFFLINE" in room.events[-1].text
    room.edge_heartbeat("cam-A", 107.0)
    assert room.edges["cam-A"]["online"] and room.cameras[A].live


@pytest.mark.parametrize("b_live", [True, False])
def test_blind_camera_gives_no_negative_evidence(room, b_live):
    live_case(room)
    room.add_sighting(A, 0.0)
    room.set_camera(B, b_live)
    passed = [any(v == B and t <= 120 for v, t in r) for r in room.belief.routes]
    assert any(passed) and not all(passed), "some particles drive through B in the first two minutes"
    w0 = list(room.belief.w)
    room.step(120.0)  # FAR is ten blocks away, so B is the only live camera a particle can pass
    through_b = room.belief.w[passed.index(True)] / w0[passed.index(True)]
    elsewhere = room.belief.w[passed.index(False)] / w0[passed.index(False)]
    if b_live:
        assert through_b < elsewhere, "passing a live camera unseen lowers a particle's weight"
    else:
        assert through_b == pytest.approx(elsewhere), "a blind camera is not evidence"


def read(name, plate, score=0.9, matched=True, hotlist_plate=None):
    return {"name": name, "plate": plate, "conf": 0.8, "score": score, "matched": matched,
            "hotlist_plate": hotlist_plate if hotlist_plate is not None else plate}


def test_edge_read_from_unregistered_camera_is_refused(room):
    assert room.edge_read(read("ghost", "TS09EK4417")) == {"ok": False, "error": "unregistered camera"}


def test_edge_read_hotlist_match_becomes_sighting(room):
    live_case(room)
    rec = room.edge_read(read("cam-A", "TS09EK4417"))
    assert rec["hit"] and rec["status"] == "accepted" and rec["junction"] == A
    assert room.edges["cam-A"]["matches"] == 1
    assert room.sightings[-1]["source"] == "cam-A" and room.belief is not None


def test_edge_read_without_match_is_only_logged(room):
    live_case(room)
    rec = room.edge_read(read("cam-A", "KA01AB1234", matched=False, hotlist_plate=""))
    assert not rec["hit"] and "status" not in rec
    assert room.reads[-1]["plate"] == "KA01AB1234"
    assert room.sightings == [] and room.edges["cam-A"]["reads"] == 1


def test_edge_read_below_threshold_or_wrong_plate_is_no_hit(room):
    live_case(room)
    assert not room.edge_read(read("cam-A", "TS09EK4417", score=0.01))["hit"]
    assert not room.edge_read(read("cam-A", "TS09EK4411", hotlist_plate="TS09EK4411"))["hit"]
    assert room.sightings == []


def test_edge_read_with_no_case_is_no_hit(room):
    room.register_edge("cam-A", junction=A)
    assert not room.edge_read(read("cam-A", "TS09EK4417"))["hit"]
