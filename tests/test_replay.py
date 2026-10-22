"""Out-of-order sightings: belief snapshot, rollback and replay by capture time (synthetic grid)."""
import math

from planner.belief import Sighting
from server.control_room import ControlRoom
from tests.synthetic import node

COLS = 9
A, B, FAR = node(4, 2, COLS), node(4, 4, COLS), node(8, 8, COLS)


def make_room(grid_world):
    room = ControlRoom(coverage=0.25, n_units=3, seed=1, world=grid_world)
    for name, j in (("cam-A", A), ("cam-B", B), ("cam-FAR", FAR)):
        room.register_edge(name, junction=j)
    room.open_case("TS09EK4417")
    return room


def status(room, rec):
    return room.db._rows("SELECT status FROM sightings WHERE id = ?", (rec["db_id"],))[0]["status"]


def test_belief_snapshot_restore_round_trip(grid_world):
    room = make_room(grid_world)
    room.add_sighting(A, 0.0)
    b = room.belief
    snap = b.snapshot()
    routes, w = list(b.routes), list(b.w)
    b.negative_evidence(60.0, {B: 0.9})
    b.condition(Sighting(B, 60.0))
    assert b.routes != routes
    b.restore(snap)
    assert b.routes == routes and b.w == w and b.t_seed == 0.0
    # the random generator is restored too, so the same update gives the same particles again
    b.condition(Sighting(B, 60.0))
    after = list(b.routes)
    b.restore(snap)
    b.condition(Sighting(B, 60.0))
    assert b.routes == after


def test_arrival_order_does_not_change_the_belief(grid_world):
    in_order, late = make_room(grid_world), make_room(grid_world)
    in_order.add_sighting(A, 0.0)
    in_order.add_sighting(B, 60.0)
    late.add_sighting(B, 60.0)   # B's message overtakes A's
    late.add_sighting(A, 0.0)
    assert late.last_ok.camera == in_order.last_ok.camera == B
    assert late.belief.routes == in_order.belief.routes
    assert late.belief.w == in_order.belief.w
    assert [(s["from"], s["to"]) for s in late.track] == [(A, B)]
    assert any(e.kind == "replay" for e in late.events) and not any(e.kind == "replay" for e in in_order.events)


def test_late_sighting_can_flag_a_later_one(grid_world):
    room = make_room(grid_world)
    t_far = math.ceil(room.W.gate.tau_min(A, FAR))
    room.add_sighting(A, 0.0)
    b = room.add_sighting(B, t_far + 5.0)
    assert b["status"] == "accepted"
    # FAR, captured 5 s before B, arrives last: FAR -> B in 5 s is impossible, so B is now the suspect read
    far = room.add_sighting(FAR, float(t_far))
    assert far["status"] == "accepted"
    assert b["status"] == "flagged: impossible journey" and "possible clone" in b["why"]
    assert status(room, b) == "flagged: impossible journey", "the stored verdict follows the replay"
    assert room.last_ok.camera == FAR
    assert [(s["from"], s["to"]) for s in room.track] == [(A, FAR)]
    assert any(e.text.startswith("Replay:") and "(was accepted)" in e.text for e in room.events)


def test_replay_rebuilds_stored_routes(grid_world):
    room = make_room(grid_world)
    a = room.add_sighting(A, 0.0)
    c = room.add_sighting(FAR, 600.0)
    b = room.add_sighting(B, 60.0)  # lands between A and FAR
    pairs = {(r["from_sighting"], r["to_sighting"]) for r in room.db._rows("SELECT * FROM routes")}
    assert pairs == {(a["db_id"], b["db_id"]), (b["db_id"], c["db_id"])}, "the old A -> FAR route is gone"


def test_late_sighting_before_the_first_restarts_from_it(grid_world):
    room = make_room(grid_world)
    room.add_sighting(B, 60.0)
    rec = room.add_sighting(A, 0.0)
    assert "first sighting" in next(e.text for e in room.events if e.data.get("id") == rec["id"])
    assert room.belief.t_seed == 60.0, "after the replay the belief is seeded at B, the latest sighting"


def test_replay_proposes_one_plan(grid_world):
    room = make_room(grid_world)
    room.add_sighting(A, 0.0)
    room.add_sighting(B, 60.0)
    plans = sum("Plan after" in e.text for e in room.events)
    room.add_sighting(A, 20.0)  # two sightings re-judged, but the officer gets a single new plan
    assert sum("Plan after" in e.text for e in room.events) == plans + 1
    assert "late sighting" in [e.text for e in room.events if "Plan after" in e.text][-1]
