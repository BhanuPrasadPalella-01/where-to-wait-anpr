"""Several open cases in the control room: severity, shared plan, unit locks, routing of sightings."""
import pytest

from server.control_room import DEFAULT_SEVERITY, ControlRoom
from tests.synthetic import node

COLS = 9
A, B, FAR = node(4, 2, COLS), node(4, 4, COLS), node(8, 8, COLS)
P1, P2 = "TS09EK4417", "KA01AB1234"


@pytest.fixture
def room(grid_world):
    return ControlRoom(coverage=0.25, n_units=4, seed=1, world=grid_world)


def two_cases(room, sev1=4, sev2=2):
    room.open_case(P1, incident=B, severity=sev1)
    room.open_case(P2, incident=FAR, severity=sev2, add=True)


def test_add_keeps_the_other_case_and_focuses_the_new_one(room):
    two_cases(room)
    assert set(room.cases) == {P1, P2}
    assert room.case["plate"] == P2 and room.belief is room.cases[P2].belief
    room.open_case("TN23L4547", incident=A)  # without add: one-suspect mode replaces everything
    assert list(room.cases) == ["TN23L4547"]


def test_severity_comes_from_the_officer_or_the_registry(room):
    room.db.register_vehicle(plate=P1, case_no="FIR 1/2026", offence="robbery", officer="SI Rao", severity=5)
    room.open_case(P1)
    room.open_case(P2, add=True)
    room.open_case("TN23L4547", severity=1.5, add=True)
    assert [c.info["severity"] for c in room.cases.values()] == [5, DEFAULT_SEVERITY, 1.5]


def test_one_plan_serves_both_cases(room):
    two_cases(room)
    assert room.pending_plan
    assert set(room.pending_case.values()) <= {P1, P2}
    assert set(room.plan_probs) == {P1, P2}
    assert all(0.0 <= p <= 1.0 for p in room.plan_probs.values())


def test_approved_units_are_locked_to_their_case(room):
    two_cases(room)
    sent = dict(room.pending_case)
    room.approve()
    assert room.unit_case and all(sent[u] == p for u, p in room.unit_case.items())
    assert set(room.unit_case.values()) == {P1, P2}, "the four units are shared between both suspects"
    # make either case far more severe: units still useful to their own case are not pulled over
    kept = 0
    for plate in (P1, P2):
        room.set_severity(plate, 50.0)
        assert set(room.plan_locks.items()) <= set(room.unit_case.items())
        for u, p in room.plan_locks.items():
            assert room.pending_case[u] == p
        kept += len(room.plan_locks)
        room.set_severity(plate, 1.0)
    assert kept, "some unit is still covering its own suspect"


def test_closing_a_case_frees_its_units(room):
    two_cases(room)
    room.approve()
    locked = [u for u, p in room.unit_case.items() if p == P1]
    assert locked
    room.close_case(P1)
    assert P1 not in room.cases and room.case["plate"] == P2
    assert all(u not in room.unit_case for u in locked)
    assert set(room.pending_case.values()) == {P2}, "every planned unit now works on the remaining case"


def test_sighting_goes_to_the_case_with_that_plate_without_moving_focus(room):
    room.register_edge("cam-A", junction=A)
    room.register_edge("cam-B", junction=B)
    room.open_case(P1)
    room.open_case(P2, add=True)
    rec = room.add_sighting(A, 0.0, plate=P1)
    assert rec["status"] == "accepted" and rec["plate"] == P1
    assert room.case["plate"] == P2 and room.belief is None, "focus stays on the officer's case"
    assert room.cases[P1].last_ok.camera == A


def test_edge_read_of_a_second_open_case_is_a_hit_for_that_case(room):
    room.register_edge("cam-A", junction=A)
    room.open_case(P1)
    room.open_case(P2, add=True)
    # OCR read one character off; the hotlist matcher maps it to P1
    rec = room.edge_read({"name": "cam-A", "plate": "TS09EK4411", "conf": 0.8, "score": 0.9,
                          "matched": True, "hotlist_plate": P1})
    assert rec["hit"] and rec["status"] == "accepted"
    assert room.cases[P1].belief is not None and room.cases[P2].belief is None


def test_snapshot_lists_cases_and_tags_particles(room):
    two_cases(room)
    room.focus(P1)
    snap = room.snapshot()
    assert [c["plate"] for c in snap["cases"]] == [P1, P2]
    assert [c["focus"] for c in snap["cases"]] == [True, False]
    assert {p[3] for p in snap["particles"]} == {0, 1}
    assert snap["case"]["plate"] == P1 and snap["plan_prob"] == round(room.plan_probs[P1], 3)
    assert not room.focus("XX00XX0000")


def test_every_case_gets_negative_evidence(room):
    two_cases(room)
    w1, w2 = list(room.cases[P1].belief.w), list(room.cases[P2].belief.w)
    room.step(240.0)
    assert room.cases[P1].belief.w != w1 and room.cases[P2].belief.w != w2
