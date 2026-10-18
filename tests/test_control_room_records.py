"""Control room with the registry, route reconstruction and police alerts (synthetic grid, in-memory DB)."""
import pytest

from server.control_room import ACK_TIMEOUT, ControlRoom
from tests.synthetic import node

COLS = 9
A, B, FAR = node(4, 2, COLS), node(4, 4, COLS), node(8, 8, COLS)


@pytest.fixture
def room(grid_world):
    r = ControlRoom(coverage=0.25, n_units=3, seed=1, world=grid_world)
    for name, j in (("cam-A", A), ("cam-B", B), ("cam-FAR", FAR)):
        r.register_edge(name, junction=j)
    return r


def register(room, plate="KL02BM4659", **kw):
    args = dict(case_no="FIR 214/2026", offence="robbery", officer="SI Rao", colour="grey", severity=5)
    args.update(kw)
    return room.register_vehicle(plate=plate, **args)


def read(room, cam, plate, matched=True, score=0.95):
    return room.edge_read({"name": cam, "plate": plate, "conf": 0.98, "matched": matched,
                           "hotlist_plate": plate if matched else None, "score": score if matched else 0.0})


def test_hotlist_contains_registry_and_typed_case(room):
    register(room)
    room.open_case("TS09EK4417")
    assert set(room.hotlist()) == {"KL02BM4659", "TS09EK4417"}


def test_registered_vehicle_read_with_no_case_starts_tracking(room):
    v = register(room)
    rec = read(room, "cam-A", "KL02BM4659")
    assert room.case["plate"] == "KL02BM4659" and room.case["vehicle"]["id"] == v["id"]
    assert rec["hit"] and rec["status"] == "accepted"


def test_other_registered_vehicle_alerts_stations_without_switching_case(room):
    register(room)
    register(room, plate="TN23L4547", case_no="FIR 98/2026")
    room.open_case("KL02BM4659")
    rec = read(room, "cam-B", "TN23L4547")
    assert not rec["hit"] and rec["hotlist"]
    assert room.case["plate"] == "KL02BM4659" and room.sightings == []
    assert any("TN23L4547" in a["message"] for a in room.db.alerts("all stations"))


def test_sightings_are_stored_against_the_vehicle(room):
    v = register(room)
    room.open_case("KL02BM4659")
    read(room, "cam-A", "KL02BM4659")
    stored = room.db.sightings(v["id"])
    assert [s["junction"] for s in stored] == [A] and stored[0]["status"] == "accepted"


def test_route_between_sightings_is_reconstructed_and_stored(room):
    v = register(room)
    room.open_case("KL02BM4659")
    read(room, "cam-A", "KL02BM4659")
    room.step(45.0)
    read(room, "cam-B", "KL02BM4659")
    (seg,) = room.track
    assert seg["from"] == A and seg["to"] == B and seg["paths"]
    best = seg["paths"][0]
    assert tuple(best["lonlat"][0]) == tuple(room.g.xy[A]) and tuple(best["lonlat"][-1]) == tuple(room.g.xy[B])
    assert room.db.routes(v["id"]), "route segments are persisted"


def test_flagged_sighting_does_not_extend_the_route(room):
    register(room)
    room.open_case("KL02BM4659")
    read(room, "cam-A", "KL02BM4659")
    room.step(5.0)
    rec = read(room, "cam-FAR", "KL02BM4659")
    assert rec["status"].startswith("flagged") and room.track == []


def test_approval_alerts_each_moved_unit_and_ack_clears_it(room):
    register(room)
    room.open_case("KL02BM4659", incident=B)
    moved = [u for u, v in room.pending_plan if room.units[u].target != v]
    room.approve()
    assert set(room.unit_alert) == set(moved)
    u = moved[0]
    (alert,) = [a for a in room.db.alerts(u) if a["kind"] == "assign"]
    assert "KL02BM4659" in alert["message"] and "case FIR 214/2026" in alert["message"]
    room.ack_alert(alert["id"])
    assert room.unit_alert[u]["acked"] and room.unit_view(u)["alert"]["acked"]


def test_unacknowledged_assignment_is_flagged_after_timeout(room):
    room.open_case("KL02BM4659", incident=B)
    room.approve()
    u = next(iter(room.unit_alert))
    sent = room.unit_alert[u]["sent_wall"]
    room.check_acks(sent + ACK_TIMEOUT - 1)
    assert not room.unit_view(u)["alert"]["late"]
    room.check_acks(sent + ACK_TIMEOUT + 1)
    assert room.unit_view(u)["alert"]["late"]


def test_decisions_reach_the_audit_log(room):
    register(room)
    room.open_case("KL02BM4659", incident=B)
    room.approve()
    kinds = {a["kind"] for a in room.db.audit_log()}
    assert {"register", "case", "plan", "approve"} <= kinds
