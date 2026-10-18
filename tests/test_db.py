"""SQLite store: hotlist registry rules, evidence retention, alerts and the audit log."""
import pytest

from server.db import Database, READ_RETENTION_S


@pytest.fixture
def db():
    return Database(":memory:")


def reg(db, plate="KL02BM4659", **kw):
    args = dict(case_no="FIR 214/2026", offence="robbery", officer="SI Rao", colour="grey", severity=5)
    args.update(kw)
    return db.register_vehicle(plate, **args)


def test_registration_normalises_plate_and_sets_expiry(db):
    v = reg(db, plate="kl 02 bm-4659", now=1000.0)
    assert v["plate"] == "KL02BM4659" and v["status"] == "active"
    assert v["expires_at"] == pytest.approx(1000.0 + 28 * 86400)
    assert db.hotlist(now=1000.0) == ["KL02BM4659"]


@pytest.mark.parametrize("missing", ["case_no", "offence", "officer"])
def test_registration_requires_case_reason_and_officer(db, missing):
    with pytest.raises(ValueError):
        reg(db, **{missing: "  "})


def test_same_plate_cannot_be_active_twice(db):
    reg(db)
    with pytest.raises(ValueError):
        reg(db, case_no="FIR 999/2026")


def test_entries_expire_and_leave_the_hotlist(db):
    v = reg(db, now=0.0, expiry_days=1)
    assert db.hotlist(now=86399.0) == ["KL02BM4659"]
    assert db.hotlist(now=86401.0) == []
    assert db.vehicle(v["id"])["status"] == "expired"
    assert any(a["kind"] == "expire" for a in db.audit_log())


def test_closed_vehicle_can_be_registered_again(db):
    v = reg(db)
    db.close_vehicle(v["id"])
    assert db.hotlist() == []
    assert reg(db, case_no="FIR 300/2026")["status"] == "active"


def test_non_matching_reads_are_pruned_hits_are_kept(db):
    db.add_read({"camera": "cam-A", "plate": "MH20BC5122", "hit": False}, now=0.0)
    db.add_read({"camera": "cam-A", "plate": "KL02BM4659", "hit": True}, now=0.0)
    db.add_read({"camera": "cam-A", "plate": "TN23L4547", "hit": False}, now=READ_RETENTION_S + 10)
    assert db.read_counts() == {"reads": 2, "hits": 1}  # the old non-match is gone, the hit stays


def test_sightings_and_routes_round_trip(db):
    v = reg(db)
    sid = db.add_sighting({"camera": 4, "t": 12.0, "plate": "KL02BM4659", "score": 0.9, "status": "accepted",
                           "source": "cam-A", "lonlat": (78.5, 17.4), "thumb": "aGk="}, v["id"])
    db.add_routes(v["id"], None, sid, [{"rank": 1, "share": 1.0, "travel_s": 60.0, "lonlat": [[78.5, 17.4]]}])
    (s,) = db.sightings(v["id"])
    assert s["junction"] == 4 and s["has_thumb"] == 1
    assert db.sighting_thumb(sid) == "aGk="
    assert db.routes(v["id"])[0]["lonlat"] == [[78.5, 17.4]]


def test_alerts_are_delivered_to_recipient_and_broadcasts(db):
    db.add_alert("Unit 1", "assign", "go to junction 7", now=1.0)
    db.add_alert("Unit 2", "assign", "go to junction 9", now=2.0)
    db.add_alert("all units", "intercept", "stopped", now=3.0)
    assert [a["message"] for a in db.alerts("Unit 1")] == ["stopped", "go to junction 7"]


def test_ack_is_recorded_once_and_audited(db):
    a = db.add_alert("Unit 1", "assign", "go to junction 7")
    first = db.ack_alert(a["id"], now=50.0)
    again = db.ack_alert(a["id"], now=99.0)
    assert first["acked_at"] == again["acked_at"] == 50.0
    assert sum(1 for x in db.audit_log() if x["kind"] == "ack") == 1
