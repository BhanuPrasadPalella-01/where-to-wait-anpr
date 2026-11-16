"""Clock-skew handling: offset estimate from heartbeats (server/clocks.py), signed send times
(server/messages.py), and capture-time correction in the control room with deliberately skewed nodes."""
import random

import pytest

from edge.camera_node import Server
from server.clocks import ClockOffset, describe
from server.control_room import ControlRoom
from server.messages import TIME_HEADER, TIME_SIG_HEADER, check_time, sign, stamp_time
from tests.synthetic import node

COLS = 9
A, B = node(4, 2, COLS), node(4, 4, COLS)
KEY = b"test-key"
PLATE = "TS09EK4417"


def heartbeats(est, skew, t0=1000.0, n=40, step=0.125, seed=0):
    """n heartbeats from a node whose clock is `skew` seconds ahead, with 5-80 ms of network delay."""
    rng = random.Random(seed)
    for i in range(n):
        sent_true = t0 + i * step
        est.add(sent_true + skew, sent_true + rng.uniform(0.005, 0.08))
    return est


# ---------------------------------------------------------------- estimator

def test_offset_is_unknown_before_the_first_heartbeat():
    est = ClockOffset()
    assert est.value is None and est.correct(123.0) == 123.0


@pytest.mark.parametrize("skew", [120.0, -90.0, 0.0])
def test_offset_recovers_the_skew_despite_network_delay(skew):
    est = heartbeats(ClockOffset(), skew)
    # high by the fastest heartbeat's delay only: between 5 ms and a few tens of ms
    assert -skew + 0.005 <= est.value < -skew + 0.03
    assert est.correct(1000.0 + skew) == pytest.approx(1000.0, abs=0.03)


def test_a_slow_heartbeat_does_not_move_the_estimate():
    est = heartbeats(ClockOffset(), 60.0)
    before = est.value
    est.add(1005.0 + 60.0, 1005.0 + 4.0)  # stuck 4 s on a bad link
    assert est.value == before


def test_window_follows_a_clock_that_is_stepped():
    est = heartbeats(ClockOffset(window=30.0), 60.0, t0=1000.0)
    assert est.value == pytest.approx(-60.0, abs=0.03)
    heartbeats(est, 0.0, t0=1010.0)  # NTP fixed the node; the old samples are still in the window
    assert est.value == pytest.approx(-60.0, abs=0.03)
    heartbeats(est, 0.0, t0=1040.0)  # ... and have left it 30 s later
    assert est.value == pytest.approx(0.0, abs=0.03)


def test_non_finite_times_are_ignored():
    est = ClockOffset()
    est.add(float("nan"), 1000.0)
    est.add(1000.0, float("inf"))
    assert est.value is None


def test_describe_names_the_direction():
    assert describe(-61.0) == "61.0 s ahead of" and describe(90.0) == "90.0 s behind"


# ---------------------------------------------------------------- signed send time

def test_send_time_round_trips_exactly():
    t = 1763280000.123456
    text, sig = stamp_time("cam-A", t, KEY)
    assert check_time("cam-A", text, sig, KEY) == t


@pytest.mark.parametrize("tamper", ["unsigned", "other camera", "altered", "wrong key"])
def test_bad_send_time_is_ignored(tamper):
    text, sig = stamp_time("cam-A", 1000.0, b"guess" if tamper == "wrong key" else KEY)
    name = "cam-B" if tamper == "other camera" else "cam-A"
    if tamper == "unsigned":
        sig = None
    elif tamper == "altered":
        text = "2000.0"
    assert check_time(name, text, sig, KEY) is None


def test_without_a_key_the_send_time_is_taken_as_is():
    text, sig = stamp_time("cam-A", 1000.5, None)
    assert sig == "" and check_time("cam-A", text, None, None) == 1000.5


@pytest.mark.parametrize("text", [None, "", "soon", "nan", "inf"])
def test_missing_or_malformed_send_time(text):
    assert check_time("cam-A", text, None, None) is None


def test_node_stamps_each_frame_with_its_clock():
    srv = Server("http://127.0.0.1:9", key=KEY)
    calls = []
    srv.call = lambda path, **kw: calls.append((path, kw))
    srv.send_frame("cam-A", b"jpg", now=1234.5)
    path, kw = calls[0]
    assert path == "/api/edge/frame/cam-A" and kw["raw"] == b"jpg"
    assert check_time("cam-A", kw["headers"][TIME_HEADER], kw["headers"][TIME_SIG_HEADER], KEY) == 1234.5


def test_node_without_a_key_sends_no_signature_header():
    srv = Server("http://127.0.0.1:9")
    calls = []
    srv.call = lambda path, **kw: calls.append(kw["headers"])
    srv.send_frame("cam-A", b"jpg", now=1234.5)
    assert calls == [{TIME_HEADER: "1234.5"}]


# ---------------------------------------------------------------- control room with skewed nodes

@pytest.fixture
def room(grid_world):
    room = ControlRoom(coverage=0.25, n_units=3, seed=1, world=grid_world, edge_key=KEY)
    room.register_edge("cam-A", junction=A)
    room.register_edge("cam-B", junction=B)
    room.open_case(PLATE)
    return room


def beat(room, name, skew, now, key=KEY):
    """Heartbeats over the 5 s before `now` from a node whose clock is `skew` seconds ahead."""
    for i in range(40):
        t = now - 5.0 + i * 0.125
        room.edge_heartbeat(name, t + 0.01, *stamp_time(name, t + skew, key))


def read(name, captured_at):
    return sign({"name": name, "plate": PLATE, "conf": 0.8, "score": 0.9, "matched": True,
                 "hotlist_plate": PLATE, "captured_at": captured_at}, KEY)


def test_skewed_nodes_get_their_true_capture_times(room):
    """cam-A's clock runs 120 s ahead, cam-B's 90 s behind. The car passes A at control-room time 1000
    and B at 1040; both reads arrive at 1050 (simulated clock 300, real time)."""
    room.t, room.clock_rate = 300.0, 1.0
    beat(room, "cam-A", +120.0, 1050.0)
    beat(room, "cam-B", -90.0, 1050.0)
    ra = room.edge_read(read("cam-A", 1000.0 + 120.0), now=1050.0)
    rb = room.edge_read(read("cam-B", 1040.0 - 90.0), now=1050.0)
    assert ra["t"] == pytest.approx(250.0, abs=0.1) and rb["t"] == pytest.approx(290.0, abs=0.1)
    assert [e["s"].camera for e in room.timeline] == [A, B]
    assert not any(e.kind == "replay" for e in room.events)


def test_the_same_reads_without_heartbeats_land_in_the_wrong_order(room):
    # what the correction prevents: A's future time is clamped to "now", B looks 100 s old
    room.t, room.clock_rate = 300.0, 1.0
    ra = room.edge_read(read("cam-A", 1000.0 + 120.0), now=1050.0)
    rb = room.edge_read(read("cam-B", 1040.0 - 90.0), now=1050.0)
    assert (ra["t"], rb["t"]) == (300.0, 200.0)
    assert [e["s"].camera for e in room.timeline] == [B, A]


def test_correction_respects_the_demo_speed(room):
    room.t, room.clock_rate = 300.0, 2.0
    beat(room, "cam-A", +120.0, 1050.0)
    assert room.edge_read(read("cam-A", 1000.0 + 120.0), now=1050.0)["t"] == pytest.approx(200.0, abs=0.1)


def test_synced_node_is_left_alone(room):
    room.t, room.clock_rate = 300.0, 1.0
    beat(room, "cam-A", 0.0, 1050.0)
    assert room.edge_read(read("cam-A", 1000.0), now=1050.0)["t"] == pytest.approx(250.0, abs=0.1)
    assert not room.edges["cam-A"]["clock_skewed"]


def test_forged_send_time_cannot_shift_a_camera(room):
    room.t, room.clock_rate = 300.0, 1.0
    beat(room, "cam-A", +500.0, 1050.0, key=b"guess")           # wrong key
    room.edge_heartbeat("cam-A", 1050.0, "550.0", None)          # no signature
    assert room.edges["cam-A"]["clock_offset"] is None
    assert room.edges["cam-A"]["last_frame"] == 1050.0, "the frame still counts as a sign of life"
    assert room.edge_read(read("cam-A", 1000.0), now=1050.0)["t"] == 250.0


def test_replayed_old_heartbeat_cannot_shift_a_camera(room):
    beat(room, "cam-A", 0.0, 1050.0)
    old = stamp_time("cam-A", 1045.0, KEY)  # a genuine signed heartbeat, captured and sent again later
    room.edge_heartbeat("cam-A", 1060.0, *old)
    assert room.edges["cam-A"]["clock_offset"] == pytest.approx(0.01, abs=0.005)


def test_skew_is_shown_and_logged_once(room):
    n = len(room.events)
    beat(room, "cam-A", +120.0, 1050.0)
    logged = [e.text for e in room.events[n:]]
    assert len(logged) == 1 and "'cam-A' clock is 120.0 s ahead of the control room's" in logged[0]
    edge = next(e for e in room.snapshot()["edges"] if e["name"] == "cam-A")
    assert edge["clock_offset"] == pytest.approx(-120.0, abs=0.02) and edge["clock_skewed"]
    other = next(e for e in room.snapshot()["edges"] if e["name"] == "cam-B")
    assert other["clock_offset"] is None and not other["clock_skewed"]


def test_offset_survives_a_case_reset(room):
    beat(room, "cam-A", +120.0, 1050.0)
    room.reset(room.seed)
    room.open_case(PLATE)
    room.t = 300.0
    assert room.edge_read(read("cam-A", 1000.0 + 120.0), now=1050.0)["t"] == pytest.approx(250.0, abs=0.1)
