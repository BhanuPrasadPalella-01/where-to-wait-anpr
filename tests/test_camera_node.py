"""Unit tests for the edge camera node helpers: OCR vote, device choice, read backlog (no camera needed)."""
import numpy as np
import pytest

from edge.camera_node import Server, pick_device, sharpness, vote


def test_vote_single_read():
    assert vote([("KL02BM4659", 0.9)]) == ("KL02BM4659", 0.9)


def test_vote_per_character_majority():
    # one misread character in each of two reads; the per-character vote recovers the plate
    reads = [("KL02BM4659", 0.8), ("KL02BM4658", 0.7), ("KLO2BM4659", 0.6)]
    plate, conf = vote(reads)
    assert plate == "KL02BM4659"
    assert conf == 0.8


def test_vote_length_by_confidence_mass():
    # two confident 10-char reads outweigh three weak 9-char reads (a dropped character)
    reads = [("KL02BM4659", 0.9), ("KL02BM4659", 0.8), ("KL02B4659", 0.3), ("KL02M4659", 0.3), ("KL0BM4659", 0.3)]
    assert vote(reads)[0] == "KL02BM4659"


def test_pick_device_explicit():
    assert pick_device("cpu") == "cpu"
    assert pick_device("cuda:1") == "cuda:1"


def test_pick_device_auto():
    # "auto" asks torch which accelerator exists; the CI test environment does not install torch
    pytest.importorskip("torch")
    assert pick_device("auto") in ("mps", "cuda:0", "cpu")


def test_sharpness_prefers_detail():
    flat = np.full((40, 120, 3), 128, np.uint8)
    edges = flat.copy()
    edges[:, ::4] = 0
    assert sharpness(edges) > sharpness(flat) == 0.0


def test_reads_queued_while_server_down_and_replayed():
    srv = Server("http://127.0.0.1:9")
    sent, up = [], False

    def fake_call(path, body=None, **_):
        if not up:
            raise OSError("connection refused")
        sent.append(body["plate"])
        return {}

    srv.call = fake_call
    assert not srv.send_read({"plate": "A1"})
    assert not srv.send_read({"plate": "B2"})
    assert len(srv.backlog) == 2
    up = True
    assert srv.send_read({"plate": "C3"})
    assert sent == ["A1", "B2", "C3"] and srv.backlog == []


def http_error(code):
    import urllib.error
    return urllib.error.HTTPError("http://cr/api/edge/read", code, "refused", {}, None)


def test_reads_are_numbered_and_signed_once():
    from server.messages import verify
    srv = Server("http://127.0.0.1:9", key=b"k")
    sent, up = [], False

    def fake_call(path, body=None, **_):
        if not up:
            raise OSError("connection refused")
        sent.append(body)
        return {}

    srv.call = fake_call
    srv.send_read({"plate": "A1"})
    first_try = dict(srv.backlog[0])
    up = True
    srv.send_read({"plate": "B2"})
    assert sent[0] == first_try, "a retry is the same message, so the server can spot duplicates"
    assert [m["seq"] for m in sent] == [1, 2] and len({m["boot"] for m in sent}) == 1
    assert all(verify(m, b"k") for m in sent)


def test_refused_read_is_dropped_and_the_queue_moves_on():
    srv = Server("http://127.0.0.1:9")
    sent = []

    def fake_call(path, body=None, **_):
        if body["plate"] == "BAD":
            raise http_error(401)
        sent.append(body["plate"])
        return {}

    srv.call = fake_call
    assert srv.send_read({"plate": "BAD"})
    assert srv.send_read({"plate": "OK"})
    assert sent == ["OK"] and srv.refused == 1 and srv.backlog == []


def test_server_error_keeps_the_read_queued():
    srv = Server("http://127.0.0.1:9")

    def fake_call(path, body=None, **_):
        raise http_error(503)

    srv.call = fake_call
    assert not srv.send_read({"plate": "A1"})
    assert len(srv.backlog) == 1 and srv.refused == 0


def test_backlog_is_bounded_and_keeps_hotlist_matches():
    srv = Server("http://127.0.0.1:9", max_backlog=3)

    def fake_call(path, body=None, **_):
        raise OSError("down")

    srv.call = fake_call
    for plate, matched in (("HIT1", True), ("MISS1", False), ("MISS2", False), ("HIT2", True)):
        srv.send_read({"plate": plate, "matched": matched})
    assert [m["plate"] for m in srv.backlog] == ["HIT1", "MISS2", "HIT2"]


def test_outage_and_lost_reply_end_to_end(grid_world):
    """Edge node and control room together: reads queued during an outage arrive late and out of
    capture order, one reply is lost so its read is sent twice; the room ends up with each read once."""
    import json
    from server.control_room import ControlRoom
    from tests.synthetic import node

    a, b = node(4, 2, 9), node(4, 4, 9)
    room = ControlRoom(coverage=0.25, n_units=3, seed=1, world=grid_world, edge_key=b"k")
    room.register_edge("cam-A", junction=a)
    room.register_edge("cam-B", junction=b)
    room.open_case("TS09EK4417")
    room.t = 200.0
    node_a, node_b = Server("http://127.0.0.1:9", key=b"k"), Server("http://127.0.0.1:9", key=b"k")
    state = {"up": False, "lose_reply": True}

    def make_call(srv):
        def call(path, body=None, **_):
            if not state["up"]:
                raise OSError("control room down")
            res = room.edge_read(json.loads(json.dumps(body)), now=1100.0)
            if state["lose_reply"]:  # processed, but the reply never reaches the node
                state["lose_reply"] = False
                raise OSError("timeout")
            return res
        return call

    node_a.call, node_b.call = make_call(node_a), make_call(node_b)
    hit = {"plate": "TS09EK4417", "conf": 0.8, "score": 0.9, "matched": True, "hotlist_plate": "TS09EK4417"}
    assert not node_a.send_read({**hit, "name": "cam-A", "captured_at": 1000.0})  # t = 100
    state["up"] = True
    assert not node_b.send_read({**hit, "name": "cam-B", "captured_at": 1060.0})  # t = 160, reply lost
    assert node_b.send_read({**hit, "name": "cam-B", "plate": "KA01AB1234", "matched": False,
                             "captured_at": 1070.0})                                 # retry + a non-match
    assert node_a.send_read({**hit, "name": "cam-A", "plate": "KA05XY0001", "matched": False,
                             "captured_at": 1080.0})                                 # queue flushed late
    assert room.edges["cam-B"]["reads"] == 2 and room.edges["cam-A"]["reads"] == 2
    assert [(s["camera"], s["t"], s["status"]) for s in room.sightings] == [(b, 160.0, "accepted"),
                                                                          (a, 100.0, "accepted")]
    assert [e["s"].camera for e in room.timeline] == [a, b], "judged in capture order"
