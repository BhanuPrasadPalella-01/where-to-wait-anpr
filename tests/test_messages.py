"""Message signing, numbering and duplicate detection (server/messages.py)."""
from server.messages import Deduper, Numberer, canonical, load_key, sign, verify

KEY = b"test-key"


def test_signed_message_verifies():
    msg = sign({"name": "cam-A", "plate": "TS09EK4417", "score": 0.93}, KEY)
    assert verify(msg, KEY)


def test_altered_or_unsigned_or_wrong_key_fails():
    msg = sign({"name": "cam-A", "plate": "TS09EK4417", "score": 0.93}, KEY)
    assert not verify({**msg, "plate": "TS09EK4418"}, KEY)
    assert not verify({k: v for k, v in msg.items() if k != "sig"}, KEY)
    assert not verify(msg, b"other-key")


def test_signature_ignores_key_order():
    a = sign({"b": 1, "a": 2}, KEY)
    assert verify({"sig": a["sig"], "a": 2, "b": 1}, KEY)
    assert canonical({"b": 1, "a": 2}) == b'{"a":2,"b":1}'


def test_signature_survives_a_json_round_trip():
    import json
    msg = sign({"plate": "KA01AB1234", "conf": 0.812, "captured_at": 1761120000.123456, "thumb": None}, KEY)
    assert verify(json.loads(json.dumps(msg)), KEY)


def test_load_key(monkeypatch):
    monkeypatch.delenv("WTW_EDGE_KEY", raising=False)
    assert load_key() is None
    assert load_key("abc") == b"abc"
    monkeypatch.setenv("WTW_EDGE_KEY", "from-env")
    assert load_key() == b"from-env"


def test_numberer_counts_per_boot():
    n = Numberer("b1")
    assert [n.stamp({})["seq"] for _ in range(3)] == [1, 2, 3]
    assert n.stamp({"x": 1}) == {"x": 1, "boot": "b1", "seq": 4}
    assert Numberer().boot != Numberer().boot


def test_deduper_drops_repeats_in_any_order():
    d = Deduper()
    assert [d.add("cam-A", "b1", s) for s in (1, 3, 2, 3, 1, 4)] == [True, True, True, False, False, True]
    assert d.seen("cam-A", "b1", 2) and not d.seen("cam-A", "b1", 5)


def test_deduper_keys_by_camera_and_boot():
    d = Deduper()
    assert d.add("cam-A", "b1", 1)
    assert d.add("cam-B", "b1", 1), "another camera may use the same number"
    assert d.add("cam-A", "b2", 1), "a restarted node starts a new numbering"


def test_deduper_memory_stays_bounded():
    d = Deduper(max_boots=2)
    for s in range(1, 1001):
        d.add("cam-A", "b1", s)
    assert d._seen[("cam-A", "b1")] == (1000, set()), "a contiguous run collapses to its high-water mark"
    d.add("cam-B", "b1", 1)
    d.add("cam-C", "b1", 1)
    assert ("cam-A", "b1") not in d._seen
