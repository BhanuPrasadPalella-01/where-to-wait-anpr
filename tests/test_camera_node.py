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
