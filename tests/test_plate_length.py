"""11-character plates and the 10-slot OCR (edge/plate_length.py), the prefix match in scripts/hotlist.py,
the fallback read on the edge node and the review flag in the control room.

Synthetic reads and a fake OCR only: no camera, no model weights.
"""
import numpy as np
import pytest

from edge import plate_length
from edge.camera_node import complete_read
from scripts.hotlist import Hotlist
from server.control_room import ControlRoom
from tests.synthetic import node


@pytest.mark.parametrize("read", ["DL14CTC015", "PB03AGT189", "MH14TCF460"])
def test_three_letter_series_with_a_short_number_at_full_length_is_flagged(read):
    cut, why = plate_length.check(read)
    assert cut and "11th character" in why


@pytest.mark.parametrize("read", ["MH43AF5037", "KA05M1234", "22BH1234AB", "DL14CTC01", "MH12AB123"])
def test_normal_plates_and_reads_with_a_spare_slot_are_not_flagged(read):
    assert plate_length.check(read) == (False, "")


def test_a_broken_format_at_full_length_is_flagged():
    cut, why = plate_length.check("KL01KL0K01")
    assert cut and "not a legal plate" in why


def test_full_plate_needs_three_series_letters_and_four_digits():
    assert plate_length.full_plate("DL14CTC0153")
    assert not plate_length.full_plate("DL14CTC015")
    assert not plate_length.full_plate("MH43AF5037")


def test_stitch_joins_on_the_overlap():
    assert plate_length.stitch("DL14CTC015", "TC0153") == "DL14CTC0153"
    assert plate_length.stitch("DL14CTC015", "C0153") == "DL14CTC0153"
    # the tail read often starts with invented characters; the overlap may sit inside it
    assert plate_length.stitch("PB03AGT897", "TLT8979") == "PB03AGT8979"


def test_stitch_refuses_joins_that_do_not_give_one_legal_extra_character():
    assert plate_length.stitch("DL14CTC015", "XX0999") is None       # no overlap of 3
    assert plate_length.stitch("DL14CTC015", "TC01537") is None      # two extra characters
    assert plate_length.stitch("PB03AGT189", "TLT18979") is None     # head misread: join is 12 long
    assert plate_length.stitch("DL14CTC015", "TC015A") is None       # 11 long but not a legal plate


def test_fallback_takes_the_most_common_join():
    assert plate_length.fallback("DL14CTC015", ["TC0153", "C0158", "TC0153", "junk"]) == "DL14CTC0153"
    assert plate_length.fallback("DL14CTC015", ["junk", ""]) is None


def test_should_retry_and_tail_crop():
    assert plate_length.should_retry(199, 61) and plate_length.should_retry(199, 155)
    assert not plate_length.should_retry(66, 38), "too few pixels for a second read"
    img = np.arange(10 * 200 * 3, dtype=np.uint8).reshape(10, 200, 3)
    tail = plate_length.tail_crop(img)
    assert tail.shape == (10, 100, 3) and (tail == img[:, 100:]).all() and tail.flags.c_contiguous


# ---------------------------------------------------------------- hotlist prefix match
WANTED = ["DL14CTC0153", "MH43AF5037", "KA05MN1234"]


def test_truncated_read_is_a_prefix_hit():
    hl = Hotlist(WANTED)
    plate, score = hl.match("DL14CTC015", 0.8)
    assert plate == "DL14CTC0153" and score == pytest.approx(0.8 * np.exp(-1.5)), "plain match pays an indel"
    assert hl.match("DL14CTC015", 0.8, truncated=True) == ("DL14CTC0153", 0.8)


def test_truncated_read_with_a_misread_compares_only_the_characters_it_has():
    hl = Hotlist(WANTED)
    plate, score = hl.match("DL14CTC0I5", 1.0, truncated=True)  # 1 -> I is a look-alike swap
    assert plate == "DL14CTC0153" and score == pytest.approx(np.exp(-1.5 * 0.3))
    assert score > hl.match("DL14CTC0I5", 1.0)[1]


def test_shared_prefix_splits_the_score():
    hl = Hotlist(WANTED + ["DL14CTC0158"])
    assert hl.match("DL14CTC015", 0.9, truncated=True) == ("DL14CTC0153", pytest.approx(0.45))


def test_exact_hit_wins_over_prefix_and_untruncated_reads_are_unchanged():
    hl = Hotlist(WANTED + ["DL14CTC015"])
    assert hl.match("DL14CTC015", 0.7, truncated=True) == ("DL14CTC015", 0.7)
    hl = Hotlist(WANTED)
    assert hl.match("MH43AF5O37", 0.9, truncated=True) == hl.match("MH43AF5O37", 0.9)


# ---------------------------------------------------------------- edge node fallback read
def crop(w=200, h=60):
    return np.zeros((h, w, 3), np.uint8)


def test_complete_read_recovers_the_cut_character():
    calls = []

    def ocr(imgs):
        calls.append([i.shape for i in imgs])
        return ["TC0153"] * len(imgs)

    plate, cut, review = complete_read(ocr, [crop(), crop(), crop(60, 40)], "DL14CTC015")
    assert (plate, cut) == ("DL14CTC0153", False) and "second read" in review
    assert calls == [[(60, 100, 3), (60, 100, 3)]], "only crops wide enough are read again, right half only"


def test_complete_read_keeps_the_flag_when_the_fallback_fails():
    plate, cut, review = complete_read(lambda imgs: ["XXXX"] * len(imgs), [crop()], "DL14CTC015")
    assert (plate, cut) == ("DL14CTC015", True) and review.startswith("check plate")
    plate, cut, _ = complete_read(lambda imgs: pytest.fail("no retry on narrow crops"), [crop(60, 40)], "DL14CTC015")
    assert cut


def test_complete_read_leaves_normal_reads_alone():
    assert complete_read(lambda imgs: pytest.fail("no second read"), [crop()], "MH43AF5037") == ("MH43AF5037", False, "")


# ---------------------------------------------------------------- control room
A = node(4, 2, 9)


@pytest.fixture
def room(grid_world):
    r = ControlRoom(coverage=0.25, n_units=3, seed=1, world=grid_world)
    r.register_edge("cam-A", junction=A)
    return r


def test_control_room_keeps_the_review_flag_and_logs_a_prefix_hit(room):
    room.open_case("DL14CTC0153")
    rec = room.edge_read({"name": "cam-A", "plate": "DL14CTC015", "conf": 0.9, "matched": True,
                          "hotlist_plate": "DL14CTC0153", "score": 0.9, "truncated": True,
                          "review": "check plate: 3-letter series with a 3-digit number"})
    assert rec["hit"] and rec["truncated"] and rec["review"].startswith("check plate")
    assert any(e.kind == "flag" and "as a prefix" in e.text for e in room.events)
    old = room.edge_read({"name": "cam-A", "plate": "KA01AB1234", "conf": 0.9, "matched": False})
    assert old["truncated"] is False and old["review"] == "", "reads from older nodes still work"
