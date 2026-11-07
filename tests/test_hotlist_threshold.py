"""Threshold sweep and choice rule for the fuzzy hotlist match (scripts/hotlist_threshold.py).

Small synthetic score tables only: no OCR, no hotlist build.
"""
import pytest

from scripts.hotlist_threshold import choose_theta, load_frozen, save_frozen, sweep


def read(wanted, truth, plate, score):
    return {"wanted": wanted, "truth": truth, "fuzzy_plate": plate, "score": score}


# 4 wanted reads (one misread to the wrong wanted plate, one with no candidate), 3 innocent reads
ROWS = [read(True, "KA01AB1234", "KA01AB1234", 0.90),
        read(True, "KA02CD5678", "KA02CD5678", 0.12),
        read(True, "MH12EF0001", "MH12EF0007", 0.30),   # matched the wrong wanted car
        read(True, "TN09GH4321", None, 0.0),             # nothing within radius 2
        read(False, "DL03JK1111", "DL03JK1117", 0.08),  # innocent near-miss of a wanted plate
        read(False, "AP05LM2222", None, 0.0),
        read(False, "GJ01NP3333", "GJ01NP3833", 0.02)]


def test_sweep_counts_recall_flags_and_wrong_alerts():
    t = sweep(ROWS, [0.0, 0.05, 0.1, 0.5])
    assert t[0.0] == {"wanted_recall": 0.5, "wrong_vehicle_alerts_per_1000_reads": round(1000 * 3 / 7, 1),
                      "innocent_cars_flagged": round(2 / 3, 4)}
    assert t[0.05]["innocent_cars_flagged"] == round(1 / 3, 4)
    assert t[0.1] == {"wanted_recall": 0.5, "wrong_vehicle_alerts_per_1000_reads": round(1000 / 7, 1),
                      "innocent_cars_flagged": 0.0}
    assert t[0.5]["wanted_recall"] == 0.25 and t[0.5]["wrong_vehicle_alerts_per_1000_reads"] == 0.0


def test_no_candidate_never_alerts_even_at_zero():
    t = sweep([read(False, "AP05LM2222", None, 0.0)], [0.0])
    assert t[0.0]["innocent_cars_flagged"] == 0.0


def test_choose_highest_recall_with_no_innocent_flagged():
    # 0.0 and 0.05 flag innocents; 0.1 and 0.12 keep recall 0.5 with none flagged -> larger wins the tie
    assert choose_theta(sweep(ROWS, [0.0, 0.05, 0.1, 0.12, 0.5])) == 0.12


def test_choose_prefers_recall_over_margin():
    t = {0.05: {"wanted_recall": 0.8, "innocent_cars_flagged": 0.0},
         0.2: {"wanted_recall": 0.7, "innocent_cars_flagged": 0.0},
         0.01: {"wanted_recall": 0.9, "innocent_cars_flagged": 0.01}}
    assert choose_theta(t) == 0.05


def test_choose_fails_loudly_when_no_theta_is_safe():
    with pytest.raises(ValueError):
        choose_theta(sweep(ROWS, [0.0, 0.05]))


def test_frozen_theta_round_trips(tmp_path):
    f = tmp_path / "theta.json"
    save_frozen(f, 0.05, "val", {"wanted_recall": 0.8})
    assert load_frozen(f) == 0.05
