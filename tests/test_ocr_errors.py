"""OCR error types and the label audit (scripts/ocr_errors.py, scripts/label_audit.py).

Synthetic (label, read) strings only: no OCR model, no images.
"""
from collections import Counter

from scripts import label_audit as la
from scripts import ocr_errors as oe


def test_align_gives_the_cheapest_script():
    assert oe.align("KA01AB1234", "KA01AB1234") == []
    assert oe.align("MH02BT6482", "MH02BY6482") == [("sub", 5, "T", "Y")]
    assert oe.align("MH47AV6753", "MH47AV675") == [("drop", 9, "3")]
    assert oe.align("TN21TC611", "TN21TC6611") == [("ins", 7, "6")]
    assert oe.align("MH47AV6753", "MH47AV673") == [("drop", 8, "5")]
    assert len(oe.align("UP16TC1366", "UP16TC336")) == 2          # no single edit turns 1366 into 336
    assert oe.align("", "AB") == [("ins", 0, "A"), ("ins", 0, "B")]
    assert oe.align("AB", "") == [("drop", 0, "A"), ("drop", 1, "B")]


def test_align_blames_the_rightmost_character_on_a_tie():
    # ...1336 -> ...133: the 6 was dropped, not "a 3 dropped and the 6 read as 3"
    assert oe.align("GJ07BR1336", "GJ07BR133") == [("drop", 9, "6")]
    # a repeated digit lost at the end counts as the last character
    assert oe.align("KA05M1233", "KA05M123") == [("drop", 8, "3")]


def test_error_types():
    assert oe.error_type("KA01AB1234", "KA01AB1234") == oe.EXACT
    assert oe.error_type("MH47AV6753", "MH47AV675") == oe.DROP_LAST
    assert oe.error_type("MH47AV6753", "MH47AV673") == oe.DROP           # a character before the last
    assert oe.error_type("KA01AB1234", "KA01AB12") == oe.DROP            # two characters, even at the end
    assert oe.error_type("MH02BT6482", "MH02BY6482") == oe.SUB_ONE
    assert oe.error_type("MH01DE2780", "MH01DC2700") == oe.SUB_MANY
    assert oe.error_type("TN21TC611", "TN21TC6611") == oe.INS
    assert oe.error_type("UP16TC1366", "UP16TC336") == oe.MIXED          # a substitution and a drop
    assert oe.error_type("MH47AV6753", "MH12FT9458") == oe.OTHER_PLATE   # 6 of 10 characters differ
    assert oe.error_type("MH47AV6753", "MH47AV0000") != oe.OTHER_PLATE   # 4 of 10: still this plate


def test_breakdown_counts_every_pair_once():
    pairs = [("KA01AB1234", "KA01AB1234"), ("KA01AB1234", "KA01AB123"), ("KA01AB1234", "KA01AB1284"),
             ("KA01AB1234", "KA01AB1284"), ("KA01AB1234", "TN99ZZ0000")]
    b = oe.breakdown(pairs)
    assert list(b) == oe.TYPES and sum(b.values()) == len(pairs)
    assert (b[oe.EXACT], b[oe.DROP_LAST], b[oe.SUB_ONE], b[oe.OTHER_PLATE]) == (1, 1, 2, 1)


def test_confusion_pairs_skip_other_plates():
    pairs = [("MH02BT6482", "MH02BI6482"), ("MH02BT6482", "MH02BI6402"), ("KA01AB1234", "KA01AB1234"),
             ("MH47AV6753", "MH12FT9458")]
    assert oe.confusion_pairs(pairs) == Counter({("T", "I"): 2, ("8", "0"): 1})


def test_error_positions():
    pairs = [("KA01AB1234", "TA01AB1234"),      # state
             ("KA01AB1234", "KA01AD1234"),      # middle (series letter)
             ("KA01AB1234", "KA01AB7234"),      # number, not its last digit
             ("KA01AB1234", "KA01AB123"),       # last (dropped)
             ("KA01AB1234", "KA01AB12345")]     # appended character counts as last
    assert oe.error_positions(pairs) == Counter({"state": 1, "middle": 1, "number": 1, "last": 2})


def test_format_effect():
    rows = [("KA01AB1234", "KA01AB1234", "KA01AB1234"),     # unchanged
            ("KA01AB1234", "KA0IAB1234", "KA01AB1234"),     # I -> 1 in the district: fixed
            ("TN19Q0835", "TN19Q0835", "TN19O0835"),        # (made up) correction broke a right read
            ("TN19Q0835", "TN1900835", "TN19O0835")]        # 0 -> O where the plate has Q: still wrong
    assert oe.format_effect(rows) == {"unchanged": 1, "fixed": 1, "broken": 1, "changed, still wrong": 1}


def test_legal_and_lookalike():
    assert la.legal("MH01DE2780") and la.legal("KA05M1234") and la.legal("22BH1234AB")
    assert not la.legal("JK1182360") and not la.legal("MH14TC206AN") and not la.legal("TERRANO")
    assert la.lookalike_only("JK1182360", "JK11B2360")            # 8 / B
    assert la.lookalike_only("WB02A03017", "WB02AD3017")          # 0 / D
    assert not la.lookalike_only("MH01CY6333", "MH01CV6333")      # Y / V are not a look-alike pair
    assert not la.lookalike_only("KA01AB1234", "KA01AB1234")
    assert not la.lookalike_only("KA01AB1234", "KA01AB123")


def test_suspect_reasons():
    assert la.suspect("JK1182360", "JK11B2360", "JK11B2360", 0.99) == la.LOOKALIKE
    assert la.suspect("JK1182360", "JK11B2360", "JK11B2360", 0.20) == la.LOOKALIKE    # no confidence needed
    assert la.suspect("MH14TC206AN", "MH14TC2061", "MH14TC2061", 0.31) == la.ILLEGAL
    assert la.suspect("TERRANO", "TERRAN0", "TERRAN0", 0.86) == la.ILLEGAL            # read is not legal either
    assert la.suspect("MH47AV6753", "MH12FT9458", "MH12FT9458", 0.99) == la.OTHER_PLATE
    assert la.suspect("MH01CY6333", "MH01CV6333", "MH01CV6333", 0.96) == la.CONFIDENT


def test_not_suspect():
    assert la.suspect("KA01AB1234", "KA01AB1234", "KA01AB1234", 0.99) is None
    assert la.suspect("DN0909704", "DN0909704", "DN09O9704", 0.92) is None    # raw read agrees with the label
    assert la.suspect("KA0IAB1234", "KA0IAB1234", "KA01AB1234", 0.9) is None
    assert la.suspect("MH47AV6753", "MH47AV675", "MH47AV675", 0.55) is None   # legal label, unsure read
    assert la.suspect("MH01CY6333", "MH01CV6333", "MH01CV6333", 0.89) is None  # just under MIN_CONF
    assert la.suspect("MH01CY6333", "MH01CV6333", "MH01CV6333", 0.89, min_conf=0.8) == la.CONFIDENT
    assert la.suspect("MH12SF3212", "MH12SF32127", "MH12SF32127", 0.95) is None   # confident, but not a legal plate


def row(label, read, conf, raw=None):
    return {"label": label, "raw": raw or read, "read": read, "conf": conf}


def test_audit_orders_and_counts_agreeing_crops():
    rows = [row("MH02ER9194", "MH02ER9194", 0.9), row("MH02ER9194", "MH02ER9194", 0.8),
            row("MH02ER9194", "MH02ER9192", 0.98),                       # odd one out of three crops
            row("MH47AV6753", "MH12FT9458", 0.99),
            row("WB02A03017", "WB02AD3017", 0.99),
            row("MH14TC206AN", "MH14TC2061", 0.3),
            row("KA01AB1234", "KA01AB1284", 0.4)]                        # ordinary misread: not listed
    out = la.audit(rows)
    assert [s["label"] for s in out] == ["WB02A03017", "MH14TC206AN", "MH47AV6753", "MH02ER9194"]
    assert [s["reason"] for s in out] == la.REASONS
    assert [s["agree"] for s in out] == [0, 0, 0, 2]
    assert out[0]["edits"] == 1 and out[2]["edits"] == 7
    assert la.summary(out) == {why: {"crops": 1, "labels": 1} for why in la.REASONS}
    assert la.audit(rows[:2]) == []
