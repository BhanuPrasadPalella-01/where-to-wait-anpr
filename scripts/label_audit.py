"""Which source labels of the OCR crops deserve a second look? (reports/RESULTS.md §2: "≥1 source label
looks wrong (MH47AV6753)")

The labels come with the public dataset and some are wrong: a look-alike typed for the real character
(JK1182360 for JK11B2360), or a crop that shows another vehicle's plate. A wrong label counts as an OCR
error it is not, so the crops where the model and the format check both speak against the label are
listed for a manual look. Per crop whose read differs from its label:

  LOOKALIKE   the label is not a legal plate, the read is, and they differ only in look-alike characters
              (0/O, 1/I, 8/B, ...): the labeller most likely typed the look-alike
  ILLEGAL     the label is not a legal plate (and the above does not hold): a typo, a trade plate, or not
              a number plate at all
  OTHER_PLATE the label is legal, but the model reads a legal plate that shares less than half its
              characters with it, and is sure of every character: the crop probably shows another plate
  CONFIDENT   the label is legal, the model reads a different legal plate close to it and is sure of
              every character: either the label or a confident misread

"Sure" is the read's lowest character confidence >= MIN_CONF. The list is a work list, not a verdict:
CONFIDENT in particular also catches confident misreads (reports/RESULTS.md §2b has the count checked by
eye). No OCR and no images here; scripts/eval_ocr_errors.py feeds it the real reads.
"""
from __future__ import annotations

import re

from scripts.indian_plate_format import STANDARD, TO_DIGIT, TO_LETTER
from scripts.ocr_errors import OTHER_PLATE as _OTHER_PLATE
from scripts.ocr_errors import align, error_type

MIN_CONF = 0.9                  # lowest character confidence of a read the audit trusts over a legal label

LOOKALIKE = "look-alike typed in the label"
ILLEGAL = "label is not a legal plate"
OTHER_PLATE = "crop shows another plate"
CONFIDENT = "confident different read"
REASONS = [LOOKALIKE, ILLEGAL, OTHER_PLATE, CONFIDENT]

_BH = re.compile(r"^\d{2}BH\d{4}[A-Z]{1,2}$")
_ALIKE = {frozenset(p) for p in list(TO_LETTER.items()) + list(TO_DIGIT.items())}


def legal(plate: str) -> bool:
    """Standard (SS DD L{0,3} D{1,4}) or BH-series plate."""
    return bool(STANDARD.match(plate) or _BH.match(plate))


def lookalike_only(label: str, read: str) -> bool:
    """True when label and read have the same length and differ only by look-alike pairs."""
    if len(label) != len(read) or label == read:
        return False
    return all(a == b or frozenset((a, b)) in _ALIKE for a, b in zip(label, read))


def suspect(label: str, raw: str, read: str, conf: float, min_conf: float = MIN_CONF) -> str | None:
    """Reason to look at this crop's label again, or None.

    raw is the model's output, read the same after format correction, conf the lowest character
    confidence. A crop the model reads as its label (before or after correction) is never a suspect.
    """
    if label in (raw, read):
        return None
    if not legal(label):
        hit = next((r for r in (raw, read) if legal(r) and lookalike_only(label, r)), None)
        return LOOKALIKE if hit else ILLEGAL
    if conf < min_conf or not legal(read):
        return None
    return OTHER_PLATE if error_type(label, read) == _OTHER_PLATE else CONFIDENT


def audit(rows, min_conf: float = MIN_CONF) -> list[dict]:
    """Suspect crops among rows of {"label", "raw", "read", "conf", ...}, strongest reason first.

    Each suspect keeps its row and gains "reason", "edits" (label -> read) and "agree": how many of the
    label's other crops the model reads as the label. A label most of whose crops are read correctly is
    probably right, and the suspect crop the odd one out.
    """
    reads_ok: dict[str, int] = {}
    for r in rows:
        reads_ok[r["label"]] = reads_ok.get(r["label"], 0) + (r["label"] in (r["raw"], r["read"]))
    out = []
    for r in rows:
        why = suspect(r["label"], r["raw"], r["read"], r["conf"], min_conf)
        if why:
            out.append({**r, "reason": why, "edits": len(align(r["label"], r["read"])),
                        "agree": reads_ok[r["label"]]})
    out.sort(key=lambda s: (REASONS.index(s["reason"]), -s["conf"], s["label"]))
    return out


def summary(suspects) -> dict[str, dict]:
    """Per reason: number of suspect crops and of distinct labels."""
    return {why: {"crops": sum(s["reason"] == why for s in suspects),
                  "labels": len({s["label"] for s in suspects if s["reason"] == why})} for why in REASONS}
