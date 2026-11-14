"""11-character plates and a 10-slot OCR model (reports/RESULTS.md §2: "the model has 10 character slots").

The fine-tuned OCR has ten classification heads, so it can never output an 11th character: a plate like
DL14CTC0153 (state, district, *three* series letters, four digits) comes back as DL14CTC015, which is
itself a legal-looking plate with a 3-digit number. Nothing downstream would notice.

check() decides whether a read was probably cut:
  * fewer characters than slots        -> the model ended the plate itself (it had a spare slot): fine
  * all slots used, standard format    -> a 3-letter series with fewer than 4 digits is what an 11-character
                                          plate looks like after its last digit is cut: flag
  * all slots used, BH / 2-letter series with 4 digits -> a normal 10-character plate: fine
  * all slots used, format broken      -> something longer than the model can spell (trade plates, odd
                                          labels): flag

For a flagged read the node can try a fallback read (tail_crop + stitch): read the right-hand part of the
plate again (the model only needs ~5 slots for it), and join it to the first read where the two overlap.
That only makes sense on a crop wide enough in pixels (should_retry). The plate aspect was tried as a second
gate (single-row plates only) and dropped: on the one two-row 11-character crop in the data the right half
read correctly (TC0153 of DL14CTC0153) while its bottom row did not, because the last characters sit at the
right edge in both layouts. If the fallback does not give a legal 11-character plate, the read
stays flagged: the read log asks for manual review and the hotlist treats it as a prefix of the real plate
(scripts/hotlist.py, Hotlist.match(truncated=True)).
"""
from __future__ import annotations

import re

from scripts.indian_plate_format import STANDARD

SLOTS = 10                      # max_plate_slots in weights/final/indian_plate_ocr_config.yaml
MIN_RETRY_WIDTH = 80            # px; a narrower crop leaves too few pixels for a second read of its tail
TAIL = 0.5                      # fraction of the plate width read again by the fallback
MIN_OVERLAP = 3                 # characters the head and tail reads must agree on to be joined

_PARTS = re.compile(r"^([A-Z]{2})(\d{1,2})([A-Z]{0,3})(\d{1,4})$")
_BH = re.compile(r"^\d{2}BH\d{4}[A-Z]{1,2}$")


def parts(read: str):
    """(state, district, series, number) of a standard plate, or None."""
    m = _PARTS.match(read)
    return m.groups() if m else None


def check(read: str, slots: int = SLOTS) -> tuple[bool, str]:
    """(likely truncated, reason) for one OCR read after format correction."""
    if len(read) < slots:
        return False, ""
    if _BH.match(read):
        return False, ""
    p = parts(read) if STANDARD.match(read) else None
    if p is None:
        return True, f"all {slots} OCR slots used and the read is not a legal plate"
    if len(p[2]) == 3 and len(p[3]) < 4:
        return True, f"3-letter series with a {len(p[3])}-digit number: an 11th character was probably cut"
    return False, ""


def full_plate(read: str) -> bool:
    """A legal plate that needs more than the model's slots (3-letter series and 4 digits, 11 characters)."""
    p = parts(read) if STANDARD.match(read) else None
    return p is not None and len(p[2]) == 3 and len(p[3]) == 4


def should_retry(width: int, height: int) -> bool:
    """A fallback tail read is only worth it on a crop wide enough that its right half still has detail."""
    return height > 0 and width >= MIN_RETRY_WIDTH


def tail_crop(img, frac: float = TAIL):
    """Right-hand part of a plate crop (numpy image, H x W x C)."""
    w = img.shape[1]
    return img[:, int(w * (1 - frac)):].copy()  # contiguous copy for the OCR's resize


def stitch(head: str, tail: str, min_overlap: int = MIN_OVERLAP) -> str | None:
    """Join a truncated head read and a tail read where they overlap: DL14CTC015 + TC0153 -> DL14CTC0153.

    The model was trained on whole plates, so a tail read often starts with invented characters
    (TLT18979 for the tail of PB03AGT8979): the overlap is the longest end of the head that appears
    anywhere in the tail. The join must add exactly one character and give a legal 11-character plate,
    otherwise None (the read stays flagged)."""
    for k in range(min(len(head), len(tail)), min_overlap - 1, -1):
        i = tail.rfind(head[-k:])
        if i >= 0:
            joined = head + tail[i + k:]
            return joined if len(joined) == len(head) + 1 and full_plate(joined) else None
    return None


def fallback(head: str, tails: list[str]) -> str | None:
    """Stitched plate from several tail reads (one per crop of the track): the most common join wins."""
    joins = [j for j in (stitch(head, t) for t in tails) if j]
    return max(set(joins), key=joins.count) if joins else None
