"""What kind of mistake is each wrong OCR read? (reports/RESULTS.md §2: "mostly a dropped last digit")

A read is aligned to its label with the cheapest edit script, and the script says what went wrong:
  drop  a label character is missing from the read      MH47AV6753 -> MH47AV675
  sub   a label character was read as another one       MH02BT6482 -> MH02BY6482
  ins   the read has a character the label does not     TN21TC611  -> TN21TC6611

error_type() names the read by its script, breakdown() counts the names over a split, and
confusion_pairs() counts which character was read as which. No OCR and no images here: the functions work
on (label, read) strings, so they are tested without the model (tests/test_ocr_errors.py).
scripts/eval_ocr_errors.py runs them on the real crops.
"""
from __future__ import annotations

from collections import Counter

EXACT = "exact"
DROP_LAST = "dropped last character"
DROP = "dropped other character(s)"
SUB_ONE = "one substitution"
SUB_MANY = "several substitutions"
INS = "insertion(s)"
MIXED = "mixed"
OTHER_PLATE = "different plate"
TYPES = [EXACT, DROP_LAST, DROP, SUB_ONE, SUB_MANY, INS, MIXED, OTHER_PLATE]


def align(label: str, read: str) -> list[tuple]:
    """Cheapest edit script from label to read: ("sub", i, a, b), ("drop", i, a), ("ins", i, b).

    i is the position in the label (for an insertion: the label position it comes before; len(label) when
    it is appended). When several scripts cost the same, the one that edits furthest to the right is
    returned: GJ07BR1336 -> GJ07BR133 is then "the last 6 was dropped", not "a 3 was dropped and the 6
    read as 3".
    """
    n, m = len(label), len(read)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (label[i - 1] != read[j - 1]))
    ops, i, j = [], n, m
    while i or j:                       # walk back from the end; drops and insertions before matches
        if i and d[i][j] == d[i - 1][j] + 1:
            ops.append(("drop", i - 1, label[i - 1]))
            i -= 1
        elif j and d[i][j] == d[i][j - 1] + 1:
            ops.append(("ins", i, read[j - 1]))
            j -= 1
        else:
            if label[i - 1] != read[j - 1]:
                ops.append(("sub", i - 1, label[i - 1], read[j - 1]))
            i, j = i - 1, j - 1
    return ops[::-1]


def error_type(label: str, read: str) -> str:
    """One of TYPES for a (label, read) pair."""
    if read == label:
        return EXACT
    ops = align(label, read)
    if 2 * len(ops) > len(label):       # more than half the plate differs: not a misread of this plate
        return OTHER_PLATE
    kinds = {op[0] for op in ops}
    if kinds == {"drop"}:
        return DROP_LAST if len(ops) == 1 and ops[0][1] == len(label) - 1 else DROP
    if kinds == {"sub"}:
        return SUB_ONE if len(ops) == 1 else SUB_MANY
    if kinds == {"ins"}:
        return INS
    return MIXED


def breakdown(pairs) -> dict[str, int]:
    """Count of each error type over (label, read) pairs, in the order of TYPES."""
    c = Counter(error_type(label, read) for label, read in pairs)
    return {t: c[t] for t in TYPES}


def confusion_pairs(pairs) -> Counter:
    """Counter of (label character, read character) over the substitutions of all misreads.

    Reads classed as a different plate are left out: their alignment is arbitrary.
    """
    c = Counter()
    for label, read in pairs:
        if error_type(label, read) in (EXACT, OTHER_PLATE):
            continue
        c.update((op[2], op[3]) for op in align(label, read) if op[0] == "sub")
    return c


def error_positions(pairs) -> Counter:
    """Counter of where the edits fall: "state" (first 2 characters), "middle", "number" (last 4 characters but the last), "last"."""
    c = Counter()
    for label, read in pairs:
        if error_type(label, read) in (EXACT, OTHER_PLATE):
            continue
        for op in align(label, read):
            i, n = min(op[1], len(label) - 1), len(label)
            c["state" if i < 2 else "last" if i == n - 1 else "number" if i >= n - 4 else "middle"] += 1
    return c


def format_effect(rows) -> dict[str, int]:
    """What the format correction did, over (label, raw read, corrected read) rows."""
    out = {"unchanged": 0, "fixed": 0, "broken": 0, "changed, still wrong": 0}
    for label, raw, read in rows:
        if raw == read:
            out["unchanged"] += 1
        elif read == label:
            out["fixed"] += 1
        elif raw == label:
            out["broken"] += 1
        else:
            out["changed, still wrong"] += 1
    return out
