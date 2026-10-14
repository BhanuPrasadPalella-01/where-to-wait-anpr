"""Indian plate-format correction for OCR output.

Standard format: SS DD [L]{1,3} DDDD  (state letters, district digits, series letters, number),
e.g. MH01DE2780, KA05M1234. BH series: YY BH DDDD LL (e.g. 22BH1234AB).
OCR confuses look-alike characters (0/O, 1/I, 8/B, 5/S, 2/Z, 6/G); the position in the format
says which one is possible, so we swap to the legal look-alike when a character is in the wrong class.
"""
import re

TO_LETTER = {"0": "O", "1": "I", "2": "Z", "4": "A", "5": "S", "6": "G", "7": "T", "8": "B"}
TO_DIGIT = {v: k for k, v in TO_LETTER.items()} | {"D": "0", "Q": "0", "U": "0", "L": "1", "J": "1"}

STANDARD = re.compile(r"^[A-Z]{2}\d{1,2}[A-Z]{0,3}\d{1,4}$")


def _force(ch: str, kind: str) -> str:
    if kind == "L":
        return ch if ch.isalpha() else TO_LETTER.get(ch, ch)
    return ch if ch.isdigit() else TO_DIGIT.get(ch, ch)


def _template(n: int, n_series: int) -> str:
    # 2 state letters + 2 district digits + series letters + remaining digits (1..4)
    rest = n - 4 - n_series
    return "LLDD" + "L" * n_series + "D" * rest if 1 <= rest <= 4 else ""


def correct(text: str) -> str:
    t = re.sub(r"[^A-Z0-9]", "", text.upper())
    if STANDARD.match(t):
        return t
    if len(t) == 10 and t[2:4] == "BH" or (len(t) == 10 and _force(t[2], "L") + _force(t[3], "L") == "BH"):
        tpl = "DDLLDDDDLL"  # BH series
        return "".join(_force(c, k) for c, k in zip(t, tpl))
    # choose the series length whose template needs the fewest substitutions
    best, best_cost = t, None
    for n_series in (2, 1, 3, 0):
        tpl = _template(len(t), n_series)
        if not tpl:
            continue
        fixed = "".join(_force(c, k) for c, k in zip(t, tpl))
        cost = sum(a != b for a, b in zip(t, fixed))
        if STANDARD.match(fixed) and (best_cost is None or cost < best_cost):
            best, best_cost = fixed, cost
    return best
