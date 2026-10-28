"""Vehicle colour and type next to the plate read (PDF Layer 3: "attribute check").

A cloned or misread plate often sits on the wrong car. The edge node therefore also looks at the body
around the plate:

* colour: HSV histogram of the paintwork just above the plate (the boot lid / bonnet). Low-saturation
  pixels vote for black / grey / white by brightness, the rest for a hue bin; the biggest bin wins.
* type: the COCO vehicle box (car, motorcycle, bus, truck) from a stock YOLO model that contains the
  plate centre, when such a model is available.

The registered colour / type of the hotlist vehicle is then compared with what the camera saw: a
mismatch lowers the match score, a near miss (white vs grey, van vs car) lowers it a little, and an
unknown on either side leaves it unchanged — the attribute check can only weaken a plate match, never
create one.
"""
from __future__ import annotations

import cv2
import numpy as np

# Hue bins on OpenCV's 0-179 scale (red wraps around 0).
HUES = [("red", 0, 8), ("orange", 8, 20), ("yellow", 20, 34), ("green", 34, 85), ("blue", 85, 130),
        ("purple", 130, 160), ("red", 160, 180)]
SAT_MIN, DARK_V, WHITE_V = 60, 60, 190  # below SAT_MIN a pixel is black / grey / white by its brightness
BROWN_V = 140                             # dark, saturated orange/red reads as brown on paintwork

# Words officers type in the registration form -> the colour names the histogram can produce.
COLOUR_WORDS = {
    "white": "white", "pearl": "white", "cream": "white", "ivory": "white",
    "silver": "grey", "grey": "grey", "gray": "grey", "ash": "grey",
    "black": "black",
    "red": "red", "maroon": "red", "wine": "red", "cherry": "red",
    "orange": "orange", "yellow": "yellow", "gold": "yellow", "golden": "yellow", "beige": "brown",
    "brown": "brown", "bronze": "brown", "green": "green", "olive": "green",
    "blue": "blue", "navy": "blue", "purple": "purple", "violet": "purple",
}
# Pairs a camera confuses under street light / shade: a softer penalty than a real mismatch.
CLOSE_COLOURS = {frozenset(p) for p in [("white", "grey"), ("grey", "black"), ("red", "orange"),
                                        ("orange", "yellow"), ("orange", "brown"), ("red", "brown"),
                                        ("yellow", "brown"), ("blue", "purple"), ("red", "purple")]}

# COCO class id -> vehicle type in the registration form's words.
COCO_TYPES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
# Form types COCO cannot tell apart from one of its own: treated as a near miss, not a mismatch.
CLOSE_TYPES = {frozenset(p) for p in [("van", "car"), ("van", "truck"), ("bus", "truck"),
                                      ("auto-rickshaw", "motorcycle"), ("auto-rickshaw", "car")]}

MISMATCH, CLOSE = 0.4, 0.8  # score multipliers


def normalise_colour(text: str | None) -> str | None:
    """'Pearl White', 'silver', 'MAROON' -> white / grey / red; None when the word is not known."""
    for w in (text or "").lower().replace("-", " ").split():
        if w in COLOUR_WORDS:
            return COLOUR_WORDS[w]
    return None


def normalise_type(text: str | None) -> str | None:
    t = (text or "").strip().lower()
    return {"bike": "motorcycle", "scooter": "motorcycle", "auto": "auto-rickshaw", "lorry": "truck",
            "suv": "car", "jeep": "car", "taxi": "car"}.get(t, t) or None


def body_region(frame: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """Paintwork above the plate: 3 plate-widths wide, from 2.5 plate-heights above it to just above it.
    The plate itself (white/yellow background) is excluded so it never decides the colour."""
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = box
    w, h, cx = x2 - x1, y2 - y1, (x1 + x2) / 2
    xa, xb = int(max(0, cx - 1.5 * w)), int(min(W, cx + 1.5 * w))
    ya, yb = int(max(0, y1 - 2.5 * h)), int(max(0, y1 - 0.3 * h))
    return frame[ya:yb, xa:xb]


def dominant_colour(bgr: np.ndarray, min_pixels: int = 50) -> tuple[str | None, float]:
    """(colour name, share of pixels voting for it); (None, 0.0) for a region too small to judge."""
    if bgr is None or bgr.size == 0 or bgr.shape[0] * bgr.shape[1] < min_pixels:
        return None, 0.0
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.int32)
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    votes: dict[str, int] = {}
    grey = s < SAT_MIN
    votes["black"] = int(np.sum(grey & (v < DARK_V)))
    votes["white"] = int(np.sum(grey & (v >= WHITE_V)))
    votes["grey"] = int(np.sum(grey & (v >= DARK_V) & (v < WHITE_V)))
    dark = ~grey & (v < DARK_V)  # saturated but nearly black: shadow, counts as black
    votes["black"] += int(np.sum(dark))
    col = ~grey & ~dark
    for name, lo, hi in HUES:
        m = col & (h >= lo) & (h < hi)
        if name == "orange":  # dull orange paint is brown (dark red stays red: maroon)
            brown = m & (v < BROWN_V)
            votes["brown"] = int(np.sum(brown))
            m = m & ~brown
        votes[name] = votes.get(name, 0) + int(np.sum(m))
    best = max(votes, key=votes.get)
    return best, votes[best] / len(hsv)


def vehicle_type(dets, box: tuple[int, int, int, int]) -> str | None:
    """Type of the smallest COCO vehicle box containing the plate centre.

    dets: iterable of (x1, y1, x2, y2, class_id) in the same pixel frame as the plate box."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    inside = [(abs((d[2] - d[0]) * (d[3] - d[1])), COCO_TYPES[int(d[4])]) for d in dets
              if int(d[4]) in COCO_TYPES and d[0] <= cx <= d[2] and d[1] <= cy <= d[3]]
    return min(inside)[1] if inside else None


def compare(seen: str | None, wanted: str | None, close: set) -> str | None:
    """'ok', 'close', 'mismatch', or None when either side is unknown."""
    if not seen or not wanted:
        return None
    if seen == wanted:
        return "ok"
    return "close" if frozenset((seen, wanted)) in close else "mismatch"


def attribute_factor(seen_colour: str | None, seen_type: str | None, want: dict | None) -> tuple[float, dict]:
    """Multiplier for the plate match score, and what was compared (shown in the read log)."""
    want = want or {}
    checks = {"colour": compare(seen_colour, normalise_colour(want.get("colour")), CLOSE_COLOURS),
              "type": compare(seen_type, normalise_type(want.get("vehicle_type")), CLOSE_TYPES)}
    factor = 1.0
    for result in checks.values():
        factor *= MISMATCH if result == "mismatch" else CLOSE if result == "close" else 1.0
    return factor, {k: v for k, v in checks.items() if v}
