"""Signed, numbered messages between edge camera nodes and the control room.

Every plate read an edge node sends carries
    boot  random id chosen when the node starts (a restarted node starts a new numbering)
    seq   1, 2, 3, ... per boot
    sig   HMAC-SHA256 over the canonical JSON of all other fields, keyed with a secret shared by the
          node and the control room (WTW_EDGE_KEY)
The control room drops a message whose signature does not check out (forged or altered on the way)
and one whose (camera, boot, seq) it has already processed (a retry after a lost reply, or a replay).
Only the standard library is used, so edge nodes and the CI tests need nothing extra.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
from collections import OrderedDict

KEY_ENV = "WTW_EDGE_KEY"


def load_key(value: str | None = None) -> bytes | None:
    """Shared key from an explicit value or the WTW_EDGE_KEY environment variable; None if neither is set."""
    value = value if value is not None else os.environ.get(KEY_ENV)
    return value.encode() if value else None


def canonical(msg: dict) -> bytes:
    """Byte string that is signed: sorted keys, no whitespace, every field except the signature."""
    return json.dumps({k: v for k, v in msg.items() if k != "sig"}, sort_keys=True,
                      separators=(",", ":"), ensure_ascii=True).encode()


def sign(msg: dict, key: bytes) -> dict:
    """Return a copy of msg with its HMAC-SHA256 signature in msg["sig"]."""
    return {**msg, "sig": hmac.new(key, canonical(msg), hashlib.sha256).hexdigest()}


def verify(msg: dict, key: bytes) -> bool:
    sig = msg.get("sig")
    if not isinstance(sig, str):
        return False
    return hmac.compare_digest(sig, hmac.new(key, canonical(msg), hashlib.sha256).hexdigest())


class Numberer:
    """Edge side: stamps each outgoing message with this boot's id and the next sequence number."""

    def __init__(self, boot: str | None = None):
        self.boot = boot or secrets.token_hex(4)
        self.seq = 0

    def stamp(self, msg: dict) -> dict:
        self.seq += 1
        return {**msg, "boot": self.boot, "seq": self.seq}


class Deduper:
    """Control-room side: remembers which (camera, boot, seq) were already processed.

    Per camera boot it keeps a high-water mark plus the numbers seen above a gap, so memory stays small
    even when a queue is replayed out of order. Only the most recent `max_boots` boots are kept.
    """

    def __init__(self, max_boots: int = 256):
        self.max_boots = max_boots
        self._seen: OrderedDict[tuple[str, str], tuple[int, set[int]]] = OrderedDict()

    def seen(self, camera: str, boot: str, seq: int) -> bool:
        low, above = self._seen.get((camera, boot), (0, set()))
        return seq <= low or seq in above

    def add(self, camera: str, boot: str, seq: int) -> bool:
        """Record a message; False if it was a duplicate."""
        key = (camera, boot)
        if self.seen(camera, boot, seq):
            return False
        low, above = self._seen.pop(key, (0, set()))
        above.add(seq)
        while low + 1 in above:  # advance the high-water mark over the contiguous run
            low += 1
            above.discard(low)
        self._seen[key] = (low, above)
        while len(self._seen) > self.max_boots:
            self._seen.popitem(last=False)
        return True
