"""Hotlist matcher (PDF Phase 7 / Layer 3): hash map for exact hits, BK-tree for near misses,
candidates re-scored with an OCR-confusion-weighted edit distance.

BK-tree: metric tree over plain Levenshtein distance. A query of radius r only descends into
children whose edge label k satisfies |k - d(query, node)| <= r (triangle inequality), so most
of a large stolen-vehicle list is never compared.
"""
import math
from dataclasses import dataclass, field

# Look-alike pairs that OCR swaps: substituting one for the other costs little.
CONFUSABLE = {frozenset(p) for p in ["0O", "0D", "0Q", "1I", "1L", "2Z", "5S", "6G", "8B", "4A", "7T", "MH", "MN", "UV", "KX"]}
CHEAP_SUB, INDEL, SUB = 0.3, 1.0, 1.0


def levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def weighted_distance(a: str, b: str) -> float:
    """Edit distance where swapping look-alike characters costs CHEAP_SUB instead of 1."""
    prev = [j * INDEL for j in range(len(b) + 1)]
    for i, ca in enumerate(a, 1):
        cur = [i * INDEL]
        for j, cb in enumerate(b, 1):
            sub = 0.0 if ca == cb else CHEAP_SUB if frozenset((ca, cb)) in CONFUSABLE else SUB
            cur.append(min(prev[j] + INDEL, cur[j - 1] + INDEL, prev[j - 1] + sub))
        prev = cur
    return prev[-1]


@dataclass
class _Node:
    plate: str
    children: dict = field(default_factory=dict)  # edge distance -> _Node


class BKTree:
    def __init__(self):
        self.root = None
        self.size = 0

    def add(self, plate: str):
        self.size += 1
        if self.root is None:
            self.root = _Node(plate)
            return
        node = self.root
        while True:
            d = levenshtein(plate, node.plate)
            if d == 0:
                return
            if d not in node.children:
                node.children[d] = _Node(plate)
                return
            node = node.children[d]

    def search(self, query: str, radius: int) -> tuple[list[tuple[str, int]], int]:
        """All plates within `radius` edits, plus how many distance computations were needed."""
        if self.root is None:
            return [], 0
        found, stack, comparisons = [], [self.root], 0
        while stack:
            node = stack.pop()
            d = levenshtein(query, node.plate)
            comparisons += 1
            if d <= radius:
                found.append((node.plate, d))
            for k, child in node.children.items():
                if d - radius <= k <= d + radius:
                    stack.append(child)
        return found, comparisons


class Hotlist:
    """Wanted plates. match() returns (plate, score) with score in (0, 1]; 1 = exact read."""

    def __init__(self, plates, lam: float = 1.5, radius: int = 2):
        self.exact = set(plates)
        self.tree = BKTree()
        for p in plates:
            self.tree.add(p)
        self.lam, self.radius = lam, radius

    def match(self, read: str, ocr_conf: float = 1.0):
        if read in self.exact:  # O(1) hash hit
            return read, ocr_conf
        cands, _ = self.tree.search(read, self.radius)
        if not cands:
            return None, 0.0
        best = min(cands, key=lambda c: weighted_distance(read, c[0]))[0]
        return best, ocr_conf * math.exp(-self.lam * weighted_distance(read, best))
