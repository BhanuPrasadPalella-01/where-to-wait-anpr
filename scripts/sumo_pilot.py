"""SUMO pilot: run a small sweep of planner/sumo_sim.py and write a Markdown table of the results.

Needs the SUMO files in sumo/ (network, background traffic, warm-up states; see README) and
data/sumo_graph.json + data/sumo_edges.json. Episodes run in parallel worker processes.

    .venv/bin/python scripts/sumo_pilot.py --runs 20 --demands normal,peak
"""
import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LABELS = {"proposed": "**Proposed**", "b2_once": "B2 plan once", "b1_static": "B1 checkpoints",
          "b0_chase": "B0 chase"}


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def summarise(rows):
    """Markdown table per (demand, coverage, method): interception rate with 95% Wilson interval."""
    groups = defaultdict(list)
    errors = [r for r in rows if "error" in r]
    for r in rows:
        if "error" not in r:
            groups[(r["demand"], r["coverage"], r["method"])].append(r)
    out = ["| Demand | Coverage | Method | Intercepted | 95% CI | Median escape s | Units parked / ep |",
           "|---|---|---|---|---|---|---|"]
    order = list(LABELS)
    for d, c, m in sorted(groups, key=lambda x: (x[0], x[1], order.index(x[2]) if x[2] in order else 99)):
        g = groups[(d, c, m)]
        k = sum(r["intercepted"] for r in g)
        p, lo, hi = wilson(k, len(g))
        esc = sorted(r["escape_t"] for r in g)[len(g) // 2]
        parked = sum(r.get("parked", 0) for r in g) / len(g)
        out.append(f"| {d} | {c:.0%} | {LABELS.get(m, m)} | {k}/{len(g)} = {p:.0%} | {lo:.0%}–{hi:.0%} "
                   f"| {esc:.0f} | {parked:.2f} |")
    out.append("")
    out.append(f"Failed episodes: {len(errors)} of {len(rows)}.")
    for e in errors[:5]:
        out.append(f"- seed {e['seed']} {e['method']} {e['demand']}: `{e['error']}`")
    return "\n".join(out)


def main():
    from planner.sumo_sim import METHODS, sweep
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="sumo_pilot")
    ap.add_argument("--coverages", default="0.3")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--demands", default="normal")
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=10000, help="unseen seeds, as in the main experiments")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    t0 = time.time()
    rows = sweep(a.name, [float(c) for c in a.coverages.split(",")], a.methods.split(","),
                 a.demands.split(","), a.runs, a.seed0, a.workers)
    table = summarise(rows)
    head = (f"# {a.name}\n\nSeeds {a.seed0}–{a.seed0 + a.runs - 1}, 4 units, odd seeds use the held-out "
            f"back-streets driver. Wall time {time.time() - t0:.0f} s. Raw rows: `reports/{a.name}.jsonl`.\n\n")
    (ROOT / "reports" / f"{a.name}.md").write_text(head + table + "\n")
    print(table)


if __name__ == "__main__":
    main()
