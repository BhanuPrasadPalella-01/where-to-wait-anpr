"""Headline plot and tables: interception rate vs camera coverage, 95% Wilson intervals."""
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LABELS = {"proposed": "Proposed (re-plan on each sighting)", "timer": "Proposed + 30 s timer re-plan",
          "no_neg": "Proposed − negative evidence", "no_gate": "Proposed − feasibility gate",
          "b2_once": "B2: plan once from incident", "b1_static": "B1: betweenness checkpoints",
          "b0_chase": "B0: nearest unit to last sighting"}
STYLE = {"proposed": ("#1f6feb", "-", "o"), "timer": ("#8957e5", "--", "s"), "no_neg": ("#2da44e", "--", "^"),
         "b2_once": ("#d1242f", "-", "D"), "b1_static": ("#bf8700", "-", "v"), "b0_chase": ("#57606a", "-", "x"),
         "no_gate": ("#0a7ea4", ":", "P")}


def wilson(k, n, z=1.96):
    if n == 0:
        return 0, 0, 0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def summarise(df, by):
    rows = []
    for key, g in df.groupby(by):
        p, lo, hi = wilson(int(g.intercepted.sum()), len(g))
        rows.append(dict(zip(by, key if isinstance(key, tuple) else (key,)), rate=p, lo=lo, hi=hi, n=len(g),
                         plan_ms=g.plan_ms_mean[g.n_plans > 0].mean() if (g.n_plans > 0).any() else float("nan"),
                         moves=g.moves.mean()))
    return pd.DataFrame(rows)


def main(name="final_coverage"):
    df = pd.read_json(ROOT / "reports" / f"{name}.jsonl", lines=True)
    df["coverage"] = df.coverage.round(2)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True)
    for ax, (title, sub) in zip(axes, [("All runs", df), ("Suspect follows a planner model", df[~df.held_out]),
                                       ("Suspect drives a model the planner did NOT assume", df[df.held_out])]):
        s = summarise(sub, ["method", "coverage"])
        for m in [m for m in LABELS if m in set(s.method)]:
            r = s[s.method == m].sort_values("coverage")
            c, ls, mk = STYLE[m]
            ax.plot(100 * r.coverage, 100 * r.rate, ls, color=c, marker=mk, label=LABELS[m], lw=2 if m == "proposed" else 1.3)
            ax.fill_between(100 * r.coverage, 100 * r.lo, 100 * r.hi, color=c, alpha=0.10)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Junctions with a camera (%)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Interception rate (%)  — shaded: 95% CI")
    axes[0].legend(fontsize=8, loc="upper left")
    n = df.groupby(["method", "coverage"]).size().min()
    fig.suptitle(f"Interception rate vs camera coverage — 4 units, {n} simulated chases per point "
                 f"(OSM road graph, 10.8k junctions)", fontsize=12)
    fig.tight_layout()
    out = ROOT / "reports" / f"{name}.png"
    fig.savefig(out, dpi=140)
    table = summarise(df, ["coverage", "method"])
    table.to_csv(ROOT / "reports" / f"{name}_table.csv", index=False)
    print(table.round(3).to_string(index=False))
    print("->", out)


if __name__ == "__main__":
    main(*sys.argv[1:])
