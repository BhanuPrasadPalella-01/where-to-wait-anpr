"""Markdown table for the two-suspect experiment (planner/multi_sim.py output in reports/<name>.jsonl).

    .venv/bin/python -m planner.multi_sim --name multi_suspect --runs 200
    .venv/bin/python scripts/multi_suspect_table.py reports/multi_suspect.jsonl
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LABELS = {"weighted": "**Severity-weighted, useful locks**", "equal": "Equal weights, useful locks",
          "hard_lock": "Severity-weighted, hard locks", "no_lock": "Severity-weighted, no lock",
          "split": "Units split per case"}


def mcnemar(wins: int, losses: int) -> float:
    """Exact two-sided McNemar (binomial) p-value on the discordant pairs."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def summarise(rows) -> str:
    by = {}
    for r in rows:
        by.setdefault(r["method"], {})[r["seed"]] = r
    out = ["| Method | Severe suspect caught | Minor suspect caught | Severity-weighted catch | Pulls / ep | Hand-offs / ep | Moves / ep | Plan ms |",
           "|---|---|---|---|---|---|---|---|"]
    for m in [m for m in LABELS if m in by] + [m for m in by if m not in LABELS]:
        g = list(by[m].values())
        n = len(g)
        out.append(f"| {LABELS.get(m, m)} | {sum(r['caught_severe'] for r in g) / n:.1%} "
                   f"| {sum(r['caught_minor'] for r in g) / n:.1%} | {sum(r['score'] for r in g) / n:.3f} "
                   f"| {sum(r['pulls'] for r in g) / n:.2f} | {sum(r.get('handoffs', 0) for r in g) / n:.2f} "
                   f"| {sum(r['moves'] for r in g) / n:.1f} "
                   f"| {sum(r['plan_ms_mean'] for r in g) / n:.0f} |")
    if "weighted" in by:
        out += ["", "Paired against severity-weighted (same seeds), severe suspect caught:", ""]
        for m in by:
            if m == "weighted":
                continue
            seeds = by["weighted"].keys() & by[m].keys()
            wins = sum(by["weighted"][s]["caught_severe"] and not by[m][s]["caught_severe"] for s in seeds)
            losses = sum(by[m][s]["caught_severe"] and not by["weighted"][s]["caught_severe"] for s in seeds)
            out.append(f"- vs {LABELS.get(m, m).strip('*')}: {wins} wins / {losses} losses, McNemar p = {mcnemar(wins, losses):.2g}")
    n = len(next(iter(by.values())))
    out += ["", f"{n} paired scenarios per method."]
    return "\n".join(out)


if __name__ == "__main__":
    path = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "reports" / "multi_suspect.jsonl")
    print(summarise([json.loads(line) for line in path.read_text().splitlines() if line.strip()]))
