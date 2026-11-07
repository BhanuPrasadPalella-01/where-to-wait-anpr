"""Alert threshold for the fuzzy hotlist match, chosen on the validation split and frozen.

Plain Python on purpose (no pandas / OCR): the sweep and the choice rule are unit-tested in CI on a
small synthetic score table, while scripts/eval_hotlist.py feeds them real OCR reads.

A row is one read: {"wanted": bool, "truth": plate, "fuzzy_plate": best hotlist candidate or None,
"score": match score}. A read alerts when score >= theta.
  wanted recall          : wanted reads that alert on their own plate
  innocent cars flagged  : innocent reads that alert at all
  wrong-vehicle alerts   : alerts on a plate other than the true one, per 1000 reads
Rule: highest wanted recall with 0 innocent cars flagged; on a tie the larger theta (more margin).
"""
import json
from pathlib import Path

THETAS = (0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7)


def sweep(rows: list, thetas=THETAS) -> dict:
    """Metrics at each theta; keys are the theta values themselves."""
    rows = list(rows)
    wanted = [r for r in rows if r["wanted"]]
    innocent = [r for r in rows if not r["wanted"]]
    out = {}
    for theta in thetas:
        alert = lambda r: r["fuzzy_plate"] is not None and r["score"] >= theta  # noqa: E731
        caught = sum(alert(r) and r["fuzzy_plate"] == r["truth"] for r in wanted)
        wrong = sum(alert(r) and r["fuzzy_plate"] != r["truth"] for r in rows)
        flagged = sum(alert(r) for r in innocent)
        out[theta] = {"wanted_recall": round(caught / len(wanted), 4) if wanted else 0.0,
                      "wrong_vehicle_alerts_per_1000_reads": round(1000 * wrong / len(rows), 1) if rows else 0.0,
                      "innocent_cars_flagged": round(flagged / len(innocent), 4) if innocent else 0.0}
    return out


def choose_theta(table: dict) -> float:
    """Highest wanted recall among thetas that flag no innocent car; ties go to the larger theta."""
    safe = [(m["wanted_recall"], theta) for theta, m in table.items() if m["innocent_cars_flagged"] == 0]
    if not safe:
        raise ValueError("every theta in the grid flags an innocent car; extend the grid upwards")
    return max(safe)[1]


def save_frozen(path: Path, theta: float, split: str, metrics: dict) -> None:
    Path(path).write_text(json.dumps({"theta": theta, "chosen_on": split, "rule":
                                      "highest wanted recall with 0 innocent cars flagged; ties -> larger theta",
                                      split: metrics}, indent=2))


def load_frozen(path: Path) -> float:
    return float(json.loads(Path(path).read_text())["theta"])
