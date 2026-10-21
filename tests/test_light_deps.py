"""Keeps the tested modules importable with requirements-test.txt alone (what CI installs).

Heavy packages (detector, OCR runtime, OSM download, SUMO, plotting) must only be imported inside the
functions that need them, so a new top-level import of one of them fails here instead of only in CI.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TESTED = ["planner.roadgraph", "planner.fugitive", "planner.belief", "planner.plan", "planner.reconstruct",
          "planner.simulate", "planner.sumo_sim", "server.db", "server.control_room", "edge.camera_node"]
HEAVY = ["torch", "ultralytics", "fast_plate_ocr", "onnxruntime", "osmnx", "traci", "sumolib", "pandas",
         "matplotlib", "fastapi"]


def test_tested_modules_skip_heavy_imports():
    # fresh interpreter: this pytest process may already have imported anything
    code = (f"import importlib, json, sys\n"
            f"for m in {TESTED!r}: importlib.import_module(m)\n"
            f"print(json.dumps(sorted(h for h in {HEAVY!r} if h in sys.modules)))")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout.strip().splitlines()[-1]) == []
