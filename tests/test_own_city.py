"""Switching the study area: --center parsing, graph files with their area, $WTW_GRAPH, simulator options.

No download: ensure_graph is only checked on a file that already exists.
"""
import argparse
import os

import pytest

from planner import roadgraph
from planner.roadgraph import GRAPH_ENV, RoadGraph, area_graph_path, default_graph_path, ensure_graph, parse_center
from planner.simulate import World, add_map_args, use_map


def test_parse_center():
    assert parse_center("12.9716,77.5946") == (12.9716, 77.5946)
    assert parse_center(" 51.5, -0.12 ") == (51.5, -0.12)
    for bad in ("12.97", "12,77,3", "95,10", "10,190", "north,east"):
        with pytest.raises(ValueError):
            parse_center(bad)


def test_save_load_keeps_area(grid, tmp_path):
    g = RoadGraph(grid.xy, grid.adj, grid.main, grid.osm_ids, {"center": [12.97, 77.59], "radius_m": 2000})
    g.save(tmp_path / "g.json")
    back = RoadGraph.load(tmp_path / "g.json")
    assert back.meta == g.meta and back.adj == g.adj and back.main == g.main


def test_old_graph_files_load(grid, tmp_path):
    # files written before the area was recorded have no "meta" key
    grid.save(tmp_path / "g.json")
    text = (tmp_path / "g.json").read_text().replace(', "meta": {}', "")
    (tmp_path / "g.json").write_text(text)
    assert RoadGraph.load(tmp_path / "g.json").meta == {}


def test_area_paths_differ():
    a = area_graph_path((12.9716, 77.5946), 2000)
    assert a != area_graph_path((12.9716, 77.5946), 3000)
    assert a != area_graph_path((17.385, 78.4867), 2000)
    assert a.parent.name == "data"


def test_ensure_graph_reuses_existing_file(grid, tmp_path, monkeypatch):
    monkeypatch.setattr(roadgraph, "ROOT", tmp_path)
    monkeypatch.setattr(roadgraph, "build_from_osm", lambda *a: pytest.fail("should not download"))
    path = area_graph_path((1.0, 2.0), 500)
    path.parent.mkdir()
    grid.save(path)
    assert ensure_graph((1.0, 2.0), 500) == path


def test_env_selects_graph(grid, tmp_path, monkeypatch):
    monkeypatch.delenv(GRAPH_ENV, raising=False)
    assert default_graph_path() == roadgraph.DEFAULT_GRAPH
    grid.save(tmp_path / "grid.json")
    monkeypatch.setenv(GRAPH_ENV, str(tmp_path / "grid.json"))
    assert default_graph_path() == tmp_path / "grid.json"
    assert World().g.n == grid.n


def test_simulator_map_options(grid, tmp_path, monkeypatch):
    monkeypatch.delenv(GRAPH_ENV, raising=False)
    ap = argparse.ArgumentParser()
    add_map_args(ap)
    grid.save(tmp_path / "grid.json")
    assert use_map(ap.parse_args(["--graph", str(tmp_path / "grid.json")])) == tmp_path / "grid.json"
    assert os.environ[GRAPH_ENV] == str((tmp_path / "grid.json").resolve())
    with pytest.raises(SystemExit):
        use_map(ap.parse_args(["--graph", str(tmp_path / "missing.json")]))
    with pytest.raises(SystemExit):
        ap.parse_args(["--center", "not-a-place"])
