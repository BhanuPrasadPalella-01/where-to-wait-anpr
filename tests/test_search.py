"""Vehicle Search index: system search never reads the answer key; routes follow the road graph."""
import pytest

from server.search import CAMS, SearchIndex

idx = SearchIndex()
pytestmark = pytest.mark.skipif(not idx.ok, reason="RoundaboutHD tracks not built (scripts/roundabout_fullpass.py)")


def test_ground_truth_lookup_is_time_ordered_with_a_route():
    r = idx.ground_truth(idx.meta()["examples"][0])
    assert len(r["stops"]) >= 2 and all(s["correct"] for s in r["stops"])
    assert [s["t_first"] for s in r["stops"]] == sorted(s["t_first"] for s in r["stops"])
    assert len(r["route"]) == len(r["stops"]) - 1 and all(len(seg["lonlat"]) >= 2 for seg in r["route"])


def test_system_search_ignores_ground_truth():
    cam, track = 1, next(iter(idx.tracks[1]))
    before = idx.search_from(cam, track)
    saved = {c: {t: tr["gt"] for t, tr in idx.tracks[c].items()} for c in CAMS}
    try:
        for c in CAMS:
            for tr in idx.tracks[c].values():
                tr["gt"] = None
        after = idx.search_from(cam, track)
    finally:
        for c in CAMS:
            for t, g in saved[c].items():
                idx.tracks[c][t]["gt"] = g
    assert [(s["cam"], s["track"]) for s in before["stops"]] == [(s["cam"], s["track"]) for s in after["stops"]]


def test_pick_finds_the_box_under_the_click():
    tr = next(t for t in idx.tracks[2].values() if len(t["frames"]) > 20)
    f, b = tr["frames"][10], tr["box"][10]
    assert idx.pick(2, f / idx.fps, (b[0] + b[2]) / 2, (b[1] + b[3]) / 2) is not None


def test_overview_counts_add_up():
    o = idx.overview()
    s = o["summary"]
    assert sum(s["by_count"].values()) == s["vehicles"] == len(o["vehicles"])
    assert s["multi"] == s["vehicles"] - s["by_count"][1]
    assert 0 <= s["system_full"] <= s["multi"]
    for r in o["vehicles"][:20]:
        assert len(r["paths"]) == len(r["cams"]) == len(r["tracks"])


def test_trail_ends_are_explained():
    r = idx.search_from(2, next(iter(idx.tracks[2])))
    assert set(r["ends"]) == {"after", "before"}
    assert all(e["why"] for e in r["ends"].values())
