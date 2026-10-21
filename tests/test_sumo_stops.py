"""Stop placement for police units in SUMO (planner/sumo_sim.py).

stop_span is pure and always runs. The TraCI checks build a three-edge network with a 0.1 m middle edge
(like the shortest edges of the city network) and skip when SUMO is not installed, as in CI.
"""
import subprocess

import pytest

from planner.sumo_sim import stop_span

NODES = """<nodes>
  <node id="a" x="0" y="0" type="priority"/><node id="b" x="300" y="0" type="priority"/>
  <node id="c" x="300.1" y="0" type="priority"/><node id="d" x="600" y="0" type="priority"/>
</nodes>"""
EDGES = """<edges>
  <edge id="ab" from="a" to="b" numLanes="1" speed="20"/><edge id="bc" from="b" to="c" numLanes="1" speed="20"/>
  <edge id="cd" from="c" to="d" numLanes="1" speed="20"/>
</edges>"""


@pytest.mark.parametrize("length", [0.1, 0.5, 1.0, 1.99, 2.0, 3.0, 6.0, 300.0])
def test_stop_span_is_valid_on_any_lane(length):
    start, end = stop_span(length)
    assert 0.0 <= start < end <= length


def test_stop_span_is_near_the_lane_end():
    assert stop_span(300.0) == (294.0, 299.0)
    assert stop_span(1.5) == (0.0, 1.5)


@pytest.fixture
def traci_net(tmp_path):
    traci = pytest.importorskip("traci")
    pytest.importorskip("sumo")
    from planner.sumo_sim import sumo_bin
    (tmp_path / "n.nod.xml").write_text(NODES)
    (tmp_path / "n.edg.xml").write_text(EDGES)
    net = tmp_path / "n.net.xml"
    netconvert = sumo_bin()[:-len("sumo")] + "netconvert"
    subprocess.run([netconvert, "-n", str(tmp_path / "n.nod.xml"), "-e", str(tmp_path / "n.edg.xml"),
                    "-o", str(net), "--no-turnarounds"], check=True, capture_output=True)
    traci.start([sumo_bin(), "-n", str(net), "--no-step-log", "--no-warnings"], label=f"stops_{tmp_path.name}")
    traci.route.add("r", ["ab", "bc", "cd"])
    yield traci
    traci.close()


def test_unit_stops_on_a_tenth_of_a_metre_edge(traci_net):
    from planner.sumo_sim import place_stop
    traci = traci_net
    assert traci.lane.getLength("bc_0") == pytest.approx(0.1)
    traci.vehicle.add("u", "r", departSpeed="max")
    assert place_stop(traci, "u", "bc")
    for _ in range(60):
        traci.simulationStep()
        if traci.vehicle.isStopped("u"):
            break
    assert traci.vehicle.isStopped("u") and traci.vehicle.getRoadID("u") == "bc"


def test_stop_too_close_to_brake_is_refused_not_raised(traci_net):
    from planner.sumo_sim import place_stop
    traci = traci_net
    traci.vehicle.add("u", "r", departSpeed="max")
    while traci.vehicle.getRoadID("u") != "ab" or traci.vehicle.getLanePosition("u") < 285:
        traci.simulationStep()
    assert traci.vehicle.getSpeed("u") > 15  # ~20 m/s needs ~40 m to stop; the lane ends in < 15 m
    assert not place_stop(traci, "u", "ab")


def test_pilot_table_counts_interceptions_and_failures():
    from scripts.sumo_pilot import summarise
    row = {"demand": "normal", "coverage": 0.3, "method": "proposed", "escape_t": 300.0, "parked": 1}
    rows = [dict(row, intercepted=True), dict(row, intercepted=False, escape_t=900.0),
            {"seed": 7, "method": "b0_chase", "demand": "normal", "error": "TraCIException('x')"}]
    table = summarise(rows)
    assert "| normal | 30% | **Proposed** | 1/2 = 50% |" in table
    assert "| 900 | 1.00 |" in table
    assert "Failed episodes: 1 of 3." in table
