"""Colour and type check (edge/attributes.py), its use on the edge node and in the control room.

Synthetic images only: no camera, no model weights.
"""
import numpy as np
import pytest

from edge.attributes import (CLOSE, MISMATCH, attribute_factor, body_region, dominant_colour, normalise_colour,
                             normalise_type, vehicle_type)
from edge.camera_node import context_crop, track_attributes
from server.control_room import ControlRoom
from tests.synthetic import node

PAINT = {"white": (235, 235, 235), "black": (20, 20, 20), "grey": (130, 130, 130), "red": (30, 30, 200),
         "orange": (20, 130, 240), "yellow": (30, 220, 230), "green": (40, 160, 40), "blue": (200, 60, 20),
         "purple": (160, 40, 120), "brown": (30, 70, 110)}


def patch(bgr, h=30, w=60):
    return np.full((h, w, 3), bgr, np.uint8)


def car(paint, H=300, W=400, plate=(170, 220, 230, 240)):
    """A painted car body with a white plate on it; returns (frame, plate box)."""
    frame = np.full((H, W, 3), (90, 90, 90), np.uint8)  # grey road
    frame[60:260, 80:320] = paint
    x1, y1, x2, y2 = plate
    frame[y1:y2, x1:x2] = 255
    return frame, plate


@pytest.mark.parametrize("name", list(PAINT))
def test_dominant_colour_of_plain_paint(name):
    colour, share = dominant_colour(patch(PAINT[name]))
    assert colour == name and share == 1.0


def test_dominant_colour_takes_the_majority_and_reports_its_share():
    img = patch(PAINT["blue"])
    img[:, :15] = PAINT["white"]  # a quarter of the region is a reflection
    colour, share = dominant_colour(img)
    assert colour == "blue" and share == pytest.approx(0.75)


def test_dominant_colour_needs_enough_pixels():
    assert dominant_colour(patch(PAINT["red"], 4, 4)) == (None, 0.0)
    assert dominant_colour(np.zeros((0, 0, 3), np.uint8)) == (None, 0.0)


def test_body_region_is_above_the_plate_and_excludes_it():
    frame, box = car(PAINT["red"])
    region = body_region(frame, box)
    assert region.size and dominant_colour(region)[0] == "red"
    assert not (region == 255).all(axis=2).any(), "plate pixels must not vote"


def test_body_region_at_the_top_edge_is_empty_not_an_error():
    frame, _ = car(PAINT["red"])
    assert dominant_colour(body_region(frame, (170, 0, 230, 20))) == (None, 0.0)


@pytest.mark.parametrize("text,want", [("Pearl White", "white"), ("silver", "grey"), ("GRAY", "grey"),
                                       ("maroon", "red"), ("dark-blue", "blue"), ("golden", "yellow"),
                                       ("", None), (None, None), ("two-tone", None)])
def test_normalise_colour(text, want):
    assert normalise_colour(text) == want


def test_normalise_type():
    assert normalise_type("Bike") == "motorcycle" and normalise_type("SUV") == "car"
    assert normalise_type("auto-rickshaw") == "auto-rickshaw" and normalise_type("") is None


def test_vehicle_type_is_the_smallest_box_holding_the_plate():
    plate = (40, 40, 60, 50)
    dets = [(0, 0, 200, 200, 5),      # bus behind, also contains the plate centre
            (20, 10, 90, 70, 2),      # the car the plate is on
            (100, 100, 150, 150, 7),  # truck elsewhere
            (30, 30, 70, 60, 0)]      # a person: not a vehicle class
    assert vehicle_type(dets, plate) == "car"
    assert vehicle_type(dets[2:], plate) is None and vehicle_type([], plate) is None


def test_attribute_factor():
    want = {"colour": "white", "vehicle_type": "car"}
    assert attribute_factor("white", "car", want) == (1.0, {"colour": "ok", "type": "ok"})
    assert attribute_factor("red", "car", want) == (MISMATCH, {"colour": "mismatch", "type": "ok"})
    assert attribute_factor("grey", "truck", want)[0] == pytest.approx(CLOSE * MISMATCH)
    assert attribute_factor("grey", None, {"colour": "silver", "vehicle_type": "van"}) == (1.0, {"colour": "ok"})
    assert attribute_factor("red", "bus", {"colour": "", "vehicle_type": ""}) == (1.0, {})
    assert attribute_factor(None, None, want) == (1.0, {})
    assert attribute_factor("red", "bus", None) == (1.0, {})


def test_track_attributes_votes_colour_over_views_and_types_the_sharpest():
    frame, box = car(PAINT["blue"])
    glare, _ = car(PAINT["white"])  # one washed-out view
    views = [context_crop(f, box) for f in (frame, glare, frame)]
    calls = []

    def det(img):
        calls.append(img.shape)
        return [(0, 0, img.shape[1], img.shape[0], 2)]

    assert track_attributes(views, det) == ("blue", "car")
    assert len(calls) == 1, "the vehicle model runs once per track"
    assert track_attributes(views) == ("blue", None)
    assert track_attributes([]) == (None, None)


def test_context_crop_keeps_the_plate_inside():
    frame, box = car(PAINT["red"])
    img, (x1, y1, x2, y2) = context_crop(frame, box)
    assert (img[y1:y2, x1:x2] == 255).all()


COLS = 9
A = node(4, 2, COLS)


@pytest.fixture
def room(grid_world):
    r = ControlRoom(coverage=0.25, n_units=3, seed=1, world=grid_world)
    r.register_edge("cam-A", junction=A)
    return r


def test_hotlist_serves_registered_attributes(room):
    room.register_vehicle(plate="KL02BM4659", case_no="FIR 1/2026", offence="robbery", officer="SI Rao",
                          colour="Pearl White", vehicle_type="car")
    room.register_vehicle(plate="TS09EK4417", case_no="FIR 2/2026", offence="theft", officer="SI Rao")
    assert room.hotlist_attributes() == {"KL02BM4659": {"colour": "Pearl White", "vehicle_type": "car"}}


def test_edge_read_keeps_the_attribute_check(room):
    room.open_case("TS09EK4417")
    rec = room.edge_read({"name": "cam-A", "plate": "TS09EK4417", "conf": 0.9, "matched": False,
                          "hotlist_plate": "TS09EK4417", "score": 0.36, "plate_score": 0.9,
                          "colour": "red", "vehicle_type": "car", "attr": {"colour": "mismatch", "type": "ok"}})
    assert not rec["hit"] and room.sightings == []
    assert (rec["colour"], rec["vehicle_type"], rec["plate_score"]) == ("red", "car", 0.9)
    assert rec["attr"] == {"colour": "mismatch", "type": "ok"}
    old = room.edge_read({"name": "cam-A", "plate": "KA01AB1234", "conf": 0.9, "matched": False})
    assert old["attr"] == {} and old["colour"] is None, "reads from older nodes still work"
