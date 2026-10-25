"""Own-photo fine-tuning: label checks, group split, dataset layout, replay list and report table.

Runs on tiny generated files; nothing here imports ultralytics or touches data/.
"""
import json

import pytest

from scripts.finetune_own import eval_sets, render_table, replay_list, train_yaml
from scripts.own_photos import build, check_label_file, split_groups


def test_label_checks():
    assert check_label_file("") == []
    assert check_label_file("0 0.5 0.5 0.2 0.1\n\n0 0.1 0.1 0.1 0.1") == []
    assert "expected 5" in check_label_file("0 0.5 0.5 0.2")[0]
    assert "not numeric" in check_label_file("0 a 0.5 0.2 0.1")[0]
    assert "class 1" in check_label_file("1 0.5 0.5 0.2 0.1")[0]
    assert "pixel" in check_label_file("0 0.95 0.5 0.2 0.1")[0]
    assert "width/height" in check_label_file("0 320 240 80 30")[0]


def _items(n_groups, per_group=2, conds=("day", "night", "two_row")):
    return [{"group": f"g{g}", "condition": conds[g % len(conds)]} for g in range(n_groups) for _ in range(per_group)]


def test_split_keeps_groups_together_and_covers_every_condition():
    items = _items(30)
    split = split_groups(items, seed=0)
    assert set(split) == {it["group"] for it in items}
    for cond in ("day", "night", "two_row"):
        assert any(split[it["group"]] == "test" for it in items if it["condition"] == cond)
        assert any(split[it["group"]] == "train" for it in items if it["condition"] == cond)
    assert split == split_groups(items, seed=0)
    assert split != split_groups(items, seed=1)


def test_split_with_a_single_group_keeps_it_for_training():
    assert split_groups(_items(1), seed=0) == {"g0": "train"}


def _make_src(tmp_path, n=24):
    src = tmp_path / "src"
    (src / "images").mkdir(parents=True)
    (src / "labels").mkdir()
    rows = ["image,condition,group"]
    for i in range(n):
        (src / "images" / f"p{i}.jpg").write_bytes(b"jpg")
        (src / "labels" / f"p{i}.txt").write_text("0 0.5 0.5 0.2 0.1\n")
        rows.append(f"p{i}.jpg,{['Day', 'night', 'two-row'][i % 3]},car{i // 2}")
    (src / "images" / "unlabelled.jpg").write_bytes(b"jpg")
    rows.append("unlabelled.jpg,day,")
    (src / "images" / "bad.jpg").write_bytes(b"jpg")
    (src / "labels" / "bad.txt").write_text("0 320 240 80 30\n")
    rows.append("bad.jpg,night,")
    (src / "images" / "extra.jpg").write_bytes(b"jpg")
    (src / "conditions.csv").write_text("\n".join(rows) + "\n")
    return src


def test_build_writes_splits_yamls_and_manifest(tmp_path):
    out = tmp_path / "own"
    m = build(_make_src(tmp_path), out)
    reasons = {d["image"]: d["reason"] for d in m["dropped"]}
    assert set(reasons) == {"unlabelled.jpg", "bad.jpg", "extra.jpg"}
    assert "conditions.csv" in reasons["extra.jpg"]
    assert sum(sum(c.values()) for c in m["counts"].values()) == 24
    assert m["test_conditions"] == ["day", "night", "two_row"] and m["unknown_conditions"] == []

    split_of_group = {}
    for it in m["items"]:
        assert split_of_group.setdefault(it["group"], it["split"]) == it["split"]
        assert (out / it["split"] / "images" / it["image"]).exists()
        assert (out / it["split"] / "labels" / it["label"]).exists()
    for cond in m["test_conditions"]:
        y = (out / f"own_test_{cond}.yaml").read_text()
        assert f"test: test_by_condition/{cond}/images" in y
        assert len(list((out / "test_by_condition" / cond / "images").iterdir())) == m["counts"]["test"][cond]
    assert "train: train/images" in (out / "own.yaml").read_text()
    assert json.loads((out / "manifest.json").read_text())["seed"] == 0


def test_build_refuses_an_empty_folder(tmp_path):
    src = tmp_path / "empty"
    (src / "images").mkdir(parents=True)
    (src / "conditions.csv").write_text("image,condition\n")
    with pytest.raises(SystemExit):
        build(src, tmp_path / "own")


def test_replay_list_and_train_yaml(tmp_path):
    out = tmp_path / "own"
    build(_make_src(tmp_path), out)
    n_own = len(list((out / "train" / "images").iterdir()))
    gen = tmp_path / "generic"
    gen.mkdir()
    for i in range(100):
        (gen / f"g{i}.jpg").write_bytes(b"jpg")
    lines = replay_list(out, gen, 2.0).read_text().split()
    assert len(lines) == 3 * n_own and len(set(lines)) == len(lines)
    assert len(replay_list(out, tmp_path / "missing", 2.0).read_text().split()) == n_own
    assert train_yaml(out, 0, 0) == out / "own.yaml"


def test_eval_sets_lists_conditions_first(tmp_path):
    out = tmp_path / "own"
    build(_make_src(tmp_path), out)
    tags = [t for t, _ in eval_sets(out)]
    assert tags[:4] == ["own_day", "own_night", "own_two_row", "own_all"]


def test_render_table_shows_change_or_dash():
    r = {"precision": 0.9, "recall": 0.5, "mAP50": 0.6, "mAP50-95": 0.4}
    better = dict(r, recall=0.7, mAP50=0.75)
    table = render_table({"shipped": {"own_night": r, "generic_test": r},
                          "finetuned": {"own_night": better}}, {"own_night": 12})
    lines = table.strip().splitlines()
    assert lines[2] == "| own_night | 12 | 0.600 | 0.750 | +0.150 | 0.500 | 0.700 | +0.200 |"
    assert lines[3] == "| generic_test |  | 0.600 | – | – | 0.500 | – | – |"
