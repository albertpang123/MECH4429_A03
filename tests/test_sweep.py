"""Config overrides and sweep planning, without starting a training run."""

import copy
from pathlib import Path

import pytest

from envs.config import apply_override, config_hash, load_config
from train.sweep import build_runs, format_value, parse_grid

ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(ROOT / "configs" / "g3_hover.yaml")


def small_config():
    return {"ppo": {"n_epochs": 10, "net_arch": [64, 64]}, "run": {"seed": 0}}


# --- apply_override ---------------------------------------------------------


def test_override_parses_yaml_types():
    config = small_config()
    apply_override(config, "ppo.n_epochs=5")
    apply_override(config, "ppo.net_arch=[64, 128]")
    apply_override(config, "run.seed=null")
    assert config["ppo"]["n_epochs"] == 5
    assert config["ppo"]["net_arch"] == [64, 128]
    assert config["run"]["seed"] is None


def test_override_does_not_touch_siblings():
    config = small_config()
    apply_override(config, "ppo.n_epochs=1")
    assert config["ppo"]["net_arch"] == [64, 64]


def test_override_changes_the_config_hash():
    config = small_config()
    before = config_hash(config)
    apply_override(config, "ppo.n_epochs=5")
    assert config_hash(config) != before


def test_override_rejects_unknown_leaf_and_lists_known_keys():
    with pytest.raises(ValueError, match="no such key at ppo"):
        apply_override(small_config(), "ppo.n_epoch=5")
    with pytest.raises(ValueError, match="n_epochs"):
        apply_override(small_config(), "ppo.n_epoch=5")


def test_override_rejects_unknown_parent():
    with pytest.raises(ValueError, match="no such key at <root>"):
        apply_override(small_config(), "nope.n_epochs=5")


def test_override_rejects_internal_keys():
    with pytest.raises(ValueError, match="'_'"):
        apply_override(small_config(), "_provenance.source=x")


def test_override_rejects_missing_value():
    with pytest.raises(ValueError, match="path=value"):
        apply_override(small_config(), "ppo.n_epochs")


def test_override_reaches_a_real_config_key():
    config = copy.deepcopy(CONFIG)
    apply_override(config, "ppo.n_steps=256")
    assert config["ppo"]["n_steps"] == 256


# --- sweep planning ---------------------------------------------------------


def test_parse_grid_types():
    path, values = parse_grid("ppo.learning_rate=1e-4,3e-4")
    assert path == "ppo.learning_rate"
    # PyYAML alone would read "1e-4" as a string; the sweep must not.
    assert values == [0.0001, 0.0003]


def test_parse_grid_scalars_of_each_kind():
    assert parse_grid("faults.rotor_id=0,3")[1] == [0, 3]
    assert parse_grid("faults.mode=fixed,partial")[1] == ["fixed", "partial"]
    assert parse_grid("observation.normalize=true,false")[1] == [True, False]


def test_parse_grid_refuses_a_comma_bearing_list_value():
    with pytest.raises(ValueError, match="commas separate grid values"):
        parse_grid("ppo.net_arch=[64,128]")


def test_parse_grid_requires_an_equals_sign():
    with pytest.raises(ValueError, match="path=v1,v2"):
        parse_grid("ppo.n_epochs")


def test_format_value_round_trips_through_apply_override():
    config = {"x": 0, "y": 0.0, "z": 0.0, "flag": False}
    for path, value in [("x", 200000), ("y", 1e-4), ("z", 1e20), ("flag", True)]:
        apply_override(config, f"{path}={format_value(value)}")
        assert config[path] == value


def test_build_runs_is_the_cartesian_product_of_grid_and_seeds():
    runs = build_runs(
        {"ppo.n_epochs": [5, 10], "ppo.batch_size": [64, 128]}, [0, 1, 2], "demo"
    )
    assert len(runs) == 2 * 2 * 3
    assert [run["index"] for run in runs] == list(range(len(runs)))
    assert len({run["name"] for run in runs}) == len(runs)
    # Seeds vary slowest: one seed's whole grid stays together in the table.
    assert [run["seed"] for run in runs] == [0] * 4 + [1] * 4 + [2] * 4
    assert runs[0]["overrides"] == {"ppo.n_epochs": "5", "ppo.batch_size": "64"}


def test_build_runs_without_a_grid_is_one_run_per_seed():
    runs = build_runs({}, [7], "demo")
    assert runs == [{"index": 0, "name": "demo-000", "seed": 7, "overrides": {}}]
