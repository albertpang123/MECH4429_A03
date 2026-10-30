"""3D replay: the drawing must follow the config, not a hard-coded airframe."""

import copy
import re
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pybullet")

from envs.config import load_config
from envs.interface import nominal_x_positions
from eval.replay import (
    airframe_dimensions,
    frame_camera,
    rotor_positions_body_m,
    urdf_from_config,
    visual_rotor_commands,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(ROOT / "configs" / "g3_hover.yaml")


def _urdf_joint_origins(urdf: str):
    """(x, y, z) of each rotor joint origin, in URDF order."""
    origins = []
    for block in urdf.split("<joint ")[1:]:
        if "rotor" not in block.split("<parent")[0]:
            continue
        match = re.search(r'<origin xyz="([^"]+)"', block)
        origins.append(tuple(float(v) for v in match.group(1).split()))
    return origins


def test_urdf_rotor_origins_match_the_config():
    config = copy.deepcopy(CONFIG)
    positions = config["dynamics"]["rotor_positions_body_m"]
    origins = _urdf_joint_origins(urdf_from_config(config))

    assert len(origins) == 4
    for origin, (x, y) in zip(origins, positions):
        assert origin[0] == pytest.approx(x, abs=1e-6)
        assert origin[1] == pytest.approx(y, abs=1e-6)


def test_airframe_follows_a_changed_rotor_layout():
    config = copy.deepcopy(CONFIG)
    doubled = [[2.0 * x, 2.0 * y] for x, y in config["dynamics"]["rotor_positions_body_m"]]
    config["dynamics"]["rotor_positions_body_m"] = doubled

    base = airframe_dimensions(rotor_positions_body_m(CONFIG))
    scaled = airframe_dimensions(rotor_positions_body_m(config))
    assert scaled["arm_m"] == pytest.approx(2.0 * base["arm_m"])
    assert scaled["rotor_radius"] == pytest.approx(2.0 * base["rotor_radius"])


def test_legacy_arm_length_still_draws():
    """Checkpoints trained before the config refactor stored only arm_length_m."""
    config = copy.deepcopy(CONFIG)
    dynamics = config["dynamics"]
    del dynamics["rotor_positions_body_m"]
    dynamics["arm_length_m"] = 0.092

    positions = rotor_positions_body_m(config)
    assert np.allclose(positions, nominal_x_positions(0.092))


def test_unfilled_dynamics_names_the_owner():
    config = copy.deepcopy(CONFIG)
    config["dynamics"]["rotor_positions_body_m"] = None

    with pytest.raises(ValueError, match="Andy"):
        urdf_from_config(config)


def test_bad_rotor_array_is_rejected():
    with pytest.raises(ValueError, match="4"):
        airframe_dimensions(np.zeros((3, 2)))


def _episode(path: np.ndarray, target=(0.0, 0.0, 1.0), fault_at: int | None = None,
             rotor_id: int = 0) -> dict:
    steps = len(path)
    actions = np.full((max(steps - 1, 0), 4), 0.5)
    active = np.zeros(steps, dtype=bool)
    if fault_at is not None:
        active[fault_at:] = True
    return {
        "target": np.asarray(target, dtype=np.float64),
        "position": np.asarray(path, dtype=np.float64),
        "quaternion": np.tile([0.0, 0.0, 0.0, 1.0], (steps, 1)),
        "action": actions if len(actions) else np.zeros((1, 4)),
        "fault_active": active,
        "error_m": np.zeros(steps),
        "time_s": np.arange(steps) / 60.0,
        "control_dt": 1.0 / 60.0,
        "scenario": {"rotor_id": rotor_id if fault_at is not None else None,
                     "effectiveness": 0.0},
    }


def test_camera_frames_the_whole_trajectory():
    """A failure can take the vehicle a metre away; it must stay on screen."""
    path = np.linspace([0.0, 0.0, 1.0], [1.2, -0.8, 0.1], 200)
    episode = _episode(path)

    centre, distance = frame_camera(episode, yaw=50.0, pitch=-22.0, aspect=800 / 450)
    assert distance >= 0.5

    from eval.replay import _camera_matrices, _ndc

    view, projection = _camera_matrices(centre, distance, 50.0, -22.0, 800, 450)
    points = np.vstack([path, episode["target"][None, :]])
    ndc = _ndc(view, projection, points)
    assert np.all(np.abs(ndc) <= 0.95)
    assert np.all(np.isfinite(ndc))


def test_visual_commands_mirror_the_injector():
    path = np.tile([0.0, 0.0, 1.0], (21, 1))
    episode = _episode(path, fault_at=11)

    commands = visual_rotor_commands(episode)
    assert commands[5, 0] == pytest.approx(0.5)    # before the failure
    assert commands[12, 0] == pytest.approx(0.0)   # failed rotor, effectiveness 0
    assert commands[12, 1] == pytest.approx(0.5)   # the others keep their command
