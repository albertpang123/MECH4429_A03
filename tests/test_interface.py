"""Gate 0 checks: the interface code and configs/base.yaml must agree.

These tests are deliberately dependency-light (numpy + pyyaml + pytest) so they
run before the simulator is installed.
"""

import inspect
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest
import yaml

from envs.interface import (
    ACTION_DIM,
    ACTION_HIGH,
    ACTION_LOW,
    CONTROL_HZ,
    HIDDEN_FROM_POLICY,
    OBS_DIM,
    OBS_FIELD_SCALES,
    OBS_FRAME_DIM,
    OBS_FRAME_FIELDS,
    OBS_HISTORY_LEN,
    PHYSICS_HZ,
    PHYSICS_STEPS_PER_CONTROL,
    ROTOR_ARM_M,
    ROTORS,
    SPIN_CCW,
    SPIN_CW,
    ObservationBuffer,
    Scenario,
    VehicleState,
    canonicalize_quaternion,
    observation_frame,
    observation_slices,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "configs" / "base.yaml").read_text(encoding="utf-8"))


def _state(position=(0.0, 0.0, 1.0), quaternion=(0.0, 0.0, 0.0, 1.0)) -> VehicleState:
    return VehicleState(
        position=np.array(position, dtype=float),
        velocity=np.zeros(3),
        quaternion=np.array(quaternion, dtype=float),
        angular_velocity=np.zeros(3),
    )


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
def test_rates_match_config_and_divide_evenly():
    assert CONFIG["env"]["physics_hz"] == PHYSICS_HZ
    assert CONFIG["env"]["control_hz"] == CONTROL_HZ
    assert PHYSICS_STEPS_PER_CONTROL == 4
    assert np.isclose(PHYSICS_HZ / CONTROL_HZ, PHYSICS_STEPS_PER_CONTROL)


# ---------------------------------------------------------------------------
# Rotor layout
# ---------------------------------------------------------------------------
def test_rotor_layout_is_a_symmetric_x_config():
    assert [rotor.index for rotor in ROTORS] == [0, 1, 2, 3]
    positions = np.array([[rotor.x_m, rotor.y_m] for rotor in ROTORS])
    assert np.allclose(positions.sum(axis=0), 0.0)

    expected_offset = ROTOR_ARM_M / np.sqrt(2.0)
    assert np.allclose(np.abs(positions), expected_offset)

    spins = [rotor.spin for rotor in ROTORS]
    assert spins.count(SPIN_CW) == 2
    assert spins.count(SPIN_CCW) == 2
    # Diagonal pairs share the spin direction.
    assert ROTORS[0].spin == ROTORS[2].spin == SPIN_CW
    assert ROTORS[1].spin == ROTORS[3].spin == SPIN_CCW


def test_rotor_labels_match_documented_order():
    assert [rotor.label for rotor in ROTORS] == [
        "front-right",
        "front-left",
        "rear-left",
        "rear-right",
    ]


# ---------------------------------------------------------------------------
# Observation and action layout
# ---------------------------------------------------------------------------
def test_observation_layout_matches_config():
    assert CONFIG["observation"]["history_len"] == OBS_HISTORY_LEN
    assert OBS_FRAME_DIM == 17
    assert OBS_DIM == 170

    for name, dim in OBS_FRAME_FIELDS:
        assert CONFIG["observation"]["scales"][name] == OBS_FIELD_SCALES[name]
        assert dim == observation_slices()[name].stop - observation_slices()[name].start


def test_action_space_matches_config():
    assert CONFIG["action"]["dim"] == ACTION_DIM == 4
    assert CONFIG["action"]["low"] == ACTION_LOW
    assert CONFIG["action"]["high"] == ACTION_HIGH


def test_quaternion_canonicalisation():
    q = canonicalize_quaternion([0.0, 0.0, 0.0, -2.0])
    assert np.isclose(np.linalg.norm(q), 1.0)
    assert q[3] >= 0.0
    assert np.allclose(q, [0.0, 0.0, 0.0, 1.0])


def test_observation_frame_has_documented_fields():
    state = VehicleState(
        position=np.array([0.1, -0.2, 0.8]),
        velocity=np.array([1.0, 2.0, 3.0]),
        quaternion=np.array([0.0, 0.0, 0.0, -1.0]),
        angular_velocity=np.array([0.1, 0.2, 0.3]),
    )
    target = np.array([0.0, 0.0, 1.0])
    command = np.array([0.1, 0.2, 0.3, 0.4])

    frame = observation_frame(state, target, command)
    assert frame.shape == (OBS_FRAME_DIM,)

    s = observation_slices()
    assert np.allclose(frame[s["position_error"]], target - state.position)
    assert np.allclose(frame[s["linear_velocity"]], state.velocity)
    assert np.allclose(frame[s["attitude_quaternion"]], [0.0, 0.0, 0.0, 1.0])
    assert np.allclose(frame[s["angular_velocity"]], state.angular_velocity)
    assert np.allclose(frame[s["last_motor_command"]], command)


def test_observation_rejects_out_of_bounds_command():
    with pytest.raises(ValueError):
        observation_frame(_state(), (0.0, 0.0, 1.0), [0.0, 0.0, 0.0, 1.5])


# ---------------------------------------------------------------------------
# History buffer
# ---------------------------------------------------------------------------
def test_buffer_history_shape_and_order():
    target = np.array([0.0, 0.0, 1.0])
    buf = ObservationBuffer(target, history_len=3, normalize=False)

    state_a = _state(position=(0.0, 0.0, 1.0))
    state_b = _state(position=(0.1, 0.0, 1.0))
    state_c = _state(position=(0.2, 0.0, 1.0))

    buf.reset(state_a, [0.1, 0.1, 0.1, 0.1])
    buf.push(state_b, [0.2, 0.2, 0.2, 0.2])
    out = buf.push(state_c, [0.3, 0.3, 0.3, 0.3])

    assert out.shape == (3 * OBS_FRAME_DIM,)
    assert np.allclose(out[-OBS_FRAME_DIM:], observation_frame(state_c, target, [0.3] * 4))
    assert np.allclose(out[:OBS_FRAME_DIM], observation_frame(state_a, target, [0.1] * 4))


def test_buffer_default_shape_dtype_and_clipping():
    buf = ObservationBuffer((0.0, 0.0, 1.0))
    state = _state(position=(1e6, 0.0, 1.0))
    out = buf.reset(state)
    assert out.shape == (OBS_DIM,)
    assert out.dtype == np.float32
    # position error = target - position = (-1e6, 0, 0) m; / 1 m scale -> clipped to -10
    assert out[0] == np.float32(-10.0)
    assert np.all(out[1:3] == np.float32(0.0))


def test_buffer_requires_reset_before_push():
    buf = ObservationBuffer((0.0, 0.0, 1.0))
    with pytest.raises(RuntimeError):
        buf.push(_state(), [0.0] * 4)


# ---------------------------------------------------------------------------
# Hidden-information boundary
# ---------------------------------------------------------------------------
def test_scenario_hidden_fields_are_declared():
    scenario_fields = {field.name for field in fields(Scenario)}
    assert set(HIDDEN_FROM_POLICY) <= scenario_fields


def test_observation_api_has_no_fault_inputs():
    functions = (
        observation_frame,
        ObservationBuffer.__init__,
        ObservationBuffer.reset,
        ObservationBuffer.push,
    )
    forbidden = set(HIDDEN_FROM_POLICY) | {"scenario", "fault", "fault_flag"}
    for function in functions:
        parameters = set(inspect.signature(function).parameters)
        assert not (parameters & forbidden), (function, parameters & forbidden)


# ---------------------------------------------------------------------------
# Scenario
# ---------------------------------------------------------------------------
def test_scenario_validates_hidden_fields():
    with pytest.raises(ValueError):
        Scenario(rotor_id=4, failure_time_s=1.0)
    with pytest.raises(ValueError):
        Scenario(rotor_id=0)  # failure time missing
    with pytest.raises(ValueError):
        Scenario(rotor_id=0, failure_time_s=-0.1)
    with pytest.raises(ValueError):
        Scenario(rotor_id=0, failure_time_s=1.0, effectiveness=1.5)


def test_scenario_failure_step_is_quantised_to_control_grid():
    scenario = Scenario(rotor_id=2, failure_time_s=4.0, effectiveness=0.0, seed=7)
    assert scenario.failure_step == 240
    assert scenario.failure_step == int(4.0 * CONTROL_HZ)
    assert Scenario(rotor_id=1, failure_time_s=4.008).failure_step == 240
    assert scenario.is_nominal is False
    assert Scenario().is_nominal is True


# ---------------------------------------------------------------------------
# Config sanity
# ---------------------------------------------------------------------------
def test_config_is_marked_proposed_and_versioned():
    assert CONFIG["schema_version"] == 1
    assert CONFIG["status"] == "proposed"
    assert CONFIG["seeds"]["train"][1] < CONFIG["seeds"]["eval"][0]
