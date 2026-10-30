"""Gate 0 checks: the interface code and the configs must agree.

These tests are deliberately dependency-light (numpy + pyyaml + pytest) so they
run before the simulator is installed.
"""

import inspect
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest
import yaml

from envs.config import (
    config_hash,
    load_config,
    unresolved_paths,
    validate_structure,
)
from envs.interface import (
    ACTION_DIM,
    ACTION_HIGH,
    ACTION_LOW,
    HIDDEN_FROM_POLICY,
    N_ROTORS,
    OBS_FRAME_DIM,
    OBS_FRAME_FIELDS,
    ROTOR_LAYOUT,
    SPIN_CCW,
    SPIN_CW,
    ObservationBuffer,
    Scenario,
    VehicleState,
    build_rotors,
    canonicalize_quaternion,
    frame_scales,
    nominal_x_positions,
    observation_dim,
    observation_frame,
    observation_slices,
    quantise_to_control_grid,
)

ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "configs" / "base.yaml"
G3_PATH = ROOT / "configs" / "g3_hover.yaml"
CONFIG = load_config(BASE_PATH)
G3 = load_config(G3_PATH)
# Raw text, for checks about what the file itself says (status, extends).
BASE_RAW = yaml.safe_load(BASE_PATH.read_text(encoding="utf-8"))
G3_RAW = yaml.safe_load(G3_PATH.read_text(encoding="utf-8"))


def _state(position=(0.0, 0.0, 1.0), quaternion=(0.0, 0.0, 0.0, 1.0)) -> VehicleState:
    return VehicleState(
        position=np.array(position, dtype=float),
        velocity=np.zeros(3),
        quaternion=np.array(quaternion, dtype=float),
        angular_velocity=np.zeros(3),
    )


# ---------------------------------------------------------------------------
# Timing and config mechanics
# ---------------------------------------------------------------------------
def test_rates_are_integral_and_consistent():
    physics_hz = float(CONFIG["env"]["physics_hz"])
    control_hz = float(CONFIG["env"]["control_hz"])
    assert np.isclose(physics_hz / control_hz, 4)
    assert float(G3["env"]["physics_hz"]) == physics_hz


def test_config_inheritance_resolves_parent_then_child():
    assert G3_RAW["extends"] == "base.yaml"
    # Inherited from base.yaml: the child file does not mention these at all.
    assert "reward" not in G3_RAW
    assert G3["reward"] == CONFIG["reward"]
    assert G3["observation"] == CONFIG["observation"]
    assert G3["env"]["control_hz"] == CONFIG["env"]["control_hz"]
    # Overridden by the child.
    assert G3_RAW["run"]["name"] == "g3_stub_hover"
    assert G3["run"]["name"] == "g3_stub_hover"
    assert G3["status"] == "g3-smoke"
    assert G3["_provenance"]["extends"] == [str(BASE_PATH)]


def test_config_hash_tracks_values_not_provenance():
    first = load_config(G3_PATH)
    second = load_config(G3_PATH)
    assert config_hash(first) == config_hash(second)

    changed = load_config(G3_PATH)
    changed["reward"]["position_error"] = 5.0
    assert config_hash(changed) != config_hash(first)


def test_unresolved_values_are_reported_for_the_base_config():
    # base.yaml is the interface skeleton: vehicle parameters stay null until
    # Andy provides them, and this is how the tooling reports what is missing.
    missing = unresolved_paths(CONFIG)
    assert "dynamics.mass_kg" in missing
    assert "dynamics.rotor_positions_body_m" in missing
    # A run config that fills them in has none of those left.
    g3_missing = unresolved_paths(G3)
    assert "dynamics.mass_kg" not in g3_missing


def test_running_refuses_a_config_with_unfilled_vehicle_parameters():
    validate_structure(CONFIG)  # structure alone is fine
    with pytest.raises(ValueError, match="dynamics"):
        validate_structure(CONFIG, expect_resolved=True)
    validate_structure(G3, expect_resolved=True)  # the stub run is complete


# ---------------------------------------------------------------------------
# Rotor layout
# ---------------------------------------------------------------------------
def test_rotor_layout_is_structural():
    assert N_ROTORS == 4
    assert [label for label, _ in ROTOR_LAYOUT] == [
        "front-right",
        "front-left",
        "rear-left",
        "rear-right",
    ]
    spins = [spin for _, spin in ROTOR_LAYOUT]
    assert spins.count(SPIN_CW) == 2
    assert spins.count(SPIN_CCW) == 2
    # Diagonal pairs share the spin direction.
    assert spins[0] == spins[2] == SPIN_CW
    assert spins[1] == spins[3] == SPIN_CCW


def test_stub_config_rotors_form_a_symmetric_x():
    positions = np.asarray(G3["dynamics"]["rotor_positions_body_m"], dtype=float)
    rotors = build_rotors(positions)

    assert [rotor.index for rotor in rotors] == [0, 1, 2, 3]
    assert np.allclose(positions.sum(axis=0), 0.0)
    assert np.allclose(np.abs(positions), abs(positions[0, 0]))
    assert np.allclose(positions, nominal_x_positions(0.092), atol=1e-7)


def test_build_rotors_rejects_the_wrong_shape():
    with pytest.raises(ValueError):
        build_rotors([[0.0, 0.0, 0.0]] * N_ROTORS)
    with pytest.raises(ValueError):
        build_rotors([[0.0, 0.0]] * (N_ROTORS - 1))


# ---------------------------------------------------------------------------
# Observation and action layout
# ---------------------------------------------------------------------------
def test_observation_layout_matches_config():
    assert OBS_FRAME_DIM == 17
    assert observation_dim(int(CONFIG["observation"]["history_len"])) == 170

    scales = frame_scales(CONFIG["observation"]["scales"])
    assert scales.shape == (OBS_FRAME_DIM,)
    for name, dim in OBS_FRAME_FIELDS:
        field = scales[observation_slices()[name]]
        assert np.all(field == float(CONFIG["observation"]["scales"][name]))
        assert dim == observation_slices()[name].stop - observation_slices()[name].start


def test_observation_dim_scales_with_history_len():
    assert observation_dim(1) == OBS_FRAME_DIM
    assert observation_dim(3) == 3 * OBS_FRAME_DIM


def test_frame_scales_rejects_an_incomplete_scale_map():
    incomplete = {"position_error": 1.0}
    with pytest.raises(KeyError):
        frame_scales(incomplete)
    with pytest.raises(ValueError):
        frame_scales({name: 0.0 for name, _ in OBS_FRAME_FIELDS})


def test_action_bounds_are_structural_not_config():
    # [0, 1] normalised motor commands define the interface; there is no
    # `action:` section in the config that could disagree with them.
    assert ACTION_DIM == N_ROTORS == 4
    assert ACTION_LOW == 0.0
    assert ACTION_HIGH == 1.0
    assert "action" not in CONFIG


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
    assert buf.dim == 3 * OBS_FRAME_DIM
    assert np.allclose(out[-OBS_FRAME_DIM:], observation_frame(state_c, target, [0.3] * 4))
    assert np.allclose(out[:OBS_FRAME_DIM], observation_frame(state_a, target, [0.1] * 4))


def test_buffer_shape_dtype_and_clipping():
    buf = ObservationBuffer((0.0, 0.0, 1.0), history_len=10)
    state = _state(position=(1e6, 0.0, 1.0))
    out = buf.reset(state)
    assert out.shape == (170,)
    assert out.dtype == np.float32
    # position error = target - position = (-1e6, 0, 0) m; / 1 m scale -> clipped to -10
    assert out[0] == np.float32(-10.0)
    assert np.all(out[1:3] == np.float32(0.0))


def test_buffer_from_config_uses_config_scales():
    buf = ObservationBuffer.from_config(G3)
    assert buf.history_len == int(G3["observation"]["history_len"])
    assert buf.dim == observation_dim(buf.history_len)
    assert np.allclose(
        buf.scales, frame_scales(G3["observation"]["scales"])
    )
    assert np.allclose(buf.target_position, G3["env"]["target_position"])


def test_buffer_requires_reset_before_push():
    buf = ObservationBuffer((0.0, 0.0, 1.0), history_len=3)
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


def test_scenario_failure_step_uses_the_rate_it_is_given():
    control_hz = float(CONFIG["env"]["control_hz"])
    scenario = Scenario(rotor_id=2, failure_time_s=4.0, effectiveness=0.0, seed=7)
    assert scenario.failure_step(control_hz) == 240
    assert scenario.failure_step(control_hz) == int(4.0 * control_hz)
    assert Scenario(rotor_id=1, failure_time_s=4.008).failure_step(control_hz) == 240
    # A faster control rate reaches the same wall-clock failure sooner in steps.
    assert Scenario(rotor_id=1, failure_time_s=4.0).failure_step(120.0) == 480
    assert scenario.is_nominal is False
    assert Scenario().is_nominal is True


def test_failure_time_is_quantised_to_the_control_grid():
    control_hz = float(CONFIG["env"]["control_hz"])
    assert np.isclose(quantise_to_control_grid(4.008, control_hz), 4.0)
    assert np.isclose(quantise_to_control_grid(4.0, control_hz), 4.0)
    snapped = quantise_to_control_grid(4.02, control_hz)
    assert snapped == pytest.approx(round(snapped * control_hz) / control_hz)


# ---------------------------------------------------------------------------
# Config sanity
# ---------------------------------------------------------------------------
def test_config_is_marked_proposed_and_versioned():
    assert BASE_RAW["schema_version"] == 1
    assert BASE_RAW["status"] == "proposed"
    assert CONFIG["seeds"]["train"][1] < CONFIG["seeds"]["eval"][0]
