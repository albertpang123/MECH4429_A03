"""The pybullet backend must reproduce the NumPy stub, then add ground contact.

Two kinds of test here:

* **equivalence** -- with the ground switched off, ``PyBulletDynamics`` and
  ``StubDynamics`` must follow the same trajectory from the same state and
  command. That is the test that makes "we added an engine" safe: if it passes,
  nothing above the dynamics interface can tell the difference.
* **the documented difference** -- with the ground on, the two models *must*
  disagree once the vehicle reaches z = 0, because the stub has no floor.

All timings below are 4 s of flight at physics 240 Hz unless stated.
"""

import copy
from pathlib import Path

import numpy as np
import pytest

pybullet = pytest.importorskip("pybullet")

from envs.config import load_config
from envs.dynamics_factory import backend_name, make_dynamics
from envs.interface import ACTION_DIM, VehicleState
from envs.pybullet_dynamics import PyBulletDynamics
from envs.quadrotor_env import QuadrotorEnv
from envs.stub_dynamics import StubDynamics, StubParams

ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(ROOT / "configs" / "g3_hover.yaml")
PARAMS = StubParams.from_config(CONFIG)
PHYSICS_DT = 1.0 / float(CONFIG["env"]["physics_hz"])
STEPS_PER_CONTROL = int(round(
    float(CONFIG["env"]["physics_hz"]) / float(CONFIG["env"]["control_hz"])
))


def _level_state() -> VehicleState:
    return VehicleState(
        position=np.array([0.0, 0.0, 1.2]),
        velocity=np.zeros(3),
        quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
        angular_velocity=np.zeros(3),
    )


def _compare(command, seconds: float, *, ground: bool):
    """Fly stub and engine side by side; return (final states, max position gap)."""
    stub = StubDynamics(PARAMS)
    engine = PyBulletDynamics(PARAMS, ground=ground)
    try:
        s_stub = stub.reset(_level_state())
        s_eng = engine.reset(_level_state())
        max_gap = 0.0
        for _ in range(int(round(seconds * float(CONFIG["env"]["control_hz"])))):
            for _ in range(STEPS_PER_CONTROL):
                s_stub = stub.step(s_stub, command, PHYSICS_DT)
                s_eng = engine.step(s_eng, command, PHYSICS_DT)
            max_gap = max(max_gap, float(np.linalg.norm(s_stub.position - s_eng.position)))
        return s_stub, s_eng, max_gap
    finally:
        engine.close()


def _hover_command() -> np.ndarray:
    return StubDynamics(PARAMS).hover_command.copy()


# ---------------------------------------------------------------------------
# Factory / config switch
# ---------------------------------------------------------------------------
def test_default_backend_is_numpy():
    """Configs written before the backend existed keep working untouched."""
    config = copy.deepcopy(CONFIG)
    config["dynamics"].pop("backend", None)
    assert backend_name(config) == "numpy"
    assert isinstance(make_dynamics(config), StubDynamics)


def test_explicit_numpy_backend():
    config = copy.deepcopy(CONFIG)
    config["dynamics"]["backend"] = "numpy"
    assert isinstance(make_dynamics(config), StubDynamics)


def test_pybullet_backend_selected_by_config():
    config = copy.deepcopy(CONFIG)
    config["dynamics"]["backend"] = "pybullet"
    dynamics = make_dynamics(config, ground=False)
    try:
        assert isinstance(dynamics, PyBulletDynamics)
    finally:
        dynamics.close()


def test_unknown_backend_is_rejected():
    config = copy.deepcopy(CONFIG)
    config["dynamics"]["backend"] = "mujoco"
    with pytest.raises(ValueError, match="dynamics.backend"):
        backend_name(config)
    with pytest.raises(ValueError, match="dynamics.backend"):
        make_dynamics(config)


# ---------------------------------------------------------------------------
# Interface compatibility
# ---------------------------------------------------------------------------
def test_engine_agrees_on_command_mapping():
    """Thrust curve, hover command and clipping are shared, not re-implemented."""
    engine = PyBulletDynamics(PARAMS, ground=False)
    stub = StubDynamics(PARAMS)
    try:
        assert engine.hover_command.shape == (ACTION_DIM,)
        np.testing.assert_allclose(engine.hover_command, stub.hover_command, atol=0.0)
        u = np.linspace(-0.2, 1.2, ACTION_DIM)   # deliberately out of range
        np.testing.assert_allclose(
            engine.applied_thrusts(u), stub.applied_thrusts(u), atol=0.0
        )
    finally:
        engine.close()


def test_engine_state_types_match_interface():
    engine = PyBulletDynamics(PARAMS, ground=False)
    try:
        state = engine.step(_level_state(), _hover_command(), PHYSICS_DT)
        assert isinstance(state, VehicleState)
        assert state.position.shape == (3,)
        assert state.velocity.shape == (3,)
        assert state.quaternion.shape == (4,)
        assert state.angular_velocity.shape == (3,)
        assert np.isclose(np.linalg.norm(state.quaternion), 1.0)
    finally:
        engine.close()


# ---------------------------------------------------------------------------
# Numerical equivalence with the stub (ground off)
# ---------------------------------------------------------------------------
def test_hover_is_identical_to_stub():
    s_stub, s_eng, _ = _compare(_hover_command(), 4.0, ground=False)
    np.testing.assert_allclose(s_eng.position, s_stub.position, atol=1e-9)
    np.testing.assert_allclose(s_eng.velocity, s_stub.velocity, atol=1e-9)
    np.testing.assert_allclose(s_eng.angular_velocity, s_stub.angular_velocity, atol=1e-9)


@pytest.mark.parametrize(
    "differential, name",
    [
        (np.array([0.01, -0.01, 0.01, -0.01]), "yaw"),
        (np.array([0.01, 0.01, -0.01, -0.01]), "pitch"),
        (np.array([0.01, 0.0, 0.0, 0.01]), "roll_plus_thrust"),
    ],
)
def test_differential_thrust_matches_stub(differential, name):
    """Sustained body torque: the integrators must not drift apart.

    This is the case that exposed the damping-semantics bug, so it stays as a
    regression test. Tolerance is 1 mm over 4 s even though the vehicle travels
    tens of metres and tumbles past 100 deg.
    """
    command = _hover_command() + differential
    s_stub, s_eng, max_gap = _compare(command, 4.0, ground=False)
    assert max_gap < 1e-3, f"{name}: diverged by {max_gap:.3e} m"
    np.testing.assert_allclose(
        s_eng.angular_velocity, s_stub.angular_velocity, atol=1e-6,
        err_msg=f"{name}: angular velocity diverged",
    )


def test_quaternion_is_the_same_rotation():
    """q and -q are the same attitude; compare |q_stub . q_engine|."""
    command = _hover_command() + np.array([0.01, 0.01, -0.01, -0.01])
    s_stub, s_eng, _ = _compare(command, 2.0, ground=False)
    assert abs(float(np.dot(s_stub.quaternion, s_eng.quaternion))) > 1.0 - 1e-9


# ---------------------------------------------------------------------------
# The documented difference: ground contact
# ---------------------------------------------------------------------------
def test_ground_contact_is_the_only_intended_difference():
    """Pitch hard enough to sink; the stub falls through z = 0, the engine does not."""
    command = _hover_command() + np.array([0.01, 0.01, -0.01, -0.01])
    s_stub, s_eng, _ = _compare(command, 6.0, ground=False)
    assert s_stub.position[2] < -1.0     # stub keeps falling, no floor

    with_ground = PyBulletDynamics(PARAMS, ground=True)
    try:
        state = with_ground.reset(_level_state())
        for _ in range(int(round(6.0 * float(CONFIG["env"]["control_hz"])))):
            for _ in range(STEPS_PER_CONTROL):
                state = with_ground.step(state, command, PHYSICS_DT)
        assert state.position[2] > 0.0, "engine let the body through the plane"
        assert state.position[2] < 0.5, "body did not come down"
    finally:
        with_ground.close()


# ---------------------------------------------------------------------------
# The environment does not notice which backend it got
# ---------------------------------------------------------------------------
def test_env_runs_unchanged_on_pybullet_backend():
    config = copy.deepcopy(CONFIG)
    config["dynamics"]["backend"] = "pybullet"
    config["faults"]["mode"] = "nominal"
    env = QuadrotorEnv(config)
    try:
        obs, info = env.reset(seed=0)
        assert obs.shape == env.observation_space.shape
        assert "fault_active" in info
        command = env.dynamics.hover_command
        for _ in range(20):
            obs, reward, terminated, truncated, info = env.step(command)
            assert obs.shape == env.observation_space.shape
            assert np.all(np.isfinite(obs))
            assert np.isfinite(reward)
            if terminated or truncated:
                break
        assert not terminated and not truncated
    finally:
        env.close()


def test_pybullet_backend_is_deterministic_for_a_fixed_seed():
    """Same seed + same actions must replay identically, or results cannot be reported."""
    config = copy.deepcopy(CONFIG)
    config["dynamics"]["backend"] = "pybullet"

    def rollout():
        env = QuadrotorEnv(config)
        try:
            env.reset(seed=7)
            command = env.dynamics.hover_command
            trace = []
            for _ in range(30):
                _, _, terminated, truncated, _ = env.step(command)
                trace.append(env._state.position.copy())
                if terminated or truncated:
                    break
            return np.array(trace)
        finally:
            env.close()

    np.testing.assert_array_equal(rollout(), rollout())


def test_env_close_is_safe_twice_on_either_backend():
    for backend in ("numpy", "pybullet"):
        config = copy.deepcopy(CONFIG)
        config["dynamics"]["backend"] = backend
        env = QuadrotorEnv(config)
        env.reset(seed=0)
        env.close()
        env.close()
