"""Checks for the G3 stub: trim, fault masking, env API, determinism."""

import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

pytest.importorskip("gymnasium")

from envs.config import load_config
from envs.interface import ACTION_DIM, Scenario, VehicleState
from envs.quadrotor_env import QuadrotorEnv
from envs.stub_dynamics import StubDynamics, StubParams
from envs.stub_faults import StubFaultInjector

ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(ROOT / "configs" / "g3_hover.yaml")
BASE_CONFIG = load_config(ROOT / "configs" / "base.yaml")
CONTROL_HZ = float(CONFIG["env"]["control_hz"])


def _trim_config():
    config = copy.deepcopy(CONFIG)
    config["reset"] = {
        "position_jitter_m": 0.0,
        "velocity_jitter_mps": 0.0,
        "tilt_jitter_deg": 0.0,
        "yaw_jitter_deg": 0.0,
    }
    config["faults"]["mode"] = "nominal"
    return config


def _level_state(position=(0.0, 0.0, 1.0)) -> VehicleState:
    return VehicleState(
        position=np.asarray(position, dtype=float),
        velocity=np.zeros(3),
        quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
        angular_velocity=np.zeros(3),
    )


def _run_hover(env: QuadrotorEnv, steps: int):
    """Step with the trim command for at most ``steps`` steps."""
    command = env.dynamics.hover_command
    rewards = []
    terminated = truncated = False
    info = {}
    for _ in range(steps):
        _, reward, terminated, truncated, info = env.step(command)
        rewards.append(reward)
        if terminated or truncated:
            break
    return rewards, terminated, truncated, info


# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------
def test_stub_parameters_match_config_geometry():
    params = StubParams.from_config(CONFIG)
    assert np.allclose(
        np.asarray(params.rotor_positions_body_m),
        np.asarray(CONFIG["dynamics"]["rotor_positions_body_m"]),
    )
    assert np.isclose(params.arm_length_m, 0.092, atol=1e-6)
    assert params.mass_kg > 0.0
    assert params.thrust_max_per_rotor_n > 0.0
    StubDynamics(params)


def test_rotor_geometry_is_driven_by_config():
    # Changing the airframe is a config edit, not a source edit: nothing in the
    # code holds an arm length of its own.
    config = copy.deepcopy(CONFIG)
    config["dynamics"]["rotor_positions_body_m"] = [
        [2.0 * x, 2.0 * y] for x, y in CONFIG["dynamics"]["rotor_positions_body_m"]
    ]
    params = StubParams.from_config(config)
    dynamics = StubDynamics(params)
    assert np.isclose(params.arm_length_m, 0.184, atol=1e-6)
    assert np.allclose(
        dynamics._positions,
        np.asarray(config["dynamics"]["rotor_positions_body_m"]),
    )


def test_unfilled_dynamics_names_the_owner():
    with pytest.raises(ValueError, match="Andy"):
        StubParams.from_config(BASE_CONFIG)


def test_hover_trim_is_stationary():
    params = StubParams.from_config(CONFIG)
    dynamics = StubDynamics(params)
    state = _level_state()
    command = dynamics.hover_command
    for _ in range(int(1.0 * CONFIG["env"]["physics_hz"])):
        state = dynamics.step(state, command, 1.0 / CONFIG["env"]["physics_hz"])

    assert np.allclose(state.position, [0.0, 0.0, 1.0], atol=1e-9)
    assert np.allclose(state.velocity, 0.0, atol=1e-9)
    assert np.allclose(state.angular_velocity, 0.0, atol=1e-9)
    assert abs(abs(state.quaternion[3]) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Fault injection
# ---------------------------------------------------------------------------
def test_fault_injector_masks_only_the_failed_rotor():
    scenario = Scenario(rotor_id=2, failure_time_s=1.0, effectiveness=0.0)
    injector = StubFaultInjector(scenario)
    command = np.array([0.6, 0.6, 0.6, 0.6])

    assert not injector.is_active(0.5)
    assert np.allclose(injector.apply(command, 0.5), command)

    applied = injector.apply(command, 1.0)
    assert injector.is_active(1.0)
    assert np.isclose(applied[2], 0.0)
    assert np.allclose(np.delete(applied, 2), 0.6)

    assert not injector.is_active(1.0 - 1.0 / CONTROL_HZ)


def test_fault_injector_supports_partial_effectiveness():
    scenario = Scenario(rotor_id=0, failure_time_s=0.0, effectiveness=0.25)
    injector = StubFaultInjector(scenario)
    applied = injector.apply([0.8, 0.8, 0.8, 0.8], 0.0)
    assert np.isclose(applied[0], 0.2)
    assert np.allclose(applied[1:], 0.8)


# ---------------------------------------------------------------------------
# Environment API
# ---------------------------------------------------------------------------
def test_spaces_and_reset():
    env = QuadrotorEnv(_trim_config())
    obs, info = env.reset(seed=0)
    assert obs.shape == (170,)
    assert obs.dtype == np.float32
    assert env.obs_dim == 170
    assert env.action_space.shape == (ACTION_DIM,)
    assert env.observation_space.shape == (170,)
    assert info["seed"] == 0
    assert info["scenario"]["rotor_id"] is None


def test_control_rate_is_taken_from_config():
    # Regression: the fault module used to import a module-level CONTROL_HZ, so
    # editing env.control_hz in the config silently left fault timing at 60 Hz.
    config = _trim_config()
    config["env"]["physics_hz"] = 240.0
    config["env"]["control_hz"] = 120.0
    config["faults"]["mode"] = "fixed"
    config["faults"]["fixed"] = {
        "rotor_id": 1,
        "failure_time_s": 2.0,
        "effectiveness": 0.0,
    }
    env = QuadrotorEnv(config)
    assert env.physics_steps_per_control == 2
    assert env.max_steps == 1200

    _, info = env.reset(seed=0)
    assert info["scenario"]["failure_time_s"] == 2.0
    # 2.0 s at 120 Hz is step 240, not step 120.
    assert env._injector.scenario.failure_step(env.control_hz) == 240
    assert not env._injector.is_active(2.0 - 1.0 / env.control_hz)
    assert env._injector.is_active(2.0)


def test_reset_is_deterministic_for_the_same_seed():
    config = _trim_config()
    config["reset"]["position_jitter_m"] = 0.1
    env_a = QuadrotorEnv(config)
    env_b = QuadrotorEnv(config)
    env_c = QuadrotorEnv(config)

    obs_a, info_a = env_a.reset(seed=123)
    obs_b, info_b = env_b.reset(seed=123)
    obs_c, info_c = env_c.reset(seed=124)

    assert np.array_equal(obs_a, obs_b)
    assert info_a["scenario"] == info_b["scenario"]
    assert not np.array_equal(obs_a, obs_c)


def test_action_is_clipped_and_reported():
    env = QuadrotorEnv(_trim_config())
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.array([-1.0, 2.0, 0.5, 0.5]))
    assert info["clip_count"] == 2


def test_unperturbed_hover_reaches_time_limit():
    env = QuadrotorEnv(_trim_config())
    env.reset(seed=0)
    rewards, terminated, truncated, info = _run_hover(env, env.max_steps + 5)

    assert not terminated
    assert truncated
    assert info["done_reason"] == "time_limit"
    assert len(rewards) == env.max_steps
    assert np.isclose(np.sum(rewards), env.max_steps, atol=1e-6)
    assert np.linalg.norm(env.state.position - env.target_position) < 1e-9


def test_zero_thrust_crashes_and_gets_the_terminal_penalty():
    env = QuadrotorEnv(_trim_config())
    env.reset(seed=0)
    total_reward = 0.0
    terminated = truncated = False
    info = {}
    for _ in range(env.max_steps):
        _, reward, terminated, truncated, info = env.step(np.zeros(ACTION_DIM))
        total_reward += reward
        if terminated or truncated:
            break

    assert terminated
    assert not truncated
    assert info["done_reason"] == "crash"
    assert total_reward < 0.0


def test_fixed_fault_scenario_is_sampled_from_config():
    config = _trim_config()
    config["faults"]["mode"] = "fixed"
    config["faults"]["fixed"] = {
        "rotor_id": 2,
        "failure_time_s": 4.0,
        "effectiveness": 0.0,
    }
    env = QuadrotorEnv(config)
    _, info = env.reset(seed=7)
    scenario = info["scenario"]
    assert scenario["rotor_id"] == 2
    assert scenario["failure_time_s"] == 4.0
    assert scenario["effectiveness"] == 0.0
