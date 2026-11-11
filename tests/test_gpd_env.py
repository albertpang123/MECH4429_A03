"""The task on gym-pybullet-drones' BaseAviary (envs/gpd_env.py).

The point of these tests is not "the physics works" -- that is gpd's job and its
own suite. It is that *our* layer on top of it is wired the right way round:
rotor indices, fault timing, observation contract.

``test_rotor_indices_match_geometry`` is the one that matters. gpd's URDF numbers
its four rotor links FR, RR, RL, FL; ours are FR, FL, RL, RR. Every command has
to be permuted on the way in. Get it wrong and hover still looks perfect (all
four motors equal), while "rotor 1 failed" silently fails rotor 3 instead. That
is the failure mode this file exists to prevent.
"""

import copy

import numpy as np
import pytest

pytest.importorskip("pybullet")
pytest.importorskip("gymnasium")

from envs.config import load_config  # noqa: E402
from envs.gpd_env import GPDQuadrotorEnv, ROTOR_GPD_INDEX  # noqa: E402

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
CONFIG = load_config(ROOT / "configs" / "g3_hover.yaml")


def _with_fault(rotor_id: int, failure_time_s: float) -> dict:
    config = copy.deepcopy(CONFIG)
    config["env"]["impl"] = "gpd"
    config["faults"] = {
        "enabled": True,
        "mode": "fixed",
        "fixed": {
            "rotor_id": rotor_id,
            "failure_time_s": failure_time_s,
            "effectiveness": 0.0,
        },
        "random": CONFIG["faults"]["random"],
    }
    return config


def _tilt_rate_signs(env, steps: int = 2):
    """Sign of (roll rate, pitch rate) right after the fault takes effect.

    Body frame FLU: +wx rolls the right side down, +wy pitches the nose down.
    Signs, not magnitudes: after a few tenths of a second the airframe has
    tumbled past 90 degrees and the Euler angles stop being interpretable.
    """
    for _ in range(steps):
        env.step(np.full(4, env.HOVER_ACTION))
    wx, wy, _ = env.state.angular_velocity
    return int(np.sign(wx)), int(np.sign(wy))


#: Rotor position in body FLU -> the torque a thrust *loss* there produces.
#: tau = (-F*y, F*x) for a lost force F at (x, y), so the sign pair is
#: (-sign(y), sign(x)).
EXPECTED = {
    0: (+1, +1),  # R0 front-right (x>0, y<0)
    1: (-1, +1),  # R1 front-left
    2: (-1, -1),  # R2 rear-left
    3: (+1, -1),  # R3 rear-right
}


def test_rotor_mapping_is_a_permutation():
    assert sorted(ROTOR_GPD_INDEX) == [0, 1, 2, 3]
    # A swap of 1 and 3 is its own inverse; assert it rather than assume it,
    # because `_preprocessAction` scatters with the forward map only.
    inverse = [ROTOR_GPD_INDEX.index(i) for i in range(4)]
    assert inverse == list(ROTOR_GPD_INDEX)


@pytest.mark.parametrize("rotor_id", [0, 1, 2, 3])
def test_rotor_indices_match_geometry(rotor_id):
    env = GPDQuadrotorEnv(_with_fault(rotor_id, failure_time_s=0.0))
    try:
        env.reset(seed=0)
        assert _tilt_rate_signs(env) == EXPECTED[rotor_id]
    finally:
        env.close()


def test_hover_holds_position_and_never_terminates():
    env = GPDQuadrotorEnv(copy.deepcopy(CONFIG))
    try:
        obs, info = env.reset(seed=0)
        z0 = float(env.state.position[2])
        assert obs.shape == (env.obs_dim,)
        for _ in range(60):                     # 1 s
            obs, reward, terminated, truncated, info = env.step(
                np.full(4, env.HOVER_ACTION))
            assert not terminated and not truncated
        # Open loop at exactly the hover command: it must not sink or climb.
        assert abs(float(env.state.position[2]) - z0) < 0.05
        assert info["fault_active"] is False
    finally:
        env.close()


def test_fault_activates_at_the_configured_step():
    env = GPDQuadrotorEnv(_with_fault(rotor_id=1, failure_time_s=2.0))
    try:
        env.reset(seed=0)
        flags = []
        for _ in range(int(2.5 * env.control_hz)):
            _, _, terminated, _, info = env.step(np.full(4, env.HOVER_ACTION))
            flags.append(info["fault_active"])
            if terminated:
                break
        # 2.0 s at 60 Hz = step 120, i.e. flags[120] is the first True.
        assert flags[:120] == [False] * 120
        assert flags[120] is True
    finally:
        env.close()


def test_failed_rotor_actually_loses_thrust():
    """A frozen policy cannot survive losing one of four rotors: it descends."""
    env = GPDQuadrotorEnv(_with_fault(rotor_id=1, failure_time_s=0.5))
    try:
        env.reset(seed=0)
        z_before = None
        terminated = False
        for step in range(int(3.0 * env.control_hz)):
            _, _, terminated, _, info = env.step(np.full(4, env.HOVER_ACTION))
            if step == int(0.4 * env.control_hz):
                z_before = float(env.state.position[2])
            if terminated:
                break
        assert terminated, "losing a rotor at hover thrust must end the episode"
        assert info["done_reason"] in ("crash", "out_of_bounds")
        assert float(env.state.position[2]) < z_before
    finally:
        env.close()


def test_action_rate_penalty_is_measured_before_the_command_is_stored():
    """BaseAviary calls ``_preprocessAction()`` *before* ``_computeReward()``.

    The obvious implementation -- overwrite ``_prev_action`` in
    ``_preprocessAction`` and subtract it in the reward -- therefore always
    subtracts a command from itself. The action-rate term would be exactly zero
    in every step of every run, and nothing would look wrong. This pins the
    value: after a jump of delta on all four rotors it must be 4 * delta^2.
    """
    env = GPDQuadrotorEnv(copy.deepcopy(CONFIG))
    try:
        env.reset(seed=0)
        env.step(np.full(4, env.HOVER_ACTION))
        assert env._delta_u_sq == pytest.approx(0.0, abs=1e-12)
        delta = 0.1
        env.step(np.full(4, env.HOVER_ACTION + delta))
        assert env._delta_u_sq == pytest.approx(4 * delta * delta, rel=1e-9)
    finally:
        env.close()


def test_scenario_is_in_info_but_not_in_observation():
    """The fault-unaware rule survives the migration."""
    env = GPDQuadrotorEnv(_with_fault(rotor_id=2, failure_time_s=1.0))
    try:
        obs, info = env.reset(seed=7)
        assert set(("rotor_id", "failure_time_s", "effectiveness", "seed")) <= set(info["scenario"])
        assert obs.shape == (env.obs_dim,)
        # 17 values per frame; the frame is position error, velocity, quaternion,
        # angular velocity and the last command -- nothing that encodes the fault.
        assert env.obs_dim == 17 * env.history_len
    finally:
        env.close()


def test_factory_returns_gpd_by_default_and_legacy_on_request():
    from envs.env_factory import impl_name, make_env

    config = copy.deepcopy(CONFIG)
    assert impl_name(config) == "gpd"

    config["env"]["impl"] = "legacy"
    legacy = make_env(config)
    assert type(legacy).__name__ == "QuadrotorEnv"
    legacy.close()

    config["env"]["impl"] = "gpd"
    gpd = make_env(config)
    assert type(gpd).__name__ == "GPDQuadrotorEnv"
    gpd.close()

    config["env"]["impl"] = "nonsense"
    with pytest.raises(ValueError):
        impl_name(config)
