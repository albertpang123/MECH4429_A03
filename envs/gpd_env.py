"""The same position-hold task, built on gym-pybullet-drones' ``BaseAviary``.

Why this file exists
--------------------
Until now the project carried its own physics (``StubDynamics``) and its own
engine wrapper (``PyBulletDynamics``). Both are correct -- the two agree to
0.04 mm over 4 s -- but neither is backed by a published simulator, and the
supervisor asked for our work to stand on gym-pybullet-drones.

This module makes that literal: ``GPDQuadrotorEnv`` subclasses ``BaseAviary``,
so the world, the URDF, the force/torque mapping, the physics stepping, the
ground/obstacle/multi-drone machinery and the GUI come from the vendored
package in ``third_party/``. Everything above physics -- observation layout,
reward, termination, fault injection, scenario sampling, seeding -- is ours and
unchanged from ``envs/quadrotor_env.py``.

What is *not* changed, deliberately
-----------------------------------
* The observation is still ``OBS_FRAME_FIELDS`` x ``history_len`` (17 x 10 = 170).
* The action is still four normalised motor commands in [0, 1].
* The fault-unaware rule still holds: ``info`` carries the hidden scenario, the
  observation never does.

What *does* change, and must be reported
----------------------------------------
The vehicle model is now gpd's ``cf2x`` instead of the placeholder values in the
config: mass 0.027 kg, arm 0.0397 m, thrust-to-weight **2.25** (the stub's
placeholder gave 1.63). That is a physically different machine, so G3/G4 numbers
from the stub path are not comparable point-by-point with numbers from this one.

Rotor numbering (read this before injecting a fault)
----------------------------------------------------
Our layout is R0 front-right, R1 front-left, R2 rear-left, R3 rear-right
(counter-clockwise seen from above). gpd's ``cf2x.urdf`` orders its four rotor
links FR, RR, RL, FL -- i.e. ``gpd 0 = our R0``, ``gpd 1 = our R3``,
``gpd 2 = our R2``, ``gpd 3 = our R1``. ``ROTOR_GPD_INDEX`` below is the only
place that mapping is written down; ``tests/test_gpd_env.py`` pins it by
failing rotor *k* and checking the resulting torque direction. Getting it
backwards silently fails the wrong propeller and still looks plausible.

Speed note
----------
One env step = ``PYB_STEPS_PER_CTRL`` (4) calls into PyBullet, each applying 4
link forces + 1 body torque -- but only +1/+2 body forces and one quaternion
read, against ``PyBulletDynamics``'s per-step URDF bookkeeping, so it is
measurably *faster* despite doing the same amount of physics. Measured on
Windows 11 / CPython 3.12 (``tools/crosscheck_gpd.py``, ms per control step):

    gpd 0.17   |   PyBulletDynamics 0.34   |   StubDynamics 0.47

Re-run that tool rather than trusting these numbers on another machine.
"""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
from gymnasium import spaces

from envs.config import validate_structure
from envs.interface import (
    ACTION_DIM,
    ACTION_HIGH,
    ACTION_LOW,
    ObservationBuffer,
    Scenario,
    VehicleState,
    canonicalize_quaternion,
    frame_scales,
    observation_dim,
    quantise_to_control_grid,
)
from envs.quat import quat_from_rpy

# ---------------------------------------------------------------------------
# Vendored dependency
# ---------------------------------------------------------------------------
# The package lives in third_party/ rather than in site-packages because it is
# pinned to a commit (see third_party/NOTICE.md) and must not be silently
# upgraded by a pip resolve. Putting it on sys.path is the smallest thing that
# keeps `import gym_pybullet_drones` working everywhere, including under
# SubprocVecEnv, where the child process re-imports this module from scratch.
_THIRD_PARTY = Path(__file__).resolve().parents[1] / "third_party"
if str(_THIRD_PARTY) not in sys.path:
    sys.path.insert(0, str(_THIRD_PARTY))

from gym_pybullet_drones.envs.BaseAviary import BaseAviary  # noqa: E402
from gym_pybullet_drones.utils.enums import DroneModel, Physics  # noqa: E402

#: our rotor index -> gpd's URDF link index. See the module docstring.
ROTOR_GPD_INDEX = (0, 3, 2, 1)
if sorted(ROTOR_GPD_INDEX) != [0, 1, 2, 3]:  # pragma: no cover - guard on edit
    raise RuntimeError("ROTOR_GPD_INDEX must be a permutation of 0..3")


class GPDQuadrotorEnv(BaseAviary):
    """Position-hold task on gym-pybullet-drones' ``BaseAviary``.

    Construct with the same merged config dict as ``QuadrotorEnv``; nothing else
    in the project has to know which of the two is running, because both expose
    ``reset/step/close``, ``state``, ``scenario`` and the same spaces.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: dict, gui: bool = False) -> None:
        # Everything BaseAviary.__init__ needs must exist *before* it runs: it
        # calls self._actionSpace() and self._observationSpace() internally.
        validate_structure(config)
        self.config = config

        env_cfg = config["env"]
        self.physics_hz = float(env_cfg["physics_hz"])
        self.control_hz = float(env_cfg["control_hz"])
        self.control_dt = 1.0 / self.control_hz
        steps = self.physics_hz / self.control_hz
        if not np.isclose(steps, round(steps)):
            raise ValueError("physics_hz / control_hz must be an integer")
        self.physics_steps_per_control = int(round(steps))
        self.max_steps = int(round(float(env_cfg["episode_length_s"]) * self.control_hz))
        self.target_position = np.asarray(env_cfg["target_position"], dtype=np.float64)
        if self.target_position.shape != (3,):
            raise ValueError("env.target_position must have shape (3,)")

        obs_cfg = config["observation"]
        self.history_len = int(obs_cfg["history_len"])
        self.normalize_obs = bool(obs_cfg["normalize"])
        self.obs_clip = float(obs_cfg["clip"])
        self.obs_dim = observation_dim(self.history_len)
        self._obs_scales = frame_scales(obs_cfg["scales"])

        self._buffer = ObservationBuffer(
            target_position=self.target_position,
            history_len=self.history_len,
            normalize=self.normalize_obs,
            clip=self.obs_clip,
            scales=self._obs_scales,
        )
        self._reward_cfg = config["reward"]
        self._termination_cfg = config["termination"]
        self._fault_cfg = config["faults"]
        self._reset_cfg = config.get("reset", {})

        self._base_seed = int(config["run"]["seed"])
        self._episode_count = 0
        self._episode_seed = self._base_seed
        self._step_count = 0
        self._scenario = Scenario(seed=self._base_seed)
        self._injector = None            # built in reset(), needs the scenario
        self._state: Optional[VehicleState] = None
        self._u_cmd = np.full(ACTION_DIM, 0.5, dtype=np.float64)
        self._prev_action = self._u_cmd.copy()
        self._fault_active = False
        self._clip_count = 0
        self._done_reason = ""
        self._reset_pending = False
        self._delta_u_sq = 0.0

        initial = np.asarray([[0.0, 0.0, float(self.target_position[2])]])

        super().__init__(
            drone_model=DroneModel.CF2X,
            num_drones=1,
            initial_xyzs=initial,
            initial_rpys=np.zeros((1, 3)),
            physics=Physics(str(env_cfg.get("gpd_physics", "pyb")).lower()),
            pyb_freq=int(self.physics_hz),
            ctrl_freq=int(self.control_hz),
            gui=gui,
            record=False,
            obstacles=False,
            user_debug_gui=bool(gui),
            vision_attributes=False,
            output_folder=str(Path(__file__).resolve().parents[1] / "results"),
        )
        # After BaseAviary has read the URDF we know MAX_RPM/HOVER_RPM; the
        # normalised hover command follows from the thrust-to-weight ratio.
        self.HOVER_ACTION = float(self.HOVER_RPM / self.MAX_RPM)
        self._u_cmd = np.full(ACTION_DIM, self.HOVER_ACTION, dtype=np.float64)
        self._prev_action = self._u_cmd.copy()

        from envs.stub_faults import StubFaultInjector  # local: keeps import graph flat

        self._injector = StubFaultInjector(self._scenario)
        self._state = self._read_state()
        self._buffer.reset(self._state, self._u_cmd)

    # ------------------------------------------------------------------ spaces
    def _actionSpace(self) -> spaces.Box:
        return spaces.Box(
            low=ACTION_LOW, high=ACTION_HIGH, shape=(ACTION_DIM,), dtype=np.float32
        )

    def _observationSpace(self) -> spaces.Box:
        low, high = (-self.obs_clip, self.obs_clip) if self.normalize_obs else (-np.inf, np.inf)
        return spaces.Box(low=low, high=high, shape=(self.obs_dim,), dtype=np.float32)

    # ------------------------------------------------------------------- reset
    def reset(self, *, seed: Optional[int] = None, options=None):
        if seed is None:
            seed = self._base_seed + self._episode_count
        self._episode_count += 1
        self._episode_seed = int(seed)
        rng = np.random.default_rng(self._episode_seed)

        self._scenario = self._sample_scenario(rng)
        self._injector.reset(self._scenario)
        self._step_count = 0
        self._fault_active = False
        self._clip_count = 0
        self._done_reason = ""
        self._delta_u_sq = 0.0
        self._u_cmd = np.full(ACTION_DIM, self.HOVER_ACTION, dtype=np.float64)
        self._prev_action = self._u_cmd.copy()

        # BaseAviary rebuilds the world in reset() and places the drones at
        # INIT_XYZS / INIT_RPYS, so the per-episode jitter goes there.
        self.INIT_XYZS = np.asarray([self._scenario.initial_position], dtype=np.float64)
        quat = np.asarray(self._scenario.initial_quaternion, dtype=np.float64)
        self.INIT_RPYS = np.asarray([_rpy_from_quat(quat)], dtype=np.float64)

        self._reset_pending = True
        obs, info = super().reset(seed=seed, options=options)
        return obs, info

    def _sample_scenario(self, rng: np.random.Generator) -> Scenario:
        """Identical to ``QuadrotorEnv._sample_scenario`` -- the task did not move."""
        position_jitter = float(self._reset_cfg.get("position_jitter_m", 0.0))
        velocity_jitter = float(self._reset_cfg.get("velocity_jitter_mps", 0.0))
        tilt_jitter = np.deg2rad(float(self._reset_cfg.get("tilt_jitter_deg", 0.0)))
        yaw_jitter = np.deg2rad(float(self._reset_cfg.get("yaw_jitter_deg", 0.0)))

        position = self.target_position + rng.uniform(-position_jitter, position_jitter, size=3)
        position[2] = max(position[2], float(self._termination_cfg["crash_z"]) + 0.2)
        roll = rng.uniform(-tilt_jitter, tilt_jitter)
        pitch = rng.uniform(-tilt_jitter, tilt_jitter)
        yaw = rng.uniform(-yaw_jitter, yaw_jitter)

        mode = str(self._fault_cfg.get("mode", "nominal"))
        rotor_id = None
        failure_time_s = None
        effectiveness = 0.0
        if mode == "fixed":
            fixed = self._fault_cfg["fixed"]
            rotor_id = int(fixed["rotor_id"])
            failure_time_s = float(fixed["failure_time_s"])
            effectiveness = float(fixed["effectiveness"])
        elif mode == "random":
            random_cfg = self._fault_cfg["random"]
            rotor_id = int(rng.choice(random_cfg["rotor_id"]))
            low, high = random_cfg["failure_time_s"]
            failure_time_s = float(rng.uniform(float(low), float(high)))
            effectiveness = float(random_cfg["effectiveness"])
        elif mode != "nominal":
            raise ValueError(f"unknown faults.mode: {mode}")

        if failure_time_s is not None:
            failure_time_s = quantise_to_control_grid(failure_time_s, self.control_hz)

        return Scenario(
            rotor_id=rotor_id,
            failure_time_s=failure_time_s,
            effectiveness=effectiveness,
            initial_position=tuple(float(x) for x in position),
            initial_velocity=tuple(float(x) for x in rng.uniform(-velocity_jitter, velocity_jitter, 3)),
            initial_quaternion=tuple(float(x) for x in quat_from_rpy(roll, pitch, yaw)),
            seed=self._episode_seed,
        )

    # -------------------------------------------------------------------- step
    def _preprocessAction(self, action) -> np.ndarray:
        """[0,1] motor commands -> RPMs, with the fault applied at the source.

        ``BaseAviary.step`` calls this once per control step and then applies the
        returned RPMs directly, so zeroing a rotor here is exactly "that motor
        produces no thrust from this step on" -- no post-hoc state surgery.
        """
        u_cmd = np.asarray(action, dtype=np.float64).reshape(-1)
        if u_cmd.shape != (ACTION_DIM,):
            raise ValueError(f"action must have shape ({ACTION_DIM},), got {u_cmd.shape}")
        u_clipped = np.clip(u_cmd, ACTION_LOW, ACTION_HIGH)
        self._clip_count = int(np.sum(u_clipped != u_cmd))

        t_s = self._step_count * self.control_dt
        u_applied = np.asarray(self._injector.apply(u_clipped, t_s), dtype=np.float64)
        self._fault_active = bool(self._injector.is_active(t_s))

        # The action-rate term has to be measured here and not in the reward:
        # BaseAviary calls _preprocessAction() before _computeReward(), so by
        # the time the reward is computed ``_prev_action`` already holds *this*
        # step's command and the difference would silently be zero. Compute it
        # against the previous step's command first, then advance.
        delta_u = u_clipped - self._prev_action
        self._delta_u_sq = float(np.dot(delta_u, delta_u))
        self._u_cmd = u_clipped
        self._prev_action = u_clipped
        self._step_count += 1

        # Reorder into gpd's URDF link order. Without this the commands land on
        # the wrong corner -- invisible at hover (all four equal) and silently
        # wrong the moment one rotor differs, which is exactly the G4 case.
        rpm_our = u_applied * self.MAX_RPM
        rpm_gpd = np.empty(ACTION_DIM, dtype=np.float64)
        rpm_gpd[list(ROTOR_GPD_INDEX)] = rpm_our
        return rpm_gpd.reshape(self.NUM_DRONES, 4)

    # ------------------------------------------------------------------ hooks
    def _read_state(self) -> VehicleState:
        i = 0
        return VehicleState(
            position=np.array(self.pos[i], dtype=np.float64),
            velocity=np.array(self.vel[i], dtype=np.float64),
            quaternion=canonicalize_quaternion(np.array(self.quat[i], dtype=np.float64)),
            angular_velocity=np.array(self.ang_v[i], dtype=np.float64),
        )

    def _computeObs(self) -> np.ndarray:
        self._state = self._read_state()
        if self._reset_pending:
            self._reset_pending = False
            return self._buffer.reset(self._state, self._u_cmd)
        return self._buffer.push(self._state, self._u_cmd)

    def _termination_reason(self) -> str:
        cfg = self._termination_cfg
        state = self._state
        if not (
            np.all(np.isfinite(state.position))
            and np.all(np.isfinite(state.velocity))
            and np.all(np.isfinite(state.quaternion))
            and np.all(np.isfinite(state.angular_velocity))
        ):
            return "non_finite"
        if state.position[2] < float(cfg["crash_z"]):
            return "crash"
        radius = float(np.hypot(state.position[0], state.position[1]))
        if radius > float(cfg["max_xy_radius"]) or state.position[2] > float(cfg["max_z"]):
            return "out_of_bounds"
        if self._step_count >= self.max_steps:
            return "time_limit"
        return ""

    def _computeReward(self) -> float:
        cfg = self._reward_cfg
        p_err = self.target_position - self._state.position
        v = self._state.velocity
        omega = self._state.angular_velocity
        reward = (
            cfg["alive"]
            - cfg["position_error"] * float(np.dot(p_err, p_err))
            - cfg["linear_velocity"] * float(np.dot(v, v))
            - cfg["action_rate"] * self._delta_u_sq
            - cfg["angular_rate_xy"] * float(omega[0] ** 2 + omega[1] ** 2)
        )
        self._done_reason = self._termination_reason()
        if self._done_reason in ("crash", "non_finite"):
            reward += float(cfg["crash"])
        elif self._done_reason == "out_of_bounds":
            reward += float(cfg["out_of_bounds"])
        return float(reward)

    def _computeTerminated(self) -> bool:
        return self._done_reason in ("crash", "non_finite", "out_of_bounds")

    def _computeTruncated(self) -> bool:
        return self._done_reason == "time_limit"

    def _computeInfo(self) -> dict:
        p_err = self.target_position - self._state.position
        return {
            "seed": self._episode_seed,
            "step": self._step_count,
            "scenario": asdict(self._scenario),   # hidden data: logging only
            "fault_active": bool(self._fault_active),
            "position": self._state.position.copy(),
            "velocity": self._state.velocity.copy(),
            "position_error_m": float(np.linalg.norm(p_err)),
            "clip_count": self._clip_count,
            "done_reason": self._done_reason,
        }

    # -------------------------------------------------------------- accessors
    @property
    def state(self) -> VehicleState:
        if self._state is None:
            raise RuntimeError("environment has not been reset")
        return self._state

    @property
    def scenario(self) -> Scenario:
        return self._scenario

    #: Vehicle description, in the same shape ``QuadrotorEnv`` exposes, so tools
    #: (``eval/fault_response.py``, ``eval/replay.py``) do not branch on which
    #: implementation is running. The numbers come from gpd's URDF, not config.
    @property
    def hover_action(self) -> float:
        return float(self.HOVER_ACTION)

    @property
    def mass_kg(self) -> float:
        return float(self.M)

    @property
    def thrust_max_per_rotor_n(self) -> float:
        return float(self.MAX_RPM**2 * self.KF)

    def applied_thrusts(self, u_applied) -> np.ndarray:
        """Thrust per rotor for normalised commands ``u_applied``, in newtons.

        Quadratic, not linear: gpd's force model is ``KF * rpm^2``, and
        ``rpm = u * MAX_RPM``. The stub's ``thrust_curve: linear`` is a modelling
        choice for Andy's placeholder; this one is the published propeller
        relation, so the same *command* does not mean the same *force* on the two.
        """
        return (np.asarray(u_applied, dtype=np.float64) * self.MAX_RPM) ** 2 * self.KF

    @property
    def rotor_positions_body_m(self) -> np.ndarray:
        """(x forward, y left) of each rotor, in our R0/R1/R2/R3 order."""
        from envs.interface import nominal_x_positions

        return nominal_x_positions(float(self.L))

    @property
    def rotor_layout(self) -> dict:
        """Diagnostics: our rotor k -> gpd link index, and the RPM it gets now."""
        return {
            "our_index_to_gpd_link": dict(enumerate(ROTOR_GPD_INDEX)),
            "max_rpm": float(self.MAX_RPM),
            "hover_rpm": float(self.HOVER_RPM),
            "hover_action": float(self.HOVER_ACTION),
            "thrust_to_weight": float(self.THRUST2WEIGHT_RATIO),
            "mass_kg": float(self.M),
            "arm_length_m": float(self.L),
        }

    def render(self):  # noqa: D401 - BaseAviary's render prints/returns nothing useful
        return super().render()

    def close(self) -> None:
        super().close()


# ---------------------------------------------------------------------------
def _rpy_from_quat(quat) -> np.ndarray:
    """Roll/pitch/yaw of a scalar-last quaternion, for INIT_RPYS."""
    x, y, z, w = np.asarray(quat, dtype=np.float64)
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sin_pitch = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sin_pitch, -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([roll, pitch, yaw], dtype=np.float64)
