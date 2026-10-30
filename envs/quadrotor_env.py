"""Gymnasium environment for the quadrotor position-hold task.

Gate 3 uses this with the stub dynamics (nominal hover).  The dynamics and fault
injector are injected through the same interfaces described in
``docs/INTERFACE.md``, so Andy's and Joe's modules can replace them later.

Fault-unaware rule: the observation is built only from the vehicle state and the
recent commands.  ``info`` may carry the hidden scenario for logging; the
training code must not feed ``info`` to the policy.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Optional

import gymnasium as gym
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
    frame_scales,
    observation_dim,
    quantise_to_control_grid,
)
from envs.quat import quat_from_rpy
from envs.stub_dynamics import StubDynamics, StubParams
from envs.stub_faults import StubFaultInjector


class QuadrotorEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: dict, dynamics: Optional[StubDynamics] = None) -> None:
        super().__init__()
        validate_structure(config)
        self.config = config

        env_cfg = config["env"]
        self.physics_hz = float(env_cfg["physics_hz"])
        self.control_hz = float(env_cfg["control_hz"])
        self.physics_dt = 1.0 / self.physics_hz
        self.control_dt = 1.0 / self.control_hz
        self.physics_steps_per_control = int(round(self.physics_hz / self.control_hz))
        if not np.isclose(
            self.physics_steps_per_control, self.physics_hz / self.control_hz
        ):
            raise ValueError("physics_hz / control_hz must be an integer")
        self.max_steps = int(round(float(env_cfg["episode_length_s"]) * self.control_hz))
        self.target_position = np.asarray(env_cfg["target_position"], dtype=np.float64)
        if self.target_position.shape != (3,):
            raise ValueError("env.target_position must have shape (3,)")

        self.dynamics = dynamics if dynamics is not None else StubDynamics(
            StubParams.from_config(config)
        )

        obs_cfg = config["observation"]
        self.history_len = int(obs_cfg["history_len"])
        self.normalize_obs = bool(obs_cfg["normalize"])
        self.obs_clip = float(obs_cfg["clip"])
        self.obs_dim = observation_dim(self.history_len)
        self._obs_scales = frame_scales(obs_cfg["scales"])

        if self.normalize_obs:
            obs_low, obs_high = -self.obs_clip, self.obs_clip
        else:
            obs_low, obs_high = -np.inf, np.inf
        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=ACTION_LOW,
            high=ACTION_HIGH,
            shape=(ACTION_DIM,),
            dtype=np.float32,
        )

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
        self._state: Optional[VehicleState] = None
        self._scenario = Scenario(seed=self._base_seed)
        self._injector = StubFaultInjector(self._scenario)
        self._prev_action = self.dynamics.hover_command
        self._clip_count = 0
        self._done_reason = ""

    # ------------------------------------------------------------------ reset
    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        if seed is None:
            seed = self._base_seed + self._episode_count
        self._episode_count += 1
        self._episode_seed = int(seed)
        rng = np.random.default_rng(self._episode_seed)

        self._scenario = self._sample_scenario(rng)
        self._injector.reset(self._scenario)
        self._state = VehicleState(
            position=np.asarray(self._scenario.initial_position, dtype=np.float64),
            velocity=np.asarray(self._scenario.initial_velocity, dtype=np.float64),
            quaternion=np.asarray(self._scenario.initial_quaternion, dtype=np.float64),
            angular_velocity=np.zeros(3),
        )
        self._step_count = 0
        self._prev_action = self.dynamics.hover_command.copy()
        self._clip_count = 0
        self._done_reason = ""
        obs = self._buffer.reset(self._state, self._prev_action)
        return obs, self._info(fault_active=False)

    def _sample_scenario(self, rng: np.random.Generator) -> Scenario:
        position_jitter = float(self._reset_cfg.get("position_jitter_m", 0.0))
        velocity_jitter = float(self._reset_cfg.get("velocity_jitter_mps", 0.0))
        tilt_jitter = np.deg2rad(float(self._reset_cfg.get("tilt_jitter_deg", 0.0)))
        yaw_jitter = np.deg2rad(float(self._reset_cfg.get("yaw_jitter_deg", 0.0)))

        position = self.target_position + rng.uniform(
            -position_jitter, position_jitter, size=3
        )
        position[2] = max(position[2], float(self._termination_cfg["crash_z"]) + 0.2)
        velocity = rng.uniform(-velocity_jitter, velocity_jitter, size=3)
        roll = rng.uniform(-tilt_jitter, tilt_jitter)
        pitch = rng.uniform(-tilt_jitter, tilt_jitter)
        yaw = rng.uniform(-yaw_jitter, yaw_jitter)
        quaternion = quat_from_rpy(roll, pitch, yaw)

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
            # Snap to the control grid so the logged scenario is the step the
            # fault actually takes effect on. Uses the configured rate, not a
            # module constant.
            failure_time_s = quantise_to_control_grid(failure_time_s, self.control_hz)

        return Scenario(
            rotor_id=rotor_id,
            failure_time_s=failure_time_s,
            effectiveness=effectiveness,
            initial_position=tuple(float(x) for x in position),
            initial_velocity=tuple(float(x) for x in velocity),
            initial_quaternion=tuple(float(x) for x in quaternion),
            seed=self._episode_seed,
        )

    # ------------------------------------------------------------------- step
    def step(self, action):
        if self._state is None:
            raise RuntimeError("step() called before reset()")
        u_cmd = np.asarray(action, dtype=np.float64).reshape(-1)
        if u_cmd.shape != (ACTION_DIM,):
            raise ValueError(f"action must have shape ({ACTION_DIM},), got {u_cmd.shape}")
        u_clipped = np.clip(u_cmd, ACTION_LOW, ACTION_HIGH)
        self._clip_count = int(np.sum(u_clipped != u_cmd))

        t_s = self._step_count * self.control_dt
        u_applied = self._injector.apply(u_clipped, t_s)
        fault_active = self._injector.is_active(t_s)

        for _ in range(self.physics_steps_per_control):
            self._state = self.dynamics.step(self._state, u_applied, self.physics_dt)

        self._step_count += 1
        obs = self._buffer.push(self._state, u_clipped)
        reward = self._reward(u_clipped)
        terminated, truncated, reason = self._check_done()
        if terminated:
            reward += float(
                self._reward_cfg["crash"]
                if reason in ("crash", "non_finite")
                else self._reward_cfg["out_of_bounds"]
            )
        self._done_reason = reason
        self._prev_action = u_clipped
        return obs, float(reward), bool(terminated), bool(truncated), self._info(fault_active)

    def _reward(self, u_cmd: np.ndarray) -> float:
        cfg = self._reward_cfg
        p_err = self.target_position - self._state.position
        v = self._state.velocity
        omega = self._state.angular_velocity
        delta_u = u_cmd - self._prev_action
        return float(
            cfg["alive"]
            - cfg["position_error"] * np.dot(p_err, p_err)
            - cfg["linear_velocity"] * np.dot(v, v)
            - cfg["action_rate"] * np.dot(delta_u, delta_u)
            - cfg["angular_rate_xy"] * (omega[0] ** 2 + omega[1] ** 2)
        )

    def _check_done(self):
        cfg = self._termination_cfg
        state = self._state
        if not (
            np.all(np.isfinite(state.position))
            and np.all(np.isfinite(state.velocity))
            and np.all(np.isfinite(state.quaternion))
            and np.all(np.isfinite(state.angular_velocity))
        ):
            return True, False, "non_finite"
        if state.position[2] < float(cfg["crash_z"]):
            return True, False, "crash"
        radius = float(np.hypot(state.position[0], state.position[1]))
        if radius > float(cfg["max_xy_radius"]) or state.position[2] > float(cfg["max_z"]):
            return True, False, "out_of_bounds"
        if self._step_count >= self.max_steps:
            return False, True, "time_limit"
        return False, False, ""

    # ------------------------------------------------------------------- info
    def _info(self, fault_active: bool) -> dict:
        p_err = self.target_position - self._state.position
        return {
            "seed": self._episode_seed,
            "step": self._step_count,
            "scenario": asdict(self._scenario),  # hidden data: logging only
            "fault_active": bool(fault_active),
            "position": self._state.position.copy(),
            "velocity": self._state.velocity.copy(),
            "position_error_m": float(np.linalg.norm(p_err)),
            "clip_count": self._clip_count,
            "done_reason": self._done_reason,
        }

    @property
    def state(self) -> VehicleState:
        if self._state is None:
            raise RuntimeError("environment has not been reset")
        return self._state

    @property
    def scenario(self) -> Scenario:
        return self._scenario

    def render(self):
        return None

    def close(self) -> None:
        return None
