"""Minimal 6-DOF quadrotor used to bring up the G3 pipeline.

This is a stand-in for Andy's model in ``src/dynamics/``.  It implements the
same ``reset(state)`` / ``step(state, u_applied, dt)`` interface, so swapping in
his module later should not touch the environment or the training code.

Modelled: rigid-body 6-DOF, gravity, rotor thrust along body +z, rotor torques,
yaw reaction torque, linear and angular drag.  Not modelled: motor lag,
aerodynamic effects, blade flapping, ground contact.

Parameters are *stub values* from ``configs/g3_hover.yaml``, not the agreed
vehicle parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

from envs.interface import (
    ACTION_DIM,
    N_ROTORS,
    VehicleState,
    build_rotors,
    nominal_x_positions,
)
from envs.quat import (
    normalize_quaternion,
    quat_derivative,
    quat_rotate,
)

GRAVITY_MPS2 = 9.81


def _require(d: dict, key: str, owner: str) -> float:
    """Read a required dynamics value, naming the owner when it is still TBD."""
    value = d.get(key)
    if value is None:
        raise ValueError(
            f"dynamics.{key} is null in the config — it is {owner}'s parameter "
            "(Gate 0 / vehicle data); fill it in before running"
        )
    return float(value)


@dataclass(frozen=True)
class StubParams:
    mass_kg: float
    inertia_kgm2: Tuple[float, float, float]
    rotor_positions_body_m: Tuple[Tuple[float, float], ...]
    thrust_max_per_rotor_n: float
    drag_torque_ratio_m: float
    linear_drag_n_per_mps: float = 0.0
    angular_drag_nm_per_radps: float = 0.0
    thrust_curve: str = "linear"
    motor_time_constant_s: float = 0.0

    def __post_init__(self) -> None:
        if self.mass_kg <= 0.0:
            raise ValueError("mass_kg must be positive")
        if len(self.inertia_kgm2) != 3 or min(self.inertia_kgm2) <= 0.0:
            raise ValueError("inertia_kgm2 must be three positive values")
        positions = np.asarray(self.rotor_positions_body_m, dtype=np.float64)
        if positions.shape != (N_ROTORS, 2):
            raise ValueError(
                f"rotor_positions_body_m must have shape ({N_ROTORS}, 2), "
                f"got {positions.shape}"
            )
        if self.thrust_max_per_rotor_n <= 0.0:
            raise ValueError("thrust_max_per_rotor_n must be positive")
        if self.thrust_curve not in ("linear", "quadratic"):
            raise ValueError("thrust_curve must be 'linear' or 'quadratic'")
        if self.motor_time_constant_s != 0.0:
            raise NotImplementedError("motor lag is not implemented in the stub")

    @property
    def arm_length_m(self) -> float:
        """CoM-to-rotor distance implied by the positions (reporting only)."""
        positions = np.asarray(self.rotor_positions_body_m, dtype=np.float64)
        return float(np.max(np.linalg.norm(positions, axis=1)))

    @classmethod
    def from_config(cls, config: dict) -> "StubParams":
        d = config["dynamics"]
        positions = d.get("rotor_positions_body_m")
        if positions is None and d.get("arm_length_m") is not None:
            # Checkpoints trained before 2026-10-30 carry a single arm length in
            # their meta.json instead of explicit positions.  Keep them
            # loadable; new configs always use rotor_positions_body_m.
            positions = nominal_x_positions(float(d["arm_length_m"]))
        if positions is None:
            raise ValueError(
                "dynamics.rotor_positions_body_m is null in the config — it is "
                "Andy's parameter (rotor positions from the CAD, Gate 0); fill it "
                "in before running"
            )
        return cls(
            mass_kg=_require(d, "mass_kg", "Andy"),
            inertia_kgm2=tuple(float(x) for x in d["inertia_kgm2"]),
            rotor_positions_body_m=tuple(
                (float(x), float(y)) for x, y in positions
            ),
            thrust_max_per_rotor_n=_require(d, "thrust_max_per_rotor_n", "Andy"),
            drag_torque_ratio_m=_require(d, "drag_torque_ratio_m", "Andy"),
            linear_drag_n_per_mps=float(d.get("linear_drag_n_per_mps", 0.0)),
            angular_drag_nm_per_radps=float(d.get("angular_drag_nm_per_radps", 0.0)),
            thrust_curve=str(d.get("thrust_curve", "linear")),
            motor_time_constant_s=float(d.get("motor_time_constant_s", 0.0)),
        )


class StubDynamics:
    """Nominal 6-DOF model with the shared ``envs.interface`` state type."""

    def __init__(self, params: StubParams) -> None:
        self.params = params
        rotors = build_rotors(params.rotor_positions_body_m)
        self._positions = np.array([[r.x_m, r.y_m] for r in rotors], dtype=np.float64)
        self._spins = np.array([r.spin for r in rotors], dtype=np.float64)
        self._inertia = np.asarray(params.inertia_kgm2, dtype=np.float64)
        self._inv_inertia = 1.0 / self._inertia

    def reset(self, state: VehicleState) -> VehicleState:
        return VehicleState(
            position=state.position.copy(),
            velocity=state.velocity.copy(),
            quaternion=state.quaternion.copy(),
            angular_velocity=state.angular_velocity.copy(),
        )

    def applied_thrusts(self, u_applied: Sequence[float]) -> np.ndarray:
        u = np.clip(np.asarray(u_applied, dtype=np.float64).reshape(-1), 0.0, 1.0)
        if u.shape != (ACTION_DIM,):
            raise ValueError(f"u_applied must have shape ({ACTION_DIM},), got {u.shape}")
        if self.params.thrust_curve == "linear":
            return self.params.thrust_max_per_rotor_n * u
        return self.params.thrust_max_per_rotor_n * u * u

    @property
    def hover_command(self) -> np.ndarray:
        weight = self.params.mass_kg * GRAVITY_MPS2
        u = weight / (ACTION_DIM * self.params.thrust_max_per_rotor_n)
        if self.params.thrust_curve == "quadratic":
            u = np.sqrt(u)
        return np.full(ACTION_DIM, float(np.clip(u, 0.0, 1.0)))

    def step(
        self,
        state: VehicleState,
        u_applied: Sequence[float],
        dt: float,
    ) -> VehicleState:
        p = state.position
        v = state.velocity
        q = normalize_quaternion(state.quaternion)
        omega = state.angular_velocity

        f = self.applied_thrusts(u_applied)
        total_thrust = float(f.sum())

        # Body torque from rotor forces plus yaw reaction torque.
        tau_x = float(np.sum(self._positions[:, 1] * f))
        tau_y = float(-np.sum(self._positions[:, 0] * f))
        tau_z = float(-self.params.drag_torque_ratio_m * np.sum(self._spins * f))
        tau = np.array([tau_x, tau_y, tau_z])

        force_world = quat_rotate(q, np.array([0.0, 0.0, total_thrust]))
        force_world[2] -= self.params.mass_kg * GRAVITY_MPS2
        force_world -= self.params.linear_drag_n_per_mps * v

        accel = force_world / self.params.mass_kg
        gyroscopic = np.cross(omega, self._inertia * omega)
        angular_accel = self._inv_inertia * (
            tau - gyroscopic - self.params.angular_drag_nm_per_radps * omega
        )

        # Semi-implicit Euler.
        v_new = v + accel * dt
        p_new = p + v_new * dt
        omega_new = omega + angular_accel * dt
        q_new = normalize_quaternion(q + quat_derivative(q, omega_new) * dt)

        return VehicleState(
            position=p_new,
            velocity=v_new,
            quaternion=q_new,
            angular_velocity=omega_new,
        )
