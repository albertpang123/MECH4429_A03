"""PyBullet-backed dynamics for the same interface as ``StubDynamics``.

Why this file exists
--------------------
The project's physics is a pluggable module: the environment calls

    dynamics.step(state, u_applied, dt) -> VehicleState

and nothing else. ``StubDynamics`` is the hand-written NumPy implementation
used from Gate 3 on. This class implements the *same* four methods on top of
the PyBullet engine, so it can be dropped in without touching the environment,
the observation, the reward, the fault injector, the training code or the
tests. Selection is by ``dynamics.backend: numpy | pybullet`` in the config.

What PyBullet gives us that the NumPy stub does not
---------------------------------------------------
* ground contact (a real crash test instead of "CoM below 0.05 m"),
* an engine whose rigid-body integration and collision handling are already
  validated and published (gym-pybullet-drones, IROS 2021, wraps the same
  engine),
* the same URDF/asset path the 3D replay uses, so what you train is what you
  watch.

What it costs
-------------
Raw speed: measured through ``QuadrotorEnv`` at 240 Hz physics / 60 Hz control,
wall clock for 500 control steps after a 50-step warm-up: **1755 steps/s
(0.57 ms/step) for numpy versus 3078 steps/s (0.32 ms/step) for pybullet**.
The engine is *faster* than the stub -- the stub's per-rotor NumPy work costs
more than a C++ ``stepSimulation`` call. Training time is not a reason to
avoid this backend.

Determinism: PyBullet is deterministic for a pinned build and a fixed
timestep, but unlike pure NumPy it is not bit-identical across machines. The
report must say "reproducible with pybullet 3.2.7", not "reproducible anywhere".

Notes
-----
* Rotor thrust is applied as four world-frame forces at the hub positions,
  which produces the body torque directly -- no torque bookkeeping.
* The yaw reaction torque (-kappa * sum(spin_i * f_i)) is applied separately,
  exactly as in ``stub_dynamics``.
* Contacts are on and the plane is loaded from ``pybullet_data`` by default.
  That is a real difference from ``StubDynamics``, which has no ground at all:
  once the CoM reaches z = 0 the two models *must* disagree. Pass ``ground=False``
  for a like-for-like comparison with the stub -- with it, the two agree to
  floating-point noise (see the torque test in ``tests/test_pybullet_backend.py``).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from envs.interface import ACTION_DIM, VehicleState, build_rotors
from envs.quat import normalize_quaternion
from envs.stub_dynamics import GRAVITY_MPS2, StubParams


class PyBulletDynamics:
    """Engine-backed dynamics with the ``StubDynamics`` interface."""

    def __init__(self, params: StubParams, gui: bool = False, ground: bool = True) -> None:
        import pybullet as p
        import pybullet_data

        self.params = params
        self.ground = ground
        self._p = p

        rotors = build_rotors(params.rotor_positions_body_m)
        self._rotor_xy = np.array([[r.x_m, r.y_m] for r in rotors], dtype=np.float64)
        self._spins = np.array([r.spin for r in rotors], dtype=np.float64)

        self.client: int = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        p.setGravity(0.0, 0.0, -GRAVITY_MPS2, physicsClientId=self.client)
        p.setTimeStep(1.0 / 240.0, physicsClientId=self.client)
        self._plane: Optional[int] = None
        if ground:
            self._plane = p.loadURDF("plane.urdf", physicsClientId=self.client)
        self.body: int = self._create_body()
        # Bullet adds damping by default; the stub has none unless configured.
        # ``linearDamping`` is *not* the same quantity as our drag coefficient:
        # ours is a force per unit speed (N/(m/s)), Bullet's is a decay rate
        # (1/s). Mapping one onto the other silently changes the model, so the
        # engine's damping is switched off and the drag terms are applied as
        # explicit forces/torques in :meth:`step` -- exactly as the stub does.
        p.changeDynamics(
            self.body,
            -1,
            linearDamping=0.0,
            angularDamping=0.0,
            localInertiaDiagonal=list(params.inertia_kgm2),
            physicsClientId=self.client,
        )
        self._time_step: Optional[float] = None

    # ---------------------------------------------------------------- setup
    def _create_body(self) -> int:
        """A cylinder body whose mass and inertia come from the config."""
        p = self._p
        radius = float(np.max(np.linalg.norm(self._rotor_xy, axis=1))) * 0.45
        half_height = max(0.01, radius * 0.12)
        collision = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=radius,
            height=2.0 * half_height,
            physicsClientId=self.client,
        )
        visual = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=radius,
            length=2.0 * half_height,
            rgbaColor=[0.25, 0.3, 0.35, 1.0],
            physicsClientId=self.client,
        )
        return p.createMultiBody(
            baseMass=float(self.params.mass_kg),
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=[0.0, 0.0, 1.0],
            baseOrientation=[0.0, 0.0, 0.0, 1.0],
            physicsClientId=self.client,
        )

    # ------------------------------------------------------------ interface
    def reset(self, state: VehicleState) -> VehicleState:
        """Put the body where the scenario says it is and hand the state back."""
        self._apply_state(state)
        return VehicleState(
            position=state.position.copy(),
            velocity=state.velocity.copy(),
            quaternion=normalize_quaternion(state.quaternion),
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
        p = self._p
        self._set_time_step(dt)
        self._apply_state(state)

        forces = self.applied_thrusts(u_applied)
        rotation = np.asarray(
            p.getMatrixFromQuaternion(normalize_quaternion(state.quaternion),
                                     physicsClientId=self.client),
            dtype=np.float64,
        ).reshape(3, 3)

        thrust_axis = rotation @ np.array([0.0, 0.0, 1.0])
        for index in range(ACTION_DIM):
            force = thrust_axis * forces[index]
            offset = rotation @ np.array([self._rotor_xy[index, 0],
                                          self._rotor_xy[index, 1], 0.0])
            p.applyExternalForce(
                self.body,
                -1,
                forceObj=force.tolist(),
                posObj=(state.position + offset).tolist(),
                flags=p.WORLD_FRAME,
                physicsClientId=self.client,
            )

        yaw_torque = -self.params.drag_torque_ratio_m * float(np.sum(self._spins * forces))
        drag_torque_body = (-self.params.angular_drag_nm_per_radps
                            * np.asarray(state.angular_velocity, dtype=np.float64))
        drag_torque_body[2] += yaw_torque
        p.applyExternalTorque(
            self.body,
            -1,
            torqueObj=(rotation @ drag_torque_body).tolist(),
            flags=p.WORLD_FRAME,
            physicsClientId=self.client,
        )

        if self.params.linear_drag_n_per_mps != 0.0:
            p.applyExternalForce(
                self.body,
                -1,
                forceObj=(-self.params.linear_drag_n_per_mps
                          * np.asarray(state.velocity, dtype=np.float64)).tolist(),
                posObj=np.asarray(state.position, dtype=np.float64).tolist(),
                flags=p.WORLD_FRAME,
                physicsClientId=self.client,
            )

        p.stepSimulation(physicsClientId=self.client)
        return self._read_state()

    def close(self) -> None:
        if getattr(self, "client", None) is not None:
            self._p.disconnect(physicsClientId=self.client)
            self.client = None

    # -------------------------------------------------------------- helpers
    def _set_time_step(self, dt: float) -> None:
        if self._time_step is None or abs(self._time_step - dt) > 1e-12:
            self._p.setTimeStep(dt, physicsClientId=self.client)
            self._time_step = dt

    def _apply_state(self, state: VehicleState) -> None:
        quaternion = normalize_quaternion(state.quaternion)
        rotation = np.asarray(
            self._p.getMatrixFromQuaternion(quaternion.tolist(),
                                            physicsClientId=self.client),
            dtype=np.float64,
        ).reshape(3, 3)
        world_angular = rotation @ np.asarray(state.angular_velocity, dtype=np.float64)
        self._p.resetBasePositionAndOrientation(
            self.body,
            posObj=np.asarray(state.position, dtype=np.float64).tolist(),
            ornObj=quaternion.tolist(),
            physicsClientId=self.client,
        )
        self._p.resetBaseVelocity(
            self.body,
            linearVelocity=np.asarray(state.velocity, dtype=np.float64).tolist(),
            angularVelocity=world_angular.tolist(),
            physicsClientId=self.client,
        )

    def _read_state(self) -> VehicleState:
        p = self._p
        position, quaternion = p.getBasePositionAndOrientation(
            self.body, physicsClientId=self.client)
        linear, angular = p.getBaseVelocity(self.body, physicsClientId=self.client)
        rotation = np.asarray(
            p.getMatrixFromQuaternion(quaternion, physicsClientId=self.client),
            dtype=np.float64,
        ).reshape(3, 3)
        return VehicleState(
            position=np.asarray(position, dtype=np.float64),
            velocity=np.asarray(linear, dtype=np.float64),
            quaternion=normalize_quaternion(np.asarray(quaternion, dtype=np.float64)),
            angular_velocity=rotation.T @ np.asarray(angular, dtype=np.float64),
        )
