"""Gate 0 interface proposal for the quadrotor fault-tolerance project.

This module is the shared boundary between the dynamics module (Andy), the
fault-injection module (Joe), the Gymnasium environment and the training
pipeline (Albert).  It fixes *structure* only -- physical parameters live in
``configs/base.yaml``.

Status: draft (2026-10-25).  See ``docs/INTERFACE.md``; the defaults are open
for comment until the 28 Oct meeting.

Nothing in the observation path may depend on a :class:`Scenario`.  The policy
is deliberately fault-unaware; the simulator keeps the scenario for applying
the fault and for logging.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Timing (docs/INTERFACE.md section 3)
# ---------------------------------------------------------------------------
PHYSICS_HZ = 240.0
CONTROL_HZ = 60.0
CONTROL_DT = 1.0 / CONTROL_HZ

_steps_per_control = PHYSICS_HZ / CONTROL_HZ
if not float(_steps_per_control).is_integer():
    raise RuntimeError(
        f"physics_hz / control_hz must be an integer, got {_steps_per_control}"
    )
PHYSICS_STEPS_PER_CONTROL = int(_steps_per_control)


# ---------------------------------------------------------------------------
# Frames and rotor layout (docs/INTERFACE.md sections 1-2)
# ---------------------------------------------------------------------------
FRAME_WORLD = "ENU (x East, y North, z Up), origin at take-off point"
FRAME_BODY = "FLU (x forward, y left, z Up)"
QUATERNION_LAYOUT = "Hamilton, body -> world, scalar-last (x, y, z, w)"

SPIN_CCW = +1  # seen from above, i.e. about +z
SPIN_CW = -1

# Provisional arm length (Crazyflie-like).  Owned by Andy; changing it changes
# only this constant and configs/base.yaml: dynamics.arm_length_m.
ROTOR_ARM_M = 0.092
_ARM_OFFSET_M = ROTOR_ARM_M / np.sqrt(2.0)


@dataclass(frozen=True)
class Rotor:
    """One rotor in the X configuration, expressed in the body frame."""

    index: int
    x_m: float
    y_m: float
    spin: int  # SPIN_CCW (+1) or SPIN_CW (-1)
    label: str


ROTORS: Tuple[Rotor, ...] = (
    Rotor(0, +_ARM_OFFSET_M, -_ARM_OFFSET_M, SPIN_CW, "front-right"),
    Rotor(1, +_ARM_OFFSET_M, +_ARM_OFFSET_M, SPIN_CCW, "front-left"),
    Rotor(2, -_ARM_OFFSET_M, +_ARM_OFFSET_M, SPIN_CW, "rear-left"),
    Rotor(3, -_ARM_OFFSET_M, -_ARM_OFFSET_M, SPIN_CCW, "rear-right"),
)

N_ROTORS = len(ROTORS)

# Fields of Scenario that must never reach the policy observation.
HIDDEN_FROM_POLICY: Tuple[str, ...] = (
    "rotor_id",
    "failure_time_s",
    "effectiveness",
    "seed",
)


# ---------------------------------------------------------------------------
# Action space (docs/INTERFACE.md section 5)
# ---------------------------------------------------------------------------
ACTION_DIM = N_ROTORS
ACTION_LOW = 0.0
ACTION_HIGH = 1.0


# ---------------------------------------------------------------------------
# Observation layout (docs/INTERFACE.md section 4)
# ---------------------------------------------------------------------------
OBS_FRAME_FIELDS: Tuple[Tuple[str, int], ...] = (
    ("position_error", 3),
    ("linear_velocity", 3),
    ("attitude_quaternion", 4),
    ("angular_velocity", 3),
    ("last_motor_command", 4),
)
OBS_FRAME_DIM = sum(dim for _, dim in OBS_FRAME_FIELDS)  # 17

OBS_HISTORY_LEN = 10
OBS_DIM = OBS_FRAME_DIM * OBS_HISTORY_LEN  # 170
OBS_CLIP = 10.0

# Fixed normalisation scales, one per field (see configs/base.yaml).
OBS_FIELD_SCALES: Dict[str, float] = {
    "position_error": 1.0,       # m
    "linear_velocity": 1.0,      # m/s
    "attitude_quaternion": 1.0,  # unit quaternion
    "angular_velocity": 10.0,    # rad/s
    "last_motor_command": 1.0,   # normalised [0, 1]
}
OBS_SCALES = np.concatenate(
    [np.full(dim, OBS_FIELD_SCALES[name], dtype=np.float64)
     for name, dim in OBS_FRAME_FIELDS]
)


def observation_slices() -> Dict[str, slice]:
    """Return the slice of one observation frame occupied by each field."""
    out: Dict[str, slice] = {}
    start = 0
    for name, dim in OBS_FRAME_FIELDS:
        out[name] = slice(start, start + dim)
        start += dim
    return out


def _as_vector(value, size: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {arr.shape}")
    return arr.copy()


def canonicalize_quaternion(quaternion) -> np.ndarray:
    """Normalise to unit length and choose the sign with w >= 0.

    q and -q are the same rotation; canonicalising removes a spurious
    discontinuity from the observation.
    """
    q = _as_vector(quaternion, 4, "quaternion")
    norm = float(np.linalg.norm(q))
    if norm < 1e-8:
        raise ValueError("quaternion has zero norm")
    q = q / norm
    if q[3] < 0.0:
        q = -q
    return q


@dataclass
class VehicleState:
    """Minimal state shared by dynamics, fault module and environment.

    Frames: position and velocity in world ENU; quaternion body -> world,
    scalar-last; angular velocity in the body frame.
    """

    position: np.ndarray
    velocity: np.ndarray
    quaternion: np.ndarray
    angular_velocity: np.ndarray

    def __post_init__(self) -> None:
        self.position = _as_vector(self.position, 3, "position")
        self.velocity = _as_vector(self.velocity, 3, "velocity")
        self.quaternion = _as_vector(self.quaternion, 4, "quaternion")
        self.angular_velocity = _as_vector(self.angular_velocity, 3, "angular_velocity")


@dataclass(frozen=True)
class Scenario:
    """Hidden episode description (Joe's module).

    This object is used to apply the fault and to score/log the episode.  It
    must never be passed to the observation builder.
    """

    rotor_id: Optional[int] = None
    failure_time_s: Optional[float] = None
    effectiveness: float = 0.0
    initial_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    initial_velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    initial_quaternion: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    seed: int = 0

    def __post_init__(self) -> None:
        if (self.rotor_id is None) != (self.failure_time_s is None):
            raise ValueError("rotor_id and failure_time_s must both be set or both be None")
        if self.rotor_id is not None and self.rotor_id not in range(N_ROTORS):
            raise ValueError(f"rotor_id must be in 0..{N_ROTORS - 1}, got {self.rotor_id}")
        if self.failure_time_s is not None and self.failure_time_s < 0.0:
            raise ValueError("failure_time_s must be >= 0")
        if not 0.0 <= self.effectiveness <= 1.0:
            raise ValueError("effectiveness must be in [0, 1]")

    @property
    def is_nominal(self) -> bool:
        return self.rotor_id is None

    @property
    def failure_step(self) -> Optional[int]:
        """Control step at which the fault takes effect, or None.

        The failure time is quantised to the control grid so that a scenario is
        reproducible regardless of physics-step ordering.
        """
        if self.failure_time_s is None:
            return None
        return int(np.floor(self.failure_time_s * CONTROL_HZ + 0.5))


def observation_frame(
    state: VehicleState,
    target_position: Sequence[float],
    last_motor_command: Sequence[float],
) -> np.ndarray:
    """Build one (17,) observation frame.

    ``last_motor_command`` is u_{t-1}: the command applied during the previous
    control step, not the action about to be applied.
    """
    target = _as_vector(target_position, 3, "target_position")
    command = _as_vector(last_motor_command, ACTION_DIM, "last_motor_command")
    if np.any(command < ACTION_LOW - 1e-9) or np.any(command > ACTION_HIGH + 1e-9):
        raise ValueError("last_motor_command must lie in [0, 1]")
    command = np.clip(command, ACTION_LOW, ACTION_HIGH)

    return np.concatenate(
        (
            target - state.position,
            state.velocity,
            canonicalize_quaternion(state.quaternion),
            state.angular_velocity,
            command,
        )
    )


class ObservationBuffer:
    """Stack the last ``history_len`` frames, oldest first.

    On reset the buffer is filled with the initial frame repeated
    ``history_len`` times, so the array always has shape (OBS_DIM,) and there
    is no artificial zero-padding transient.

    Usage::

        buf = ObservationBuffer(target_position=(0, 0, 1))
        buf.reset(state)
        obs = buf.as_array()              # after reset
        buf.push(state, u_prev)           # once per control step
        obs = buf.as_array()
    """

    def __init__(
        self,
        target_position: Sequence[float],
        history_len: int = OBS_HISTORY_LEN,
        normalize: bool = True,
        clip: float = OBS_CLIP,
        scales: Optional[Sequence[float]] = None,
    ) -> None:
        if history_len < 1:
            raise ValueError("history_len must be >= 1")
        self.target_position = _as_vector(target_position, 3, "target_position")
        self.history_len = int(history_len)
        self.normalize = bool(normalize)
        self.clip = float(clip)
        if scales is None:
            self.scales = OBS_SCALES.copy()
        else:
            self.scales = _as_vector(scales, OBS_FRAME_DIM, "scales")
            if np.any(self.scales <= 0.0):
                raise ValueError("scales must be positive")
        self._frames: Deque[np.ndarray] = deque(maxlen=self.history_len)

    def reset(
        self,
        state: VehicleState,
        last_motor_command: Optional[Sequence[float]] = None,
    ) -> np.ndarray:
        if last_motor_command is None:
            last_motor_command = np.zeros(ACTION_DIM, dtype=np.float64)
        frame = observation_frame(state, self.target_position, last_motor_command)
        self._frames.clear()
        for _ in range(self.history_len):
            self._frames.append(frame.copy())
        return self.as_array()

    def push(
        self,
        state: VehicleState,
        last_motor_command: Sequence[float],
    ) -> np.ndarray:
        if not self._frames:
            raise RuntimeError("ObservationBuffer.push() called before reset()")
        self._frames.append(
            observation_frame(state, self.target_position, last_motor_command)
        )
        return self.as_array()

    def as_array(self, normalize: Optional[bool] = None) -> np.ndarray:
        if len(self._frames) != self.history_len:
            raise RuntimeError("ObservationBuffer is not full; call reset() first")
        use_normalize = self.normalize if normalize is None else bool(normalize)
        out = np.concatenate(tuple(self._frames))
        if use_normalize:
            out = np.clip(
                out / np.tile(self.scales, self.history_len), -self.clip, self.clip
            )
        return out.astype(np.float32)
