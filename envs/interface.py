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
from typing import Deque, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Timing (docs/INTERFACE.md section 3)
# ---------------------------------------------------------------------------
# The rates themselves live in configs/base.yaml (env.physics_hz / control_hz).
# Nothing here caches them: a module-level copy is exactly how a config edit
# ends up silently ignored by half the pipeline.


# ---------------------------------------------------------------------------
# Frames and rotor layout (docs/INTERFACE.md sections 1-2)
# ---------------------------------------------------------------------------
FRAME_WORLD = "ENU (x East, y North, z Up), origin at take-off point"
FRAME_BODY = "FLU (x forward, y left, z Up)"
QUATERNION_LAYOUT = "Hamilton, body -> world, scalar-last (x, y, z, w)"

SPIN_CCW = +1  # seen from above, i.e. about +z
SPIN_CW = -1

# Structure, not geometry: index order, label and spin direction are part of the
# interface and must not be reordered.  The *positions* are vehicle data and
# come from configs/base.yaml: dynamics.rotor_positions_body_m, so a new airframe
# does not require a code change.
ROTOR_LAYOUT: Tuple[Tuple[str, int], ...] = (
    ("front-right", SPIN_CW),
    ("front-left", SPIN_CCW),
    ("rear-left", SPIN_CW),
    ("rear-right", SPIN_CCW),
)

N_ROTORS = len(ROTOR_LAYOUT)


@dataclass(frozen=True)
class Rotor:
    """One rotor in the body frame, at an index fixed by ``ROTOR_LAYOUT``."""

    index: int
    x_m: float
    y_m: float
    spin: int  # SPIN_CCW (+1) or SPIN_CW (-1)
    label: str

    @property
    def position_m(self) -> np.ndarray:
        return np.array([self.x_m, self.y_m], dtype=np.float64)


def nominal_x_positions(arm_length_m: float) -> np.ndarray:
    """Positions for a symmetric X layout, ``(N_ROTORS, 2)``, body frame.

    ``arm_length_m`` is the CoM-to-rotor distance, so each rotor sits at
    ``(±L/√2, ±L/√2)``.  This is a helper for writing a config; the config
    itself always stores explicit positions.
    """
    if arm_length_m <= 0.0:
        raise ValueError("arm_length_m must be positive")
    offset = float(arm_length_m) / np.sqrt(2.0)
    return np.array(
        [
            [+offset, -offset],  # 0 front-right
            [+offset, +offset],  # 1 front-left
            [-offset, +offset],  # 2 rear-left
            [-offset, -offset],  # 3 rear-right
        ],
        dtype=np.float64,
    )


def build_rotors(rotor_positions_body_m: Sequence[Sequence[float]]) -> Tuple[Rotor, ...]:
    """Pair the configured positions with the fixed index/label/spin order."""
    positions = np.asarray(rotor_positions_body_m, dtype=np.float64)
    if positions.shape != (N_ROTORS, 2):
        raise ValueError(
            f"rotor positions must have shape ({N_ROTORS}, 2) [x, y] in metres, "
            f"got {positions.shape}"
        )
    if not np.all(np.isfinite(positions)):
        raise ValueError("rotor positions must be finite")
    return tuple(
        Rotor(index, float(x), float(y), ROTOR_LAYOUT[index][1], ROTOR_LAYOUT[index][0])
        for index, (x, y) in enumerate(positions)
    )

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
# Bounds are structural: a "normalised motor command" is defined as [0, 1].
# Changing them changes the interface, not a tunable, so they are not in config.
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


def observation_dim(history_len: int) -> int:
    """Flat observation size for a given history length."""
    if int(history_len) < 1:
        raise ValueError("history_len must be >= 1")
    return OBS_FRAME_DIM * int(history_len)


def frame_scales(scale_map: Mapping[str, float]) -> np.ndarray:
    """Expand ``observation.scales`` to one positive scale per observation value.

    Lives here rather than in the environment so the buffer and the env cannot
    disagree about the field order.
    """
    values = []
    for name, dim in OBS_FRAME_FIELDS:
        if name not in scale_map:
            raise KeyError(f"observation.scales is missing '{name}'")
        scale = float(scale_map[name])
        if scale <= 0.0:
            raise ValueError(f"observation.scales.{name} must be positive")
        values.extend([scale] * dim)
    return np.asarray(values, dtype=np.float64)


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

    def failure_step(self, control_hz: float) -> Optional[int]:
        """Control step at which the fault takes effect, or None.

        The control rate is passed in rather than read from a module constant:
        it is an ``env.control_hz`` setting, and a cached copy here would drift
        out of sync the moment someone changed the config.
        """
        if self.failure_time_s is None:
            return None
        return int(np.floor(self.failure_time_s * float(control_hz) + 0.5))


def quantise_to_control_grid(t_s: float, control_hz: float) -> float:
    """Snap a time to the nearest control step.

    The environment quantises the sampled failure time, so the value recorded in
    the scenario is the step the fault actually takes effect on — no off-by-one
    between the log and the physics.
    """
    if control_hz <= 0.0:
        raise ValueError("control_hz must be positive")
    step = int(np.floor(float(t_s) * float(control_hz) + 0.5))
    return step / float(control_hz)


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
    ``history_len`` times, so the array always has shape ``(dim,)`` and there is
    no artificial zero-padding transient.

    Usage::

        buf = ObservationBuffer.from_config(config)
        obs = buf.reset(state)            # after reset
        obs = buf.push(state, u_prev)     # once per control step
    """

    def __init__(
        self,
        target_position: Sequence[float],
        history_len: int,
        normalize: bool = True,
        clip: float = 10.0,
        scales: Optional[Sequence[float]] = None,
    ) -> None:
        if history_len < 1:
            raise ValueError("history_len must be >= 1")
        self.target_position = _as_vector(target_position, 3, "target_position")
        self.history_len = int(history_len)
        self.normalize = bool(normalize)
        self.clip = float(clip)
        if scales is None:
            # No normalisation table baked in here: the scales are config, and a
            # default copy is how two sources of truth appear.
            self.scales = np.ones(OBS_FRAME_DIM, dtype=np.float64)
        else:
            self.scales = _as_vector(scales, OBS_FRAME_DIM, "scales")
            if np.any(self.scales <= 0.0):
                raise ValueError("scales must be positive")
        self._frames: Deque[np.ndarray] = deque(maxlen=self.history_len)

    @property
    def dim(self) -> int:
        return observation_dim(self.history_len)

    @classmethod
    def from_config(
        cls,
        config: Mapping,
        target_position: Optional[Sequence[float]] = None,
    ) -> "ObservationBuffer":
        """Build from a resolved config (see :func:`envs.config.load_config`)."""
        obs_cfg = config["observation"]
        if target_position is None:
            target_position = config["env"]["target_position"]
        return cls(
            target_position=target_position,
            history_len=int(obs_cfg["history_len"]),
            normalize=bool(obs_cfg["normalize"]),
            clip=float(obs_cfg["clip"]),
            scales=frame_scales(obs_cfg["scales"]),
        )

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
