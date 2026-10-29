"""Quaternion helpers for the stub physics.

Convention: Hamilton product, ``q`` rotates body -> world, scalar-last
``(x, y, z, w)``.  This matches ``envs/interface.py`` and PyBullet.
"""

from __future__ import annotations

import numpy as np


def normalize_quaternion(q) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        raise ValueError("quaternion has zero norm")
    return q / norm


def quat_multiply(a, b) -> np.ndarray:
    ax, ay, az, aw = np.asarray(a, dtype=np.float64)
    bx, by, bz, bw = np.asarray(b, dtype=np.float64)
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


def quat_rotate(q, v) -> np.ndarray:
    """Rotate vector ``v`` from body to world."""
    q = normalize_quaternion(q)
    u = q[:3]
    w = q[3]
    v = np.asarray(v, dtype=np.float64)
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def quat_derivative(q, omega) -> np.ndarray:
    """q_dot = 0.5 * q ⊗ (omega, 0), with omega in the body frame."""
    omega_q = np.array([omega[0], omega[1], omega[2], 0.0], dtype=np.float64)
    return 0.5 * quat_multiply(q, omega_q)


def quat_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """ZYX (yaw-pitch-roll) Euler angles -> quaternion."""
    cr, sr = np.cos(roll / 2.0), np.sin(roll / 2.0)
    cp, sp = np.cos(pitch / 2.0), np.sin(pitch / 2.0)
    cy, sy = np.cos(yaw / 2.0), np.sin(yaw / 2.0)
    return np.array(
        [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ]
    )
