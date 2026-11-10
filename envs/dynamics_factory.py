"""Pick a dynamics backend from the config.

The environment only ever talks to the four methods in ``envs.interface``, so
which integrator runs underneath is a one-line config choice:

    dynamics:
      backend: numpy      # default -- StubDynamics (hand-written 6-DOF Euler)
      backend: pybullet   # PyBullet engine, same model, real ground contact

Nothing else in the project has to know. The config digest already covers the
``dynamics`` section, so a checkpoint records which backend produced it.

Why a factory and not an import in ``quadrotor_env``
----------------------------------------------------
``PyBulletDynamics`` imports ``pybullet`` at construction time, not at module
import time. Keeping the import inside this module's branch means a NumPy-only
run never touches the engine (and a machine without pybullet can still train on
the stub).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Union

from envs.stub_dynamics import StubDynamics, StubParams

if TYPE_CHECKING:  # pragma: no cover - typing only
    from envs.pybullet_dynamics import PyBulletDynamics

    Dynamics = Union[StubDynamics, PyBulletDynamics]

BACKENDS = ("numpy", "pybullet")
DEFAULT_BACKEND = "numpy"


def backend_name(config: Mapping[str, Any]) -> str:
    """The backend this config asks for, with the default applied."""
    raw = config.get("dynamics") or {}
    name = str(raw.get("backend", DEFAULT_BACKEND)).strip().lower()
    if name not in BACKENDS:
        raise ValueError(
            f"dynamics.backend must be one of {BACKENDS}, got {name!r}"
        )
    return name


def make_dynamics(
    config: Mapping[str, Any],
    *,
    gui: bool = False,
    ground: bool = True,
) -> "Dynamics":
    """Build the dynamics object for ``config``.

    ``gui`` / ``ground`` only apply to the pybullet backend and are here for
    debugging and for cross-validation runs; ``ground=False`` reproduces the
    stub exactly (the stub has no floor).
    """
    name = backend_name(config)
    params = StubParams.from_config(dict(config))
    if name == "numpy":
        return StubDynamics(params)

    from envs.pybullet_dynamics import PyBulletDynamics

    return PyBulletDynamics(params, gui=gui, ground=ground)
