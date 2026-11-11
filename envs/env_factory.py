"""Pick the environment implementation from the config.

    env:
      impl: gpd       # default -- the task on gym-pybullet-drones' BaseAviary
      impl: legacy    # the original envs.quadrotor_env, with our own physics

Why both exist
--------------
``legacy`` is not dead weight kept out of politeness. It is the reference the
migration is checked against:

* every number in the G3/G4 results so far came from it, and the crosscheck
  tool (``tools/crosscheck_gpd.py``) runs the *same* episode through both
  implementations and through gpd's own ``CtrlAviary``;
* if ``gpd`` ever disagrees with ``legacy`` in a way the vehicle model does not
  explain, having both runnable is what makes that a five-minute diagnosis
  instead of an archaeology project.

The two are not numerically equal and are not meant to be: ``legacy`` uses the
placeholder vehicle in ``configs/*.yaml`` (thrust-to-weight 1.63), ``gpd`` uses
the published ``cf2x`` URDF (2.25). Same task, same observation, same action
semantics, same fault injection -- different machine.

Never import ``envs.gpd_env`` at module scope here
--------------------------------------------------
It pulls in the vendored package and, at construction time, a PyBullet client.
Keeping that inside the branch means a machine without pybullet can still run
the ``legacy``/NumPy path (and the tests that cover it).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:  # pragma: no cover - typing only
    import gymnasium as gym

IMPLS = ("gpd", "legacy")
DEFAULT_IMPL = "gpd"


def impl_name(config: Mapping[str, Any]) -> str:
    """The implementation this config asks for, with the default applied."""
    env_cfg = config.get("env") or {}
    name = str(env_cfg.get("impl", DEFAULT_IMPL)).strip().lower()
    if name not in IMPLS:
        raise ValueError(f"env.impl must be one of {IMPLS}, got {name!r}")
    return name


def make_env(config: Mapping[str, Any], *, gui: bool = False) -> "gym.Env":
    """Build the environment named by ``env.impl``.

    ``gui`` only means something for ``gpd``; ``legacy`` renders through
    ``eval/replay.py`` and ignores it.
    """
    name = impl_name(config)
    if name == "gpd":
        from envs.gpd_env import GPDQuadrotorEnv

        return GPDQuadrotorEnv(dict(config), gui=gui)

    from envs.quadrotor_env import QuadrotorEnv

    return QuadrotorEnv(dict(config))
