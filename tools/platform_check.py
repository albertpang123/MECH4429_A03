#!/usr/bin/env python3
"""Can this machine run the project? Prints one report you can paste into chat.

Run it after creating the environment:

    python tools/platform_check.py          # or: python run.py --run doctor

Checks, in order:

1. the versions of everything the project depends on,
2. the NumPy dynamics backend: 10 steps of hover command, state stays finite,
3. the PyBullet backend: same, plus whether it is installed at all,
4. **cross-machine determinism**: 120 control steps of a fixed, published action
   sequence, printed to full precision. Two machines that agree here produce the
   same training data; two that disagree mean a dependency differs.

Exit code 0 = everything needed for training works. The PyBullet section failing
only matters if you plan to use the 3D replay or `dynamics.backend: pybullet`.
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REQUIRED = (
    ("numpy", None),
    ("gymnasium", None),
    ("stable_baselines3", None),
    ("torch", None),
    ("pandas", None),
    ("matplotlib", None),
    ("yaml", "pyyaml"),
)
OPTIONAL = (("pybullet", None),)

#: The published action sequence for the determinism check. Do not change it: its
#: output is what other machines compare against.
CHECK_STEPS = 120
CHECK_CONFIG = "configs/g3_hover.yaml"


def _version(name: str) -> str:
    module = __import__(name)
    version = getattr(module, "__version__", None)
    if version:
        return str(version)
    # pybullet has no __version__; the installed distribution does.
    from importlib.metadata import version as dist_version

    return dist_version(name)


def _section(title: str) -> None:
    print(f"\n--- {title} " + "-" * max(0, 60 - len(title)))


def _versions(pairs, required: bool) -> bool:
    ok = True
    for name, package in pairs:
        try:
            print(f"  {name:<20} {_version(name)}")
        except Exception as exc:                      # noqa: BLE001 - report, don't crash
            label = "MISSING" if required else "not installed"
            print(f"  {name:<20} {label} ({exc.__class__.__name__})")
            ok = ok and not required
    return ok


def _hover_steps(backend: str) -> str | None:
    from envs.config import load_config
    from envs.quadrotor_env import QuadrotorEnv

    config = load_config(ROOT / CHECK_CONFIG)
    config["dynamics"]["backend"] = backend
    env = QuadrotorEnv(config)
    try:
        _, info = env.reset(seed=0)
        start = info["position"].copy()
        command = env.dynamics.hover_command
        for _ in range(10):
            _, _, terminated, truncated, info = env.step(command)
            if terminated or truncated:
                return f"episode ended early: {info['done_reason']}"
        state = env.state
        # The reset jitters position and velocity, so hover holds *near* the start,
        # not exactly on it; 0.1 s at the 0.1 m/s velocity jitter is 1 cm.
        if abs(state.position[2] - start[2]) > 0.05:
            return f"height moved {state.position[2] - start[2]:+.3f} m in 0.1 s"
        if not all(abs(x) < 1e9 for x in state.position):
            return "non-finite state"
        return None
    finally:
        env.close()


def _determinism_digest() -> str:
    """A fixed, published rollout -- the thing two machines must agree on."""
    import numpy as np

    from envs.config import load_config
    from envs.quadrotor_env import QuadrotorEnv

    config = load_config(ROOT / CHECK_CONFIG)
    env = QuadrotorEnv(config)
    try:
        env.reset(seed=0)
        hover = env.dynamics.hover_command
        for step in range(CHECK_STEPS):
            # Same schedule on every machine: hover plus a small deterministic wobble.
            command = hover + 0.02 * np.sin(0.13 * step + np.arange(4))
            env.step(command)
        position = env.state.position
        velocity = env.state.velocity
        return (f"pos {position[0]:.17g} {position[1]:.17g} {position[2]:.17g}\n"
                f"      vel {velocity[0]:.17g} {velocity[1]:.17g} {velocity[2]:.17g}")
    finally:
        env.close()


def main() -> int:
    print("MECH4429 A03 platform check")
    print(f"  platform     {platform.platform()}")
    print(f"  machine      {platform.machine()}")
    print(f"  python       {sys.version.split()[0]}")
    print(f"  executable   {sys.executable}")
    print(f"  repository   {ROOT}")

    _section("dependencies (required)")
    required_ok = _versions(REQUIRED, required=True)
    _section("dependencies (optional)")
    _versions(OPTIONAL, required=False)

    _section("numpy dynamics backend")
    try:
        problem = _hover_steps("numpy")
    except Exception as exc:                          # noqa: BLE001
        print(f"  FAILED: {exc.__class__.__name__}: {exc}")
        problem = "exception"
    print("  ok" if problem is None else f"  FAILED: {problem}")

    _section("pybullet backend")
    try:
        problem = _hover_steps("pybullet")
    except Exception as exc:                          # noqa: BLE001
        print(f"  not usable: {exc.__class__.__name__}: {exc}")
        problem = "exception"
    print("  ok" if problem is None else f"  skipped/failed: {problem}")

    _section(f"determinism digest ({CHECK_STEPS} control steps)")
    digest = None
    try:
        digest = _determinism_digest()
        print("  " + digest)
        print("\n  Compare this digest with the other machine. Identical numbers = the")
        print("  same training data; different numbers = a dependency differs.")
    except Exception as exc:                          # noqa: BLE001
        print(f"  FAILED: {exc.__class__.__name__}: {exc}")

    _section("verdict")
    if required_ok and digest is not None:
        print("  training-ready")
        return 0
    print("  NOT training-ready -- see the FAILED lines above")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
