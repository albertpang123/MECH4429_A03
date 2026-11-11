#!/usr/bin/env python3
"""Three implementations of the same task, side by side, in one table.

    python tools/crosscheck_gpd.py
    python tools/crosscheck_gpd.py --config configs/g4_fixed_rotor1.yaml --json out.json

What is compared
----------------
* ``stub``     -- ``env.impl: legacy``, ``dynamics.backend: numpy`` (the hand-written
                  6-DOF model that produced the G0-G4 numbers)
* ``pybullet`` -- ``env.impl: legacy``, ``dynamics.backend: pybullet`` (the same model
                  integrated by the engine, ``ground=False`` so the floor does not
                  introduce a difference the stub cannot express)
* ``gpd``      -- ``env.impl: gpd`` (gym-pybullet-drones ``BaseAviary`` + ``cf2x.urdf``)

The three are **not** the same vehicle, and this tool does not pretend otherwise:
the first two use ``dynamics.*`` from the config (the placeholder airframe), the
third uses the published ``cf2x`` URDF. What can be compared, and what this
prints, are the properties that must be internally consistent in each one:

1. the vehicle each implementation believes it is flying (mass, arm, hover
   command, thrust-to-weight, force model);
2. **hover**: at that implementation's own hover command, from the same start,
   does altitude hold for 1 s? A wrong force mapping or a wrong rotor order shows
   up here immediately.
3. **thrust response**: command ``hover + 0.1`` for 0.13 s; the measured vertical
   velocity gain is compared with the one predicted from that implementation's
   own ``applied_thrusts()``. A large residual means the force it *reports* is
   not the force it *applies* -- a real bug, in whichever column it appears.
4. **cost**: milliseconds per control step, measured, because gpd steps the
   engine ``physics_hz/control_hz`` times per action.

Exit code 0 = every implementation is self-consistent. Rows that disagree with
their own prediction are flagged.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

# Windows: torch/numpy OpenMP duplication. See tests/conftest.py.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.config import load_config  # noqa: E402
from envs.env_factory import make_env  # noqa: E402

GRAVITY = 9.81
HOVER_STEPS = 60          # 1 s at 60 Hz
STEP_ALLOWANCE = 0.05     # 5 % relative residual is "the same statement"
TIMING_STEPS = 120


def _config(path: str, impl: str, backend: str | None = None) -> dict:
    """A config with jitter removed, so all three start from the same state."""
    config = copy.deepcopy(load_config(ROOT / path))
    config["env"]["impl"] = impl
    config["reset"] = {
        "position_jitter_m": 0.0,
        "velocity_jitter_mps": 0.0,
        "tilt_jitter_deg": 0.0,
        "yaw_jitter_deg": 0.0,
    }
    if backend is not None:
        config["dynamics"]["backend"] = backend
    return config


def _vehicle(env, label: str) -> dict:
    return {
        "label": label,
        "mass_kg": float(env.mass_kg),
        "arm_m": float(np.max(np.linalg.norm(env.rotor_positions_body_m, axis=1))),
        "hover_command": float(env.hover_action),
        "thrust_max_per_rotor_n": float(env.thrust_max_per_rotor_n),
        "weight_n": float(env.mass_kg * GRAVITY),
    }


def _measure(env, label: str) -> dict:
    row = _vehicle(env, label)

    # --- 1. hover: hold altitude at this implementation's own hover command ----
    obs, info = env.reset(seed=0)
    z0 = float(info["position"][2])
    hover = np.full(4, env.hover_action)
    for _ in range(HOVER_STEPS):
        _, _, terminated, truncated, info = env.step(hover)
        if terminated or truncated:
            break
    row["hover_dz_m"] = float(env.state.position[2] - z0)
    row["hover_ended"] = str(info["done_reason"] or "")
    row["hover_tilt_deg"] = float(np.rad2deg(
        np.arccos(np.clip(abs(env.state.quaternion[3]), -1.0, 1.0)) * 2.0))

    # --- 2. thrust response: reported force vs observed acceleration ----------
    env.reset(seed=0)
    command = np.clip(hover + 0.1, 0.0, 1.0)
    dt = float(env.control_dt)
    predicted = (float(np.sum(env.applied_thrusts(command))) / row["mass_kg"]
                 - GRAVITY) * dt
    vz0 = float(env.state.velocity[2])
    env.step(command)
    row["dvz_measured_mps"] = float(env.state.velocity[2] - vz0)
    row["dvz_predicted_mps"] = float(predicted)
    residual = row["dvz_measured_mps"] - row["dvz_predicted_mps"]
    row["dvz_residual_mps"] = float(residual)
    row["dvz_residual_relative"] = (
        float(abs(residual) / abs(predicted)) if predicted else float("inf"))

    # --- 3. cost --------------------------------------------------------------
    start = time.perf_counter()
    for _ in range(TIMING_STEPS):
        env.step(hover)
        if env.state.position[2] < 0.0:      # keep it off the floor while timing
            env.reset(seed=0)
    row["ms_per_step"] = (time.perf_counter() - start) * 1000.0 / TIMING_STEPS
    return row


def _print_table(rows: list[dict]) -> None:
    def line(label: str, key: str, fmt: str) -> None:
        cells = " ".join(f"{fmt.format(row[key]):>14}" for row in rows)
        print(f"  {label:<30}{cells}")

    header = " ".join(f"{row['label']:>14}" for row in rows)
    print(f"  {'':<30}{header}")
    print("  " + "-" * (30 + 15 * len(rows)))
    line("mass (kg)", "mass_kg", "{:.4f}")
    line("arm length (m)", "arm_m", "{:.4f}")
    line("hover command", "hover_command", "{:.4f}")
    line("max thrust / rotor (N)", "thrust_max_per_rotor_n", "{:.4f}")
    line("weight (N)", "weight_n", "{:.4f}")
    line("thrust / weight", "tw_ratio", "{:.3f}")

    print()
    line("hover drift after 1 s (m)", "hover_dz_m", "{:+.4f}")
    line("hover tilt after 1 s (deg)", "hover_tilt_deg", "{:.2f}")
    print()
    line("dv_z measured (m/s)", "dvz_measured_mps", "{:+.5f}")
    line("dv_z predicted (m/s)", "dvz_predicted_mps", "{:+.5f}")
    line("residual (m/s)", "dvz_residual_mps", "{:+.5f}")
    line("residual / predicted", "dvz_residual_relative", "{:.2%}")
    print()
    line("cost (ms / control step)", "ms_per_step", "{:.3f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/g3_hover.yaml")
    parser.add_argument("--json", default=None, help="also write the rows here")
    args = parser.parse_args()

    specs = [
        ("stub", dict(impl="legacy", backend="numpy")),
        ("pybullet", dict(impl="legacy", backend="pybullet")),
        ("gpd", dict(impl="gpd", backend=None)),
    ]

    rows = []
    problems = []
    for label, spec in specs:
        config = _config(args.config, spec["impl"], spec["backend"])
        try:
            env = make_env(config)
        except Exception as exc:                      # noqa: BLE001 - report, don't crash
            print(f"[{label}] could not build: {exc.__class__.__name__}: {exc}")
            problems.append(f"{label}: not runnable")
            continue
        try:
            row = _measure(env, label)
        finally:
            env.close()
        row["tw_ratio"] = (row["thrust_max_per_rotor_n"] * 4.0
                           / row["weight_n"] if row["weight_n"] else float("nan"))
        rows.append(row)
        if row["dvz_residual_relative"] > STEP_ALLOWANCE:
            problems.append(
                f"{label}: thrust response is off by "
                f"{row['dvz_residual_relative']:.1%} from its own prediction")
        if abs(row["hover_dz_m"]) > 0.05:
            problems.append(f"{label}: hover drifted {row['hover_dz_m']:+.3f} m in 1 s")

    if not rows:
        print("nothing ran")
        return 1

    print("\nSame task, three implementations "
          f"({args.config}, jitter removed, seed 0)\n")
    _print_table(rows)

    print("\n  Note: 'stub' and 'pybullet' fly the placeholder airframe in")
    print("  dynamics.*; 'gpd' flies the published cf2x URDF. Equal columns are a")
    print("  coincidence, not the check. The check is that each column agrees with")
    print("  its own prediction.")

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2)
        print(f"\n  written to {out}")

    print()
    if problems:
        for problem in problems:
            print(f"  FAIL  {problem}")
        return 1
    print("  ok    every implementation applies the force it reports")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
