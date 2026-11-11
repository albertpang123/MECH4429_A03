#!/usr/bin/env python3
"""What does a policy actually do when a known rotor stops?

Trained on a *fixed* fault (``faults.mode: fixed``), the policy has one job: after
rotor ``k`` stops, decide what the other three do. This script reads that decision
out of the checkpoint instead of describing it in words.

For every evaluation episode it records the commanded motor signal per control step
and splits the episode at the failure instant:

* **before** -- the last 0.5 s before failure (the nominal trim the policy learned),
* **after**  -- from 0.5 s after failure to the end (the post-fault response).

and reports the mean command on each rotor in both windows, in normalised units and
in newtons, together with the attitude the vehicle ends up holding. Outputs go to
``results/<run>/fault_response.{json,csv}``.

    python -m eval.fault_response --checkpoint checkpoints/g4_rotor1_fixed

Numbers here are reproducible: the policy is deterministic, the episode seeds are
the reserved evaluation seeds, and the config is the one stored in ``meta.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

# Windows: two OpenMP runtimes in one process abort at the next numpy.linalg call.
# Must precede the heavy imports; see tests/conftest.py. No effect on results.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.env_factory import make_env  # noqa: E402


def _resolve_checkpoint(value: str) -> Path:
    checkpoint = Path(value)
    if checkpoint.is_dir():
        checkpoint = checkpoint / "model.zip"
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    return checkpoint


def _rpy(quaternion: np.ndarray) -> tuple[float, float, float]:
    """Roll, pitch, yaw in radians from a scalar-last Hamilton quaternion."""
    x, y, z, w = quaternion
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return float(roll), float(pitch), float(yaw)


def _mean_or_nan(stack: list[np.ndarray]) -> np.ndarray:
    return np.mean(np.asarray(stack), axis=0) if stack else np.full(4, np.nan)


def run(checkpoint: Path, episodes: int | None, seed_start: int | None, out: Path | None) -> int:
    run_dir = checkpoint.parent
    with open(run_dir / "meta.json", "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    config = meta["config"]

    fault_cfg = config.get("faults", {})
    if fault_cfg.get("mode") != "fixed":
        raise SystemExit(
            "fault_response.py is for `faults.mode: fixed` runs; this checkpoint used "
            f"{fault_cfg.get('mode')!r}. For `random` there is no single answer to report."
        )
    fixed = fault_cfg["fixed"]
    failed_rotor = int(fixed["rotor_id"])

    env = make_env(config)          # whatever the checkpoint trained on
    model = PPO.load(checkpoint, env=env, device="cpu")
    model.policy.set_training_mode(False)
    torch.set_num_threads(1)

    eval_cfg = config.get("evaluation", {})
    n_episodes = int(episodes if episodes is not None else eval_cfg.get("episodes", 10))
    first_seed = int(seed_start if seed_start is not None else eval_cfg.get("seed_start", 100000))
    dt = env.control_dt
    hover = float(env.hover_action)

    rows: list[dict] = []
    pre_commands: list[np.ndarray] = []
    post_commands: list[np.ndarray] = []

    for episode in range(n_episodes):
        seed = first_seed + episode
        obs, info = env.reset(seed=seed)
        t_fail = float(env.scenario.failure_time_s)

        before: list[np.ndarray] = []
        after: list[np.ndarray] = []
        done = terminated = truncated = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            command = np.clip(np.asarray(action, dtype=float).reshape(-1),
                              env.action_space.low, env.action_space.high)
            obs, _, terminated, truncated, info = env.step(action)
            t_s = float(info["step"]) * dt
            if t_fail - 0.5 <= t_s < t_fail - 1e-9:
                before.append(command)
            elif t_s >= t_fail + 0.5:
                after.append(command)
            done = bool(terminated or truncated)

        mean_before = _mean_or_nan(before)
        mean_after = _mean_or_nan(after)
        state = env.state
        roll, pitch, _ = _rpy(state.quaternion)
        survived = bool(truncated)  # reached the time limit

        row = {
            "episode": episode,
            "seed": seed,
            "survived": survived,
            "done_reason": info["done_reason"],
            "length": int(info["step"]),
            **{f"u_before_{i}": float(mean_before[i]) for i in range(4)},
            **{f"u_after_{i}": float(mean_after[i]) for i in range(4)},
            "tilt_end_deg": float(np.rad2deg(np.hypot(roll, pitch))),
            "z_end_m": float(state.position[2]),
            "position_error_end_m": float(info["position_error_m"]),
        }
        rows.append(row)
        # Averaged over every episode that reached the window, not only survivors:
        # when the policy fails, "what did it command while failing" is the whole
        # point of looking. Survival is reported separately.
        if len(before):
            pre_commands.append(mean_before)
        if len(after):
            post_commands.append(mean_after)

        print(f"  episode {episode:2d} seed {seed}: "
              f"{'survived' if survived else info['done_reason']:<12} "
              f"before {np.round(mean_before, 3)}  after {np.round(mean_after, 3)}")

    mean_pre = _mean_or_nan(pre_commands)
    mean_post = _mean_or_nan(post_commands)
    # Through the environment, not through its dynamics object: the two
    # implementations have different force models (the stub's placeholder is
    # linear in the command, gpd's cf2x is KF*rpm^2), and this report has to be
    # right for whichever one the checkpoint was trained on.
    thrust_pre = env.applied_thrusts(mean_pre)
    thrust_post = env.applied_thrusts(mean_post)
    weight = env.mass_kg * 9.81
    survivors = sum(int(r["survived"]) for r in rows)

    summary = {
        "checkpoint": str(checkpoint),
        "episodes": n_episodes,
        "survived": survivors,
        "episodes_with_after_window": len(post_commands),
        "failed_rotor": failed_rotor,
        "failure_time_s": float(fixed["failure_time_s"]),
        "hover_command": hover,
        "weight_n": float(weight),
        "u_before_mean": mean_pre.tolist(),
        "u_after_mean": mean_post.tolist(),
        "thrust_before_n": thrust_pre.tolist(),
        "thrust_after_n": thrust_post.tolist(),
        "thrust_after_total_n": float(np.sum(thrust_post)),
        "remaining_share_of_weight": float(np.sum(thrust_post) / weight) if weight else None,
        "tilt_end_deg_mean": float(np.mean([r["tilt_end_deg"] for r in rows])),
        "position_error_end_m_mean": float(np.mean([r["position_error_end_m"] for r in rows])),
    }

    out_dir = out if out is not None else ROOT / "results" / run_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "fault_response.json", "w", encoding="utf-8") as handle:
        json.dump({**summary, "episodes_rows": rows}, handle, indent=2)
    with open(out_dir / "fault_response.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== what the policy commands after rotor "
          f"{failed_rotor} stops at {fixed['failure_time_s']} s "
          f"({survivors}/{n_episodes} episodes survived) ===")
    print("  rotor   before   after   change   after (N)")
    for i in range(4):
        tag = "  <- failed" if i == failed_rotor else ""
        print(f"    R{i}   {mean_pre[i]:6.3f}  {mean_post[i]:6.3f}  "
              f"{mean_post[i] - mean_pre[i]:+7.3f}  {thrust_post[i]:8.3f}{tag}")
    print(f"\n  hover command was {hover:.3f} on all four = {np.sum(env.applied_thrusts(np.full(4, hover))):.3f} N"
          f" for a {weight:.3f} N weight")
    print(f"  after the fault the three survivors sum to {np.sum(thrust_post):.3f} N "
          f"= {np.sum(thrust_post) / weight * 100:.1f} % of the weight")
    print(f"  mean final tilt {summary['tilt_end_deg_mean']:.1f} deg, "
          f"mean final position error {summary['position_error_end_m_mean']:.3f} m")
    env.close()
    print(f"\n  written to {out_dir / 'fault_response.json'} and .csv")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="model.zip or its run directory")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed-start", type=int, default=None)
    parser.add_argument("--out", default=None, help="output directory (default: results/<run>)")
    args = parser.parse_args()
    return run(_resolve_checkpoint(args.checkpoint), args.episodes, args.seed_start,
               Path(args.out) if args.out else None)


if __name__ == "__main__":
    raise SystemExit(main())
