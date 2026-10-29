#!/usr/bin/env python3
"""Gate 3 entry point: load a saved policy, evaluate it, no further learning.

Example:
    python -m eval.evaluate --checkpoint checkpoints/g3_stub_hover/model.zip
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.quadrotor_env import QuadrotorEnv  # noqa: E402
from eval.plotting import plot_evaluation  # noqa: E402


def resolve_checkpoint(path: str | Path) -> Path:
    checkpoint = Path(path)
    if checkpoint.is_dir():
        checkpoint = checkpoint / "model.zip"
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    return checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="model.zip or its run directory")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed-start", type=int, default=None)
    parser.add_argument("--out", default=None, help="output directory (default: results/<run>)")
    args = parser.parse_args()

    checkpoint = resolve_checkpoint(args.checkpoint)
    run_dir = checkpoint.parent
    with open(run_dir / "meta.json", "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    config = meta["config"]

    env = QuadrotorEnv(config)
    if int(env.observation_space.shape[0]) != int(meta["observation"]["dim"]):
        raise ValueError("checkpoint observation dimension does not match the environment")

    model = PPO.load(checkpoint, env=env, device="cpu")
    model.policy.set_training_mode(False)
    torch.set_num_threads(1)

    eval_cfg = config.get("evaluation", {})
    episodes = int(args.episodes if args.episodes is not None else eval_cfg.get("episodes", 20))
    seed_start = int(
        args.seed_start if args.seed_start is not None else eval_cfg.get("seed_start", 100000)
    )
    seed_end = seed_start + episodes - 1
    eval_range = config.get("seeds", {}).get("eval", [100000, 100999])
    if seed_start < int(eval_range[0]) or seed_end > int(eval_range[1]):
        raise ValueError(
            f"evaluation seeds {seed_start}..{seed_end} fall outside the reserved "
            f"evaluation range {eval_range}"
        )

    results_dir = Path(args.out) if args.out else ROOT / "results" / run_dir.name
    results_dir.mkdir(parents=True, exist_ok=True)

    trajectories = []
    rows = []
    for episode_index in range(episodes):
        seed = seed_start + episode_index
        obs, info = env.reset(seed=seed)
        position = info["position"].copy()
        times = [0.0]
        positions = [position]
        errors = [float(np.linalg.norm(env.target_position - position))]
        rewards = []
        clip_counts = []
        done = False
        terminated = truncated = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            rewards.append(float(reward))
            clip_counts.append(int(info["clip_count"]))
            times.append(float(info["step"]) * env.control_dt)
            positions.append(info["position"].copy())
            errors.append(float(info["position_error_m"]))
            done = bool(terminated or truncated)

        position_array = np.asarray(positions, dtype=float)
        error_array = np.asarray(errors, dtype=float)
        row = {
            "episode": episode_index,
            "seed": seed,
            "return": float(np.sum(rewards)),
            "length": int(len(rewards)),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "done_reason": info["done_reason"],
            "crashed": info["done_reason"] in ("crash", "non_finite"),
            "out_of_bounds": info["done_reason"] == "out_of_bounds",
            "timeout": bool(truncated),
            "fault_active": bool(info["fault_active"]),
            "final_position_error_m": float(error_array[-1]),
            "max_position_error_m": float(error_array.max()),
            "position_rmse_m": float(np.sqrt(np.mean(error_array**2))),
            "final_z_m": float(position_array[-1, 2]),
            "max_clip_count": int(max(clip_counts) if clip_counts else 0),
            "mean_clip_count": float(np.mean(clip_counts) if clip_counts else 0.0),
        }
        rows.append(row)
        trajectories.append(
            {
                "seed": seed,
                "time_s": times,
                "position": position_array,
                "target": env.target_position.copy(),
                "return": row["return"],
                "done_reason": row["done_reason"],
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "eval_episodes.csv", index=False)
    plot_evaluation(trajectories, results_dir / "eval_trajectory.png")

    summary = {
        "run_name": run_dir.name,
        "checkpoint": str(checkpoint),
        "episodes": episodes,
        "seed_start": seed_start,
        "seed_end": seed_end,
        "deterministic": True,
        "return_mean": float(df["return"].mean()),
        "return_std": float(df["return"].std(ddof=0)),
        "length_mean": float(df["length"].mean()),
        "final_position_error_mean_m": float(df["final_position_error_m"].mean()),
        "final_position_error_max_m": float(df["final_position_error_m"].max()),
        "max_position_error_m": float(df["max_position_error_m"].max()),
        "position_rmse_mean_m": float(df["position_rmse_m"].mean()),
        "crash_rate": float(df["crashed"].mean()),
        "out_of_bounds_rate": float(df["out_of_bounds"].mean()),
        "timeout_rate": float(df["timeout"].mean()),
        "all_survived": bool(not df["crashed"].any() and not df["out_of_bounds"].any()),
        "observation_normalisation": meta["observation"],
    }
    with open(results_dir / "eval.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
