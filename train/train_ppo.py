#!/usr/bin/env python3
"""Gate 3 entry point: train PPO, save a checkpoint, write the learning curve.

Example:
    python -m train.train_ppo --config configs/g3_hover.yaml
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.quadrotor_env import QuadrotorEnv  # noqa: E402
from eval.plotting import plot_learning_curve  # noqa: E402


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def unique_run_dir(base: Path, name: str) -> Path:
    candidate = base / name
    suffix = 1
    while candidate.exists():
        suffix += 1
        candidate = base / f"{name}-{suffix:03d}"
    candidate.mkdir(parents=True)
    return candidate


def make_env(rank: int, config: dict):
    def _init():
        local = copy.deepcopy(config)
        local["run"]["seed"] = int(config["run"]["seed"]) + rank
        return QuadrotorEnv(local)

    return _init


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.timesteps is not None:
        config["run"]["total_timesteps"] = int(args.timesteps)
    if args.n_envs is not None:
        config["run"]["n_envs"] = int(args.n_envs)
    if args.name is not None:
        config["run"]["name"] = args.name
    if args.device is not None:
        config["run"]["device"] = args.device

    seed = int(config["run"]["seed"])
    n_envs = int(config["run"]["n_envs"])
    total_timesteps = int(config["run"]["total_timesteps"])
    device = str(config["run"]["device"])

    set_random_seed(seed, using_cuda=device.startswith("cuda"))
    torch.set_num_threads(1)

    checkpoint_root = ROOT / "checkpoints"
    checkpoint_root.mkdir(exist_ok=True)
    run_dir = unique_run_dir(checkpoint_root, str(config["run"]["name"]))
    run_name = run_dir.name
    config["run"]["name"] = run_name
    results_dir = ROOT / "results" / run_name
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    log_dir = run_dir / "logs"
    log_dir.mkdir()
    # SB3 2.2+ loggers are per-model: the configured logger must be attached
    # with model.set_logger(), otherwise learn() installs its own default logger.
    sb3_logger = configure(folder=str(log_dir), format_strings=["stdout", "csv"])

    vec_env_class = DummyVecEnv if n_envs == 1 else SubprocVecEnv
    vec_env = vec_env_class([make_env(rank, config) for rank in range(n_envs)])
    # SB3 only adds a Monitor for a bare Gym env; without VecMonitor an
    # already-vectorised env never reports rollout/ep_rew_mean to the logger.
    vec_env = VecMonitor(vec_env, filename=str(log_dir / "episodes"))
    obs = vec_env.reset()
    expected_obs_dim = vec_env.observation_space.shape[0]
    if obs.shape != (n_envs, expected_obs_dim):
        raise RuntimeError(f"unexpected reset shape: {obs.shape}")

    ppo_cfg = config["ppo"]
    policy_kwargs = {"net_arch": list(ppo_cfg.get("net_arch", [64, 64]))}
    model = PPO(
        ppo_cfg["policy"],
        vec_env,
        learning_rate=float(ppo_cfg["learning_rate"]),
        n_steps=int(ppo_cfg["n_steps"]),
        batch_size=int(ppo_cfg["batch_size"]),
        n_epochs=int(ppo_cfg["n_epochs"]),
        gamma=float(ppo_cfg["gamma"]),
        gae_lambda=float(ppo_cfg["gae_lambda"]),
        clip_range=float(ppo_cfg["clip_range"]),
        ent_coef=float(ppo_cfg["ent_coef"]),
        vf_coef=float(ppo_cfg["vf_coef"]),
        max_grad_norm=float(ppo_cfg["max_grad_norm"]),
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=seed,
        device=device,
    )

    model.set_logger(sb3_logger)

    print(f"run: {run_name}")
    print(f"checkpoints: {run_dir}")
    print(f"obs shape: {vec_env.observation_space.shape}, action: {vec_env.action_space.shape}")
    print(f"training {total_timesteps} steps on {n_envs} env(s), device={device}")

    model.learn(total_timesteps=total_timesteps)
    model.save(run_dir / "model")
    sb3_logger.close()
    vec_env.close()

    progress = log_dir / "progress.csv"
    if progress.exists():
        shutil.copy2(progress, results_dir / "progress.csv")
        plot_learning_curve(progress, results_dir / "learning_curve.png")

    metadata = {
        "schema_version": int(config.get("schema_version", 1)),
        "run_name": run_name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_file": "model.zip",
        "total_timesteps": int(model.num_timesteps),
        "seed": seed,
        "n_envs": n_envs,
        "device": device,
        "observation": {
            "dim": int(vec_env.observation_space.shape[0]),
            "history_len": int(config["observation"]["history_len"]),
            "normalize": bool(config["observation"]["normalize"]),
            "clip": float(config["observation"]["clip"]),
            "scales": config["observation"]["scales"],
        },
        "action": {
            "dim": int(vec_env.action_space.shape[0]),
            "low": float(config["action"]["low"]),
            "high": float(config["action"]["high"]),
        },
        "target_position": config["env"]["target_position"],
        "versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "config": config,
    }
    with open(run_dir / "meta.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    summary = {
        "run_name": run_name,
        "checkpoint": str(run_dir / "model.zip"),
        "meta": str(run_dir / "meta.json"),
        "total_timesteps": int(model.num_timesteps),
        "results": str(results_dir),
    }
    with open(results_dir / "train_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
