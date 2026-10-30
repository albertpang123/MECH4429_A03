"""G3 acceptance: train a tiny PPO, save, reload, and predict identically."""

import copy
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("stable_baselines3")
pytest.importorskip("gymnasium")

from stable_baselines3 import PPO

from envs.config import load_config
from envs.quadrotor_env import QuadrotorEnv

ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(ROOT / "configs" / "g3_hover.yaml")


def test_train_save_reload_predict(tmp_path):
    config = copy.deepcopy(CONFIG)
    config["run"]["n_envs"] = 1
    config["run"]["seed"] = 0
    config["ppo"].update(
        {
            "n_steps": 64,
            "batch_size": 64,
            "n_epochs": 1,
            "net_arch": [16, 16],
        }
    )
    env = QuadrotorEnv(config)
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=64,
        batch_size=64,
        n_epochs=1,
        gamma=0.99,
        gae_lambda=0.95,
        policy_kwargs={"net_arch": [16, 16]},
        seed=0,
        device="cpu",
        verbose=0,
    )
    model.learn(total_timesteps=128)

    checkpoint = tmp_path / "model"
    model.save(checkpoint)
    assert checkpoint.with_suffix(".zip").exists()

    obs, _ = env.reset(seed=999)
    action_before, _ = model.predict(obs, deterministic=True)

    loaded = PPO.load(checkpoint, env=env, device="cpu")
    action_after, _ = loaded.predict(obs, deterministic=True)
    assert np.allclose(action_before, action_after, atol=1e-6)
