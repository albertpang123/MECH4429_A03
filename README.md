# MECH4429 Project A03 — Quadrotor Position Hold After Single-Rotor Failure

A reinforcement-learning controller that keeps a quadrotor near a set point after one rotor
fails, trained in simulation.

**Current state: Gate 3 — the RL pipeline runs end to end on stub dynamics.** The physics and
fault modules are placeholders written so the learning side could be built before Gates 1–2
were ready. **No number in this repo is a vehicle result.** See [docs/G3.md](docs/G3.md).

## Team and ownership

| Who | Gate | Scope |
|---|---|---|
| Andy | G1 | 6-DOF dynamics model, rotor thrust/torque, vehicle parameters |
| Joe | G2 | Fault model, scenario sampling |
| Albert | G3–G4 | Gymnasium environment, PPO training, evaluation, checkpoints |

Albert's machine is the reference platform (Ubuntu 22.04 dual boot): code that runs there is
the code we compare against. Everyone else should be able to reproduce the same numbers.

## Gate status

| Gate | Content | State |
|---|---|---|
| G0 | Interface: frames, rates, observation, action, reward, seeds | **Proposal circulating** — [docs/INTERFACE.md](docs/INTERFACE.md), still changeable |
| G1 | 6-DOF dynamics | Not started in this repo (`src/dynamics/` empty) |
| G2 | Single-rotor failure model | Not started in this repo (`src/faults/` empty) |
| G3 | RL pipeline, stub dynamics | **Done** — train → save → reload → evaluate works |
| G4 | Curriculum over failure time / rotor | Not started |
| G5 | Final evaluation and report | Not started |

## Repository layout

```
configs/            base.yaml (G0 spec) + one YAML per run, each extending base
envs/
  interface.py      shared types, observation layout, structural rotor order  <- start here
  config.py         config loading, inheritance, digest, validation
  quat.py           quaternion helpers
  quadrotor_env.py  Gymnasium task (observation, action, reward, termination)
  stub_dynamics.py  G1 placeholder: 6-DOF rigid body, NumPy
  stub_faults.py    G2 placeholder: masks one rotor after failure_time_s
train/train_ppo.py  PPO training, checkpoint + metadata + CSV logs
eval/               deterministic evaluation, metrics, plots
tests/              interface, stub physics, train/save/reload
docs/               INTERFACE.md (G0), G3.md (pipeline + results), rotor_layout.svg
src/dynamics/       Andy: replace stub_dynamics.py
src/faults/         Joe: replace stub_faults.py
results/            metrics and figures that are worth keeping (in git)
checkpoints/        weights (not in git)
```

## Quick start

Ubuntu 22.04, Python 3.10, conda env named `uav-g3`:

```bash
conda create -n uav-g3 python=3.10 -y
conda activate uav-g3
pip install numpy==1.26.4 gymnasium==0.29.1 stable-baselines3==2.2.1 \
            pandas==2.0.3 matplotlib==3.7.5 pyyaml pytest==7.4.4
# CPU-only is fine for everything in this repo. torch comes with stable-baselines3.
```

All commands below run **from the repository root**.

### Three things that will bite you

1. **`env -u PYTHONPATH`** — if ROS is installed on the same machine it exports a
   `python3.10` ROS path into `PYTHONPATH`, and the conda interpreter will then import ROS's
   NumPy instead of the env's. Drop the variable for every command.
2. **`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`** — a third-party pytest plugin installed in the base
   environment can abort collection. This flag stops auto-loading plugins; `tests/` only needs
   `pytest-*.py` from this repo.
3. **`CUDA_VISIBLE_DEVICES=""`** — the reported numbers were produced on CPU. Set it to keep
   your own GPU from silently changing them.

## Commands

```bash
# tests: interface + stub physics + train/save/reload
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 CUDA_VISIBLE_DEVICES="" \
  python -m pytest tests -q

# train (200k steps, ~200 s on CPU)
env -u PYTHONPATH CUDA_VISIBLE_DEVICES="" \
  python -m train.train_ppo --config configs/g3_hover.yaml

# evaluate a saved policy — no learning
env -u PYTHONPATH CUDA_VISIBLE_DEVICES="" \
  python -m eval.evaluate --checkpoint checkpoints/g3_stub_hover/model.zip
```

Overrides: `--timesteps`, `--n-envs`, `--name`, `--device` for training;
`--episodes`, `--seed-start`, `--out` for evaluation.

### Changing parameters

Every tunable value lives in `configs/base.yaml`: vehicle parameters, reward weights,
termination thresholds, observation scales, reset jitter. Run configs extend it and override
only what differs, so a shared value is edited in **one** place:

```yaml
# configs/my_run.yaml
extends: base.yaml
dynamics:
  mass_kg: 0.5
```

```bash
# show the fully merged config and its digest, without training
env -u PYTHONPATH /home/albertpang/miniconda3/envs/uav-g3/bin/python \
  -m train.train_ppo --config configs/g3_hover.yaml --print-config
```

Load configs with `envs.config.load_config`, not `yaml.safe_load` — a bare load returns the
child file alone and silently drops everything it inherits. Each run writes the **resolved**
config and its `config_sha256` into `meta.json`, so a result can always be traced back to the
values that produced it.

Structural values are deliberately *not* configurable: the rotor index/label/spin order, the
`[0, 1]` action bounds and the 17-field observation layout are the interface, and changing
them invalidates every existing checkpoint.

**GPU (opt-in).** Pass `--device cuda` and do **not** set `CUDA_VISIBLE_DEVICES=""`. On the
RTX 4080 Laptop this works through CUDA forward compatibility (compute capability 8.9,
driver 580.178.04 / CUDA 13.0) — a supported-enough combination for experiments, not an
officially supported one. For long-term stability install `torch==2.4.1+cu121` in a dedicated
env and run with `PYTHONNOUSERSITE=1`. Details in [docs/G3.md](docs/G3.md).

## The interface — start here

Any of the three of us will need these, so they are the first thing to read:

| Question | Answer |
|---|---|
| World frame | ENU, x East / y North / z Up, origin at take-off |
| Body frame | FLU, x forward / y left / z up |
| Attitude | Quaternion `(x, y, z, w)`, body → world, Hamilton, `w ≥ 0` |
| Physics / control rate | 240 Hz / 60 Hz (exactly 4 physics steps per action) |
| Episode | 10 s = 600 control steps |
| Rotor order | 0 front-right, 1 front-left, 2 rear-left, 3 rear-right (counter-clockwise from above) |
| Action | `(4,)` in `[0, 1]`, normalised motor command, not newtons |
| Observation | 17 values × 10 frames = `(170,)` `float32` |
| Seeds | train 0–999, evaluation 100000–100999 (never both) |

Full tables, reward terms and open questions: [docs/INTERFACE.md](docs/INTERFACE.md) and
`configs/base.yaml`. Shared types: `envs/interface.py`. Rotor geometry:
[docs/rotor_layout.svg](docs/rotor_layout.svg).

### The policy must not see the fault

Failed rotor ID, failure time, effectiveness, fault flag and scenario seed go into `info`
(logging only). The observation builder takes `(state, target, recent commands)` and nothing
else — `tests/test_interface.py` enforces this by inspecting the signatures.

## Results so far

Nominal hover, `configs/g3_hover.yaml`, 200,704 steps, evaluated on 20 unused seeds:
20/20 episodes reached the time limit, final position error ≈ 0.24 m with a steady bias of
about 0.17 m in z. The policy learned to *survive*, not to hold tightly, and training had not
converged at the 200k cutoff. Numbers and interpretation: [docs/G3.md](docs/G3.md).

## What stays out of git

`checkpoints/`, `*.zip`, `*.pt`, `runs/`, `wandb/`, `__pycache__/`, `.pytest_cache/`.
Configs, metrics (`results/*.json`, `results/*.csv`) and figures **do** go in — they are how we
compare runs.

## Known gaps in the stubs

These are deliberate; they go away when G1/G2 land, but do not build on them:

* `QuadrotorEnv` constructs `StubFaultInjector` itself (`envs/quadrotor_env.py`) — the fault
  module is not injectable yet, unlike the dynamics.
* Rotor positions are vehicle data in the config, but the stub assumes a symmetric X for the
  *thrust* terms only through the positions themselves; a non-symmetric airframe is accepted
  without warning. Compare against `nominal_x_positions()` if that matters.
* Action → thrust is assumed **linear** (`f = T_max · u`). If the real mapping becomes
  quadratic in RPM, existing checkpoints stop meaning the same thing.

## Working with this repo

* One `main` branch. Push directly for small changes; open a PR for anything that touches
  `envs/interface.py`, `configs/base.yaml` or `docs/INTERFACE.md` — those are the shared
  contract and other people's code depends on them.
* Add a `results/<run>/` entry for every headline number you report, so it can be traced back.
* Update the gate table above when a gate moves.
