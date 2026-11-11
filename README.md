# MECH4429 Project A03 — Quadrotor Position Hold After Single-Rotor Failure

**English** · [简体中文](README.zh-CN.md)

A reinforcement-learning controller that keeps a quadrotor near a set point after one rotor
fails, trained in simulation.

**Current state: Gate 3 — the RL pipeline runs end to end on stub dynamics, on top of
gym-pybullet-drones.** The fault module is still a placeholder; the vehicle is no longer ours
(it is gpd's published `cf2x`). **No number in this repo is a vehicle result.** See
[docs/G3.md](docs/G3.md).

Since 2026-11-10 the task is built on **[gym-pybullet-drones](third_party/NOTICE.md)** —
`envs/gpd_env.py` subclasses its `BaseAviary`, so the world, URDF, force mapping and physics
stepping come from a published simulator rather than from code we wrote. The two earlier
implementations (our own NumPy 6-DOF model and the same model on the PyBullet engine) are still
runnable and are kept as the reference the migration is checked against — see
[Three implementations, one task](#three-implementations-one-task).

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
| G3 | RL pipeline, stub dynamics | **Done** — train → save → reload → evaluate works; 89 tests; on gym-pybullet-drones, with the two earlier implementations still runnable |
| G4 | Curriculum over failure time / rotor | Not started |
| G5 | Final evaluation and report | Not started |

## Repository layout

```
environment.yml     one conda command for Linux and Windows, versions pinned
configs/            base.yaml (G0 spec) + one YAML per run, each extending base
third_party/
  gym_pybullet_drones/  vendored gym-pybullet-drones, pinned to a commit (MIT, see NOTICE.md)
    examples/         upstream demos (pid.py, learn.py, ...); `python run.py --run gpd_<name>` runs one
envs/
  interface.py      shared types, observation layout, structural rotor order  <- start here
  config.py         config loading, inheritance, digest, validation
  quat.py           quaternion helpers
  env_factory.py    picks the implementation from env.impl: gpd (default) | legacy
  gpd_env.py        the task on gym-pybullet-drones' BaseAviary + cf2x URDF  <- default
  quadrotor_env.py  the same task on our own dynamics (legacy implementation)
  stub_dynamics.py  G1 placeholder: 6-DOF rigid body, NumPy
  pybullet_dynamics.py  the same model on the PyBullet engine, + ground contact
  dynamics_factory.py   picks a backend from dynamics.backend: numpy | pybullet (legacy only)
  stub_faults.py    G2 placeholder: masks one rotor after failure_time_s
train/train_ppo.py  PPO training, checkpoint + metadata + CSV logs
train/sweep.py      grid of comparison/ablation runs -> plan.json + summary.csv
eval/               deterministic evaluation, metrics, plots, 3D replay
eval/fault_response.py  what the policy commands after a known rotor stops
tools/platform_check.py versions + three implementations + cross-machine determinism digest
tools/crosscheck_gpd.py the three implementations side by side, one table
tools/gpd_demo.py   launcher for the vendored demos: legacy GUI renderer, window restore, run(**kwargs)
run.py              click-to-run menu: our train/evaluate/replay/tests/doctor + 12 upstream gpd demos
run_replay.py       click-to-run launcher for the 3D replay (no arguments needed)
tests/              interface, stub physics, train/save/reload
docs/               INTERFACE.md (G0), G3.md (pipeline + results), rotor_layout.svg
src/dynamics/       Andy: replace stub_dynamics.py
src/faults/         Joe: replace stub_faults.py
results/            metrics and figures that are worth keeping (in git)
checkpoints/        weights (not in git)
```

## Quick start

Python 3.12, conda env named `uav-g3`, same file on Linux and Windows:

```bash
conda env create -f environment.yml     # one command, Linux or Windows
conda activate uav-g3
```

`environment.yml` pins exactly the versions every reported number was produced with. If you
prefer to build it by hand instead:

```bash
conda create -n uav-g3 python=3.12.16 -y
conda activate uav-g3
conda install -c conda-forge numpy=2.5.4 scipy=1.18.1 matplotlib=3.11.2 pillow=13.0.0 \
      gymnasium=1.3.0 transforms3d=0.4.2 cloudpickle=3.1.2 pybullet=3.25 \
      pyyaml pandas=3.1.1 pytest
pip install --extra-index-url https://download.pytorch.org/whl/cpu \
      torch==2.15.0+cpu stable-baselines3==2.9.2
# CPU-only is fine for everything in this repo.
```

`pybullet` comes from conda-forge on purpose: PyPI has **no Windows wheel** for any modern
version, so `pip install pybullet` fails on Windows without MSVC Build Tools. conda-forge calls
it 3.25; the package is pybullet 3.2.5.

After that, run `python tools/platform_check.py` (or menu item `doctor`). It prints the versions,
exercises all three implementations and emits a **determinism digest** — a fixed 120-step rollout.
Two machines that print the same digest produce the same training data; that is how the Ubuntu and
Windows copies are checked against each other.

All commands below run **from the repository root**.

### Don't want to type commands? `run.py`

Open `run.py` and press **Run** (▶ top-right in VS Code), or run `python3 run.py`.
It shows a menu — train, evaluate, replay, tests, backend comparison, print
config — and starts whatever you pick. It needs no arguments and no activated
environment: it uses only the standard library itself, finds the `uav-g3`
interpreter, drops `PYTHONPATH` and sets the CPU flag for you. `--list` prints
the entries, `--run <key>` runs one directly.

The last twelve entries are the **upstream gym-pybullet-drones examples**, run
from the vendored copy on this same environment: `gpd_pid` (three drones
circling, RPM sliders), `gpd_hover`, `gpd_pid_velocity`, `gpd_mrac`,
`gpd_downwash`, `gpd_pettingzoo`, `gpd_debug`, `gpd_learn_small` /
`gpd_learn_full` / `gpd_play`, `gpd_tests` (upstream's own 13 tests) and
`gpd_beta` (needs a compiled Betaflight SITL; the menu tells you so instead of
failing). They are demos of *their* environments — they do not go through our
observation, reward, termination or fault logic, so they verify nothing we
claim. Their logs land in `.scratch/gpd_example_results/`, never in `results/`.

They are launched through [tools/gpd_demo.py](tools/gpd_demo.py), a thin wrapper
that exists for two reasons. **Text clipping:** pybullet's default `OpenGL3+`
window clips the first letter of every label (`File`→`rile`, `Explorer`→
`xplorer`, `Params`→`arams`) — it is not a DPI problem (declaring DPI awareness
and enlarging the window both change nothing), the renderer just draws every
string one character left of its clip rectangle. The legacy `--opengl2`
renderer draws them whole, at the price of no shadows, floor textures or the ball
texture. It is off by default; `UAV_GPD_GL=2` turns it on when you need to read
every label. **Duration:** upstream's demos end by themselves after 5–15 s
(`pid.py` uses 12 s), which looks like a timer someone set. The menu runs the
PID/MRAC/downwash demos for 60 s; `UAV_GPD_DURATION_SEC=300` before starting
`run.py` makes it longer, closing the window or Ctrl-C makes it shorter.

`run_replay.py` is the older single-purpose launcher for just the 3D replay;
the menu calls it, so both stay in sync.

### Five things that will bite you

1. **`env -u PYTHONPATH`** — if ROS is installed on the same machine it exports a
   `python3.10` ROS path into `PYTHONPATH`, and the conda interpreter will then import ROS's
   NumPy instead of the env's. Drop the variable for every command.
2. **`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`** — a third-party pytest plugin installed in the base
   environment can abort collection. This flag stops auto-loading plugins; `tests/` only needs
   `pytest-*.py` from this repo.
3. **`CUDA_VISIBLE_DEVICES=""`** — the reported numbers were produced on CPU. Set it to keep
   your own GPU from silently changing them.
4. **`KMP_DUPLICATE_LIB_OK=TRUE` on Windows** — torch ships Intel's OpenMP runtime and
   numpy/scipy from conda-forge ship another; the second one to load aborts the process at the
   next `numpy.linalg` call, with a message that blames LAPACK and prints no traceback. Every
   entry point sets it before importing torch (`train/`, `eval/`, `run.py`,
   `tools/`, `tests/conftest.py`), so you do not need to set it yourself. If you write a new
   script, do the same — see the note at the top of [tests/conftest.py](tests/conftest.py).
5. **Import `torch` before the engine (Windows, upstream examples only)** — once a
   pybullet simulation has called into numpy's BLAS, MKL's `libiomp5md.dll` is
   loaded, and a later `import torch` fails to load *its* copy of the same DLL:
   `WinError 127` on `torch\lib\shm.dll`, no traceback from our code. Importing
   torch first avoids it. `gpd_tests` does exactly that, and it is why the
   upstream suite shows 13 passed here but fails 2 tests if you run
   `pytest third_party/gym_pybullet_drones/tests` directly. Our own suite is not
   affected — it imports torch during collection.

## Commands

```bash
# tests: interface + stub physics + gpd env + train/save/reload
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 CUDA_VISIBLE_DEVICES="" \
  python -m pytest tests -q

# train (200k steps, ~20 min on CPU: the default env steps the engine)
env -u PYTHONPATH CUDA_VISIBLE_DEVICES="" \
  python -m train.train_ppo --config configs/g3_hover.yaml

# evaluate a saved policy — no learning
env -u PYTHONPATH CUDA_VISIBLE_DEVICES="" \
  python -m eval.evaluate --checkpoint checkpoints/g3_stub_hover/model.zip

# watch one of those episodes in 3D (needs pybullet, see below)
env -u PYTHONPATH CUDA_VISIBLE_DEVICES="" \
  python -m eval.replay --checkpoint checkpoints/g3_stub_hover/model.zip --episode 0

# one run with two values changed — still one config file
env -u PYTHONPATH CUDA_VISIBLE_DEVICES="" \
  python -m train.train_ppo --config configs/g4_fixed_rotor1.yaml \
  --set ppo.n_epochs=5 --set ppo.learning_rate=1.0e-4 --seed 3

# comparison / ablation: 2 learning rates x 2 seeds = 4 runs, then evaluation
env -u PYTHONPATH CUDA_VISIBLE_DEVICES="" \
  python -m train.sweep --config configs/g4_fixed_rotor1.yaml \
  --grid ppo.learning_rate=1e-4,3e-4 --seeds 0,1 --timesteps 200000
```

Overrides: `--timesteps`, `--n-envs`, `--name`, `--device`, `--seed`, and repeatable
`--set path=value` for training; `--episodes`, `--seed-start`, `--out` for evaluation.

`--set` writes any dotted key in the resolved config, so an ablation does not need a YAML
file per combination. The value is parsed as YAML (`2.0` float, `true` bool, `[64, 128]`
list, `null` null) — **spell the path exactly**: an unknown key is an error that lists the
keys that do exist, rather than a silent no-op. Write scientific notation with a decimal
point (`1.0e-4`); PyYAML reads bare `1e-4` as a string. `--print-config` shows the merged
config after the overrides, and the `config_sha256` in `meta.json` covers the overridden
values, so two `--set` runs remain distinguishable months later.

`train/sweep.py` runs the Cartesian product of `--grid` values and `--seeds`, one
`train.train_ppo` subprocess per combination, each named `<run name>-000`, `-001`, … so
nothing overwrites anything. It writes the full plan to `plan.json` before the first run
starts (a sweep that dies halfway still records what it was going to do), appends a row per
run to `summary.csv` as it finishes, and keeps each run's stdout in `runs/<name>.log`.
Evaluation runs afterwards unless `--no-eval`; the metrics land in the same row. Use
`--dry-run` to print the plan, `--stop-on-error` to stop at the first failure, and `--out`
to choose the directory (default `results/sweeps/<UTC timestamp>/`). Grid values are
scalars only — a list such as `net_arch` needs its own config file, because commas
separate values.

### 3D replay (optional)

**Easiest: open [`run_replay.py`](run_replay.py) and press ▶ Run.** It takes no arguments,
plays episode 0 of `g3_stub_hover`, re-executes itself with the `uav-g3` interpreter if the one
the editor picked is missing packages, and writes `replay.gif` instead of failing when no window
can be opened. `.vscode/settings.json` pins the interpreter and clears the ROS `PYTHONPATH`
entry; `.vscode/` is gitignored, so that file stays local.

`eval/replay.py` draws one evaluation episode in PyBullet. PyBullet is **display only** — the
trajectory comes from the environment's own dynamics, so a replay cannot change any number in
`eval.json`. Episode `n` uses the evaluator's seed (`seed_start + n`), so `--episode 0` is the
episode in [results/g3_stub_hover/eval_trajectory.png](results/g3_stub_hover/eval_trajectory.png).

PyBullet is **not** optional any more — it is what `env.impl: gpd` runs on — so it is in
`environment.yml` (from conda-forge, not PyPI: no Windows wheel exists).

In the window: **Ctrl + left-drag** orbits (a plain drag is object picking in PyBullet's
ExampleBrowser, and picking is switched off here so it does nothing), scroll zooms, space pauses,
q quits. When the episode ends the window holds the last frame until you press q, so a 10 s
replay does not look like a crash. The model is generated from
`dynamics.rotor_positions_body_m`, so its rotor layout is whatever the config says — there is no
second copy of the airframe to keep in sync. The failed rotor turns orange at the failure step.
The rotor discs spin cosmetically (the rate is not a physical quantity).

`--record out.gif` writes a ~1 MB GIF instead of opening a window, for people without PyBullet.
It takes about 1m45s: offscreen rendering costs ~0.45 s a frame here (the offscreen renderer falls
back to software rasterisation), and only the frames the GIF keeps are rendered. Rendering happens
offscreen, so the file is the same on any machine. Committed example:
[results/g3_stub_hover/replay_seed100000.gif](results/g3_stub_hover/replay_seed100000.gif).

The current checkpoint was trained with `faults.mode: nominal`, so nothing fails in it — the
failure visuals (`faults.mode: fixed` or `random`) need a checkpoint trained under faults, which
is G4.

### Three implementations, one task

`env.impl` in the config picks the *implementation* of the whole task; for the two legacy ones
`dynamics.backend` then picks the integrator underneath. Everything above physics — observation,
reward, termination, fault injection, training, evaluation — is untouched. That is what made the
migration cheap and reversible.

| `env.impl` | Class | Vehicle | What it is |
|---|---|---|---|
| `gpd` (default) | `envs/gpd_env.py` | gpd `cf2x` URDF | `BaseAviary` subclass, vendored gym-pybullet-drones |
| `legacy` + `dynamics.backend: numpy` | `envs/quadrotor_env.py` | `dynamics.*` in the config | our hand-written 6-DOF semi-implicit Euler |
| `legacy` + `dynamics.backend: pybullet` | `envs/quadrotor_env.py` | `dynamics.*` in the config | the same model on the PyBullet engine, plus a ground plane |

`envs/env_factory.py` builds whichever the config asks for. The legacy path is not dead weight:
it produced every G0–G4 number so far, and it is what makes a disagreement with gpd diagnosable
in five minutes instead of an afternoon.

**The vehicles differ, and that matters.** `gpd` flies the published `cf2x`: 0.027 kg, arm
0.0397 m, thrust-to-weight **2.25**, thrust quadratic in the normalised command (`KF · rpm²`).
The legacy path flies the placeholder airframe in `configs/*.yaml`: 0.5 kg, arm 0.092 m, T/W
**1.63**, thrust linear. Same task, different machine — numbers from before 2026-11-10 are not
comparable point by point with numbers from `gpd`.

```bash
# put all three side by side: vehicle, hover, thrust self-consistency, cost per step
env -u PYTHONPATH CUDA_VISIBLE_DEVICES="" \
  python tools/crosscheck_gpd.py
```

Measured on Windows 11 / CPython 3.12 (2026-11-10): each implementation holds altitude at its own
hover command, and each applies the thrust it reports to within 0.06 % (legacy) and 0.29 %
(gpd). Cost per control step: **gpd 0.17 ms**, PyBulletDynamics 0.34 ms, NumPy stub 0.47 ms —
gpd is the fastest of the three, because it does no per-step URDF bookkeeping.

The legacy engine backend is verified against the legacy NumPy one by
[tests/test_pybullet_backend.py](tests/test_pybullet_backend.py) (15 tests): with the plane
switched off the two agree to **0.041 mm over 4 s** while airborne — including cases that fly
35 m and tumble past 100° — and hover is bit-for-bit identical. With the plane on they differ
exactly where they should: the stub has no floor and falls through z = 0, the engine lands.

* **PyBullet's `linearDamping` is not our drag coefficient.** Ours is a force per unit speed
  (N/(m/s)); Bullet's is a decay rate (1/s). Mapping one onto the other silently changes the
  model — it showed up as a 15 m divergence in 4 s. The engine's damping is therefore set to zero
  and the drag terms are applied as explicit forces, exactly as the stub does.
* **Ground contact is the feature, not an accident.** `termination.use_ground_contact` in
  `base.yaml` is still an open G0 question, so the legacy pybullet backend has a floor and the
  NumPy one does not. `gpd` always has one, and its own ground-effect model is available through
  `env.gpd_physics: pyb_gnd`.

#### Rotor order: the one thing to read before injecting a fault

Our layout is R0 front-right, R1 front-left, R2 rear-left, R3 rear-right. gpd's `cf2x.urdf`
numbers its rotor links **FR, RR, RL, FL** — so `gpd 0 = R0`, `gpd 1 = R3`, `gpd 2 = R2`,
`gpd 3 = R1`. `ROTOR_GPD_INDEX` in [envs/gpd_env.py](envs/gpd_env.py) is the only place that
mapping is written down, and every command is permuted through it in `_preprocessAction`.

Get it wrong and hover still looks perfect (all four motors equal) while "rotor 1 failed"
silently kills rotor 3. [tests/test_gpd_env.py](tests/test_gpd_env.py) pins it by failing each
rotor in turn and checking the sign of the resulting roll and pitch rates against the geometry.

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

Those numbers came from the **legacy** implementation (`env.impl: legacy` + NumPy). The
2026-11-10 switch to `gpd` as the default did not touch them: a checkpoint records the resolved
config it was trained with in `meta.json`, and `make_env` builds exactly that — so the original
checkpoint re-evaluates to the same figures. `gpd` is a different machine (0.027 kg, thrust-to-
weight 2.25); comparable results on it need a fresh training run, not a translation.

## What stays out of git

`checkpoints/`, `*.zip`, `*.pt`, `runs/`, `wandb/`, `__pycache__/`, `.pytest_cache/`.
Configs, metrics (`results/*.json`, `results/*.csv`) and figures **do** go in — they are how we
compare runs.

## Known gaps in the stubs

These are deliberate; they go away when G1/G2 land, but do not build on them:

* Both implementations construct `StubFaultInjector` themselves (`envs/quadrotor_env.py`,
  `envs/gpd_env.py`) — the fault module is not injectable yet, unlike the dynamics.
* Rotor positions are vehicle data in the config, but the stub assumes a symmetric X for the
  *thrust* terms only through the positions themselves; a non-symmetric airframe is accepted
  without warning. Compare against `nominal_x_positions()` if that matters. (`gpd` is exempt:
  its geometry comes from the URDF.)
* Legacy action → thrust is assumed **linear** (`f = T_max · u`). `gpd` uses `KF · rpm²`, which is
  quadratic in the command — the same number means a different force in the two.
* The pybullet backend's ground contact is not wired into the termination condition yet — a
  crash is still decided by `termination.crash_z`. Same for `gpd`: it has a floor, but the crash
  test is still an altitude threshold.

## Working with this repo

* One `main` branch. Push directly for small changes; open a PR for anything that touches
  `envs/interface.py`, `configs/base.yaml` or `docs/INTERFACE.md` — those are the shared
  contract and other people's code depends on them.
* Add a `results/<run>/` entry for every headline number you report, so it can be traced back.
* Update the gate table above when a gate moves.
