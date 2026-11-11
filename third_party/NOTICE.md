# Third-party code in this directory

## gym-pybullet-drones

| | |
|---|---|
| Upstream | https://github.com/utiasDSL/gym-pybullet-drones |
| Commit | `348b5c18a4bb770a48928b9d044decf77ca33302` |
| Version reported by upstream at that commit | 2.2.0 |
| Licence | MIT — full text in `gym_pybullet_drones/LICENSE` |
| Copyright | (c) 2020 Jacopo Panerati |
| Vendored on | 2026-11-10 |
| Location | `third_party/gym_pybullet_drones/` (48 files, ~15 MB) |

### Why it is vendored rather than installed

* It is pinned to a commit. `pip install gym-pybullet-drones` resolves to
  whatever is newest, which means two machines running "the same" project can
  step different physics — and every number in `results/` becomes unattributable.
* The package must be importable from a repository that predates it. Keeping it
  in-tree makes the checkout self-contained: clone, create the env, run.
* An upstream upgrade becomes a reviewable diff in the same pull request as the
  code that depends on it, instead of an invisible change in someone's
  site-packages.

`envs/gpd_env.py` puts this directory on `sys.path` and imports
`gym_pybullet_drones.envs.BaseAviary`. `envs/env_factory.py` only reaches that
module when `env.impl: gpd`. Two other entry points use it, both opt-in:
`run.py` runs the upstream examples (`--run gpd_<name>`, with `PYTHONPATH` set to
this directory), and the upstream test suite below.

### What is in here besides the package

* `gym_pybullet_drones/tests/` — upstream's own three test files (13 tests),
  copied verbatim from the same commit. Upstream keeps `tests/` at the *repo*
  root; here it has to sit inside our tree, because pytest prepends the test
  file's parent directory to `sys.path` — run from a neighbouring checkout it
  would import *that* copy of `gym_pybullet_drones`, not the one we ship.
  Run it with `python run.py --run gpd_tests`.

### What was removed from the upstream tree

Only material that has no bearing on behaviour:

* `.git/`, so the vendored copy cannot be mistaken for a checkout;
* `tmp/` and `results/` (upstream's example output directories);
* `__pycache__/` where present;
* `pyproject.toml`, `README.md`, `CITATION.cff` (packaging and description; the
  dependency set that matters is the one in our `environment.yml`).

No source file was modified. If one ever has to be, patch it here and record the
line below — a silent local edit is indistinguishable from upstream code when
someone reads it six months from now.

### Local modifications

None.
