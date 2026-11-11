#!/usr/bin/env python3
"""Run a grid of comparison or ablation experiments, one training run each.

Example:
    python -m train.sweep --config configs/g3_hover.yaml \
        --grid ppo.learning_rate=1e-4,3e-4 --grid ppo.n_epochs=5,10 \
        --seeds 0,1 --timesteps 50000

Every run is a plain ``train.train_ppo`` subprocess, so a sweep cannot drift
from the single-run path: the same validation, the same ``meta.json``, the same
``unique_run_dir`` collision handling.  What this adds is the bookkeeping that
makes a table of results trustworthy:

* the plan is written to ``plan.json`` *before* the first run starts, so a sweep
  that dies halfway still records what it was going to do;
* ``summary.csv`` gains a row as each run finishes, with the resolved run
  directory parsed from that run's own stdout (a name can be suffixed ``-002``);
* each run keeps its own ``.log``, so a failure is diagnosable without re-running;
* every run gets a distinct name (``<prefix>-000``, ``-001``, ...) and therefore
  its own checkpoint and results directory, so nothing is overwritten.

Ctrl-C stops after the current run; the rows already written stay valid.
"""

from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.config import apply_override, load_config  # noqa: E402

SUMMARY_COLUMNS = [
    "index",
    "name",
    "seed",
    "overrides",
    "status",
    "seconds",
    "return_mean",
    "return_std",
    "crash_rate",
    "out_of_bounds_rate",
    "timeout_rate",
    "final_position_error_mean_m",
    "all_survived",
    "checkpoint",
    "eval_error",
]

EVAL_METRICS = (
    "return_mean",
    "return_std",
    "crash_rate",
    "out_of_bounds_rate",
    "timeout_rate",
    "final_position_error_mean_m",
    "all_survived",
)


def parse_grid(specification: str) -> tuple[str, List[Any]]:
    """Parse ``path=v1,v2`` into the path and its parsed values.

    Values are parsed as int, then float, then ``yaml.safe_load``.  The int/float
    attempts are not redundant: PyYAML reads ``1e-4`` as a string (its float rule
    requires a decimal point) and a learning-rate sweep would otherwise set a
    string where a number belongs.

    Commas separate values, so a list value such as ``net_arch=[64,128]`` cannot
    be swept here -- it would be split in two.  That is a deliberate limit: give
    the list its own config file, or use ``--set`` on a single run.
    """
    path, separator, raw_values = specification.partition("=")
    path = path.strip()
    if not separator or not path:
        raise ValueError(f"expected path=v1,v2, got {specification!r}")
    values: List[Any] = []
    for token in raw_values.split(","):
        try:
            value = _parse_scalar(token)
        except yaml.YAMLError:
            raise ValueError(
                f"{path}: could not parse {token.strip()!r}; a list or mapping value "
                "needs its own config file, because commas separate grid values"
            ) from None
        if isinstance(value, (list, dict)):
            raise ValueError(
                f"{path}: {value!r} is not a scalar; grid values cannot contain commas"
            )
        values.append(value)
    if not values:
        raise ValueError(f"{path}: no values")
    return path, values


def _parse_scalar(token: str) -> Any:
    token = token.strip()
    if not token:
        raise ValueError("empty value in grid")
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        pass
    return yaml.safe_load(token)


def format_value(value: Any) -> str:
    """One value as a ``--set``-ready string that parses back to the same thing."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        text = repr(value)
        # "1e+20" has no decimal point, which PyYAML reads as a string.
        if "e" in text and "." not in text:
            text = text.replace("e", ".0e", 1)
        return text
    return str(value)


def build_runs(
    grid: Dict[str, Sequence[Any]], seeds: Sequence[int], prefix: str
) -> List[Dict[str, Any]]:
    """Cartesian product of the grid, crossed with the seeds, numbered in order."""
    paths = list(grid)
    combinations = list(itertools.product(*(grid[path] for path in paths)))
    runs: List[Dict[str, Any]] = []
    for seed in seeds:
        for combination in combinations:
            index = len(runs)
            overrides = {
                path: format_value(value) for path, value in zip(paths, combination)
            }
            runs.append(
                {
                    "index": index,
                    "name": f"{prefix}-{index:03d}",
                    "seed": int(seed),
                    "overrides": overrides,
                }
            )
    return runs


def validate_overrides(config: Dict[str, Any], runs: Sequence[Dict[str, Any]]) -> None:
    """Reject a bad path or value before the first run starts, not after ten."""
    for run in runs:
        trial = copy.deepcopy(config)
        for path, value in run["overrides"].items():
            try:
                apply_override(trial, f"{path}={value}")
            except ValueError as error:
                raise ValueError(f"run {run['name']}: {error}") from None


def train_command(args: argparse.Namespace, run: Dict[str, Any]) -> List[str]:
    command = [
        sys.executable,
        "-m",
        "train.train_ppo",
        "--config",
        args.config,
        "--name",
        run["name"],
        "--seed",
        str(run["seed"]),
    ]
    for path, value in run["overrides"].items():
        command += ["--set", f"{path}={value}"]
    if args.timesteps is not None:
        command += ["--timesteps", str(args.timesteps)]
    if args.n_envs is not None:
        command += ["--n-envs", str(args.n_envs)]
    if args.device is not None:
        command += ["--device", args.device]
    return command


def eval_command(args: argparse.Namespace, checkpoint: Path) -> List[str]:
    command = [
        sys.executable,
        "-m",
        "eval.evaluate",
        "--checkpoint",
        str(checkpoint),
    ]
    if args.episodes is not None:
        command += ["--episodes", str(args.episodes)]
    return command


def run_logged(command: Sequence[str], log_path: Path) -> tuple[int, str]:
    """Run ``command`` with its combined output in ``log_path``; return it too."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as handle:
        process = subprocess.run(
            list(command),
            cwd=str(ROOT),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return process.returncode, log_path.read_text(encoding="utf-8", errors="replace")


def parse_run_dir(output: str) -> Path | None:
    """The run directory train_ppo reports as ``checkpoints: <path>``."""
    for line in reversed(output.splitlines()):
        if line.startswith("checkpoints: "):
            return Path(line.split(": ", 1)[1].strip())
    return None


def read_eval(candidate: Path) -> Dict[str, Any]:
    path = candidate / "eval.json"
    if not path.is_file():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--grid",
        action="append",
        default=[],
        metavar="PATH=V1,V2",
        help="sweep one config key over scalar values (repeatable)",
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help="comma-separated training seeds (default: run.seed from the config)",
    )
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="evaluation episodes per run (default: configs' evaluation.episodes)",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="train only; skip eval.evaluate",
    )
    parser.add_argument(
        "--name-prefix",
        default=None,
        help="run name prefix (default: run.name from the config)",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="stop the sweep at the first failed run",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan and exit without training",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="sweep output directory (default: results/sweeps/<UTC timestamp>)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    try:
        grid: Dict[str, Sequence[Any]] = {}
        for specification in args.grid:
            path, values = parse_grid(specification)
            if path in grid:
                raise ValueError(f"{path}: given more than once")
            grid[path] = values
        if args.seeds:
            seeds = [int(token) for token in args.seeds.split(",") if token.strip()]
        else:
            seeds = [int(config["run"]["seed"])]
        if not seeds:
            raise ValueError("--seeds is empty")
        prefix = args.name_prefix or str(config["run"]["name"])
        runs = build_runs(grid, seeds, prefix)
        validate_overrides(config, runs)
    except ValueError as error:
        parser.error(str(error))

    out_dir = (
        Path(args.out)
        if args.out
        else ROOT / "results" / "sweeps" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    plan = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(Path(args.config).resolve()),
        "grid": {path: [format_value(value) for value in values] for path, values in grid.items()},
        "seeds": seeds,
        "timesteps": args.timesteps,
        "n_envs": args.n_envs,
        "device": args.device,
        "evaluate": not args.no_eval,
        "episodes": args.episodes,
        "runs": [
            {
                "index": run["index"],
                "name": run["name"],
                "seed": run["seed"],
                "overrides": run["overrides"],
                "train_command": train_command(args, run),
            }
            for run in runs
        ],
    }

    print(f"sweep: {len(runs)} run(s), {len(grid)} swept key(s), {len(seeds)} seed(s)")
    for run in runs:
        detail = ", ".join(f"{path}={value}" for path, value in run["overrides"].items())
        print(f"  {run['index']:03d}  {run['name']}  seed={run['seed']}  {detail}")
    if args.dry_run:
        print("dry run: nothing started")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "plan.json", "w", encoding="utf-8") as handle:
        json.dump(plan, handle, indent=2)

    summary_path = out_dir / "summary.csv"
    failures = 0
    with open(summary_path, "w", encoding="utf-8", newline="") as summary:
        writer = csv.DictWriter(summary, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        summary.flush()

        for position, run in enumerate(runs, start=1):
            row = {
                "index": run["index"],
                "name": run["name"],
                "seed": run["seed"],
                "overrides": ";".join(
                    f"{path}={value}" for path, value in run["overrides"].items()
                ),
                "status": "failed",
            }
            print(
                f"\n[{position}/{len(runs)}] {run['name']} seed={run['seed']}",
                flush=True,
            )
            started = time.monotonic()
            try:
                code, output = run_logged(
                    train_command(args, run), out_dir / "runs" / f"{run['name']}.log"
                )
                run_dir = parse_run_dir(output)
                if code != 0 or run_dir is None:
                    row["status"] = "train_failed"
                else:
                    row["checkpoint"] = str(run_dir / "model.zip")
                    row["status"] = "trained"
                    if not args.no_eval:
                        eval_code, _ = run_logged(
                            eval_command(args, run_dir),
                            out_dir / "runs" / f"{run['name']}.eval.log",
                        )
                        metrics = read_eval(ROOT / "results" / run_dir.name)
                        if eval_code != 0:
                            row["status"] = "eval_failed"
                            row["eval_error"] = f"exit code {eval_code}"
                        elif not metrics:
                            row["status"] = "eval_failed"
                            row["eval_error"] = "eval.json not found"
                        else:
                            row["status"] = "ok"
                            for key in EVAL_METRICS:
                                if key in metrics:
                                    row[key] = metrics[key]
            except KeyboardInterrupt:
                row["status"] = "interrupted"
                row["seconds"] = round(time.monotonic() - started, 1)
                writer.writerow(row)
                summary.flush()
                print(f"\ninterrupted; partial results in {summary_path}")
                return 130
            row["seconds"] = round(time.monotonic() - started, 1)
            writer.writerow(row)
            summary.flush()
            if row["status"] != "ok":
                failures += 1
                print(f"  {row['status']} -- see {out_dir / 'runs' / (run['name'] + '.log')}")
                if args.stop_on_error:
                    print(f"stopping after the first failure; results in {summary_path}")
                    break

    print(f"\nsummary: {summary_path}")
    if failures:
        print(f"{failures} run(s) did not finish cleanly")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
