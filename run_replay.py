#!/usr/bin/env python3
"""Click-to-run launcher for the 3D replay. No arguments needed.

Press Run in the editor (or run ``python3 run_replay.py``) and it plays episode 0
of the ``g3_stub_hover`` checkpoint. Passing the usual ``eval.replay`` flags
still works and is forwarded unchanged.

Why this file exists: pressing Run starts the script with whatever interpreter
the editor has selected, which on this machine is ``/usr/bin/python3`` -- a
system Python with no torch, no stable-baselines3 and no pybullet. There is also
no ``python`` command at all outside a conda shell, which is why
``python -m eval.replay`` came back as "command not found". So this launcher:

* re-executes itself with the project environment (``uav-g3``) when the current
  interpreter lacks the dependencies,
* drops ROS's ``PYTHONPATH`` entries, which otherwise shadow the env's NumPy,
* locates the repository from ``__file__``, so the working directory is irrelevant,
* writes a GIF instead of failing when no window can be opened.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REEXEC_MARKER = "RUN_REPLAY_REEXECUTED"

#: Interpreters to try, in order, when the current one is missing packages.
INTERPRETER_CANDIDATES = (
    os.environ.get("UAV_PYTHON"),
    str(Path.home() / "miniconda3" / "envs" / "uav-g3" / "bin" / "python"),
    "/home/albertpang/miniconda3/envs/uav-g3/bin/python",
)

REQUIRED = ("numpy", "gymnasium", "stable_baselines3", "torch", "pybullet")


def _drop_ros_paths() -> None:
    """ROS exports its own python3.10 tree into PYTHONPATH; the env's NumPy loses."""
    os.environ.pop("PYTHONPATH", None)
    for entry in list(sys.path):
        if entry.startswith("/opt/ros") or "/ros/" in entry:
            sys.path.remove(entry)


def _missing_packages() -> list[str]:
    import importlib.util

    return [name for name in REQUIRED
            if importlib.util.find_spec(name) is None]


def _re_exec(candidates) -> bool:
    """Replace this process with one running a complete interpreter."""
    current = Path(sys.executable).resolve()
    for candidate in candidates:
        if not candidate:
            continue
        interpreter = Path(candidate)
        if not interpreter.exists() or interpreter.resolve() == current:
            continue
        print(f"[run_replay] {Path(sys.executable).name} is missing packages; "
              f"switching to {interpreter}")
        sys.stdout.flush()
        os.environ[REEXEC_MARKER] = "1"
        os.execv(str(interpreter), [str(interpreter), str(Path(__file__).resolve()),
                                    *sys.argv[1:]])
    return False


def _default_checkpoint() -> Path:
    run_dir = ROOT / "checkpoints" / "g3_stub_hover"
    return run_dir / "model.zip"


def _build_arguments(argv: list[str]) -> list[str]:
    args = list(argv)
    if not any(arg == "--checkpoint" or arg.startswith("--checkpoint=")
               for arg in args):
        checkpoint = _default_checkpoint()
        if checkpoint.exists():
            args += ["--checkpoint", str(checkpoint)]
        else:
            available = sorted(p.parent.name for p in
                               (ROOT / "checkpoints").glob("*/model.zip"))
            raise SystemExit(
                "No checkpoint found at\n"
                f"  {checkpoint}\n\n"
                "checkpoints/ is not in git -- train one first:\n"
                "  python -m train.train_ppo --config configs/g3_hover.yaml\n\n"
                + (f"Existing runs: {', '.join(available)}\n" if available else "")
            )
    if not any(arg.startswith("--episode") or arg.startswith("--seed")
               for arg in args):
        args += ["--episode", "0"]
    return args


def main() -> int:
    _drop_ros_paths()

    missing = _missing_packages()
    if missing:
        if not os.environ.get(REEXEC_MARKER) and _re_exec(INTERPRETER_CANDIDATES):
            return 1  # unreachable: os.execv replaces the process
        raise SystemExit(
            "This interpreter is missing: " + ", ".join(missing) + "\n"
            f"  interpreter: {sys.executable}\n\n"
            "Run with the project environment instead:\n"
            "  /home/albertpang/miniconda3/envs/uav-g3/bin/python run_replay.py\n"
            "In VS Code: click the Python version in the bottom-right corner and\n"
            "pick the uav-g3 interpreter."
        )

    sys.path.insert(0, str(ROOT))
    # The replay is CPU-only; keep it from picking up a GPU and changing nothing
    # but the startup cost.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    from eval import replay  # imported late: needs the environment in place

    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    args = _build_arguments(sys.argv[1:])

    if not has_display and "--record" not in args:
        url = ROOT / "replay.gif"
        print(f"[run_replay] no display detected; writing {url} instead")
        args += ["--record", str(url)]

    sys.argv = [str(Path(__file__).resolve()), *args]
    try:
        return replay.main()
    except RuntimeError as exc:
        # Window could not be opened even though DISPLAY is set.
        if "--record" in args or "could not open a window" not in str(exc):
            raise
        url = ROOT / "replay.gif"
        print(f"[run_replay] {exc}\n[run_replay] writing {url} instead")
        sys.argv = [sys.argv[0], *args, "--record", str(url)]
        return replay.main()


if __name__ == "__main__":
    raise SystemExit(main())
