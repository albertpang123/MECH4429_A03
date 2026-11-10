#!/usr/bin/env python3
"""Click-to-run menu for the whole project. No arguments needed.

Open this file and press Run (▶ top-right) in VS Code; pick a number and it
starts. Everything the project can do is one menu item:

    python3 run.py            # 菜单
    python3 run.py --list     # 只列条目
    python3 run.py --run train_smoke

为什么需要这个文件
------------------
1. 编辑器按「当前选中的解释器」启动脚本，这台机器上通常是 /usr/bin/python3，
   里面没有 torch / stable-baselines3 / pybullet。本文件只用标准库，自己去找
   uav-g3 的解释器再启动子进程，所以编辑器选哪个解释器都能点。
2. 本机装了 ROS humble，它往 PYTHONPATH 里塞 python3.10 的路径，会串味；启动
   子进程时统一去掉。
3. 不需要先 cd、不需要 activate、不需要记 -m 模块路径：工作目录和路径都从
   __file__ 推出来。

和 run_replay.py 的关系：那个是 3D 回放的专用启动器（带窗口失败时写 GIF 的兜底）。
本文件里的「回放」一项直接调它，不重复实现。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CHECKPOINTS = ROOT / "checkpoints"

#: 按顺序尝试；第一个能 import numpy/gymnasium 的就是项目环境。
INTERPRETER_CANDIDATES = (
    os.environ.get("UAV_PYTHON"),
    # conda puts the environment here on both platforms, with a different tail.
    str(Path.home() / "miniconda3" / "envs" / "uav-g3"
        / ("python.exe" if os.name == "nt" else "bin/python")),
    "/home/albertpang/miniconda3/envs/uav-g3/bin/python",
)

_PROBE = "import numpy, gymnasium, yaml"


def _env() -> dict:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)              # ROS humble 会污染解释器路径
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = ""         # 默认 CPU，数字才可复现
    return env


def _python() -> str:
    """The project interpreter: the first candidate that has the dependencies."""
    tried = []
    for candidate in INTERPRETER_CANDIDATES:
        if not candidate:
            continue
        if not Path(candidate).exists():
            tried.append(f"{candidate} (not found)")
            continue
        probe = subprocess.run([candidate, "-c", _PROBE], env=_env(),
                               capture_output=True, text=True)
        if probe.returncode == 0:
            return candidate
        tried.append(f"{candidate} (missing packages)")
    raise SystemExit(
        "找不到项目的 Python 环境。尝试过：\n  " + "\n  ".join(tried) +
        "\n\n要么装好 conda env uav-g3，要么设置 UAV_PYTHON 指向它的 python。"
    )


def _latest_checkpoint() -> Path | None:
    found = sorted(CHECKPOINTS.glob("*/model.zip"), key=lambda p: p.stat().st_mtime)
    return found[-1] if found else None


def _items(python: str) -> list[tuple[str, str, list[str], str]]:
    """(key, 菜单标题, 命令, 说明)"""
    train = [python, "-m", "train.train_ppo"]
    return [
        ("replay", "3D 回放最近一次训练的 checkpoint",
         [python, str(ROOT / "run_replay.py")],
         "弹 PyBullet 窗口播放一整集；没有显示器时自动写 replay.gif。"),
        ("train_smoke", "冒烟训练：numpy 后端，4096 步（约 10 s）",
         train + ["--config", "configs/g3_hover.yaml", "--timesteps", "4096",
                  "--n-envs", "2", "--name", "smoke_numpy", "--device", "cpu"],
         "只证明训练→存档链路通，不代表学到东西。"),
        ("train_pybullet", "冒烟训练：pybullet 后端，4096 步（单环境）",
         train + ["--config", "configs/g3_pybullet.yaml", "--timesteps", "4096",
                  "--n-envs", "1", "--name", "smoke_pybullet", "--device", "cpu"],
         "同样的任务换成引擎后端，用来对比两个积分器。"
         "PyBullet 客户端属于单个进程，多环境并行会被拒绝。"),
        ("train_full", "完整训练：numpy 后端，200k 步（CPU 约 2 分钟）",
         train + ["--config", "configs/g3_hover.yaml", "--device", "cpu"],
         "参数取自 configs/g3_hover.yaml 的 run.total_timesteps。"),
        ("evaluate", "评估最近一次训练的 checkpoint（20 集）",
         [python, "-m", "eval.evaluate", "--checkpoint", "<latest>", "--episodes", "20"],
         "用保留的评估种子 100000+，输出 eval.json / csv / 轨迹图。"),
        ("tests", "跑全部测试",
         [python, "-m", "pytest", "tests/", "-q"],
         "61 项：接口、物理、训练存档、回放、pybullet 后端。"),
        ("test_backend", "只跑后端等价性测试（15 项，约 2 s）",
         [python, "-m", "pytest", "tests/test_pybullet_backend.py", "-q"],
         "证明 PyBullet 后端复现了 NumPy stub；地面是唯一有意的差别。"),
        ("config", "打印合并后的配置和 digest（两个后端各一份）",
         [],
         "检查 extends 有没有正确合并，以及这次运行的 config_sha256。"),
        ("doctor", "环境体检：版本、两个后端、跨机器一致性指纹",
         [python, str(ROOT / "tools" / "platform_check.py")],
         "换机器（比如 Windows）后先跑这个，把输出发回来比对。"),
    ]


def _print_list(items) -> None:
    print("\n可运行的条目：\n")
    for key, label, _, note in items:
        print(f"  {key:<15} {label}")
        print(f"  {'':<15} {note}")
    print("\n直接跑某一个： python3 run.py --run <key>")


def _run(key: str, items, python: str) -> int:
    for item_key, label, cmd, _ in items:
        if item_key != key:
            continue
        if key == "evaluate":
            latest = _latest_checkpoint()
            if latest is None:
                print("[run.py] checkpoints/ 里还没有模型，先跑 train_smoke 或 train_full。")
                return 1
            cmd = [str(latest.parent) if arg == "<latest>" else arg for arg in cmd]
            print(f"[run.py] 使用模型：{latest}")
        if key == "config":
            # 两个后端各打一份，方便直接比对唯一不同的那一行。
            code = 0
            for name in ("g3_hover", "g3_pybullet"):
                print(f"\n── configs/{name}.yaml ──")
                sys.stdout.flush()
                code |= subprocess.run(
                    [python, "-m", "train.train_ppo",
                     "--config", f"configs/{name}.yaml", "--print-config"],
                    cwd=str(ROOT), env=_env()).returncode
            return code
        print(f"\n▶ {label}")
        print("  等价命令： " + " ".join(
            [Path(cmd[0]).name] + cmd[1:]).replace(str(ROOT) + "/", ""))
        print()
        sys.stdout.flush()
        try:
            code = subprocess.run(cmd, cwd=str(ROOT), env=_env()).returncode
        except KeyboardInterrupt:
            print("\n[run.py] 已中断")
            return 130
        if code == 0:
            print("\n[run.py] ✓ 正常结束")
        else:
            print(f"\n[run.py] ✗ 退出码 {code}——往上翻到以 Traceback 开头、"
                  "以 xxxError 结尾的那段，那才是真正的原因。")
        return code
    print(f"[run.py] 没有这个条目：{key}")
    _print_list(items)
    return 2


def _menu(items, python: str) -> int:
    while True:
        print("\n" + "=" * 62)
        print(" MECH4429 单旋翼失效定点保持 — 启动器")
        print("=" * 62)
        for i, (_, label, _, _) in enumerate(items, start=1):
            print(f" {i:2d}. {label}")
        print("  0. 退出")
        try:
            choice = input("\n选一个编号（回车 = 退出）：").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not choice or choice == "0":
            return 0
        if not choice.isdigit() or not 1 <= int(choice) <= len(items):
            print("  编号不对，再来一次。")
            continue
        key = items[int(choice) - 1][0]
        print(f"\n  {items[int(choice) - 1][3]}")
        code = _run(key, items, python)
        print(f"\n  [结束，退出码 {code}]")


def main(argv: list[str]) -> int:
    python = _python()
    items = _items(python)
    if "--list" in argv:
        _print_list(items)
        return 0
    if "--run" in argv:
        idx = argv.index("--run")
        if idx + 1 >= len(argv):
            _print_list(items)
            return 2
        return _run(argv[idx + 1], items, python)
    return _menu(items, python)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
