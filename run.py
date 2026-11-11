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

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

def _mark(fancy: str, plain: str) -> str:
    """控制台编码打不出的符号退回 ASCII。

    Windows 默认控制台是 cp936（GBK），没有 ▶ ✓ ✗，print 会抛
    UnicodeEncodeError 把启动器打断——而且崩的位置正是给 Andy / Joe 的
    那条 `python run.py --run doctor`。中文本身在 GBK 里，所以只退符号，
    不强行切 UTF-8（切了在 cmd.exe 里反而全是乱码）。
    """
    try:
        fancy.encode(sys.stdout.encoding or "ascii")
    except (UnicodeEncodeError, LookupError, AttributeError):
        return plain
    return fancy


PLAY = _mark("▶", ">")
OK = _mark("✓", "OK")
FAIL = _mark("✗", "X")

ROOT = Path(__file__).resolve().parent
CHECKPOINTS = ROOT / "checkpoints"

#: Windows 上为了不动系统 PATH、不要管理员权限，用的是一份工作区内的便携环境
#: （micromamba 自带的 root + env）。它的 python.exe 不能直接调用：Library/bin
#: 不在 PATH 上，numpy 第一次进 LAPACK 就会 abort。必须经 micromamba run。
PORTABLE_ROOT = ROOT.parents[1] / ".scratch" / "gpd_win"
PORTABLE_MAMBA = PORTABLE_ROOT / "Library" / "bin" / "micromamba.exe"
PORTABLE_ENV = PORTABLE_ROOT / "env"

#: 按顺序尝试；第一个能 import numpy/gymnasium/pybullet 的「命令前缀」就是项目环境。
#: 每一项是命令前缀的列表：普通解释器是 [python]，micromamba 环境多两段。
INTERPRETER_CANDIDATES = (
    [os.environ["UAV_PYTHON"]] if os.environ.get("UAV_PYTHON") else None,
    # conda puts the environment here on both platforms, with a different tail.
    [str(Path.home() / "miniconda3" / "envs" / "uav-g3"
         / ("python.exe" if os.name == "nt" else "bin/python"))],
    ["/home/albertpang/miniconda3/envs/uav-g3/bin/python"],
    [str(PORTABLE_MAMBA), "run", "-p", str(PORTABLE_ENV), "python"],
)

#: 上游示例（`third_party/gym_pybullet_drones/examples/`）以包的形式 import，
#: 所以它们的父目录 `third_party/` 必须在 PYTHONPATH 上。平时 `_env()` 会把这个
#: 变量清掉（挡 ROS），只有跑示例时按需设回来。
THIRD_PARTY = ROOT / "third_party"

#: 示例的日志（Logger 每次建一个 `save-<时间戳>/`）不写进项目的 `results/`，
#: 那里放的是我们自己的训练产物，混进去会让人以为是我们跑出来的。
EXAMPLE_RESULTS = ROOT.parents[1] / ".scratch" / "gpd_example_results"

_PROBE = "import numpy, gymnasium, pybullet, yaml"


def _env(pythonpath: Path | None = None) -> dict:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)              # ROS humble 会污染解释器路径
    if pythonpath is not None:
        env["PYTHONPATH"] = str(pythonpath)
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = ""         # 默认 CPU，数字才可复现
    # torch 自带 Intel OpenMP，numpy/scipy 自带另一份；两者同时在进程里时，
    # numpy 第一次进 LAPACK 就会 abort（报 numpy.linalg.inv 处的 Fatal Python
    # error，没有 Python 回溯）。Windows 上这是必踩的一条。数值结果不受影响。
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    return env


def _python() -> list[str]:
    """The project interpreter, as a command prefix.

    A list rather than a path because the portable Windows environment is only
    usable through ``micromamba run``: the entries it adds to PATH (Library/bin)
    are what make numpy find its MKL DLLs.
    """
    tried = []
    for candidate in INTERPRETER_CANDIDATES:
        if not candidate:
            continue
        executable = Path(candidate[0])
        if not executable.exists():
            tried.append(f"{candidate[0]} (not found)")
            continue
        probe = subprocess.run(candidate + ["-c", _PROBE], env=_env(),
                               capture_output=True, text=True)
        if probe.returncode == 0:
            return candidate
        tried.append(f"{candidate[0]} (missing packages)")
    raise SystemExit(
        "找不到项目的 Python 环境。尝试过：\n  " + "\n  ".join(tried) +
        "\n\n要么装好 conda env uav-g3，要么设置 UAV_PYTHON 指向它的 python。"
    )


def _latest_checkpoint() -> Path | None:
    found = sorted(CHECKPOINTS.glob("*/model.zip"), key=lambda p: p.stat().st_mtime)
    return found[-1] if found else None


class Item(NamedTuple):
    """一条可运行的菜单项。

    ``cwd`` / ``pythonpath`` 只对上游示例有意义：它们的日志默认写到当前目录的
    ``results/``，而导入 ``gym_pybullet_drones`` 又需要 ``third_party`` 在
    ``PYTHONPATH`` 上。其余条目两者都保持默认（项目根目录、PYTHONPATH 清空）。
    """

    key: str
    label: str
    cmd: list[str]
    note: str
    cwd: Path | None = None
    pythonpath: Path | None = None


#: 上游 ``beta.py`` 要先编译 Betaflight SITL 才有东西可连（``assets/clone_bfs.sh 1``，
#: 几百 MB）。没编译过就不启动，只打印怎么编译。
BETA_SITL = THIRD_PARTY / "gym_pybullet_drones" / "betaflight_sitl"

#: 上游测试目录（从同一个 commit 拷进来；上游把它放在仓库根，我们放在包旁边）。
UPSTREAM_TESTS = THIRD_PARTY / "gym_pybullet_drones" / "tests"

#: 上游的 ``debug.py`` / ``beta.py`` 用 ``"../assets/..."`` 这种**相对当前目录**的
#: 路径，只能在 examples 目录里跑；其余例子走 ``files('gym_pybullet_drones')``，
#: 与工作目录无关。
EXAMPLES = THIRD_PARTY / "gym_pybullet_drones" / "examples"

#: 上游示例的启动壳：换 GUI 渲染器（默认那个 OpenGL3+ 窗口会把每个标签的第一个
#: 字母裁掉）、把最小化的窗口还原、以及统一「module / call」两种调法。
GPD_DEMO = ROOT / "tools" / "gpd_demo.py"

#: 演示跑多久。上游各例的默认值都是 5~15 s（``pid.py`` 是 12 s），跑着跑着自己
#: 就退出了，看起来像「设了定时」。这里统一给 60 s，想更长就设这个环境变量。
DEFAULT_DEMO_SEC = 60


def _legacy_gui_renderer() -> bool:
    """是否用 ``--opengl2`` 那个 legacy 渲染器。默认不用。

    pybullet 默认的 ``OpenGL3+ [btgl]`` 窗口在这台机器上把每个标签的第一个字符
    裁掉了（``File``→``rile``、``Params``→``arams``），换 legacy 字就完整——不是
    DPI 的问题，声明 DPI 感知、把窗口开大都试过。可 legacy 没有阴影、地面贴图和
    球体纹理，所以默认还是上游用哪个就用哪个，要看全标签时 ``UAV_GPD_GL=2``。
    """
    return os.environ.get("UAV_GPD_GL", "").strip() == "2"


def _demo_sec() -> str:
    return os.environ.get("UAV_GPD_DURATION_SEC", "").strip() or str(DEFAULT_DEMO_SEC)


def _gpd(python: list[str], mode: str, name: str, *args: str, window: bool = True) -> list[str]:
    """``tools/gpd_demo.py <mode> <name> ...`` 的完整命令。

    ``-u`` 让窗口和打印不被重定向缓冲住；``module`` 是把示例当脚本跑、参数原样
    透传，``call`` 是直接 ``run(**kwargs)``（给不接受命令行参数的示例用）。

    ``window=False`` 的条目（训练）不加 ``--gl2``——注意别和示例自己的
    ``gui=False`` 混了：那样启动壳不会提前 import pybullet，示例照旧先 import
    torch，Windows 上 torch 的 DLL 才加载得干净。
    """
    cmd = python + ["-u", str(GPD_DEMO)]
    if window and _legacy_gui_renderer():
        cmd.append("--gl2")
    return cmd + [mode, name, *args]


def _gpd_example(python: list[str], name: str, *args: str, window: bool = True) -> list[str]:
    return _gpd(python, "module", name, *args, window=window)


def _gpd_call(python: list[str], name: str, *, window: bool = True, **kwargs) -> list[str]:
    """不接受命令行参数的示例（``mrac.py``）、或者默认值不适合当演示的示例
    （训练 1e6 步），用这个直接调 ``run()``，关键字参数走 JSON。"""
    return _gpd(python, "call", name, json.dumps(kwargs), window=window)


def _latest_example_model() -> Path | None:
    """上游 ``learn.py`` 存下的 ``best_model.zip``（没有就退回 ``final_model.zip``）。

    模型在 ``<output_folder>/save-<时间戳>/`` 里，而每个示例各用自己的一层
    output_folder，所以按递归 glob 找，不假设深度。
    """
    for pattern in ("**/best_model.zip", "**/final_model.zip"):
        found = sorted(EXAMPLE_RESULTS.glob(pattern), key=lambda p: p.stat().st_mtime)
        if found:
            return found[-1]
    return None


def _items(python: list[str]) -> list[Item]:
    train = python + ["-m", "train.train_ppo"]
    pid = str(EXAMPLE_RESULTS / "pid")
    return [
        Item("replay", "3D 回放最近一次训练的 checkpoint",
             python + [str(ROOT / "run_replay.py")],
             "弹 PyBullet 窗口播放一整集；没有显示器时自动写 replay.gif。"),
        Item("train_smoke", "冒烟训练：gpd 后端，4096 步（约 20 s，两个进程）",
             train + ["--config", "configs/g3_hover.yaml", "--timesteps", "4096",
                      "--n-envs", "2", "--name", "smoke_gpd", "--device", "cpu"],
             "只证明训练→存档链路通，不代表学到东西。默认环境是 gym-pybullet-drones "
             "的 BaseAviary（env.impl: gpd）。"),
        Item("train_legacy", "冒烟训练：老的自研物理 + pybullet，4096 步（单环境）",
             train + ["--config", "configs/g3_pybullet.yaml", "--timesteps", "4096",
                      "--n-envs", "1", "--name", "smoke_pybullet", "--device", "cpu"],
             "对照用：env.impl: legacy，跑自己的 URDF 和积分器。"
             "PyBullet 客户端属于单个进程，所以只能单环境。"),
        Item("train_full", "完整训练：gpd 后端，200k 步（CPU 约 20 分钟）",
             train + ["--config", "configs/g3_hover.yaml", "--device", "cpu"],
             "参数取自 configs/g3_hover.yaml 的 run.total_timesteps。"),
        Item("evaluate", "评估最近一次训练的 checkpoint（20 集）",
             python + ["-m", "eval.evaluate", "--checkpoint", "<latest>", "--episodes", "20"],
             "用保留的评估种子 100000+，输出 eval.json / csv / 轨迹图。"),
        Item("crosscheck", "三套实现并排对比（stub / pybullet / gpd）",
             python + [str(ROOT / "tools" / "crosscheck_gpd.py")],
             "同一任务三套实现：各自悬停、推力响应自洽性、每步耗时。"),
        Item("tests", "跑全部测试（我们的）",
             python + ["-m", "pytest", "tests/", "-q"],
             "89 项：接口、物理、训练存档、回放、gpd 环境、pybullet 后端、配置覆盖与批量实验。"),
        Item("test_backend", "只跑后端等价性测试（15 项，约 2 s）",
             python + ["-m", "pytest", "tests/test_pybullet_backend.py", "-q"],
             "证明 PyBullet 后端复现了 NumPy stub；地面是唯一有意的差别。"),
        Item("config", "打印合并后的配置和 digest（g3_hover / g3_pybullet 各一份）",
             [],
             "检查 extends 有没有正确合并，以及这次运行的 config_sha256。"),
        Item("doctor", "环境体检：版本、三套实现、跨机器一致性指纹",
             python + [str(ROOT / "tools" / "platform_check.py")],
             "换机器（比如 Windows）后先跑这个，把输出发回来比对。"),

        # ---- 上游 gym-pybullet-drones 自带的例子（不是我们的任务） --------------
        Item("gpd_pid", "上游：PID 三机盘旋（GUI，带 RPM 滑块）",
             _gpd_example(python, "pid", "--gui", "True", "--user_debug_gui", "True",
                          "--plot", "False", "--duration_sec", _demo_sec(),
                          "--output_folder", pid),
             "就是导师说的那个例子。右边滑块可以手动给转速，点 Use GUI RPM 接管。"
             f"默认跑 {_demo_sec()} s 就自己停（上游写死 12 s），"
             "要更长设 UAV_GPD_DURATION_SEC。不经过我们的任务/奖励/故障逻辑，是演示不是验证。",
             pythonpath=THIRD_PARTY),
        Item("gpd_hover", "上游：单机悬停（GUI，带 RPM 滑块）",
             _gpd_example(python, "pid", "--gui", "True", "--user_debug_gui", "True",
                          "--plot", "False", "--num_drones", "1",
                          "--duration_sec", _demo_sec(),
                          "--output_folder", pid),
             "一架飞机，界面最干净。想手动演示掉一个旋翼就拖滑块。",
             pythonpath=THIRD_PARTY),
        Item("gpd_pid_velocity", "上游：速度控制例子（GUI）",
             _gpd_example(python, "pid_velocity", "--gui", "True", "--plot", "False",
                          "--duration_sec", _demo_sec(),
                          "--output_folder", str(EXAMPLE_RESULTS / "pid_velocity")),
             "同样的 PID，输入换成速度而不是位置。上游默认只跑 5 s。",
             pythonpath=THIRD_PARTY),
        Item("gpd_mrac", "上游：MRAC 自适应控制（GUI）",
             _gpd_call(python, "mrac",
                       gui=True, plot=False, duration_sec=int(_demo_sec()),
                       output_folder=str(EXAMPLE_RESULTS / "mrac")),
             "模型参考自适应控制，抗参数摄动的另一种传统方法。这个脚本不接受命令行参数，"
             "上游把参数写在 DEFAULT_* 常量里，所以这里直接调 run()。",
             pythonpath=THIRD_PARTY),
        Item("gpd_downwash", "上游：下洗效应（GUI，两机）",
             # downwash.py 的 CLI 没有 --plot，而 run() 的 plot 默认是 True：
             # 传 --plot 会 argparse 报错，不传则跑完后卡在 matplotlib 窗口里
             # 不退出。只能直接调 run(plot=False)。
             _gpd_call(python, "downwash",
                       gui=True, plot=False, duration_sec=int(_demo_sec()),
                       output_folder=str(EXAMPLE_RESULTS / "downwash")),
             "一机在另一机正上方，下面的会被气流往下压。",
             pythonpath=THIRD_PARTY),
        Item("gpd_pettingzoo", "上游：多智能体（2 机，100 步训练）",
             _gpd_call(python, "pettingzoo_multiagent",
                       gui=True, plot=False, local=False,
                       output_folder=str(EXAMPLE_RESULTS / "multiagent")),
             "并行 API，每架机一个 agent。上游默认训练 1e6 步（要几小时），这里用 "
             "local=False 的 100 步版本，只证明多智能体管线通。",
             pythonpath=THIRD_PARTY),
        Item("gpd_debug", "上游：PyBullet 最小调试例子（两个盒子）",
             _gpd_example(python, "debug"),
             "跟无人机无关，讲清楚 addUserDebugLine / applyExternalForce 怎么用。",
             cwd=EXAMPLES, pythonpath=THIRD_PARTY),
        Item("gpd_learn_small", "上游：小规模 PPO 训练（100 步，验证流程）",
             _gpd_call(python, "learn",
                       gui=False, plot=False, local=False,
                       output_folder=str(EXAMPLE_RESULTS / "learn"), window=False),
             "几十秒跑完，只证明训练链路通。它训的是上游的 HoverAviary，不是我们的任务。",
             pythonpath=THIRD_PARTY),
        Item("gpd_learn_full", "上游：完整 PPO 训练（1e7 步，CPU 要几小时）",
             _gpd_example(python, "learn", "--gui", "False", "--plot", "False",
                          "--output_folder", str(EXAMPLE_RESULTS / "learn"),
                          window=False),
             "上游论文里的设定，1e7 步。跑完才有能看的 best_model.zip。",
             pythonpath=THIRD_PARTY),
        Item("gpd_play", "上游：回放训练好的策略（GUI）",
             _gpd_example(python, "play"),
             "自动找最近一次 gpd_learn_* 留下的 best_model.zip。"
             "跑完会弹出曲线窗口，关掉它进程才结束（上游脚本就是这样，不是卡死）。",
             pythonpath=THIRD_PARTY),
        Item("gpd_tests", "上游：跑它自带的测试（13 项，约 55 s）",
             python + ["-c",
                       # 必须先 import torch：Windows 上如果 numpy/MKL 的
                       # libiomp5md.dll 先被加载（跑过一个 pybullet 仿真就会），
                       # torch 再 import 时它自己的同名 DLL 加载不到，
                       # 报 WinError 127 / shm.dll。预导入 torch 让顺序反过来。
                       "import torch, pytest, sys; sys.exit(pytest.main(['-q', "
                       f"{str(UPSTREAM_TESTS)!r}]))"],
             "上游 3 个测试文件、13 项：imports、6 个例子能否跑完、PettingZoo 接口。",
             cwd=EXAMPLE_RESULTS / "upstream_tests", pythonpath=THIRD_PARTY),
        Item("gpd_beta", "上游：Betaflight SITL（要先编译 SITL，Ubuntu）",
             _gpd_example(python, "beta", "--gui", "True"),
             "半实物接口。没执行过 assets/clone_bfs.sh 就跑不起来，菜单会先告诉你。",
             cwd=EXAMPLES, pythonpath=THIRD_PARTY),
    ]


def _print_list(items) -> None:
    print("\n可运行的条目：\n")
    for item in items:
        print(f"  {item.key:<17} {item.label}")
        print(f"  {'':<17} {item.note}")
    print("\n直接跑某一个： python3 run.py --run <key>")


def _run(key: str, items, python: str) -> int:
    for item in items:
        if item.key != key:
            continue
        cmd = item.cmd
        if key == "evaluate":
            latest = _latest_checkpoint()
            if latest is None:
                print("[run.py] checkpoints/ 里还没有模型，先跑 train_smoke 或 train_full。")
                return 1
            cmd = [str(latest.parent) if arg == "<latest>" else arg for arg in cmd]
            print(f"[run.py] 使用模型：{latest}")
        if key == "gpd_play":
            model = _latest_example_model()
            if model is None:
                print("[run.py] 还没有上游示例训出来的模型，先跑 gpd_learn_small 或 gpd_learn_full。")
                return 1
            cmd = cmd + ["--model_path", str(model), "--gui", "True"]
            print(f"[run.py] 使用模型：{model}")
        if key == "gpd_beta" and not BETA_SITL.exists():
            print("\n[run.py] Betaflight SITL 还没编译，这个例子跑不起来。")
            print("  需要时先执行（会下载并编译一份 Betaflight，几百 MB，和我们的任务无关）：")
            print(f"    cd {THIRD_PARTY / 'gym_pybullet_drones'} && ./assets/clone_bfs.sh 1")
            return 1
        if key == "config":
            # 两份配置各打一份：g3_hover 走默认的 gpd，g3_pybullet 走 legacy +
            # 自研 pybullet。并排看就能确认 extends 只合并了该合并的那几行。
            code = 0
            for name in ("g3_hover", "g3_pybullet"):
                print(f"\n── configs/{name}.yaml ──")
                sys.stdout.flush()
                code |= subprocess.run(
                    python + ["-m", "train.train_ppo",
                              "--config", f"configs/{name}.yaml", "--print-config"],
                    cwd=str(ROOT), env=_env()).returncode
            return code
        print(f"\n{PLAY} {item.label}")
        # 便携环境的前缀是 micromamba run -p ... python，等价命令照原样打印。
        printable = [Path(cmd[0]).name, *cmd[1:]]
        print("  等价命令： " + " ".join(printable).replace(str(ROOT) + os.sep, ""))
        cwd = item.cwd or ROOT
        if cwd != ROOT:
            print(f"  工作目录： {cwd}")
        print()
        sys.stdout.flush()
        if not cwd.exists():
            cwd.mkdir(parents=True, exist_ok=True)
        try:
            code = subprocess.run(cmd, cwd=str(cwd), env=_env(item.pythonpath)).returncode
        except KeyboardInterrupt:
            print("\n[run.py] 已中断")
            return 130
        if code == 0:
            print(f"\n[run.py] {OK} 正常结束")
        else:
            print(f"\n[run.py] {FAIL} 退出码 {code}——往上翻到以 Traceback 开头、"
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
        for i, item in enumerate(items, start=1):
            print(f" {i:2d}. {item.label}")
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
        item = items[int(choice) - 1]
        print(f"\n  {item.note}")
        code = _run(item.key, items, python)
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
