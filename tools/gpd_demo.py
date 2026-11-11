#!/usr/bin/env python3
"""跑上游 gym-pybullet-drones 示例的薄壳，只补上游自己做不到的两件事。

一、换 GUI 渲染器
-----------------
pybullet 默认那个 ``OpenGL3+ [btgl]`` 窗口在这台机器上会把**每个标签的第一个
字符裁掉**：``File``→``rile``、``Explorer``→``xplorer``、``Test``→``est``、
``Physics Server``→``hysics Server``、参数面板的 ``Params``→``arams``、
``Propeller 0 RPM``→``ropeller 0 RPM``。不是 DPI 的问题——声明 DPI 感知后
（2560x1440 @125%，进程报 125 dpi）照样裁；把窗口开大到 1396x805 也照样裁；
换成 ``--opengl2``（legacy 渲染器）文字就完整了。所以这是 btg 渲染器自己的
文字绘制偏了一个字宽，Python 侧没有别的开关。

代价是 legacy 渲染器没有阴影、地面贴图和球体纹理（窗口标题会变成 "limited
OpenGL2 fallback"）。两个都要不了，所以默认不换（画面好看），只有连着
``--gl2`` 才换——``run.py`` 那边由 ``UAV_GPD_GL=2`` 控制。

上游的 env 写死了 ``p.connect(p.GUI)``（``envs/BaseAviary.py:149``），没有参数
能把 options 传进去，所以只能在这里把 ``pybullet.connect`` 包一层，在连 GUI
时补上 ``--opengl2``。

二、把窗口从最小化里拉出来
--------------------------
legacy 渲染器的窗口在这台机器上创建出来就是最小化的：任务栏里有，屏幕上看不
到，``GetWindowRect`` 返回 Windows 给最小化窗口的 -24576 坐标。所以连上之后
延时几次 ``ShowWindow(SW_RESTORE)``，不打扰用户。

用法（``run.py`` 就是这么调的）
-------------------------------
    python tools/gpd_demo.py module pid --gui True --plot False
    python tools/gpd_demo.py --gl2 call mrac '{"gui": true, "plot": false}'

``module`` 是 ``python -m gym_pybullet_drones.examples.<name>`` 的等价物（参数
原样透传，示例自己的 argparse 负责解析）；``call`` 用于不接受命令行参数的示例
（``mrac.py``）或者默认值不适合当演示的示例（训练 1e6 步），直接
``run(**kwargs)``。

不加 ``--gl2`` 时本文件只做转发，不 import pybullet——保持和上游一样的导入顺序
（Windows 上先跑 pybullet 再 import torch 会报 WinError 127 / shm.dll）。
"""

from __future__ import annotations

import importlib
import json
import runpy
import sys

PACKAGE = "gym_pybullet_drones.examples"


def _patch_gui_renderer() -> None:
    """让示例里的 ``p.connect(p.GUI)`` 带上 ``--opengl2``。"""
    import pybullet

    original = pybullet.connect

    def connect(mode=0, *args, **kwargs):
        if mode != pybullet.GUI:
            return original(mode, *args, **kwargs)
        if args:                                    # options 按位置传进来的情况
            args = ((args[0] + " --opengl2").strip(), *args[1:])
        else:
            options = str(kwargs.get("options") or "")
            kwargs["options"] = (options + " --opengl2").strip()
        client = original(mode, *args, **kwargs)
        _restore_window_soon()
        return client

    pybullet.connect = connect


def _restore_window_soon() -> None:
    """连上 GUI 后隔几拍把窗口还原。窗口不一定在 connect 返回时就已经建好，
    所以试三次；daemon 线程，不影响进程退出。"""
    if sys.platform != "win32":
        return
    import threading

    for delay in (1.0, 3.0, 6.0):
        timer = threading.Timer(delay, _restore_bullet_window)
        timer.daemon = True
        timer.start()


def _restore_bullet_window() -> None:
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    handles: list[int] = []
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def visit(hwnd, _):
        length = user32.GetWindowTextLengthW(hwnd)
        if length:
            buffer = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buffer, length + 1)
            if buffer.value.startswith("Bullet Physics"):
                handles.append(hwnd)
        return True

    user32.EnumWindows(callback_type(visit), 0)     # 回调对象要在调用期间活着
    for hwnd in handles:
        if user32.IsIconic(hwnd):
            user32.ShowWindow(hwnd, 9)              # SW_RESTORE


def main(argv: list[str]) -> int:
    legacy = "--gl2" in argv
    argv = [arg for arg in argv if arg != "--gl2"]
    if len(argv) < 2 or argv[0] not in ("module", "call"):
        print(__doc__)
        return 2
    mode, name, rest = argv[0], argv[1], argv[2:]

    if legacy:
        _patch_gui_renderer()

    if mode == "call":
        kwargs = json.loads(rest[0]) if rest else {}
        importlib.import_module(f"{PACKAGE}.{name}").run(**kwargs)
        return 0

    sys.argv = [name, *rest]
    runpy.run_module(f"{PACKAGE}.{name}", run_name="__main__", alter_sys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
