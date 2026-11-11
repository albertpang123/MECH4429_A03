# MECH4429 项目 A03 —— 单旋翼完全失效后的四旋翼定点保持

[English](README.md) · **简体中文**

用强化学习，在仿真中训练一个控制器：一个旋翼失效之后，仍把四旋翼保持在目标点附近。

**当前状态：Gate 3 —— RL 管线在 stub 动力学上已端到端跑通，底座是 gym-pybullet-drones。**
故障模块仍是占位实现；机体也不再是我们自己的——是 gpd 公开的 `cf2x`。
**本仓库里没有任何一个数字可以当作真机结果。** 详见 [docs/G3.md](docs/G3.md)。

2026-11-10 起，整个任务搭在 **[gym-pybullet-drones](third_party/NOTICE.md)** 上：`envs/gpd_env.py`
继承它的 `BaseAviary`，世界、URDF、力/力矩映射、物理步进都来自一个有论文和社区的公开仿真器，
而不是我们手写的代码。之前的两套实现（自研 NumPy 6-DOF 模型，以及同一模型跑在 PyBullet 引擎上）
仍然可以运行，保留下来当迁移的参照，见[三套实现，一个任务](#三套实现一个任务)。

## 团队与分工

| 成员 | Gate | 范围 |
|---|---|---|
| Andy | G1 | 6-DOF 动力学模型、旋翼推力/力矩、车辆参数 |
| Joe | G2 | 故障模型、场景采样 |
| Albert | G3–G4 | Gymnasium 环境、PPO 训练、评估、checkpoint |

Albert 的机器是基准平台（Ubuntu 22.04 双系统）：能在那里跑通的代码，才是我们拿来比较的代码。
其他人应当能复现出同样的数字。

## Gate 状态

| Gate | 内容 | 状态 |
|---|---|---|
| G0 | 接口：坐标系、频率、观测、动作、奖励、种子 | **方案在流转中** —— [docs/INTERFACE.md](docs/INTERFACE.md)，仍可修改 |
| G1 | 6-DOF 动力学 | 本仓库未开始（`src/dynamics/` 为空） |
| G2 | 单旋翼失效模型 | 本仓库未开始（`src/faults/` 为空） |
| G3 | RL 管线、stub 动力学 | **完成** —— 训练 → 保存 → 重载 → 评估可用；89 项测试；底座换成 gym-pybullet-drones，两套旧实现仍可运行 |
| G4 | 按失效时刻/失效旋翼做课程学习 | 未开始 |
| G5 | 最终评估与报告 | 未开始 |

## 仓库结构

```
environment.yml     一条 conda 命令，Linux / Windows 通用，版本钉死
configs/            base.yaml（G0 规格）+ 每次运行一个 YAML，均 extends base
third_party/
  gym_pybullet_drones/  内嵌的 gym-pybullet-drones，钉死在一个 commit（MIT，见 NOTICE.md）
    examples/         上游示例（pid.py、learn.py 等）；`python run.py --run gpd_<名字>` 跑其中一个
envs/
  interface.py      共享类型、观测布局、结构性旋翼顺序  <- 从这里开始读
  config.py         配置加载、继承、digest、校验
  quat.py           四元数工具
  env_factory.py    按 env.impl 选实现：gpd（默认）| legacy
  gpd_env.py        任务跑在 gym-pybullet-drones 的 BaseAviary + cf2x URDF 上  <- 默认
  quadrotor_env.py  同一任务跑在我们自己的动力学上（legacy 实现）
  stub_dynamics.py  G1 占位：6-DOF 刚体，NumPy
  pybullet_dynamics.py  同一模型跑在 PyBullet 引擎上，另有地面接触
  dynamics_factory.py   按 dynamics.backend 选后端：numpy | pybullet（仅 legacy）
  stub_faults.py    G2 占位：failure_time_s 之后屏蔽一个旋翼
train/train_ppo.py  PPO 训练、checkpoint + 元数据 + CSV 日志
train/sweep.py      对比/消融实验网格 -> plan.json + summary.csv
eval/               确定性评估、指标、作图、3D 回放
eval/fault_response.py  已知旋翼失效后，策略到底给其余旋翼下什么指令
tools/platform_check.py 版本 + 三套实现 + 跨机器一致性指纹
tools/crosscheck_gpd.py 三套实现并排一张表
tools/gpd_demo.py   上游示例的启动壳：换 GUI 渲染器、还原窗口、直接调 run(**kwargs)
run.py              点 ▶ 就出菜单：训练 / 评估 / 回放 / 测试 / 体检 / 打印配置 / 上游示例
run_replay.py       3D 回放专用的点击启动器（无需参数）
tests/              接口、stub 物理、训练/保存/重载、pybullet 后端
docs/               INTERFACE.md（G0）、G3.md（管线 + 结果）、rotor_layout.svg
src/dynamics/       Andy：替换 stub_dynamics.py
src/faults/         Joe：替换 stub_faults.py
results/            值得留下的指标与图（进版本控制）
checkpoints/        权重（不进版本控制）
```

## 从 GitHub 下载后怎么跑起来

只需要 Git 和 conda，CPU 就够 —— 不需要 GPU，不需要 ROS，也不需要 C++ 编译器。

**1. 装好前置工具**（每台机器一次）：[Git](https://git-scm.com/downloads) 和
[Miniconda](https://docs.conda.io/en/latest/miniconda.html)（Anaconda 也行）。Windows 上在
**Anaconda Prompt** 里运行下面的命令，或者先执行一次 `conda init powershell` 再重开终端；
Ubuntu 上先执行一次 `conda init bash` 再重开 shell。环境加上内置模拟器大约占 2 GB 磁盘。

**2. 克隆仓库。**

```bash
git clone https://github.com/albertpang123/MECH4429_A03.git
cd MECH4429_A03
```

GitHub 上的仓库叫 `MECH4429_A03`；本地工作副本的文件夹名可能不同。下面所有命令都在
**仓库根目录** 下运行。

**3. 创建环境** —— Python 3.12，环境名 `uav-g3`，Linux 和 Windows 用同一个文件：

```bash
conda env create -f environment.yml     # 几分钟，约 1 GB 下载
conda activate uav-g3
```

`environment.yml` 把每个报告数字对应的版本都钉死了。想自己手装也行：

```bash
conda create -n uav-g3 python=3.12.16 -y
conda activate uav-g3
conda install -c conda-forge numpy=2.5.4 scipy=1.18.1 matplotlib=3.11.2 pillow=13.0.0 \
      gymnasium=1.3.0 transforms3d=0.4.2 cloudpickle=3.1.2 pybullet=3.25 \
      pyyaml pandas=3.1.1 pytest
pip install --extra-index-url https://download.pytorch.org/whl/cpu \
      torch==2.15.0+cpu stable-baselines3==2.9.2
# 本仓库所有东西都用 CPU 就够。
```

`pybullet` 走 conda-forge 是有原因的：PyPI 上**没有任何一个现代版本的 Windows wheel**，
`pip install pybullet` 在 Windows 上要先装 MSVC Build Tools 才能从源码编译。conda-forge 把它
编号为 3.25，包本身是 pybullet 3.2.5。

**4. 确认环境对上了。** 跑一次 `python tools/platform_check.py`（菜单里的 `doctor` 项）。它会
打印版本、把三套实现各跑一遍，并输出一个**一致性指纹**——一段固定的 120 步 rollout。两台机器
打出来的指纹相同，说明它们会产生同样的训练数据；Ubuntu 和 Windows 两份代码就是这样互相校验的。
（指纹来自 legacy 的 NumPy 实现；2026-11-10 升级依赖后 numpy 从 1.26 到 2.5，和那之前的输出
对不上是依赖变了，不是机器坏了。）

**5. 跑测试。** 应该看到 `89 passed`，约 7 秒：

```bash
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 CUDA_VISIBLE_DEVICES="" \
  python -m pytest tests -q
```

`env -u PYTHONPATH` 只在同机装了 ROS 时才有必要（ROS 会往 `PYTHONPATH` 里塞 `python3.10`，
遮住环境里的包）；Windows 上没装 ROS 的话，直接 `python -m pytest tests -q` 就行。这一步过了，
说明装对了，后面的命令都能跑。

**6. 训练第一个策略。** 先来个 4k 步的冒烟（约 10 秒），再跑正式的：

```bash
python -m train.train_ppo --config configs/g3_hover.yaml --timesteps 4096 --name first_run
python -m train.train_ppo --config configs/g3_hover.yaml
```

第二条是 200,704 步——参考机器上约 3 分钟（4 个 env，约 1200 步/秒）；旧机器会慢一些。每次运行
会留下 `checkpoints/<名字>/model.zip`、`meta.json`、`logs/`，以及 `results/<名字>/` 里的学习曲线
和 `progress.csv`。

**7. 评估和回放。**

```bash
python -m eval.evaluate --checkpoint checkpoints/first_run/model.zip --episodes 20
python -m eval.replay   --checkpoint checkpoints/first_run/model.zip --episode 0   # 需要显示器
```

**权重不在仓库里。** `.gitignore` 排除了 `checkpoints/` 和 `*.zip`，所以你能在 `results/` 里看到
的结果，其 checkpoint 并**没有**上传。后面示例里写 `checkpoints/g3_stub_hover/...` 的命令，只有
你自己训练过（或者从别处拿到 checkpoint）之后才能跑——换成你自己的
`checkpoints/<名字>/model.zip`。

**8. 改参数、做对比。** `--set` 可以覆盖任意配置键，`train/sweep.py` 跑网格——见下面的
[常用命令](#常用命令)。

### 启动失败时看这里

| 症状 | 原因 | 解决 |
|---|---|---|
| `conda: command not found` | 当前 shell 没初始化 conda | 用 **Anaconda Prompt**，或执行一次 `conda init bash` / `conda init powershell` 后重开 |
| `ModuleNotFoundError: No module named 'torch'`（或 `gymnasium`） | 命令跑在别的解释器里 | `conda activate uav-g3`；用 `python -c "import sys; print(sys.executable)"` 确认路径 |
| numpy/ROS 相关的 import 错误 | ROS 的 `python3.10` 路径进了 `PYTHONPATH` | `env -u PYTHONPATH ...`（Linux），或在 Windows 上清掉 `PYTHONPATH` |
| `FileNotFoundError: checkpoints/...` | 权重不在 git 里 | 先训练一次（第 6 步） |
| Windows 上 `pip install pybullet` 失败 | PyPI 没有 wheel | 用 `environment.yml`（conda-forge） |
| `Fatal Python error: Aborted`，栈顶在 `numpy.linalg` | 两份 OpenMP 运行时（Windows） | 仓库里每个入口都会设 `KMP_DUPLICATE_LIB_OK=TRUE`；你自己写脚本时照做 |
| `git clone` 报代理错误 | git 配置里有个失效的 `http.proxy` | `git config --global --unset http.proxy`，或 `git -c http.proxy= clone ...` |

更细的环境坑见 [五个一定会坑到你的地方](#五个一定会坑到你的地方)。

### 不想敲命令？用 `run.py`

打开 `run.py`，点右上角的 **Run**（▶），或者直接 `python3 run.py`。它弹一个菜单——训练、评估、
回放、测试、后端对比、打印配置——选一个就启动。不需要参数，也不要求先 activate 环境：它自身只
用标准库，会自己找到 `uav-g3` 的解释器，顺手去掉 `PYTHONPATH`、设好 CPU 标志。`--list` 列出全部
条目，`--run <key>` 直接跑某一个。

最后 12 项是**上游 gym-pybullet-drones 自带的示例**，直接用内嵌副本和同一个环境跑：
`gpd_pid`（三机盘旋、带 RPM 滑块）、`gpd_hover`、`gpd_pid_velocity`、`gpd_mrac`、
`gpd_downwash`、`gpd_pettingzoo`、`gpd_debug`、`gpd_learn_small` / `gpd_learn_full` /
`gpd_play`、`gpd_tests`（上游自己的 13 项测试）和 `gpd_beta`（要先把 Betaflight SITL 编译好，
没编译时菜单直接告诉你，不会先崩一次）。它们演示的是**上游的环境**，不经过我们的观测、奖励、
终止或故障逻辑——证明不了我们任何一条结论。日志落在 `.scratch/gpd_example_results/`，
不会混进 `results/`。

这 12 项都经由 [tools/gpd_demo.py](tools/gpd_demo.py) 启动，这个薄壳只解决两件事。
**首字母被裁**：pybullet 默认的 `OpenGL3+` 窗口会把每个标签的第一个字母裁掉（`File`→`rile`、
`Explorer`→`xplorer`、`Params`→`arams`）。不是 DPI 的问题——声明 DPI 感知、把窗口开大到
1600×900 都试过，照裁；是这个渲染器把每段文字都往左画了一个字宽，正好超出裁剪区。换
legacy 的 `--opengl2` 渲染器字就完整了，代价是没阴影、没地面贴图、球也没纹理。默认不用它
（画面好看），要读全标签时设 `UAV_GPD_GL=2`。**自动结束**：上游示例的时长写死在各自脚本里，5~15 s
（`pid.py` 是 12 s），跑着跑着就退出了，看起来像有人设了定时。菜单把 PID / MRAC / 下洗
这几个演示统一拉到 60 s；启动 `run.py` 前设 `UAV_GPD_DURATION_SEC=300` 可以更长，
关窗口或 Ctrl-C 可以更短。

`run_replay.py` 是更早的、只负责 3D 回放的启动器；菜单里的「回放」直接调它，两者不会走偏。

### 五个一定会坑到你的地方

1. **`env -u PYTHONPATH`** —— 如果同一台机器装了 ROS，它会往 `PYTHONPATH` 里塞一个 `python3.10`
   的 ROS 路径，于是 conda 解释器导入的是 ROS 的 NumPy 而不是环境里的。每条命令都要去掉这个变量。
2. **`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`** —— 基础环境里装的第三方 pytest 插件可能导致收集阶段直接
   报错。这个标志禁止自动加载插件；`tests/` 只需要本仓库里的 `pytest-*.py`。
3. **`CUDA_VISIBLE_DEVICES=""`** —— 报告里的数字是在 CPU 上跑出来的。设置它，免得你自己的 GPU
   悄悄把数字改掉。
4. **Windows 上的 `KMP_DUPLICATE_LIB_OK=TRUE`** —— torch 自带 Intel 的 OpenMP，conda-forge 的
   numpy/scipy 自带另一份；两者同时进一个进程，后加载的那个会在下一次 `numpy.linalg` 调用时
   直接 abort，报错指向 LAPACK，而且没有 Python 回溯。所有入口（`train/`、`eval/`、`run.py`、
   `tools/`、`tests/conftest.py`）都已经在 import torch 之前设好了，你不用手动设。自己写新脚本时
   照做，原因写在 [tests/conftest.py](tests/conftest.py) 顶部。
5. **先 import torch 再碰引擎（Windows，只影响上游示例）** —— pybullet 跑过一次仿真、调用过 numpy
   的 BLAS 之后，MKL 的 `libiomp5md.dll` 就已经在进程里；此时再 `import torch`，它自己那份同名
   DLL 加载不上，报 `WinError 127`（`torch\lib\shm.dll`），而且堆栈里没有我们的一行代码。
   先 import torch 就能绕开。`gpd_tests` 就是这么做的——所以它在这里是 13 passed，而你直接
   `pytest third_party/gym_pybullet_drones/tests` 会挂 2 项。我们自己的测试不受影响（收集阶段
   就 import 了 torch）。

## 常用命令

```bash
# 测试：接口 + stub 物理 + gpd 环境 + 训练/保存/重载
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 CUDA_VISIBLE_DEVICES="" \
  python -m pytest tests -q

# 训练（200k 步，参考机器约 3 分钟：默认环境要真的步进引擎）
env -u PYTHONPATH CUDA_VISIBLE_DEVICES="" \
  python -m train.train_ppo --config configs/g3_hover.yaml

# 评估一个已保存的策略 —— 不学习
env -u PYTHONPATH CUDA_VISIBLE_DEVICES="" \
  python -m eval.evaluate --checkpoint checkpoints/g3_stub_hover/model.zip

# 在 3D 里看其中一集（需要 pybullet，见下文）
env -u PYTHONPATH CUDA_VISIBLE_DEVICES="" \
  python -m eval.replay --checkpoint checkpoints/g3_stub_hover/model.zip --episode 0

# 只改两个值跑一次 —— 不用再复制一份 YAML
env -u PYTHONPATH CUDA_VISIBLE_DEVICES="" \
  python -m train.train_ppo --config configs/g4_fixed_rotor1.yaml \
  --set ppo.n_epochs=5 --set ppo.learning_rate=1.0e-4 --seed 3

# 对比/消融：2 个学习率 × 2 个种子 = 4 次运行，跑完自动评估
env -u PYTHONPATH CUDA_VISIBLE_DEVICES="" \
  python -m train.sweep --config configs/g4_fixed_rotor1.yaml \
  --grid ppo.learning_rate=1e-4,3e-4 --seeds 0,1 --timesteps 200000
```

可覆盖项：训练用 `--timesteps`、`--n-envs`、`--name`、`--device`、`--seed`，以及可重复的
`--set path=value`；评估用 `--episodes`、`--seed-start`、`--out`。

`--set` 写的是合并后配置里的任意点路径，所以做消融不需要一组组合一个 YAML 文件。值按 YAML
解析（`2.0` 是浮点、`true` 是布尔、`[64, 128]` 是列表、`null` 是空）。**路径必须拼对**：
不存在的键会直接报错并列出该层已有的键，而不是静默地什么都不改。科学计数法要写小数点
（`1.0e-4`）——PyYAML 会把裸 `1e-4` 读成字符串。`--print-config` 打出来的就是覆盖之后的
配置，`meta.json` 里的 `config_sha256` 也覆盖了被改的值，所以两次 `--set` 运行几个月后仍能
区分。

`train/sweep.py` 跑 `--grid` 各值和 `--seeds` 的笛卡尔积，每个组合一个
`train.train_ppo` 子进程，名字是 `<运行名>-000`、`-001`……所以不会互相覆盖。它**在第一次
训练开始前**就把完整计划写进 `plan.json`（中途挂掉也留得下要做什么），每跑完一次就往
`summary.csv` 追加一行，并把该次 stdout 存在 `runs/<name>.log`。除非 `--no-eval`，跑完会
自动评估，指标写进同一行。`--dry-run` 只打印计划，`--stop-on-error` 第一次失败就停，
`--out` 指定目录（默认 `results/sweeps/<UTC 时间戳>/`）。网格值只能是标量 ——
`net_arch` 这类列表要给单独的配置文件，因为逗号是值分隔符。

### 3D 回放（可选）

**最简单的办法：打开 [`run_replay.py`](run_replay.py)，点 ▶ Run。** 它不接受参数，播放 `g3_stub_hover`
的第 0 集；如果编辑器选中的解释器缺包，它会自动换到 `uav-g3` 的解释器重新执行；开不了窗口时
改写成 `replay.gif` 而不是报错。`.vscode/settings.json` 固定了解释器并清掉了 ROS 的 `PYTHONPATH`
条目；`.vscode/` 已被 gitignore，那个文件留在本地。

`eval/replay.py` 用 PyBullet 画出一集评估数据。PyBullet **只负责显示** —— 轨迹来自环境自己的
动力学，所以回放不可能改变 `eval.json` 里的任何数字。第 `n` 集用的是评估器的种子
（`seed_start + n`），所以 `--episode 0` 就是
[results/g3_stub_hover/eval_trajectory.png](results/g3_stub_hover/eval_trajectory.png) 里那一集。

PyBullet 现在**不是可选项**——`env.impl: gpd` 就跑在它上面——所以它写在 `environment.yml` 里
（来自 conda-forge，不是 PyPI：Windows 没有 wheel）。

窗口里的操作：**Ctrl + 左键拖拽**转视角（普通拖拽在 PyBullet 的 ExampleBrowser 里是"抓取物体"，
这里已经关掉了，所以普通拖拽什么也不会发生），滚轮缩放，空格暂停，q 退出。一集结束后窗口会停在
最后一帧，直到你按 q —— 这样 10 s 的回放看起来不像崩溃。模型由
`dynamics.rotor_positions_body_m` 现场生成，所以旋翼布局就是配置里写的那个，不存在第二份需要同步
的机架数据。失效旋翼在失效那一步变成橙色。桨盘会转，但转速只是视觉效果，不是物理量。

`--record out.gif` 会写一个约 1 MB 的 GIF 而不开窗口，给没装 PyBullet 的人看。大约要 1 分 45 秒：
这台机器上离屏渲染约 0.45 s 一帧（离屏渲染器退化成软件光栅化），而且只渲染 GIF 真正保留的帧。
渲染全程离屏，所以文件在任何机器上都一样。已提交的示例：
[results/g3_stub_hover/replay_seed100000.gif](results/g3_stub_hover/replay_seed100000.gif)。

当前 checkpoint 是用 `faults.mode: nominal` 训的，所以里面不会发生失效——失效的视觉效果
（`faults.mode: fixed` 或 `random`）需要一个在故障下训练过的 checkpoint，那是 G4 的事。

### 三套实现，一个任务

配置里的 `env.impl` 决定用哪一套**实现**；对两套 legacy 实现，`dynamics.backend` 再决定下面积分器。
它之上的一切——观测、奖励、终止、故障注入、训练、评估——都不受影响。迁移之所以便宜、之所以可回退，
就靠这一层。

| `env.impl` | 类 | 机体 | 是什么 |
|---|---|---|---|
| `gpd`（默认） | `envs/gpd_env.py` | gpd 的 `cf2x` URDF | `BaseAviary` 子类，内嵌的 gym-pybullet-drones |
| `legacy` + `dynamics.backend: numpy` | `envs/quadrotor_env.py` | 配置里的 `dynamics.*` | 手写 6-DOF 半隐式欧拉 |
| `legacy` + `dynamics.backend: pybullet` | `envs/quadrotor_env.py` | 配置里的 `dynamics.*` | 同一模型跑在 PyBullet 引擎上，另有地面 |

`envs/env_factory.py` 按配置构造。legacy 不是留着好看的：到目前为止所有 G0–G4 的数字都是它产出的，
而且它让「和 gpd 的结果不一致」变成五分钟能查清的事，而不是一下午的考古。

**机体不一样，这点必须记住。** `gpd` 飞的是公开的 `cf2x`：0.027 kg、臂长 0.0397 m、推重比 **2.25**、
推力与归一化指令成**平方**关系（`KF · rpm²`）。legacy 飞的是 `configs/*.yaml` 里的占位机体：
0.5 kg、臂长 0.092 m、推重比 **1.63**、推力**线性**。同一个任务，不同的机器——2026-11-10 之前的
数字不能和 `gpd` 的数字逐点比较。

```bash
# 三套并排：机体、悬停、推力自洽性、每步耗时
env -u PYTHONPATH CUDA_VISIBLE_DEVICES="" \
  python tools/crosscheck_gpd.py
```

Windows 11 / CPython 3.12 实测（2026-11-10）：三套实现都能用自己的悬停指令保持高度，并且各自
「报告的推力」和「实际施加的推力」在 0.06 %（legacy）和 0.29 %（gpd）以内一致。每个控制步耗时：
**gpd 0.17 ms**，PyBulletDynamics 0.34 ms，NumPy stub 0.47 ms——gpd 最快，因为它不做每步的
URDF 记账。

legacy 引擎后端与 legacy NumPy 后端之间由
[tests/test_pybullet_backend.py](tests/test_pybullet_backend.py)（15 项）验证：关掉地面时，两者在
空中 **4 s 内位置差不超过 0.041 mm** —— 包括飞了 35 m、翻滚超过 100° 的用例 —— 悬停更是逐位相同。
开启地面后，它们正好在该不同的地方不同：stub 没有地板，会穿过 z = 0 掉下去；引擎会落地。

* **PyBullet 的 `linearDamping` 不是我们的阻尼系数。** 我们的是每单位速度的力（N/(m/s)），
  Bullet 的是衰减率（1/s）。把两者直接对应会悄悄改掉模型——实测表现为 4 s 内 15 m 的发散。
  所以引擎阻尼被设为 0，阻尼项按显式的力/力矩施加，和 stub 完全一致。
* **地面接触是特性，不是意外。** `base.yaml` 里的 `termination.use_ground_contact` 仍是 G0 的
  待定问题，所以 legacy pybullet 后端有地面、NumPy 后端没有。`gpd` 永远有地面，它自己的地面效应
  模型可以通过 `env.gpd_physics: pyb_gnd` 打开。

#### 旋翼编号：注入故障前唯一必须读的东西

我们的编号是 R0 前右、R1 前左、R2 后左、R3 后右。gpd 的 `cf2x.urdf` 把旋翼链节编号为
**FR、RR、RL、FL**——所以 `gpd 0 = R0`、`gpd 1 = R3`、`gpd 2 = R2`、`gpd 3 = R1`。
这个映射只写在 [envs/gpd_env.py](envs/gpd_env.py) 的 `ROTOR_GPD_INDEX` 一处，所有指令在
`_preprocessAction` 里按它重排。

搞错的话，悬停看起来完美（四个电机一样），而「旋翼 1 失效」实际上打掉的是旋翼 3。
[tests/test_gpd_env.py](tests/test_gpd_env.py) 逐个打掉旋翼、对照几何检查角速度符号，把这个映射钉住了。

### 改参数

所有可调值都在 `configs/base.yaml` 里：车辆参数、奖励权重、终止阈值、观测缩放、reset 抖动。
运行配置继承它、只覆盖不同的部分，所以共享的值只需要改**一处**：

```yaml
# configs/my_run.yaml
extends: base.yaml
dynamics:
  mass_kg: 0.5
```

```bash
# 打印完全合并后的配置及其 digest，不训练
env -u PYTHONPATH /home/albertpang/miniconda3/envs/uav-g3/bin/python \
  -m train.train_ppo --config configs/g3_hover.yaml --print-config
```

加载配置请用 `envs.config.load_config`，不要用 `yaml.safe_load` —— 后者只返回子文件，会悄悄丢掉
所有继承来的值。每次运行都会把**解析后**的配置及其 `config_sha256` 写进 `meta.json`，所以任何结果
都能追溯回产生它的那组参数。

结构性取值是刻意**不可**配置的：旋翼编号/标签/转向、`[0, 1]` 动作界、17 字段观测布局就是接口，
改它们会让所有已有 checkpoint 失效。

**GPU（可选启用）。** 传 `--device cuda`，且**不要**设 `CUDA_VISIBLE_DEVICES=""`。在 RTX 4080 Laptop
上这能跑，靠的是 CUDA 向前兼容（计算能力 8.9，驱动 580.178.04 / CUDA 13.0）——这组配置"够用于实验"，
但不是官方支持组合。要长期稳定，请在独立环境里装 `torch==2.4.1+cu121` 并用 `PYTHONNOUSERSITE=1`
运行。细节见 [docs/G3.md](docs/G3.md)。

## 接口 —— 从这里开始读

我们三个人都会用到，所以这是最先该读的东西：

| 问题 | 答案 |
|---|---|
| 世界坐标系 | ENU，x 东 / y 北 / z 上，原点在起飞点 |
| 机体坐标系 | FLU，x 前 / y 左 / z 上 |
| 姿态 | 四元数 `(x, y, z, w)`，body → world，Hamilton，`w ≥ 0` |
| 物理 / 控制频率 | 240 Hz / 60 Hz（每个动作恰好 4 个物理步） |
| 回合 | 10 s = 600 个控制步 |
| 旋翼顺序 | 0 右前、1 左前、2 左后、3 右后（俯视逆时针） |
| 动作 | `(4,)`，范围 `[0, 1]`，归一化电机指令，不是牛顿 |
| 观测 | 17 个值 × 10 帧 = `(170,)` `float32` |
| 种子 | 训练 0–999，评估 100000–100999（两者永不重叠） |

完整表格、奖励项和待定问题：[docs/INTERFACE.md](docs/INTERFACE.md) 与 `configs/base.yaml`。
共享类型：`envs/interface.py`。旋翼几何：[docs/rotor_layout.svg](docs/rotor_layout.svg)。

### 策略绝不能看到故障

失效旋翼 ID、失效时刻、剩余效率、故障标志、场景种子只进 `info`（仅用于记录）。观测构造函数只接受
`(state, target, recent commands)`，别的一概不给 —— `tests/test_interface.py` 通过检查函数签名来
强制这一点。

## 目前的结果

名义悬停，`configs/g3_hover.yaml`，200,704 步，在 20 个未参与训练的种子上评估：20/20 集跑满时限，
最终位置误差约 0.24 m，z 方向有约 0.17 m 的稳定偏置。策略学会的是**存活**，不是紧密保持，而且训练在
200k 处还没收敛。数字与解读：[docs/G3.md](docs/G3.md)。

上面这些数字来自 **legacy 实现**（`env.impl: legacy` + NumPy）。2026-11-10 把默认实现换成 `gpd`
之后，同一个 checkpoint 仍然要在 legacy 上评估——`meta.json` 里存了它训练时的配置，`make_env`
会照着构造——所以这些数字没有被迁移改掉。`gpd` 是另一台机器（0.027 kg、推重比 2.25），要在它上面
得到可比的结果需要重新训练；旧数字不能平移过来。

`gpd` 的第一批基线（2026-11-11，各 200,704 步、4 个 env）：名义悬停和"1 号桨在 2.0 s 停转"
**都是 20/20 坠机**，平均每集 37–40 步（10 s 的一集只跑了约 0.6 s）。故障那次连触发都没有
（评估里每集 `fault_active = False`），所以这两次还说明不了故障恢复——两个策略都还没学会悬停。
数字见 [results/g3_nominal_gpd](results/g3_nominal_gpd/) 和
[results/g4_rotor1_gpd](results/g4_rotor1_gpd/)。

## 不进版本控制的东西

`checkpoints/`、`*.zip`、`*.pt`、`runs/`、`wandb/`、`.scratch/`、`__pycache__/`、
`.pytest_cache/`。配置、指标（`results/*.json`、`results/*.csv`）和图片**要**进 —— 它们是我们
比较各次运行的依据。

## stub 的已知缺口

这些是刻意的；G1/G2 落地后会消失，但不要把设计建立在它们之上：

* 两套实现都自己构造 `StubFaultInjector`（`envs/quadrotor_env.py`、`envs/gpd_env.py`）——故障模块
  还不能注入，和动力学不一样。
* 旋翼位置是配置里的车辆数据，但 stub 只通过位置本身假设对称 X 形机架；非对称机架会被接受而不报
  警。在意的话拿 `nominal_x_positions()` 对一下。（`gpd` 这条不适用：位置来自 URDF。）
* legacy 动作到推力假定是**线性**的（`f = T_max · u`）。`gpd` 用的是 `KF · rpm²`，对指令是二次的。
  同一个指令在两套实现里不代表同一个力。
* `pybullet` 后端的地面接触还没有接进终止条件 —— 现在是否坠机仍由 `termination.crash_z` 判定。
  `gpd` 也一样：它有地面，但坠机判据仍是高度。

## 在这个仓库里工作

* 只有一个 `main` 分支。小改动直接推；任何触及 `envs/interface.py`、`configs/base.yaml` 或
  `docs/INTERFACE.md` 的改动请提 PR —— 那是共享契约，别人的代码依赖它。
* 每个你在报告里引用的关键数字，都要对应一个 `results/<run>/` 条目，这样能追溯回去。
* 某个 gate 有进展时，更新上面的 gate 表。
