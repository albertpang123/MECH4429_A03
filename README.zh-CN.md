# MECH4429 项目 A03 —— 单旋翼完全失效后的四旋翼定点保持

[English](README.md) · **简体中文**

用强化学习，在仿真中训练一个控制器：一个旋翼失效之后，仍把四旋翼保持在目标点附近。

**当前状态：Gate 3 —— RL 管线在 stub 动力学上已端到端跑通。** 物理模块和故障模块都是占位实现，
它们存在的意义是让学习这一侧能在 Gate 1–2 就绪前先做起来。**本仓库里没有任何一个数字可以当作真机结果。**
详见 [docs/G3.md](docs/G3.md)。

同一条接口下有两个可互换的动力学后端——NumPy stub（默认）和 PyBullet 引擎。两者在 4 s 飞行中的
位置差不超过 0.041 mm，见[两个动力学后端](#两个动力学后端)。

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
| G3 | RL 管线、stub 动力学 | **完成** —— 训练 → 保存 → 重载 → 评估可用；61 项测试；两个可互换的动力学后端 |
| G4 | 按失效时刻/失效旋翼做课程学习 | 未开始 |
| G5 | 最终评估与报告 | 未开始 |

## 仓库结构

```
environment.yml     一条 conda 命令，Linux / Windows 通用，版本钉死
configs/            base.yaml（G0 规格）+ 每次运行一个 YAML，均 extends base
envs/
  interface.py      共享类型、观测布局、结构性旋翼顺序  <- 从这里开始读
  config.py         配置加载、继承、digest、校验
  quat.py           四元数工具
  quadrotor_env.py  Gymnasium 任务（观测、动作、奖励、终止）
  stub_dynamics.py  G1 占位：6-DOF 刚体，NumPy（默认后端）
  pybullet_dynamics.py  同一模型跑在 PyBullet 引擎上，另有地面接触
  dynamics_factory.py   按 dynamics.backend 选后端：numpy | pybullet
  stub_faults.py    G2 占位：failure_time_s 之后屏蔽一个旋翼
train/train_ppo.py  PPO 训练、checkpoint + 元数据 + CSV 日志
eval/               确定性评估、指标、作图、3D 回放
eval/fault_response.py  已知旋翼失效后，策略到底给其余旋翼下什么指令
tools/platform_check.py 版本 + 两个后端 + 跨机器一致性指纹
run.py              点 ▶ 就出菜单：训练 / 评估 / 回放 / 测试 / 体检 / 打印配置
run_replay.py       3D 回放专用的点击启动器（无需参数）
tests/              接口、stub 物理、训练/保存/重载、pybullet 后端
docs/               INTERFACE.md（G0）、G3.md（管线 + 结果）、rotor_layout.svg
src/dynamics/       Andy：替换 stub_dynamics.py
src/faults/         Joe：替换 stub_faults.py
results/            值得留下的指标与图（进版本控制）
checkpoints/        权重（不进版本控制）
```

## 快速开始

Ubuntu 22.04，Python 3.10，conda 环境名 `uav-g3`：

```bash
conda env create -f environment.yml     # 一条命令，Linux / Windows 通用
conda activate uav-g3
```

`environment.yml` 把每个报告数字对应的版本都钉死了。想自己手装也行：

```bash
conda create -n uav-g3 python=3.10 -y
conda activate uav-g3
pip install numpy==1.26.4 gymnasium==0.29.1 stable-baselines3==2.2.1 \
            pandas==2.0.3 matplotlib==3.7.5 pyyaml pytest==7.4.4
# 本仓库所有东西都用 CPU 就够。torch 会随 stable-baselines3 一起装上。
```

装完之后跑一次 `python tools/platform_check.py`（菜单里的 `doctor` 项）。它会打印版本、把两个
动力学后端各跑一遍，并输出一个**一致性指纹**——一段固定的 120 步 rollout。两台机器打出来的指纹
相同，说明它们会产生同样的训练数据；Ubuntu 和 Windows 两份代码就是这样互相校验的。

下面所有命令都在 **仓库根目录** 下运行。

### 不想敲命令？用 `run.py`

打开 `run.py`，点右上角的 **Run**（▶），或者直接 `python3 run.py`。它弹一个菜单——训练、评估、
回放、测试、后端对比、打印配置——选一个就启动。不需要参数，也不要求先 activate 环境：它自身只
用标准库，会自己找到 `uav-g3` 的解释器，顺手去掉 `PYTHONPATH`、设好 CPU 标志。`--list` 列出全部
条目，`--run <key>` 直接跑某一个。

`run_replay.py` 是更早的、只负责 3D 回放的启动器；菜单里的「回放」直接调它，两者不会走偏。

### 三个一定会坑到你的地方

1. **`env -u PYTHONPATH`** —— 如果同一台机器装了 ROS，它会往 `PYTHONPATH` 里塞一个 `python3.10`
   的 ROS 路径，于是 conda 解释器导入的是 ROS 的 NumPy 而不是环境里的。每条命令都要去掉这个变量。
2. **`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`** —— 基础环境里装的第三方 pytest 插件可能导致收集阶段直接
   报错。这个标志禁止自动加载插件；`tests/` 只需要本仓库里的 `pytest-*.py`。
3. **`CUDA_VISIBLE_DEVICES=""`** —— 报告里的数字是在 CPU 上跑出来的。设置它，免得你自己的 GPU
   悄悄把数字改掉。

## 常用命令

```bash
# 测试：接口 + stub 物理 + 训练/保存/重载
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 CUDA_VISIBLE_DEVICES="" \
  python -m pytest tests -q

# 训练（200k 步，CPU 约 200 s）
env -u PYTHONPATH CUDA_VISIBLE_DEVICES="" \
  python -m train.train_ppo --config configs/g3_hover.yaml

# 评估一个已保存的策略 —— 不学习
env -u PYTHONPATH CUDA_VISIBLE_DEVICES="" \
  python -m eval.evaluate --checkpoint checkpoints/g3_stub_hover/model.zip

# 在 3D 里看其中一集（需要 pybullet，见下文）
env -u PYTHONPATH CUDA_VISIBLE_DEVICES="" \
  python -m eval.replay --checkpoint checkpoints/g3_stub_hover/model.zip --episode 0
```

可覆盖项：训练用 `--timesteps`、`--n-envs`、`--name`、`--device`；
评估用 `--episodes`、`--seed-start`、`--out`。

### 3D 回放（可选）

**最简单的办法：打开 [`run_replay.py`](run_replay.py)，点 ▶ Run。** 它不接受参数，播放 `g3_stub_hover`
的第 0 集；如果编辑器选中的解释器缺包，它会自动换到 `uav-g3` 的解释器重新执行；开不了窗口时
改写成 `replay.gif` 而不是报错。`.vscode/settings.json` 固定了解释器并清掉了 ROS 的 `PYTHONPATH`
条目；`.vscode/` 已被 gitignore，那个文件留在本地。

`eval/replay.py` 用 PyBullet 画出一集评估数据。PyBullet **只负责显示** —— 轨迹来自环境自己的
动力学，所以回放不可能改变 `eval.json` 里的任何数字。第 `n` 集用的是评估器的种子
（`seed_start + n`），所以 `--episode 0` 就是
[results/g3_stub_hover/eval_trajectory.png](results/g3_stub_hover/eval_trajectory.png) 里那一集。

```bash
pip install pybullet==3.2.7     # 多一个依赖，训练不需要
```

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

### 两个动力学后端

配置里的 `dynamics.backend` 决定用哪个积分器。它之上的一切——观测、奖励、故障注入、训练、评估——
都不受影响，这正是重点：环境只调用 `reset`、`step`、`applied_thrusts` 和 `hover_command`。

| `dynamics.backend` | 类 | 是什么 |
|---|---|---|
| `numpy`（默认） | `envs/stub_dynamics.py` | 手写 6-DOF 半隐式欧拉 |
| `pybullet` | `envs/pybullet_dynamics.py` | 同一模型跑在 PyBullet 引擎上，另有地面 |

`envs/dynamics_factory.py` 按配置构造；[configs/g3_pybullet.yaml](configs/g3_pybullet.yaml) 与
`g3_hover.yaml` 只差这一行，所以两者的对比只隔离出积分器这一个变量。

```bash
# 在引擎后端上训练（同样 200k 步，其他一切不变）
env -u PYTHONPATH CUDA_VISIBLE_DEVICES="" \
  python -m train.train_ppo --config configs/g3_pybullet.yaml
```

由 [tests/test_pybullet_backend.py](tests/test_pybullet_backend.py)（15 项）验证：关掉地面时，两个
后端在空中 **4 s 内位置差不超过 0.041 mm** —— 包括飞了 35 m、翻滚超过 100° 的用例 —— 悬停更是
逐位相同。开启地面后，它们正好在该不同的地方不同：stub 没有地板，会穿过 z = 0 掉下去；引擎会落地。
这是唯一有意的差别。

两件事值得记住：

* **PyBullet 的 `linearDamping` 不是我们的阻尼系数。** 我们的是每单位速度的力（N/(m/s)），
  Bullet 的是衰减率（1/s）。把两者直接对应会悄悄改掉模型——实测表现为 4 s 内 15 m 的发散。
  所以引擎阻尼被设为 0，阻尼项按显式的力/力矩施加，和 stub 完全一致。
* **引擎更快，不是更慢。** 通过 `QuadrotorEnv` 实测：3078 步/s（每个控制步 0.32 ms），
  NumPy stub 是 1755 步/s（0.57 ms）。

地面接触是特性，不是意外：`base.yaml` 里的 `termination.use_ground_contact` 仍是 G0 的待定问题，
所以 pybullet 后端就是有地面，numpy 后端就是没有。

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

默认后端仍是 `numpy`，所以加入 PyBullet 后端没有改变上述任何数字 —— 原来的 checkpoint 重新评估
得到完全相同的值。

## 不进版本控制的东西

`checkpoints/`、`*.zip`、`*.pt`、`runs/`、`wandb/`、`__pycache__/`、`.pytest_cache/`。
配置、指标（`results/*.json`、`results/*.csv`）和图片**要**进 —— 它们是我们比较各次运行的依据。

## stub 的已知缺口

这些是刻意的；G1/G2 落地后会消失，但不要把设计建立在它们之上：

* `QuadrotorEnv` 自己构造 `StubFaultInjector`（`envs/quadrotor_env.py`）—— 故障模块还不能注入，
  和动力学不一样。
* 旋翼位置是配置里的车辆数据，但 stub 只通过位置本身假设对称 X 形机架；非对称机架会被接受而不报
  警。在意的话拿 `nominal_x_positions()` 对一下。
* 动作到推力假定是**线性**的（`f = T_max · u`）。如果真实映射变成 RPM 的二次关系，已有 checkpoint
  的含义就变了。
* `pybullet` 后端的地面接触还没有接进终止条件 —— 现在是否坠机仍由 `termination.crash_z` 判定。

## 在这个仓库里工作

* 只有一个 `main` 分支。小改动直接推；任何触及 `envs/interface.py`、`configs/base.yaml` 或
  `docs/INTERFACE.md` 的改动请提 PR —— 那是共享契约，别人的代码依赖它。
* 每个你在报告里引用的关键数字，都要对应一个 `results/<run>/` 条目，这样能追溯回去。
* 某个 gate 有进展时，更新上面的 gate 表。
