#!/usr/bin/env python3
"""3D replay of one evaluation episode in PyBullet.

PyBullet is used **for display only**.  The trajectory being replayed was produced
by the environment's own dynamics (the NumPy stub today, Andy's G1 later): this
tool records the episode headless, then teleports a visual model along it.
PyBullet never integrates the flight, so nothing here can change a number in
``eval.json``.

Example:
    python -m eval.replay --checkpoint checkpoints/g3_stub_hover/model.zip --episode 0
    python -m eval.replay --checkpoint checkpoints/g3_stub_hover/model.zip \\
        --episode 0 --record replay_seed100000.gif

``--episode n`` uses the same seed as the evaluator (``seed_start + n``), so
episode 0 is the episode plotted in ``results/<run>/eval_trajectory.png``.

In the window: drag to orbit, scroll to zoom, space pauses, q quits.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.interface import ROTOR_LAYOUT, nominal_x_positions  # noqa: E402
from envs.quadrotor_env import QuadrotorEnv  # noqa: E402
from eval.evaluate import resolve_checkpoint  # noqa: E402

try:  # optional dependency: only this module needs it
    import pybullet as p
    import pybullet_data
except ImportError as exc:  # pragma: no cover - environment dependent
    raise ImportError(
        "3D replay needs PyBullet. Install it into the same environment:\n"
        "    pip install pybullet==3.2.7"
    ) from exc

# rgba in [0, 1]. Blue = healthy rotor, orange = failed rotor, aqua = target:
# the same three hues the evaluation plots use.
BODY_RGBA = [0.17, 0.22, 0.28, 1.0]
ARM_RGBA = [0.55, 0.58, 0.62, 1.0]
ROTOR_RGBA = [0.165, 0.471, 0.839, 1.0]
FAILED_RGBA = [0.922, 0.408, 0.204, 1.0]
TARGET_RGBA = [0.106, 0.686, 0.478, 1.0]
START_RGBA = [0.35, 0.37, 0.40, 1.0]
PATH_RGBA = [0.62, 0.65, 0.69, 1.0]
POLE_RGBA = [0.106, 0.686, 0.478, 0.35]


# --------------------------------------------------------------------- geometry
def rotor_positions_body_m(config: dict) -> np.ndarray:
    """Rotor positions from the run config, with the pre-refactor fallback.

    Checkpoints trained before ``dynamics.arm_length_m`` became
    ``dynamics.rotor_positions_body_m`` still load in ``StubDynamics``; replay has
    to read them through the same door or those runs cannot be drawn.
    """
    dynamics = config["dynamics"]
    positions = dynamics.get("rotor_positions_body_m")
    if positions is None and dynamics.get("arm_length_m") is not None:
        positions = nominal_x_positions(float(dynamics["arm_length_m"]))
    if positions is None:
        raise ValueError(
            "dynamics.rotor_positions_body_m is null — there is no airframe to draw "
            "until Andy's CAD values are filled in"
        )
    return np.asarray(positions, dtype=np.float64)


def airframe_dimensions(rotor_positions: np.ndarray) -> dict:
    """Drawing dimensions derived from the configured rotor positions.

    Only the layout is derived — nothing here is a vehicle parameter, so the
    model follows the config when Andy's CAD numbers land.
    """
    positions = np.asarray(rotor_positions, dtype=np.float64)
    if positions.shape != (4, 2):
        raise ValueError(
            f"dynamics.rotor_positions_body_m must be 4 x (x, y), got {positions.shape}"
        )
    arm = float(np.max(np.linalg.norm(positions, axis=1)))
    if arm <= 0.0:
        raise ValueError("rotor positions are all at the centre of mass")
    return {
        "arm_m": arm,
        "body_size": 0.66 * arm,
        "body_height": 0.20 * arm,
        "arm_radius": max(0.0015, 0.035 * arm),
        "rotor_radius": 0.33 * arm,
        "rotor_thickness": max(0.002, 0.030 * arm),
        "rotor_z": 0.05 * arm,
    }


def _inertial(mass: float, ixx: float, iyy: float, izz: float) -> str:
    return (
        f'<inertial><origin xyz="0 0 0" rpy="0 0 0"/><mass value="{mass:.6g}"/>'
        f'<inertia ixx="{ixx:.6g}" ixy="0" ixz="0" iyy="{iyy:.6g}" iyz="0" izz="{izz:.6g}"/>'
        f"</inertial>"
    )


def _material(name: str, rgba) -> str:
    return (
        f'<material name="{name}"><color rgba="'
        f'{rgba[0]:.4f} {rgba[1]:.4f} {rgba[2]:.4f} {rgba[3]:.4f}"/></material>'
    )


def urdf_from_config(config: dict, name: str = "quadrotor_replay") -> str:
    """Build a URDF whose rotor layout matches ``configs/*.yaml``.

    Generated rather than checked in so the drawing cannot drift from the
    physics: change ``dynamics.rotor_positions_body_m`` and the model moves with
    it.  Rotors are revolute joints about +z so they can be spun cosmetically.
    Deliberately has no collision shapes — this model is never simulated.
    """
    positions = rotor_positions_body_m(config)
    dims = airframe_dimensions(positions)
    dynamics = config["dynamics"]

    mass = float(dynamics.get("mass_kg") or 0.5)
    inertia = dynamics.get("inertia_kgm2") or [mass * 0.01, mass * 0.01, mass * 0.02]
    rotor_mass = 0.02 * mass
    r = dims["rotor_radius"]
    rotor_izz = 0.5 * rotor_mass * r * r
    rotor_ixx = rotor_mass * (3.0 * r * r + dims["rotor_thickness"] ** 2) / 12.0

    lines = [
        '<?xml version="1.0"?>',
        f'<robot name="{name}">',
        '  <link name="base_link">',
        "    " + _inertial(mass, float(inertia[0]), float(inertia[1]), float(inertia[2])),
        "    <visual>",
        '      <origin xyz="0 0 0" rpy="0 0 0"/>',
        f'      <geometry><box size="{dims["body_size"]:.6f} {dims["body_size"]:.6f} '
        f'{dims["body_height"]:.6f}"/></geometry>',
        "      " + _material("body", BODY_RGBA),
        "    </visual>",
    ]
    for index, (x, y) in enumerate(positions):
        angle = float(np.arctan2(y, x))
        # A URDF cylinder points along +z. Roll 90 deg puts it along -y; yawing by
        # angle + 90 deg then aims it along the arm.
        rpy = f"1.5707963 0 {angle + np.pi / 2:.6f}"
        lines += [
            "    <visual>",
            f'      <origin xyz="{0.5 * x:.6f} {0.5 * y:.6f} 0" rpy="{rpy}"/>',
            f'      <geometry><cylinder radius="{dims["arm_radius"]:.6f}" '
            f'length="{dims["arm_m"]:.6f}"/></geometry>',
            "      " + _material(f"arm{index}", ARM_RGBA),
            "    </visual>",
        ]
    lines.append("  </link>")

    for index, ((x, y), (label, spin)) in enumerate(zip(positions, ROTOR_LAYOUT)):
        lines += [
            f'  <link name="rotor{index}">',
            "    " + _inertial(rotor_mass, rotor_ixx, rotor_ixx, rotor_izz),
            "    <visual>",
            '      <origin xyz="0 0 0" rpy="0 0 0"/>',
            f'      <geometry><cylinder radius="{r:.6f}" '
            f'length="{dims["rotor_thickness"]:.6f}"/></geometry>',
            "      " + _material(f"disc{index}", ROTOR_RGBA),
            "    </visual>",
            "  </link>",
            f'  <joint name="rotor{index}_joint" type="revolute">',
            '    <parent link="base_link"/>',
            f'    <child link="rotor{index}"/>',
            f'    <origin xyz="{x:.6f} {y:.6f} {dims["rotor_z"]:.6f}" rpy="0 0 0"/>',
            '    <axis xyz="0 0 1"/>',
            '    <limit lower="-1000" upper="1000" effort="1" velocity="1000"/>',
            f'    <!-- {label}, spins {"CCW" if spin > 0 else "CW"} seen from above -->',
            "  </joint>",
        ]
    lines.append("</robot>")
    return "\n".join(lines) + "\n"


def _quat_aligning_z_with(axis: np.ndarray) -> list:
    """Shortest-arc quaternion rotating +z onto ``axis`` (x, y, z, w)."""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    dot = float(axis[2])
    if dot > 1.0 - 1e-9:
        return [0.0, 0.0, 0.0, 1.0]
    if dot < -1.0 + 1e-9:
        return [1.0, 0.0, 0.0, 0.0]
    cross = np.cross([0.0, 0.0, 1.0], axis)
    quat = np.array([cross[0], cross[1], cross[2], 1.0 + dot])
    return (quat / np.linalg.norm(quat)).tolist()


# --------------------------------------------------------------------- recording
def record_episode(model: PPO, env: QuadrotorEnv, seed: int) -> dict:
    """Roll out one deterministic episode and keep everything needed to draw it."""
    obs, info = env.reset(seed=seed)
    scenario = dict(info["scenario"])
    positions = [info["position"].copy()]
    quaternions = [env.state.quaternion.copy()]
    actions = []
    times = [0.0]
    errors = [float(info["position_error_m"])]
    fault_active = [False]
    done = False
    terminated = truncated = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _reward, terminated, truncated, info = env.step(action)
        actions.append(np.asarray(action, dtype=np.float64).reshape(-1))
        positions.append(info["position"].copy())
        quaternions.append(env.state.quaternion.copy())
        times.append(float(info["step"]) * env.control_dt)
        errors.append(float(info["position_error_m"]))
        fault_active.append(bool(info["fault_active"]))
        done = bool(terminated or truncated)

    return {
        "seed": int(seed),
        "target": env.target_position.copy(),
        "scenario": scenario,
        "time_s": np.asarray(times, dtype=np.float64),
        "position": np.asarray(positions, dtype=np.float64),
        "quaternion": np.asarray(quaternions, dtype=np.float64),
        "action": np.asarray(actions, dtype=np.float64),
        "error_m": np.asarray(errors, dtype=np.float64),
        "fault_active": np.asarray(fault_active, dtype=bool),
        "done_reason": info["done_reason"],
        "control_dt": float(env.control_dt),
    }


def visual_rotor_commands(episode: dict) -> np.ndarray:
    """Per-frame, per-rotor 0..1 command, mirroring what the injector did.

    Cosmetic only: it drives the drawn disc rotation, never thrust.  Mirrors
    ``StubFaultInjector.apply`` so the failed rotor visibly stops.
    """
    commands = episode["action"].copy()
    rotor_id = episode["scenario"].get("rotor_id")
    if rotor_id is not None:
        effectiveness = float(episode["scenario"].get("effectiveness") or 0.0)
        mask = episode["fault_active"][1:]  # state after action k is frame k + 1
        commands[mask, int(rotor_id)] *= effectiveness
    return np.clip(commands, 0.0, 1.0)


# ------------------------------------------------------------------------- scene
def _sphere(client: int, radius: float, rgba, position) -> int:
    shape = p.createVisualShape(
        p.GEOM_SPHERE, radius=radius, rgbaColor=rgba, physicsClientId=client
    )
    return p.createMultiBody(
        baseMass=0, baseVisualShapeIndex=shape,
        basePosition=[float(v) for v in position], physicsClientId=client,
    )


def _segment(client: int, a, b, radius: float, rgba) -> int | None:
    """A thin cylinder from ``a`` to ``b``: the trajectory trail.

    Drawn as geometry rather than ``addUserDebugLine`` because debug lines are
    not rendered by ``getCameraImage``, so a recorded GIF would lose the trail.
    """
    start = np.asarray(a, dtype=np.float64)
    end = np.asarray(b, dtype=np.float64)
    delta = end - start
    length = float(np.linalg.norm(delta))
    if length < 1e-9:
        return None
    shape = p.createVisualShape(
        p.GEOM_CYLINDER, radius=radius, length=length, rgbaColor=rgba,
        physicsClientId=client,
    )
    return p.createMultiBody(
        baseMass=0, baseVisualShapeIndex=shape,
        basePosition=((start + end) / 2.0).tolist(),
        baseOrientation=_quat_aligning_z_with(delta),
        physicsClientId=client,
    )


def _load_scene(client: int, config: dict, episode: dict,
                clear_world: bool = False) -> dict:
    """Ground, target, start marker, trail and airframe.  Returns handles."""
    if clear_world:
        # The GUI is the ExampleBrowser, which opens with its own demo world (a
        # rover, sample boxes, a few robots). Those bodies are not ours and cannot
        # be animated, so clear them first. Only in the GUI: resetSimulation also
        # drops the hardware GL renderer that --record relies on, taking a 20 s
        # GIF to several minutes of software rasterisation.
        p.resetSimulation(physicsClientId=client)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
    p.setGravity(0, 0, 0, physicsClientId=client)
    p.setTimeStep(episode["control_dt"], physicsClientId=client)
    p.loadURDF("plane.urdf", physicsClientId=client)

    target = episode["target"]
    positions = rotor_positions_body_m(config)
    dims = airframe_dimensions(positions)
    marker_radius = max(0.015, 0.50 * dims["rotor_radius"])

    _sphere(client, marker_radius, TARGET_RGBA, target)
    # A thin pole from the ground to the hold point: without it a 1 m altitude
    # reads as flat in a perspective view.
    _segment(client, [target[0], target[1], 0.0], target, 0.30 * marker_radius, POLE_RGBA)
    _sphere(client, marker_radius * 0.85, START_RGBA, episode["position"][0])

    # Full path up front: this is a replay, so the whole trajectory is known and
    # the moving model shows where along it we are.
    path = episode["position"]
    stride = max(1, (len(path) - 1) // 300)
    for k in range(0, len(path) - 1, stride):
        _segment(client, path[k], path[min(k + stride, len(path) - 1)],
                 0.35 * dims["arm_radius"], PATH_RGBA)

    tmpdir = tempfile.mkdtemp(prefix="uav_replay_")
    urdf_path = Path(tmpdir) / "quadrotor_replay.urdf"
    urdf_path.write_text(urdf_from_config(config), encoding="utf-8")
    quad = p.loadURDF(
        str(urdf_path),
        basePosition=episode["position"][0].tolist(),
        baseOrientation=episode["quaternion"][0].tolist(),
        useFixedBase=False,
        # No collision shapes in the generated model; without this flag Bullet
        # derives zero inertia from them.
        flags=p.URDF_USE_INERTIA_FROM_FILE,
        physicsClientId=client,
    )
    joints = {
        p.getJointInfo(quad, i, physicsClientId=client)[1].decode(): i
        for i in range(p.getNumJoints(quad, physicsClientId=client))
    }
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=client)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1, physicsClientId=client)
    # Orbiting is Ctrl+left-drag in this build; a plain left-drag is object
    # picking, which would yank the drone off its recorded path and read as "the
    # replay is broken". Nothing here is meant to be dragged, so turn it off: a
    # stray plain drag then does nothing at all.
    p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0, physicsClientId=client)
    return {"quad": quad, "joints": joints, "dims": dims, "tmpdir": tmpdir}


def _set_camera(client: int, target, distance: float, yaw: float, pitch: float) -> None:
    p.resetDebugVisualizerCamera(
        cameraDistance=distance, cameraYaw=yaw, cameraPitch=pitch,
        cameraTargetPosition=[float(v) for v in target], physicsClientId=client,
    )


def _hud_text(episode: dict, k: int) -> str:
    rotor_id = episode["scenario"].get("rotor_id")
    if rotor_id is not None and bool(episode["fault_active"][k]):
        label = ROTOR_LAYOUT[int(rotor_id)][0]
        status = f"fault: R{int(rotor_id)} {label} OFF"
    else:
        status = "all rotors nominal"
    return (
        f"t = {episode['time_s'][k]:5.2f} s   {status}   "
        f"err = {episode['error_m'][k]:.3f} m"
    )


# ------------------------------------------------------------------- frame capture
def _camera_matrices(target, distance, yaw, pitch, width, height):
    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[float(v) for v in target], distance=distance,
        yaw=yaw, pitch=pitch, roll=0, upAxisIndex=2,
    )
    projection = p.computeProjectionMatrixFOV(
        fov=60.0, aspect=width / height, nearVal=0.01, farVal=50.0
    )
    return view, projection


def _ndc(view, projection, points) -> np.ndarray:
    """Project world points to normalised device coordinates."""
    to_matrix = lambda m: np.asarray(m, dtype=np.float64).reshape(4, 4).T  # noqa: E731
    transform = to_matrix(projection) @ to_matrix(view)
    homogeneous = np.hstack([np.asarray(points, dtype=np.float64),
                             np.ones((len(points), 1))])
    clip = homogeneous @ transform.T
    return (clip[:, :2] / clip[:, 3:4])


def frame_camera(episode: dict, yaw: float, pitch: float, aspect: float,
                 margin: float = 0.88, minimum: float = 0.55,
                 maximum: float = 6.0) -> tuple[np.ndarray, float]:
    """Choose a camera target and distance that hold the whole trajectory.

    A fixed camera framing the hold point drops the vehicle out of view exactly
    when a failure makes it interesting, and the failure case is the one worth
    watching.  This is a replay of a known path, so the framing can be computed
    from the path instead of guessed: centre on the bounding box of the path and
    the hold point, then pull back until every sample is on screen.
    """
    path = np.asarray(episode["position"], dtype=np.float64)
    points = np.vstack([path, np.asarray(episode["target"], dtype=np.float64)[None, :]])
    centre = 0.5 * (points.min(axis=0) + points.max(axis=0))

    distance = maximum
    for candidate in np.arange(minimum, maximum + 1e-9, 0.05):
        view, projection = _camera_matrices(
            centre, float(candidate), yaw, pitch, 1.0, aspect
        )
        if np.all(np.abs(_ndc(view, projection, points)) <= margin):
            distance = float(candidate)
            break
    return centre, distance


def _grab_frame(client, view, projection, width, height) -> np.ndarray:
    _w, _h, rgba, _depth, _seg = p.getCameraImage(
        width, height, view, projection,
        # Flat lighting: the default light distance puts a pool on the ground
        # centred on the camera target, and the shadow map adds another blob on
        # top of it. Neither carries information, and both look worse after the
        # GIF palette is cut down.
        shadow=0, lightDistance=1000.0, lightSpecularCoeff=0.0,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
        physicsClientId=client,
    )
    return np.reshape(np.asarray(rgba, dtype=np.uint8), (height, width, 4))[:, :, :3]


def _hud_font(size: int):
    from PIL import ImageFont

    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def _burn_hud(frame: np.ndarray, text: str, font):
    """Draw a legible caption band over a rendered frame.

    ``getCameraImage`` does not render ``addUserDebugText``, so recorded frames
    would otherwise carry no timestamp or fault status.
    """
    from PIL import Image, ImageDraw

    image = Image.fromarray(frame).convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    margin, padding = 16, 10
    box = draw.textbbox((0, 0), text, font=font)
    band = (margin, margin, margin + box[2] + 2 * padding,
            margin + box[3] + 2 * padding)
    draw.rectangle(band, fill=(0, 0, 0, 150))
    draw.text((margin + padding, margin + padding), text, fill=(255, 255, 255), font=font)
    return image.convert("RGB")


# ------------------------------------------------------------------------ driver
def _connect(prefer_gui: bool) -> tuple[int, bool]:
    """Return ``(client, is_gui)``; fall back to DIRECT when there is no display."""
    if prefer_gui:
        try:
            client = p.connect(p.GUI)
        except Exception:  # pragma: no cover - depends on the machine
            client = -1
        if client >= 0:
            return client, True
    return p.connect(p.DIRECT), False


def run_replay(
    episode: dict,
    config: dict,
    *,
    live: bool = True,
    speed: float = 1.0,
    camera_distance: float | None = None,
    camera_yaw: float = 50.0,
    camera_pitch: float = -22.0,
    rotor_rps: float = 3.0,
    loop: bool = False,
    gif_path: Path | None = None,
    gif_fps: int = 20,
    gif_colors: int = 64,
    width: int = 800,
    height: int = 450,
) -> None:
    # Recording deliberately uses DIRECT: ``getCameraImage`` through a GUI client
    # inherits that window's lighting and puts a lit ellipse on the ground that
    # the offscreen renderer does not, and a GIF should not depend on whether a
    # display happened to be open.
    client, is_gui = _connect(prefer_gui=live)
    if live and not is_gui:
        # A DIRECT client has no window and no keyboard: the loop below would run
        # to completion invisibly, which looks exactly like a crash. Say so.
        p.disconnect(physicsClientId=client)
        raise RuntimeError(
            "PyBullet could not open a window (no DISPLAY?). Use --record out.gif "
            "to write a GIF instead."
        )
    if live:
        print("Ctrl+left-drag to orbit · scroll to zoom · space pauses · q quits")
    tmpdir = None
    try:
        scene = _load_scene(client, config, episode, clear_world=is_gui)
        tmpdir = scene["tmpdir"]
        quad, joints = scene["quad"], scene["joints"]
        commands = visual_rotor_commands(episode)
        if camera_distance is None:
            camera_target, distance = frame_camera(
                episode, camera_yaw, camera_pitch, aspect=width / height
            )
        else:
            camera_target, distance = episode["target"], float(camera_distance)
        _set_camera(client, camera_target, distance, camera_yaw, camera_pitch)
        view, projection = _camera_matrices(
            camera_target, distance, camera_yaw, camera_pitch, width, height
        )
        font = _hud_font(max(14, height // 30)) if gif_path is not None else None
        # A GIF keeps every record_every-th step, so only render those. An
        # offscreen frame costs ~0.45 s here (DIRECT falls back to the software
        # rasteriser), so rendering the 450 frames that get thrown away was most
        # of a four-minute recording.
        record_every = max(
            1, int(round((1.0 / episode["control_dt"]) / gif_fps))
        ) if gif_path is not None else 0

        frames = []
        angles = np.zeros(4)
        rotor_id = episode["scenario"].get("rotor_id")
        hud_id = -1
        paused = False

        while True:
            for k in range(len(episode["time_s"])):
                if live:
                    keys = p.getKeyboardEvents(physicsClientId=client)
                    if ord("q") in keys and keys[ord("q")] & p.KEY_WAS_TRIGGERED:
                        return
                    if p.B3G_SPACE in keys and keys[p.B3G_SPACE] & p.KEY_WAS_TRIGGERED:
                        paused = not paused
                    while paused:
                        time.sleep(0.02)
                        keys = p.getKeyboardEvents(physicsClientId=client)
                        if p.B3G_SPACE in keys and keys[p.B3G_SPACE] & p.KEY_WAS_TRIGGERED:
                            paused = False
                        elif ord("q") in keys and keys[ord("q")] & p.KEY_WAS_TRIGGERED:
                            return

                p.resetBasePositionAndOrientation(
                    quad, episode["position"][k].tolist(),
                    episode["quaternion"][k].tolist(), physicsClientId=client,
                )

                if k > 0:
                    applied = commands[k - 1]
                    for index, (_label, spin) in enumerate(ROTOR_LAYOUT):
                        angles[index] += (
                            spin * 2.0 * np.pi * rotor_rps
                            * episode["control_dt"] * applied[index]
                        )
                for index in range(4):
                    angle = float(np.arctan2(np.sin(angles[index]), np.cos(angles[index])))
                    p.resetJointState(
                        quad, joints[f"rotor{index}_joint"], angle, physicsClientId=client
                    )

                if rotor_id is not None:
                    colour = FAILED_RGBA if episode["fault_active"][k] else ROTOR_RGBA
                    p.changeVisualShape(
                        quad, joints[f"rotor{int(rotor_id)}_joint"], -1,
                        rgbaColor=colour, physicsClientId=client,
                    )

                p.stepSimulation(physicsClientId=client)

                if gif_path is not None and k % record_every == 0:
                    frames.append(
                        _burn_hud(
                            _grab_frame(client, view, projection, width, height),
                            _hud_text(episode, k), font,
                        )
                    )
                elif live:
                    hud_id = p.addUserDebugText(
                        _hud_text(episode, k),
                        [episode["position"][k][0], episode["position"][k][1],
                         episode["position"][k][2] + 0.09],
                        textColorRGB=[1.0, 1.0, 1.0], textSize=1.1, lifeTime=0,
                        replaceItemUniqueId=hud_id, physicsClientId=client,
                    )
                    if speed > 0:
                        time.sleep(episode["control_dt"] / speed)
                    if not p.getConnectionInfo(physicsClientId=client)["isConnected"]:
                        return

            if not loop:
                break

        if gif_path is not None and frames:
            from PIL import Image

            # Explicit quantization: letting Pillow convert RGB with dithering
            # produced a 13 MB file for these 200 frames. FASTOCTREE, not
            # MEDIANCUT -- MEDIANCUT allocates palette entries by pixel count and
            # drops the target marker's green (a few dozen pixels) into blue-grey,
            # which is precisely the identity the colour is carrying.
            kept = [
                frame.quantize(colors=gif_colors, method=Image.FASTOCTREE,
                               dither=Image.NONE)
                for frame in frames
            ]
            gif_path.parent.mkdir(parents=True, exist_ok=True)
            kept[0].save(
                gif_path, save_all=True, append_images=kept[1:],
                duration=int(round(1000.0 / gif_fps)), loop=0, optimize=True,
            )
            print(f"wrote {gif_path} ({len(kept)} frames at {gif_fps} fps)")
        elif live and speed > 0:
            # Hold the last frame instead of snapping shut: at --speed 1 the
            # window otherwise disappears 10 s after it appears, which reads as a
            # crash rather than "the episode ended".
            print("finished — close the window or press q to exit")
            while p.getConnectionInfo(physicsClientId=client)["isConnected"]:
                keys = p.getKeyboardEvents(physicsClientId=client)
                if ord("q") in keys and keys[ord("q")] & p.KEY_WAS_TRIGGERED:
                    break
                time.sleep(0.05)
    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)
        if p.getConnectionInfo(physicsClientId=client)["isConnected"]:
            p.disconnect(physicsClientId=client)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", required=True, help="model.zip or its run directory")
    parser.add_argument("--episode", type=int, default=0,
                        help="evaluation episode index; seed = seed_start + index")
    parser.add_argument("--seed", type=int, default=None, help="use this seed instead")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="playback speed multiplier, 0 = as fast as possible")
    parser.add_argument("--rotor-rps", type=float, default=3.0,
                        help="cosmetic rotor revolutions per second at full command")
    parser.add_argument("--camera-distance", type=float, default=None,
                        help="fixed camera distance; default fits the whole trajectory")
    parser.add_argument("--camera-yaw", type=float, default=50.0)
    parser.add_argument("--camera-pitch", type=float, default=-22.0)
    parser.add_argument("--loop", action="store_true", help="replay until closed")
    parser.add_argument("--record", default=None,
                        help="write a GIF here instead of playing it on screen")
    parser.add_argument("--gif-fps", type=int, default=15,
                        help="GIF frame rate (viewers clamp anything above 50 fps)")
    parser.add_argument("--gif-colors", type=int, default=64,
                        help="palette size per GIF frame; fewer colours, smaller file")
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=450)
    args = parser.parse_args()

    checkpoint = resolve_checkpoint(args.checkpoint)
    run_dir = checkpoint.parent
    with open(run_dir / "meta.json", "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    config = meta["config"]

    env = QuadrotorEnv(config)
    model = PPO.load(checkpoint, env=env, device="cpu")
    model.policy.set_training_mode(False)
    torch.set_num_threads(1)

    seed_start = int(config.get("evaluation", {}).get("seed_start", 100000))
    seed = int(args.seed) if args.seed is not None else seed_start + int(args.episode)

    episode = record_episode(model, env, seed)
    print(
        json.dumps(
            {
                "seed": episode["seed"],
                "steps": int(len(episode["action"])),
                "done_reason": episode["done_reason"],
                "final_position_error_m": float(episode["error_m"][-1]),
                "failed_rotor": episode["scenario"].get("rotor_id"),
                "failure_time_s": episode["scenario"].get("failure_time_s"),
            },
            indent=2,
        )
    )

    gif_path = Path(args.record) if args.record else None
    run_replay(
        episode,
        config,
        live=gif_path is None,
        speed=args.speed,
        camera_distance=args.camera_distance,
        camera_yaw=args.camera_yaw,
        camera_pitch=args.camera_pitch,
        rotor_rps=args.rotor_rps,
        loop=args.loop,
        gif_path=gif_path,
        gif_fps=args.gif_fps,
        gif_colors=args.gif_colors,
        width=args.width,
        height=args.height,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
