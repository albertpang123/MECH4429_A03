"""Shared matplotlib style for the G3 report figures.

Palette: the project's validated categorical slots (blue, orange, aqua) on a
light surface.  Aqua is below 3:1 contrast, so every multi-series line is also
direct-labelled (the relief rule).  The underlying numbers are in the CSV/JSON
artefacts, which serve as the table view.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]
BLUE_LIGHT = "#9ec5f4"


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.labelcolor": INK_SECONDARY,
            "axes.labelsize": 10,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 1.4,
        }
    )


def _style_axis(ax, title: str, xlabel: str, ylabel: str, grid_axis: str = "y") -> None:
    ax.set_title(title, loc="left", pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis=grid_axis, alpha=1.0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(length=0)


def plot_learning_curve(csv_path: str | Path, out_path: str | Path) -> bool:
    """Plot episode return and length from an SB3 progress.csv."""
    df = pd.read_csv(csv_path)
    if "rollout/ep_rew_mean" not in df.columns:
        return False
    steps = df["time/total_timesteps"]
    window = 5 if len(df) >= 5 else max(1, len(df))
    reward = df["rollout/ep_rew_mean"]
    reward_smooth = reward.rolling(window, min_periods=1).mean()

    fig, axes = plt.subplots(2, 1, figsize=(7, 5.2), sharex=True)
    ax = axes[0]
    ax.plot(steps, reward, color=BLUE_LIGHT, linewidth=0.9, label="per update")
    ax.plot(steps, reward_smooth, color=SERIES[0], linewidth=1.6, label=f"{window}-update mean")
    _style_axis(ax, "PPO learning curve — nominal hover", "", "episode return")
    ax.legend(loc="lower right")
    ax.annotate(
        f"{reward_smooth.iloc[-1]:.0f}",
        xy=(steps.iloc[-1], reward_smooth.iloc[-1]),
        xytext=(-6, 6),
        textcoords="offset points",
        ha="right",
        color=INK_SECONDARY,
        fontsize=9,
    )

    ax = axes[1]
    if "rollout/ep_len_mean" in df.columns:
        length = df["rollout/ep_len_mean"]
        length_smooth = length.rolling(window, min_periods=1).mean()
        ax.plot(steps, length, color=BLUE_LIGHT, linewidth=0.9, label="per update")
        ax.plot(steps, length_smooth, color=SERIES[0], linewidth=1.6, label=f"{window}-update mean")
        _style_axis(ax, "Episode length", "timesteps", "control steps")
        ax.legend(loc="lower right")
    else:
        ax.set_visible(False)

    fig.align_ylabels(axes)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_evaluation(trajectories: list[dict], out_path: str | Path) -> None:
    """Two panels: top-view path, and x/y/z position error vs time."""
    episode = trajectories[0]
    pos = np.asarray(episode["position"], dtype=float)
    time_s = np.asarray(episode["time_s"], dtype=float)
    target = np.asarray(episode["target"], dtype=float)
    error = target[None, :] - pos

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    ax = axes[0]
    ax.plot(pos[:, 0], pos[:, 1], color=SERIES[0], linewidth=1.6)
    ax.scatter(
        pos[0, 0], pos[0, 1], s=36, facecolor=SURFACE, edgecolor=SERIES[0],
        linewidth=1.4, zorder=3,
    )
    ax.scatter(pos[-1, 0], pos[-1, 1], s=36, color=SERIES[0], zorder=3)
    ax.scatter(
        target[0], target[1], s=54, marker="x", color=INK, linewidth=1.6, zorder=4,
        label="target",
    )
    ax.annotate(
        "start", xy=pos[0, :2], xytext=(5, 5), textcoords="offset points",
        color=MUTED, fontsize=9,
    )
    ax.annotate(
        "end", xy=pos[-1, :2], xytext=(5, -12), textcoords="offset points",
        color=INK_SECONDARY, fontsize=9,
    )
    ax.legend(loc="upper right")
    ax.set_aspect("equal", adjustable="datalim")
    _style_axis(
        ax,
        f"Top view — seed {episode['seed']}",
        "x East (m)",
        "y North (m)",
        grid_axis="both",
    )
    ax.text(
        0.02, 0.98,
        f"return {episode['return']:.1f}\nfinal error {np.linalg.norm(error[-1]):.3f} m",
        transform=ax.transAxes,
        va="top",
        color=INK_SECONDARY,
        fontsize=9,
    )

    ax = axes[1]
    for index, label in enumerate(("x", "y", "z")):
        ax.plot(time_s, error[:, index], color=SERIES[index], linewidth=1.5, label=f"{label} error")

    # Direct labels are required (aqua fails the contrast check), but x/y often
    # end at almost the same value.  Stagger them by end-value rank so they
    # never print on top of each other.
    vertical_offsets = {}
    for rank, index in enumerate(np.argsort(error[-1])):
        vertical_offsets[index] = (rank - 1) * 12
    for index, label in enumerate(("x", "y", "z")):
        ax.annotate(
            label,
            xy=(time_s[-1], error[-1, index]),
            xytext=(6, vertical_offsets[index]),
            textcoords="offset points",
            color=SERIES[index],
            fontsize=9,
            va="center",
            annotation_clip=False,
        )

    _style_axis(ax, "Position error components", "time (s)", "error (m)")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=3,
        borderaxespad=0.0,
    )
    ax.margins(x=0.10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
