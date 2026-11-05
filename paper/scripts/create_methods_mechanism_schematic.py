#!/usr/bin/env python3
"""Create a schematic figure explaining OGT uniqueness mechanisms.

The figure is intentionally conceptual: it uses deterministic synthetic
trajectories to explain how each behavioural feature is computed, rather than
plotting participant-level data.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("analysis_outputs/.matplotlib_cache").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("analysis_outputs/.cache").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402


OUTPUT_DIR = Path("paper/figures")
OUTPUT_BASENAME = "methods_mechanism_schematic"

CANVAS_W = 16
CANVAS_H = 24

INK = "#222222"
MUTED = "#777777"
GRID = "#D9DEE3"
PANEL_BG = "#FAFAF8"
BLUE = "#4F86A8"
BLUE_DARK = "#2F617E"
ORANGE = "#C86F4A"
GREEN = "#6F9A63"
RED = "#B84A42"
YELLOW = "#E5B84A"
PURPLE = "#7C6A9E"


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 8.5,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def bezier(points: list[tuple[float, float]], n: int = 120) -> np.ndarray:
    """Sample a Bezier curve through control points."""
    pts = np.asarray(points, dtype=float)
    t = np.linspace(0, 1, n)[:, None]
    curve = np.zeros((n, 2), dtype=float)
    order = len(pts) - 1
    for i, point in enumerate(pts):
        coef = math.comb(order, i) * (1 - t) ** (order - i) * t**i
        curve += coef * point
    return curve


def interpolate_polyline(points: list[tuple[float, float]], n: int = 140) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    dist = np.concatenate([[0], np.cumsum(seg)])
    target = np.linspace(0, dist[-1], n)
    x = np.interp(target, dist, pts[:, 0])
    y = np.interp(target, dist, pts[:, 1])
    return np.column_stack([x, y])


def setup_canvas(ax: plt.Axes, *, grid: bool = False) -> None:
    ax.set_xlim(0, CANVAS_W)
    ax.set_ylim(0, CANVAS_H)
    ax.set_aspect("equal")
    ax.set_facecolor(PANEL_BG)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_edgecolor("#BFC5CA")
    if grid:
        ax.set_xticks(np.arange(0, CANVAS_W + 1, 1), minor=True)
        ax.set_yticks(np.arange(0, CANVAS_H + 1, 1), minor=True)
        ax.grid(which="minor", color=GRID, linewidth=0.35)
        ax.tick_params(which="minor", length=0)


def add_panel_label(ax: plt.Axes, label: str, title: str) -> None:
    ax.text(
        0.0,
        1.09,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="bottom",
        color=INK,
    )
    ax.text(
        0.13,
        1.095,
        title,
        transform=ax.transAxes,
        fontsize=8.5,
        fontweight="bold",
        ha="left",
        va="bottom",
        color=INK,
    )


def plot_path(ax: plt.Axes, path: np.ndarray, color: str, lw: float = 1.8, alpha: float = 1.0) -> None:
    ax.plot(path[:, 0], path[:, 1], color=color, lw=lw, alpha=alpha, solid_capstyle="round")
    ax.scatter(path[0, 0], path[0, 1], s=12, color=color, edgecolor="white", linewidth=0.4, zorder=4)


def plot_ordered_paths(ax: plt.Axes, paths: list[np.ndarray], colors: list[str]) -> None:
    for i, (path, color) in enumerate(zip(paths, colors), start=1):
        plot_path(ax, path, color=color, lw=1.6)
        ax.text(
            path[0, 0],
            path[0, 1] - 0.75,
            str(i),
            ha="center",
            va="top",
            fontsize=6.5,
            fontweight="bold",
            color=color,
        )


def draw_uniqueness_panel(ax: plt.Axes) -> None:
    setup_canvas(ax)
    add_panel_label(ax, "A", "Uniqueness")

    previous = [
        bezier([(2, 5), (4, 18), (8, 10), (13, 19)]),
        bezier([(3, 20), (6, 14), (10, 15), (13, 5)]),
        bezier([(2, 9), (6, 8), (8, 20), (14, 13)]),
        bezier([(5, 4), (7, 9), (11, 8), (14, 17)]),
    ]
    target = bezier([(2.5, 18.5), (5.5, 5.5), (10.5, 6.5), (13.5, 18)])

    for path in previous:
        plot_path(ax, path, color="#B8BEC4", lw=1.1, alpha=0.9)
    plot_path(ax, target, color=BLUE_DARK, lw=2.4)

    similar = previous[0]
    idx = 68
    ax.annotate(
        "",
        xy=target[idx],
        xytext=similar[idx],
        arrowprops=dict(arrowstyle="<->", color=RED, lw=1.1, shrinkA=2, shrinkB=2),
    )
    ax.text(8.2, 16.8, "distance to\nnearest route", fontsize=6.5, color=RED, ha="center")
    ax.text(0.7, 1.0, "new option vs. preceding / sample routes", fontsize=6.5, color=MUTED)


def shade_occupied_cells(ax: plt.Axes, paths: list[np.ndarray], color: str) -> None:
    occupied: set[tuple[int, int]] = set()
    for path in paths:
        cells_x = np.clip(path[:, 0].astype(int), 0, CANVAS_W - 1)
        cells_y = np.clip(path[:, 1].astype(int), 0, CANVAS_H - 1)
        occupied.update(zip(cells_x, cells_y))
    for x, y in occupied:
        rect = plt.Rectangle((x, y), 1, 1, facecolor=color, edgecolor="none", alpha=0.24, zorder=0)
        ax.add_patch(rect)


def draw_entropy_panel(ax_low: plt.Axes, ax_high: plt.Axes) -> None:
    for ax in (ax_low, ax_high):
        setup_canvas(ax, grid=True)
    add_panel_label(ax_low, "B", "Roaming Entropy")

    low_paths = [
        bezier([(7.2, 2), (7.8, 8), (7.1, 16), (8.0, 22)], n=150),
        bezier([(8.4, 2), (7.4, 7), (8.6, 15), (7.7, 22)], n=150),
        bezier([(6.7, 3), (8.4, 9), (6.9, 16), (8.2, 21)], n=150),
    ]
    high_paths = [
        bezier([(1.5, 4), (5, 22), (9, 3), (14.5, 20)], n=150),
        bezier([(2, 19), (12, 18), (4, 6), (14, 5)], n=150),
        bezier([(1.5, 9), (5, 2), (12, 23), (15, 12)], n=150),
    ]
    shade_occupied_cells(ax_low, low_paths, BLUE)
    shade_occupied_cells(ax_high, high_paths, GREEN)
    for path in low_paths:
        plot_path(ax_low, path, color=BLUE_DARK, lw=1.4, alpha=0.85)
    for path in high_paths:
        plot_path(ax_high, path, color=GREEN, lw=1.4, alpha=0.9)
    ax_low.text(0.7, 1.0, "low: concentrated cells", fontsize=6.5, color=MUTED)
    ax_high.text(0.7, 1.0, "high: distributed cells", fontsize=6.5, color=MUTED)


def draw_elaboration_panel(ax_path: plt.Axes, ax_speed: plt.Axes) -> None:
    setup_canvas(ax_path)
    add_panel_label(ax_path, "C", "Option Elaboration")

    path = interpolate_polyline(
        [(2, 3), (5, 17), (4, 8), (8.5, 20), (11, 7), (10, 14), (14, 21)],
        n=180,
    )
    plot_path(ax_path, path, color=PURPLE, lw=2.0)
    minima_idx = np.array([34, 64, 96, 125, 150])
    ax_path.scatter(path[minima_idx, 0], path[minima_idx, 1], s=28, color=YELLOW, edgecolor=INK, linewidth=0.4, zorder=5)
    ax_path.text(0.7, 1.0, "velocity minima", fontsize=6.5, color=MUTED)

    t = np.linspace(0, 1, path.shape[0])
    speed = 0.72 + 0.16 * np.sin(8 * np.pi * t)
    for loc in t[minima_idx]:
        speed -= 0.30 * np.exp(-((t - loc) ** 2) / 0.0008)
    speed = np.clip(speed, 0.18, None)
    ax_speed.plot(t, speed, color=PURPLE, lw=1.7)
    ax_speed.scatter(t[minima_idx], speed[minima_idx], s=24, color=YELLOW, edgecolor=INK, linewidth=0.35, zorder=4)
    ax_speed.set_xlim(0, 1)
    ax_speed.set_ylim(0, 1.05)
    ax_speed.set_xlabel("within-path time")
    ax_speed.set_ylabel("speed")
    ax_speed.set_xticks([])
    ax_speed.set_yticks([])
    ax_speed.spines["top"].set_visible(False)
    ax_speed.spines["right"].set_visible(False)
    ax_speed.text(0.04, 0.08, "more minima", transform=ax_speed.transAxes, fontsize=6.5, color=MUTED)


def draw_perseveration_panel(ax_local: plt.Axes, ax_jump: plt.Axes) -> None:
    for ax in (ax_local, ax_jump):
        setup_canvas(ax)
    add_panel_label(ax_local, "D", "Perseveration")

    local_paths = [
        bezier([(4.0 + 0.35 * i, 5.5 + 0.3 * i), (6, 15), (9, 10), (11 + 0.25 * i, 17)], n=100)
        for i in range(4)
    ]
    jump_paths = [
        bezier([(2, 4), (5, 9), (4, 14), (7, 19)], n=100),
        bezier([(11, 20), (14, 16), (13, 8), (9, 5)], n=100),
        bezier([(2, 18), (5, 21), (8, 12), (13, 16)], n=100),
        bezier([(5, 3), (11, 4), (10, 19), (14, 22)], n=100),
    ]
    colors = [BLUE, GREEN, ORANGE, PURPLE]
    plot_ordered_paths(ax_local, local_paths, colors)
    plot_ordered_paths(ax_jump, jump_paths, colors)
    for paths, ax in ((local_paths, ax_local), (jump_paths, ax_jump)):
        starts = np.array([p[0] for p in paths])
        ax.plot(starts[:, 0], starts[:, 1], color=INK, lw=0.8, alpha=0.45, linestyle="--")
    ax_local.text(0.7, 1.0, "low distance ratio", fontsize=6.5, color=MUTED)
    ax_jump.text(0.7, 1.0, "high distance ratio", fontsize=6.5, color=MUTED)


def draw_novelty_panel(ax: plt.Axes) -> None:
    add_panel_label(ax, "E", "Novelty Maintenance")
    rng = np.random.default_rng(7)
    path_no = np.arange(1, 25)
    maintained = 0.55 + 0.06 * np.log(path_no) + rng.normal(0, 0.025, path_no.size)
    declining = 0.72 - 0.018 * path_no + rng.normal(0, 0.025, path_no.size)
    maintained = np.clip(maintained, 0.05, 0.96)
    declining = np.clip(declining, 0.05, 0.96)

    for values, color, label in (
        (maintained, GREEN, "maintained"),
        (declining, RED, "declining"),
    ):
        ax.scatter(path_no, values, s=14, color=color, alpha=0.55, edgecolor="none")
        coef = np.polyfit(path_no, values, 1)
        ax.plot(path_no, np.polyval(coef, path_no), color=color, lw=2.0, label=label)

    ax.set_xlim(0.5, 24.5)
    ax.set_ylim(0, 1)
    ax.set_xlabel("path number")
    ax.set_ylabel("uniqueness vs. own\npreceding paths", labelpad=5)
    ax.set_xticks([1, 8, 16, 24])
    ax.set_yticks([0, 0.5, 1.0])
    ax.legend(frameon=False, loc="lower left", fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(0.60, 0.12, "Spearman rho over session", transform=ax.transAxes, fontsize=6.5, color=MUTED)


def main() -> None:
    configure_matplotlib()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(11.8, 7.4))
    gs = GridSpec(
        3,
        6,
        figure=fig,
        height_ratios=[1.0, 1.0, 0.24],
        left=0.045,
        right=0.985,
        top=0.91,
        bottom=0.08,
        hspace=0.46,
        wspace=0.46,
    )

    ax_a = fig.add_subplot(gs[0:2, 0:2])
    ax_b1 = fig.add_subplot(gs[0, 2])
    ax_b2 = fig.add_subplot(gs[0, 3])
    ax_c1 = fig.add_subplot(gs[0, 4])
    ax_c2 = fig.add_subplot(gs[0, 5])
    ax_d1 = fig.add_subplot(gs[1, 2])
    ax_d2 = fig.add_subplot(gs[1, 3])
    ax_e = fig.add_subplot(gs[1, 4:6])
    ax_caption = fig.add_subplot(gs[2, :])

    draw_uniqueness_panel(ax_a)
    draw_entropy_panel(ax_b1, ax_b2)
    draw_elaboration_panel(ax_c1, ax_c2)
    draw_perseveration_panel(ax_d1, ax_d2)
    draw_novelty_panel(ax_e)

    ax_caption.axis("off")
    caption = (
        "Schematic decomposition of option-generation behaviour in the test block. "
        "Uniqueness is illustrated as distance from previous or sample routes; roaming entropy as the "
        "spread of occupied 16 x 24 grid cells; option elaboration as within-path velocity minima; "
        "perseveration as local similarity between consecutive paths; and novelty maintenance as the "
        "association between path order and uniqueness relative to one's own preceding paths."
    )
    ax_caption.text(0.0, 0.70, caption, ha="left", va="top", fontsize=8, color=INK, wrap=True)

    png = OUTPUT_DIR / f"{OUTPUT_BASENAME}.png"
    pdf = OUTPUT_DIR / f"{OUTPUT_BASENAME}.pdf"
    tiff = OUTPUT_DIR / f"{OUTPUT_BASENAME}.tiff"
    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(tiff, dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)

    print(f"Wrote {png}")
    print(f"Wrote {pdf}")
    print(f"Wrote {tiff}")


if __name__ == "__main__":
    main()
