#!/usr/bin/env python3
"""
Combined static trajectory plot for all methods in 4_agents_0_humans_0_rogue.
Produces a 2x2 grid, one subplot per method.
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as ticker
import numpy as np

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT   = os.path.join(SCRIPT_DIR, "..", "data")
PLOTS_ROOT  = os.path.join(SCRIPT_DIR, "..", "plots")

SCENARIO    = "4_agents_0_humans_0_rogue"
SCENARIO_DIR = os.path.join(DATA_ROOT, SCENARIO)

TRIAL_LABELS = {
    "mpc_cbf_trial":        "MPC-CBF",
    "nh_orca_trial":        "NH-ORCA",
    "nod_cooperation_trial":"NOD-COOPERATION (ours)",
    "orca_trial":           "ORCA",
}

ROBOT_RADIUS = 0.2   # D_SAFE / 2.0 — agent radius from constants.py
N_GHOST      = 20


def load_trial(trial_dir):
    robot_data = {}
    for rd in sorted(glob.glob(os.path.join(trial_dir, "*"))):
        if not os.path.isdir(rd):
            continue
        robot_name = os.path.basename(rd)
        csv_path = os.path.join(rd, f"{robot_name}_data.csv")
        if not os.path.isfile(csv_path):
            continue
        df = pd.read_csv(csv_path)
        if len(df) < 2:
            continue
        robot_data[robot_name] = df
    return robot_data


def draw_trajectories(ax, robot_data, title):
    colors = cm.tab10(np.linspace(0, 1, len(robot_data)))

    x_offset = np.mean([df["x"].values[0] for df in robot_data.values()])
    y_offset = np.mean([df["y"].values[0] for df in robot_data.values()])

    for (robot_name, df), color in zip(robot_data.items(), colors):
        x = df["x"].values - x_offset
        y = df["y"].values - y_offset

        # Clip trajectory at first point that exits [-2, 2] on either axis
        out_of_bounds = np.where((np.abs(x) >= 2) | (np.abs(y) >= 2))[0]
        if len(out_of_bounds) > 0:
            stop = out_of_bounds[0] + 1   # include the boundary point
            x = x[:stop]
            y = y[:stop]

        # Fading ghost circles
        indices = np.linspace(0, len(x) - 1, N_GHOST, dtype=int)
        for k, idx in enumerate(indices):
            alpha = 0.03 + 0.28 * (k / (N_GHOST - 1))
            ax.add_patch(plt.Circle((x[idx], y[idx]), ROBOT_RADIUS,
                                    color=color, alpha=alpha, zorder=2))

        # Trajectory line
        ax.plot(x, y, "-", color=color, linewidth=1.5, alpha=0.9,
                label=robot_name, zorder=3)

        # Start dot
        ax.plot(x[0], y[0], "o", color=color, markersize=4, zorder=5)

        # End: bold circle + X
        ax.add_patch(plt.Circle((x[-1], y[-1]), ROBOT_RADIUS,
                                color=color, fill=False, linewidth=2.5, zorder=6))
        r = ROBOT_RADIUS * 0.65
        ax.plot([x[-1] - r, x[-1] + r], [y[-1] + r, y[-1] - r],
                color=color, linewidth=2, zorder=7)
        ax.plot([x[-1] - r, x[-1] + r], [y[-1] - r, y[-1] + r],
                color=color, linewidth=2, zorder=7)

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(title, fontsize=11, fontweight="normal", pad=6)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.autoscale()
    ax.legend().remove()


# ── main ──────────────────────────────────────────────────────────────────────
trial_names = list(TRIAL_LABELS.keys())

fig, axes = plt.subplots(2, 2, figsize=(10, 9))

for (row, col), ax, trial_name in zip(np.ndindex(2, 2), axes.flat, trial_names):
    trial_dir  = os.path.join(SCENARIO_DIR, trial_name)
    robot_data = load_trial(trial_dir)
    if not robot_data:
        ax.set_title(f"{TRIAL_LABELS[trial_name]}\n(no data)", fontsize=11, fontweight="normal")
        ax.axis("off")
        continue
    draw_trajectories(ax, robot_data, TRIAL_LABELS[trial_name])
    if col == 0:
        ax.set_ylabel("y (m)", fontsize=9)
    if row == 1:
        ax.set_xlabel("x (m)", fontsize=9)

# Unify axis limits across all subplots and make them square so
# equal-aspect ratio leaves no padding
x_mins, x_maxs, y_mins, y_maxs = [], [], [], []
for ax in axes.flat:
    x_mins.append(ax.get_xlim()[0])
    x_maxs.append(ax.get_xlim()[1])
    y_mins.append(ax.get_ylim()[0])
    y_maxs.append(ax.get_ylim()[1])

cx = (min(x_mins) + max(x_maxs)) / 2
cy = (min(y_mins) + max(y_maxs)) / 2
half = max(max(x_maxs) - min(x_mins), max(y_maxs) - min(y_mins)) / 2 * 1.02  # 2% margin

for ax in axes.flat:
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))

plt.tight_layout(h_pad=0.5, w_pad=0.2, pad=0.3)

out_dir  = os.path.join(PLOTS_ROOT, SCENARIO)
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "combined_trajectories.png")
fig.savefig(out_path, dpi=150)
print(f"Saved: {out_path}")
