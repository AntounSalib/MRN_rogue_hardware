#!/usr/bin/env python3
"""
Combined static trajectory plot for all methods in 4_agents_0_humans_2_rogue.
Produces a 2x2 grid, one subplot per method.
"""

import os
import re
import glob
import itertools
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.cm as cm
import matplotlib.ticker as ticker
import numpy as np

COLOR_ROGUES   = ["#d62728", "#8b0000"]   # bright red, dark red
COLOR_NORMAL   = ["#1f77b4", "#2ca02c"]


def load_rogue_agents(trial_dir):
    path = os.path.join(trial_dir, "constants_copy.py")
    if not os.path.isfile(path):
        return set()
    with open(path) as f:
        content = f.read()
    match = re.search(r'ROGUE_AGENTS\s*=\s*\{([^}]*)\}', content)
    if not match:
        return set()
    return set(re.findall(r'"(\w+)"', match.group(1)))


SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT    = os.path.join(SCRIPT_DIR, "..", "data")
PLOTS_ROOT   = os.path.join(SCRIPT_DIR, "..", "plots")

SCENARIO     = "4_agents_0_humans_2_rogue"
SCENARIO_DIR = os.path.join(DATA_ROOT, SCENARIO)

TRIAL_LABELS = {
    "mpc_cbf_trial":           "MPC-CBF",
    "nh_orca_trial_2":         "NH-ORCA",
    "nod_cooperation_trial_5": "NOD-COOPERATION (ours)",
    "orca_trial_2":            "ORCA",
}

ROBOT_RADIUS   = 0.2
N_GHOST        = 20
MOVE_THRESHOLD = 0.05

# Which agent pairs collide in each subplot (flat index → (name1, name2))
COLLISION_AGENTS = {
    0: ("tb1", "tb2"),   # MPC-CBF: blue vs bright red
    3: ("tb1", "tb6"),   # ORCA: blue vs dark red
}


def find_collision_point(robot_data, name1, name2, x_offset, y_offset):
    """Return (mx, my, t_collision): midpoint of closest approach in plot coords."""
    df1, df2 = robot_data[name1], robot_data[name2]
    t1, t2 = df1["t"].values, df2["t"].values
    t_min = max(t1[0], t2[0])
    t_max = min(t1[-1], t2[-1])
    t_grid = np.linspace(t_min, t_max, 2000)
    x1 = np.interp(t_grid, t1, df1["x"].values) - x_offset
    y1 = np.interp(t_grid, t1, df1["y"].values) - y_offset
    x2 = np.interp(t_grid, t2, df2["x"].values) - x_offset
    y2 = np.interp(t_grid, t2, df2["y"].values) - y_offset
    dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    idx = np.argmin(dist)
    return (x1[idx] + x2[idx]) / 2, (y1[idx] + y2[idx]) / 2, t_grid[idx]


def load_trial(trial_dir):
    rogue_agents = load_rogue_agents(trial_dir)
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

    t_start = None
    for df in robot_data.values():
        x0, y0 = df["x"].values[0], df["y"].values[0]
        dist = np.sqrt((df["x"].values - x0)**2 + (df["y"].values - y0)**2)
        moving = np.where(dist > MOVE_THRESHOLD)[0]
        if len(moving) > 0:
            t_candidate = df["t"].values[moving[0]]
            if t_start is None or t_candidate < t_start:
                t_start = t_candidate

    if t_start is not None:
        robot_data = {name: df[df["t"] >= t_start].reset_index(drop=True)
                      for name, df in robot_data.items()}

    return robot_data, rogue_agents


def draw_trajectories(ax, robot_data, title, rogue_agents=None, collision_times=None):
    if rogue_agents is None:
        rogue_agents = set()
    if collision_times is None:
        collision_times = {}
    if rogue_agents:
        rogue_cycle  = itertools.cycle(COLOR_ROGUES)
        normal_cycle = itertools.cycle(COLOR_NORMAL)
        colors = [next(rogue_cycle) if name in rogue_agents else next(normal_cycle)
                  for name in robot_data]
    else:
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"][:len(robot_data)]

    x_offset = np.mean([df["x"].values[0] for df in robot_data.values()])
    y_offset = np.mean([df["y"].values[0] for df in robot_data.values()])

    t_global_min = min(df["t"].values[0]  for df in robot_data.values())
    t_global_max = max(df["t"].values[-1] for df in robot_data.values())
    t_span = t_global_max - t_global_min if t_global_max > t_global_min else 1.0

    for (robot_name, df), color in zip(robot_data.items(), colors):
        x = df["x"].values - x_offset
        y = df["y"].values - y_offset

        # Only clip at ±2m once the agent has moved away from its start
        moved = np.where(np.sqrt((x - x[0])**2 + (y - y[0])**2) > ROBOT_RADIUS)[0]
        if len(moved) > 0:
            move_start = moved[0]
            oob = np.where((np.abs(x[move_start:]) >= 2) | (np.abs(y[move_start:]) >= 2))[0]
            if len(oob) > 0:
                x = x[:move_start + oob[0] + 1]
                y = y[:move_start + oob[0] + 1]

        t_robot = df["t"].values[:len(x)]
        indices = np.linspace(0, len(x) - 1, N_GHOST, dtype=int)
        last_x, last_y = None, None
        for k, idx in enumerate(indices):
            if last_x is not None and np.sqrt((x[idx]-last_x)**2 + (y[idx]-last_y)**2) < ROBOT_RADIUS * 0.5:
                continue
            t_norm = (t_robot[idx] - t_global_min) / t_span
            alpha = 0.03 + 0.28 * t_norm
            ax.add_patch(plt.Circle((x[idx], y[idx]), ROBOT_RADIUS,
                                    color=color, alpha=alpha, zorder=2))
            last_x, last_y = x[idx], y[idx]

        # Trajectory line: solid before collision, dotted after
        if robot_name in collision_times:
            t_coll = collision_times[robot_name]
            split = int(np.searchsorted(t_robot, t_coll))
            split = min(max(split, 0), len(x) - 1)
            ax.plot(x[:split+1], y[:split+1], "-", color=color,
                    linewidth=1.5, alpha=0.9, label=robot_name, zorder=3)
            if split < len(x) - 1:
                ax.plot(x[split:], y[split:], ":", color=color,
                        linewidth=2.0, alpha=0.9, zorder=3)
        else:
            ax.plot(x, y, "-", color=color, linewidth=1.5, alpha=0.9,
                    label=robot_name, zorder=3)

        ax.plot(x[0], y[0], "o", color=color, markersize=4, zorder=5)

        ax.add_patch(plt.Circle((x[-1], y[-1]), ROBOT_RADIUS,
                                color=color, fill=False, linewidth=2.5, zorder=6))
        r = ROBOT_RADIUS * 0.65
        ax.plot([x[-1] - r, x[-1] + r], [y[-1] + r, y[-1] - r],
                color=color, linewidth=2, zorder=7)
        ax.plot([x[-1] - r, x[-1] + r], [y[-1] - r, y[-1] + r],
                color=color, linewidth=2, zorder=7)

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(title, fontsize=18, fontweight="normal", pad=6)
    ax.tick_params(labelsize=15)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.autoscale()
    ax.legend().remove()


# ── main ──────────────────────────────────────────────────────────────────────
trial_names = list(TRIAL_LABELS.keys())

fig, axes = plt.subplots(2, 2, figsize=(10, 9))

# Pass 1: load all trial data
all_robot_data = {}
all_rogue_agents = {}
for flat_idx, trial_name in enumerate(trial_names):
    trial_dir = os.path.join(SCENARIO_DIR, trial_name)
    robot_data, rogue_agents = load_trial(trial_dir)
    all_robot_data[flat_idx] = robot_data
    all_rogue_agents[flat_idx] = rogue_agents

# Compute collision times for each colliding subplot
collision_times_per_subplot = {}
for flat_idx, (n1, n2) in COLLISION_AGENTS.items():
    robot_data = all_robot_data.get(flat_idx, {})
    if n1 in robot_data and n2 in robot_data:
        x_off = np.mean([df["x"].values[0] for df in robot_data.values()])
        y_off = np.mean([df["y"].values[0] for df in robot_data.values()])
        _, _, t_coll = find_collision_point(robot_data, n1, n2, x_off, y_off)
        collision_times_per_subplot[flat_idx] = {n1: t_coll, n2: t_coll}

# Pass 2: draw trajectories
for flat_idx, (ax, trial_name) in enumerate(zip(axes.flat, trial_names)):
    row, col = divmod(flat_idx, 2)
    robot_data = all_robot_data[flat_idx]
    rogue_agents = all_rogue_agents[flat_idx]
    collision_times = collision_times_per_subplot.get(flat_idx, {})
    if not robot_data:
        ax.set_title(f"{TRIAL_LABELS[trial_name]}\n(no data)", fontsize=15, fontweight="normal")
        ax.axis("off")
        continue
    draw_trajectories(ax, robot_data, TRIAL_LABELS[trial_name], rogue_agents, collision_times)
    if col == 0:
        ax.set_ylabel("y (m)", fontsize=15)
    if row == 1:
        ax.set_xlabel("x (m)", fontsize=15)

x_mins, x_maxs, y_mins, y_maxs = [], [], [], []
for ax in axes.flat:
    x_mins.append(ax.get_xlim()[0])
    x_maxs.append(ax.get_xlim()[1])
    y_mins.append(ax.get_ylim()[0])
    y_maxs.append(ax.get_ylim()[1])

cx = (min(x_mins) + max(x_maxs)) / 2
cy = (min(y_mins) + max(y_maxs)) / 2
half = max(max(x_maxs) - min(x_mins), max(y_maxs) - min(y_mins)) / 2 * 1.02

for ax in axes.flat:
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1.0))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))

# Mark collision points with a black X at the closest approach midpoint
for flat_idx, (n1, n2) in COLLISION_AGENTS.items():
    robot_data = all_robot_data.get(flat_idx, {})
    if n1 in robot_data and n2 in robot_data:
        ax = list(axes.flat)[flat_idx]
        x_off = np.mean([df["x"].values[0] for df in robot_data.values()])
        y_off = np.mean([df["y"].values[0] for df in robot_data.values()])
        cx_coll, cy_coll, _ = find_collision_point(robot_data, n1, n2, x_off, y_off)
        ax.plot(cx_coll, cy_coll, "x", color="black", markersize=14,
                markeredgewidth=2.5, zorder=15)

# Collision annotations: flat index → colliding agent colors
# MPC-CBF (idx 0): dark red + blue; ORCA (idx 3): blue + dark red
COLLISION_PAIRS = {
    0: ["#1f77b4", "#d62728"],
    3: ["#1f77b4", "#8b0000"],
}
for flat_idx, pair_colors in COLLISION_PAIRS.items():
    ax = list(axes.flat)[flat_idx]
    handles = [Line2D([0], [0], marker="o", color="w",
                      markerfacecolor=c, markersize=10, label="")
               for c in pair_colors]
    ax.legend(handles=handles, loc="upper right", fontsize=8,
              title="COLLISION", title_fontsize=9,
              framealpha=0.85, handletextpad=0,
              labelspacing=0, borderpad=0.5, ncol=2,
              columnspacing=0.3)

plt.tight_layout(h_pad=0.5, w_pad=0.2, pad=0.3)

out_dir  = os.path.join(PLOTS_ROOT, SCENARIO)
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "combined_trajectories.png")
fig.savefig(out_path, dpi=150)
print(f"Saved: {out_path}")
