#!/usr/bin/env python3
"""
Plot robot trajectories from a trial data folder.

Usage:
    python3 plot_trajectories.py [trial_folder]

If no argument is given, defaults to the first (oldest) trial folder
under data/.  Plots are saved to experiments/plots/<trial_name>/.
"""

import sys
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation

# ── locate trial folder ────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT  = os.path.join(SCRIPT_DIR, "..", "data")
PLOTS_ROOT = os.path.join(SCRIPT_DIR, "..", "plots")

if len(sys.argv) > 1:
    trial_dir = sys.argv[1]
else:
    candidates = glob.glob(os.path.join(DATA_ROOT, "*", "*"))
    trial_dirs = [p for p in candidates if os.path.isdir(p)]
    if not trial_dirs:
        sys.exit(f"No trial folders found under {DATA_ROOT}")
    trial_dir = min(trial_dirs, key=os.path.getmtime)   # oldest = first
    print(f"Using trial folder: {trial_dir}")

trial_name  = os.path.basename(trial_dir)
trial_id    = os.path.basename(os.path.dirname(trial_dir))
plot_dir    = os.path.join(PLOTS_ROOT, trial_id, trial_name)
os.makedirs(plot_dir, exist_ok=True)
print(f"Plots will be saved to: {plot_dir}")

# ── load per-robot CSVs ────────────────────────────────────────────────────────
robot_dirs = sorted(glob.glob(os.path.join(trial_dir, "*")))
robot_data = {}
for rd in robot_dirs:
    if not os.path.isdir(rd):
        continue
    robot_name = os.path.basename(rd)
    csv_path = os.path.join(rd, f"{robot_name}_data.csv")
    if not os.path.isfile(csv_path):
        continue
    df = pd.read_csv(csv_path)
    if len(df) < 2:
        print(f"  Skipping {robot_name}: only {len(df)} row(s)")
        continue
    robot_data[robot_name] = df

if not robot_data:
    sys.exit("No robot data found.")

print(f"Loaded robots: {list(robot_data.keys())}")

# ── colour palette ─────────────────────────────────────────────────────────────
colors = cm.tab10(np.linspace(0, 1, len(robot_data)))

# ── centre-of-gravity normalisation (plot only, data files unchanged) ──────────
x_offset = np.mean([df["x"].values[0] for df in robot_data.values()])
y_offset = np.mean([df["y"].values[0] for df in robot_data.values()])

# ── static trajectory plot ─────────────────────────────────────────────────────
ROBOT_RADIUS = 0.17   # visual footprint radius (m)
N_GHOST      = 20     # ghost circles per robot trail

fig, ax = plt.subplots(figsize=(7, 7))
ax.set_aspect("equal")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title(f"Robot Trajectories\n{trial_name}")
ax.grid(True, linestyle="--", alpha=0.4)

for (robot_name, df), color in zip(robot_data.items(), colors):
    x = df["x"].values - x_offset
    y = df["y"].values - y_offset

    # Fading ghost circles: transparent at start, opaque toward end
    indices = np.linspace(0, len(x) - 1, N_GHOST, dtype=int)
    for k, idx in enumerate(indices):
        alpha = 0.03 + 0.28 * (k / (N_GHOST - 1))
        circle = plt.Circle((x[idx], y[idx]), ROBOT_RADIUS,
                             color=color, alpha=alpha, zorder=2)
        ax.add_patch(circle)

    # Trajectory centre-line
    ax.plot(x, y, "-", color=color, linewidth=1.5, alpha=0.9, zorder=3,
            label=robot_name)

    # Start: small filled dot
    ax.plot(x[0], y[0], "o", color=color, markersize=4, zorder=5)

    # End: bold circle outline + X
    end_circ = plt.Circle((x[-1], y[-1]), ROBOT_RADIUS,
                           color=color, fill=False, linewidth=2.5, zorder=6)
    ax.add_patch(end_circ)
    r = ROBOT_RADIUS * 0.65
    ax.plot([x[-1] - r, x[-1] + r], [y[-1] + r, y[-1] - r],
            color=color, linewidth=2, zorder=7)
    ax.plot([x[-1] - r, x[-1] + r], [y[-1] - r, y[-1] + r],
            color=color, linewidth=2, zorder=7)

ax.autoscale()
ax.legend(loc="upper right", fontsize=8)
plt.tight_layout()
static_path = os.path.join(plot_dir, "trajectories_static.png")
fig.savefig(static_path, dpi=150)
print(f"Saved: {static_path}")

# ── animated plot ──────────────────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(8, 8))
ax2.set_aspect("equal")
ax2.set_xlabel("x (m)")
ax2.set_ylabel("y (m)")
ax2.set_title(f"Robot Trajectories (animated)\n{trial_name}")
ax2.grid(True, linestyle="--", alpha=0.4)

t_min = min(df["t"].values[0]  for df in robot_data.values())
t_max = max(df["t"].values[-1] for df in robot_data.values())
N_FRAMES = 120
t_grid = np.linspace(t_min, t_max, N_FRAMES)

all_x = np.concatenate([df["x"].values for df in robot_data.values()]) - x_offset
all_y = np.concatenate([df["y"].values for df in robot_data.values()]) - y_offset
margin = 0.5
ax2.set_xlim(all_x.min() - margin, all_x.max() + margin)
ax2.set_ylim(all_y.min() - margin, all_y.max() + margin)

lines = {}
dots  = {}
for (robot_name, df), color in zip(robot_data.items(), colors):
    line, = ax2.plot([], [], "-", color=color, linewidth=2,
                     label=f"{robot_name}", alpha=0.8)
    dot,  = ax2.plot([], [], "o", color=color, markersize=9, zorder=5)
    lines[robot_name] = line
    dots[robot_name]  = dot

time_text = ax2.text(0.02, 0.96, "", transform=ax2.transAxes, fontsize=9,
                     verticalalignment="top")
ax2.legend(loc="upper right", fontsize=8)


def _interp(df, t_query):
    t = df["t"].values
    x = np.interp(t_query, t, df["x"].values, left=np.nan, right=np.nan) - x_offset
    y = np.interp(t_query, t, df["y"].values, left=np.nan, right=np.nan) - y_offset
    return x, y


def init():
    for line in lines.values():
        line.set_data([], [])
    for dot in dots.values():
        dot.set_data([], [])
    time_text.set_text("")
    return list(lines.values()) + list(dots.values()) + [time_text]


def update(frame_idx):
    t_now   = t_grid[frame_idx]
    artists = []
    for robot_name, df in robot_data.items():
        mask = df["t"].values <= t_now
        if mask.sum() == 0:
            lines[robot_name].set_data([], [])
            dots[robot_name].set_data([], [])
        else:
            lines[robot_name].set_data(df["x"].values[mask] - x_offset, df["y"].values[mask] - y_offset)
            cx, cy = _interp(df, t_now)
            if not np.isnan(cx):
                dots[robot_name].set_data([cx], [cy])
        artists += [lines[robot_name], dots[robot_name]]
    time_text.set_text(f"t = {t_now - t_min:.1f} s")
    artists.append(time_text)
    return artists


anim = FuncAnimation(fig2, update, frames=N_FRAMES, init_func=init,
                     interval=50, blit=True)

plt.tight_layout()
anim_path = os.path.join(plot_dir, "trajectories_animated.gif")
anim.save(anim_path, writer="pillow", fps=6)
print(f"Saved: {anim_path}")

# ── tb6 cooperation with tb1 and tb3 ──────────────────────────────────────────
if "tb6" in robot_data:
    df6 = robot_data["tb6"]
    t6  = df6["t"].values - df6["t"].values[0]   # relative time

    fig3, ax3 = plt.subplots(figsize=(9, 4))
    ax3.set_xlabel("time (s)")
    ax3.set_ylabel("cooperation")
    ax3.set_title(f"tb6 Cooperation\n{trial_name}")
    ax3.set_ylim(-1.05, 1.05)
    ax3.grid(True, linestyle="--", alpha=0.4)

    for target, color in [("tb1", "tab:blue"), ("tb3", "tab:orange")]:
        col = f"p_coop_{target}"
        if col in df6.columns:
            ax3.plot(t6, df6[col].values, color=color, linewidth=1.8,
                     label=f"tb6 → {target}")
        else:
            print(f"  Warning: column {col} not found in tb6 data")

    ax3.legend()
    plt.tight_layout()
    coop_path = os.path.join(plot_dir, "tb6_cooperation.png")
    fig3.savefig(coop_path, dpi=150)
    print(f"Saved: {coop_path}")
else:
    print("tb6 data not available, skipping cooperation plot.")

# ── tb1 cooperation with tb2 ───────────────────────────────────────────────────
if "tb1" in robot_data:
    df1 = robot_data["tb1"]
    t1  = df1["t"].values - df1["t"].values[0]

    fig4, ax4 = plt.subplots(figsize=(9, 4))
    ax4.set_xlabel("time (s)")
    ax4.set_ylabel("cooperation")
    ax4.set_title(f"tb1 Cooperation → tb2\n{trial_name}")
    ax4.set_ylim(-1.05, 1.05)
    ax4.grid(True, linestyle="--", alpha=0.4)

    col = "p_coop_tb2"
    if col in df1.columns:
        ax4.plot(t1, df1[col].values, color="tab:green", linewidth=1.8,
                 label="tb1 → tb2")
    else:
        print(f"  Warning: column {col} not found in tb1 data")

    ax4.legend()
    plt.tight_layout()
    coop1_path = os.path.join(plot_dir, "tb1_cooperation_tb2.png")
    fig4.savefig(coop1_path, dpi=150)
    print(f"Saved: {coop1_path}")
else:
    print("tb1 data not available, skipping tb1 cooperation plot.")

# ── tb3 cooperation with tb6 ───────────────────────────────────────────────────
if "tb3" in robot_data:
    df3 = robot_data["tb3"]
    t3  = df3["t"].values - df3["t"].values[0]

    fig5, ax5 = plt.subplots(figsize=(9, 4))
    ax5.set_xlabel("time (s)")
    ax5.set_ylabel("cooperation")
    ax5.set_title(f"tb3 Cooperation → tb6\n{trial_name}")
    ax5.set_ylim(-1.05, 1.05)
    ax5.grid(True, linestyle="--", alpha=0.4)

    col = "p_coop_tb6"
    if col in df3.columns:
        ax5.plot(t3, df3[col].values, color="tab:purple", linewidth=1.8,
                 label="tb3 → tb6")
    else:
        print(f"  Warning: column {col} not found in tb3 data")

    ax5.legend()
    plt.tight_layout()
    coop3_path = os.path.join(plot_dir, "tb3_cooperation_tb6.png")
    fig5.savefig(coop3_path, dpi=150)
    print(f"Saved: {coop3_path}")
else:
    print("tb3 data not available, skipping tb3 cooperation plot.")

