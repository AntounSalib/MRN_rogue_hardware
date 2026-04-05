#!/usr/bin/env python3
"""
Interactive two-panel figure for nod_cooperation_test_14.

Top panel   : trajectories of tb3 (NOD) and tb6 (ROGUE).
              Filled dots show current robot positions.
Bottom panel: tb3's cooperation estimate of tb6 and tb3's opinion.
              Vertical cursor shows the current time.

A slider at the bottom lets you scrub through time.

Usage:
    python3 explore_test14_tb3_tb6.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider

# ── paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRIAL_DIR  = os.path.join(SCRIPT_DIR, "..", "data",
                           "4_agents_0_humans_2_rogue", "nod_cooperation_test_14")

# ── load data ──────────────────────────────────────────────────────────────────
df3 = pd.read_csv(os.path.join(TRIAL_DIR, "tb3", "tb3_data.csv"))
df6 = pd.read_csv(os.path.join(TRIAL_DIR, "tb6", "tb6_data.csv"))

# ── trim startup stillness ─────────────────────────────────────────────────────
MOVE_THRESHOLD = 0.05
def first_move_time(df):
    x0, y0 = df["x"].values[0], df["y"].values[0]
    dist = np.sqrt((df["x"].values - x0)**2 + (df["y"].values - y0)**2)
    idx = np.where(dist > MOVE_THRESHOLD)[0]
    return df["t"].values[idx[0]] if len(idx) > 0 else None

t_starts = [t for t in [first_move_time(df3), first_move_time(df6)] if t is not None]
if t_starts:
    t_start = min(t_starts)
    df3 = df3[df3["t"] >= t_start].reset_index(drop=True)
    df6 = df6[df6["t"] >= t_start].reset_index(drop=True)

t0    = min(df3["t"].values[0], df6["t"].values[0])
t3    = df3["t"].values - t0
t6    = df6["t"].values - t0
t_max = max(t3[-1], t6[-1])

# ── colours ────────────────────────────────────────────────────────────────────
COLOR_TB3  = "#1f77b4"
COLOR_TB6  = "#d62728"
COLOR_COOP = "#2ca02c"
COLOR_OPN  = "#ff7f0e"

# ── figure layout ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(7, 11))
gs  = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1, 1], hspace=0.4)
gs.update(top=0.95, bottom=0.12)          # leave room for the slider
ax_traj = fig.add_subplot(gs[0])
ax_sig  = fig.add_subplot(gs[1])

# ══ TOP: static trajectory + moving dots ═════════════════════════════════════
ROBOT_RADIUS = 0.17
N_GHOST      = 20

x_off = np.mean([df3["x"].values[0], df6["x"].values[0]])
y_off = np.mean([df3["y"].values[0], df6["y"].values[0]])

for df, color, label in [(df3, COLOR_TB3, "tb3 (NOD)"),
                          (df6, COLOR_TB6, "tb6 (ROGUE)")]:
    x = df["x"].values - x_off
    y = df["y"].values - y_off

    indices = np.linspace(0, len(x) - 1, N_GHOST, dtype=int)
    for k, idx in enumerate(indices):
        alpha = 0.03 + 0.28 * (k / (N_GHOST - 1))
        ax_traj.add_patch(plt.Circle((x[idx], y[idx]), ROBOT_RADIUS,
                                     color=color, alpha=alpha, zorder=2))

    ax_traj.plot(x, y, "-", color=color, linewidth=1.8, alpha=0.4, zorder=3)
    ax_traj.plot(x[0], y[0], "o", color=color, markersize=5, zorder=5)

    end_circ = plt.Circle((x[-1], y[-1]), ROBOT_RADIUS,
                           color=color, fill=False, linewidth=2.5, zorder=6)
    ax_traj.add_patch(end_circ)
    r = ROBOT_RADIUS * 0.65
    ax_traj.plot([x[-1] - r, x[-1] + r], [y[-1] + r, y[-1] - r],
                 color=color, linewidth=2, zorder=7)
    ax_traj.plot([x[-1] - r, x[-1] + r], [y[-1] - r, y[-1] + r],
                 color=color, linewidth=2, zorder=7)

    ax_traj.plot([], [], "-", color=color, linewidth=1.8, label=label)

ax_traj.set_aspect("equal")
ax_traj.autoscale()
ax_traj.set_xlabel("x (m)", fontsize=11)
ax_traj.set_ylabel("y (m)", fontsize=11)
ax_traj.set_title("Trajectories", fontsize=12)
ax_traj.legend(loc="upper right", fontsize=9)
ax_traj.grid(True, linestyle="--", alpha=0.4)

dot3, = ax_traj.plot([], [], "o", color=COLOR_TB3, markersize=10, zorder=8)
dot6, = ax_traj.plot([], [], "o", color=COLOR_TB6, markersize=10, zorder=8)

# ══ BOTTOM: signals + cursor ══════════════════════════════════════════════════
if "p_coop_tb6" in df3.columns:
    ax_sig.plot(t3, df3["p_coop_tb6"].values,
                color=COLOR_COOP, linewidth=1.8,
                label="tb3 cooperation estimate of tb6")

if "opinion" in df3.columns:
    ax_sig.plot(t3, df3["opinion"].values,
                color=COLOR_OPN, linewidth=1.8,
                label="tb3 opinion")

ax_sig.set_xlabel("time (s)", fontsize=11)
ax_sig.set_ylabel("value", fontsize=11)
ax_sig.set_title("tb3 Cooperation Estimate of tb6  &  tb3 Opinion", fontsize=12)
ax_sig.set_ylim(-1.1, 1.1)
ax_sig.set_xlim(0, t_max)
ax_sig.legend(loc="upper right", fontsize=9)
ax_sig.grid(True, linestyle="--", alpha=0.4)

time_cursor = ax_sig.axvline(x=0, color="black", linewidth=1.2,
                              linestyle="--", alpha=0.8, zorder=10)

# ── slider ─────────────────────────────────────────────────────────────────────
ax_slider = fig.add_axes([0.12, 0.04, 0.76, 0.025])
slider = Slider(ax_slider, "time (s)", 0.0, t_max, valinit=0.0, color="steelblue")

def update(val):
    t_now = slider.val

    x3 = np.interp(t_now, t3, df3["x"].values - x_off)
    y3 = np.interp(t_now, t3, df3["y"].values - y_off)
    dot3.set_data([x3], [y3])

    x6 = np.interp(t_now, t6, df6["x"].values - x_off)
    y6 = np.interp(t_now, t6, df6["y"].values - y_off)
    dot6.set_data([x6], [y6])

    time_cursor.set_xdata([t_now])
    fig.canvas.draw_idle()

slider.on_changed(update)
update(0.0)   # initialise dots at t=0

plt.show()
