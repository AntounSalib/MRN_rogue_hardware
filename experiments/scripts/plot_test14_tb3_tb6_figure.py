#!/usr/bin/env python3
"""
Two-panel figure for nod_cooperation_test_14 (4_agents_0_humans_2_rogue).

Top panel   : trajectories of tb3 (NOD) and tb6 (ROGUE).
Bottom panel: tb3's cooperation estimate of tb6  (p_coop_tb6 from tb3_data.csv)
              tb3's opinion signal                (opinion      from tb3_data.csv)

Outputs:
    tb3_tb6_figure.png          – static version
    tb3_tb6_figure_animated.gif – animated version with time cursor
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation

# ── paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRIAL_DIR  = os.path.join(SCRIPT_DIR, "..", "data",
                           "4_agents_0_humans_2_rogue", "nod_cooperation_test_14")
PLOT_DIR   = os.path.join(SCRIPT_DIR, "..", "plots",
                           "4_agents_0_humans_2_rogue", "nod_cooperation_test_14")
os.makedirs(PLOT_DIR, exist_ok=True)

# ── load data ──────────────────────────────────────────────────────────────────
# Load all 4 active robots; tb1/tb2 used only for COM offset calculation
df1 = pd.read_csv(os.path.join(TRIAL_DIR, "tb1", "tb1_data.csv"))
df2 = pd.read_csv(os.path.join(TRIAL_DIR, "tb2", "tb2_data.csv"))
df3 = pd.read_csv(os.path.join(TRIAL_DIR, "tb3", "tb3_data.csv"))
df6 = pd.read_csv(os.path.join(TRIAL_DIR, "tb6", "tb6_data.csv"))

# ── trim startup stillness ─────────────────────────────────────────────────────
MOVE_THRESHOLD = 0.05
def first_move_time(df):
    x0, y0 = df["x"].values[0], df["y"].values[0]
    dist = np.sqrt((df["x"].values - x0)**2 + (df["y"].values - y0)**2)
    idx = np.where(dist > MOVE_THRESHOLD)[0]
    return df["t"].values[idx[0]] if len(idx) > 0 else None

t_starts = [t for t in [first_move_time(df) for df in [df1, df2, df3, df6]] if t is not None]
if t_starts:
    t_start = min(t_starts)
    df1 = df1[df1["t"] >= t_start].reset_index(drop=True)
    df2 = df2[df2["t"] >= t_start].reset_index(drop=True)
    df3 = df3[df3["t"] >= t_start].reset_index(drop=True)
    df6 = df6[df6["t"] >= t_start].reset_index(drop=True)

# COM offset from all 4 active robots' starting positions
x_off = np.mean([df["x"].values[0] for df in [df1, df2, df3, df6]])
y_off = np.mean([df["y"].values[0] for df in [df1, df2, df3, df6]])

# ── clip trajectories at ±2 m (after offset), same logic as combined plot ──────
BOUNDARY    = 1.5
ROBOT_RADIUS = 0.17

def clip_at_boundary(df):
    """Trim start to when this robot first moves; trim end when it exits ±BOUNDARY."""
    x     = df["x"].values - x_off
    y     = df["y"].values - y_off
    t     = df["t"].values
    speed = df["target_speed"].values if "target_speed" in df.columns else np.ones(len(t))

    dist = np.sqrt((x - x[0])**2 + (y - y[0])**2)

    # Per-robot start: skip still period until this robot individually moves
    moved = np.where(dist > MOVE_THRESHOLD)[0]
    si    = moved[0] if len(moved) > 0 else 0

    # Begin oob check only once the robot has entered the bounded zone — prevents
    # robots that start outside the boundary from being clipped immediately.
    in_zone = np.where((np.abs(x) < BOUNDARY) & (np.abs(y) < BOUNDARY))[0]
    if len(in_zone) > 0:
        si_oob = in_zone[0]
        oob = np.where((np.abs(x[si_oob:]) >= BOUNDARY) | (np.abs(y[si_oob:]) >= BOUNDARY))[0]
        if len(oob) > 0:
            clip = si_oob + oob[0] + 1
            return x[si:clip], y[si:clip], t[si:clip], speed[si:clip]
    return x[si:], y[si:], t[si:], speed[si:]

x3, y3, t3_abs, spd3 = clip_at_boundary(df3)
x6, y6, t6_abs, spd6 = clip_at_boundary(df6)

t0    = min(t3_abs[0], t6_abs[0])
t3    = t3_abs - t0
t6    = t6_abs - t0
t_max = max(t3[-1], t6[-1])

# Signals plot: start at t=0, end when the last agent hits the boundary
t_end_abs = max(t3_abs[-1], t6_abs[-1])
df3_sig   = df3[df3["t"] <= t_end_abs].reset_index(drop=True)
t3_sig    = df3_sig["t"].values - t0

# Find second zero-crossing of cooperation; everything after is referenced to this
coop_vals_full = df3_sig["p_coop_tb6"].values if "p_coop_tb6" in df3_sig.columns else np.array([])
t_cross2 = 0.0
if len(coop_vals_full) > 0:
    signs     = np.sign(coop_vals_full)
    crossings = np.where(np.diff(signs) != 0)[0]
    if len(crossings) >= 2:
        t_cross2 = t3_sig[crossings[1]]
    elif len(crossings) == 1:
        t_cross2 = t3_sig[crossings[0]]

# Trim trajectory arrays to [t_cross2, t_cross2 + PLOT_DURATION]
PLOT_DURATION = 10.0   # seconds to show after t_cross2
t_plot_end = t_cross2 + PLOT_DURATION

mask3 = (t3 >= t_cross2) & (t3 <= t_plot_end)
mask6 = (t6 >= t_cross2) & (t6 <= t_plot_end)
x3_plot, y3_plot, t3_plot = x3[mask3], y3[mask3], t3[mask3]
x6_plot, y6_plot, t6_plot = x6[mask6], y6[mask6], t6[mask6]

# ── colours ────────────────────────────────────────────────────────────────────
COLOR_TB3  = "#2ca02c"   # green  (NOD trajectory)
COLOR_TB6  = "#d62728"   # red    (ROGUE trajectory)
COLOR_OPN  = "#ff7f0e"   # orange (opinion)

# Blues colormap range matching the ghost circle shading (light → dark)
CMAP_BLUES = cm.Greens
BLUES_NORM = mcolors.Normalize(vmin=0, vmax=t_max)

# ── figure layout ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(7, 10))
gs  = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1, 0.4], hspace=0.4)
ax_traj = fig.add_subplot(gs[0])
ax_sig  = fig.add_subplot(gs[1])

# ══ TOP: trajectories ═════════════════════════════════════════════════════════
N_GHOST = 20

# Global time range for consistent alpha normalisation across both robots
t_global_min = min(t3_plot[0], t6_plot[0])
t_global_max = max(t3_plot[-1], t6_plot[-1])
t_span       = t_global_max - t_global_min

for x, y, t_arr, color, label in [(x3_plot, y3_plot, t3_plot, COLOR_TB3, "tb3 (NOD)"),
                                    (x6_plot, y6_plot, t6_plot, COLOR_TB6, "tb6 (ROGUE)")]:
    # ghost circles at evenly-spaced indices; alpha reflects absolute time
    indices = np.linspace(0, len(x) - 1, N_GHOST, dtype=int)
    last_x, last_y = None, None
    for k, idx in enumerate(indices):
        if last_x is not None and np.sqrt((x[idx]-last_x)**2 + (y[idx]-last_y)**2) < ROBOT_RADIUS * 0.5:
            continue
        t_norm = (t_arr[idx] - t_global_min) / t_span
        alpha  = 0.03 + 0.28 * t_norm
        ax_traj.add_patch(plt.Circle((x[idx], y[idx]), ROBOT_RADIUS,
                                     color=color, alpha=alpha, zorder=2))
        last_x, last_y = x[idx], y[idx]

    # trajectory line (faded so animated dot pops)
    ax_traj.plot(x, y, "-", color=color, linewidth=1.8, alpha=0.4, zorder=3)

    # start dot
    ax_traj.plot(x[0], y[0], "o", color=color, markersize=5, zorder=5)

    # end marker: circle outline + X
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
ax_traj.legend().set_visible(False)
ax_traj.grid(True, linestyle="--", alpha=0.4)

dot3, = ax_traj.plot([], [], "o", color=COLOR_TB3, markersize=10, zorder=8)
dot6, = ax_traj.plot([], [], "o", color=COLOR_TB6, markersize=10, zorder=8)



# ══ BOTTOM: signals + time cursor ════════════════════════════════════════════
coop_vals = df3_sig["p_coop_tb6"].values if "p_coop_tb6" in df3_sig.columns else None

t3_sig_plot = t3_sig - t_cross2   # shifted time axis for the cooperation plot

if coop_vals is not None:
    # x positions use shifted time; color still tracks real elapsed time
    points = np.array([t3_sig_plot, coop_vals]).T.reshape(-1, 1, 2)
    segs   = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segs, cmap=CMAP_BLUES, norm=BLUES_NORM,
                        linewidth=1.8, zorder=3)
    lc.set_array(t3_sig[:-1])
    ax_sig.add_collection(lc)
    ax_sig.plot([], [], "-", color=CMAP_BLUES(0.7), linewidth=1.8,
                label="tb3 cooperation estimate of tb6")
else:
    print("Warning: p_coop_tb6 not found in tb3 data")

if "opinion" in df3_sig.columns:
    opn_vals = df3_sig["opinion"].values
    points_o = np.array([t3_sig_plot, opn_vals]).T.reshape(-1, 1, 2)
    segs_o   = np.concatenate([points_o[:-1], points_o[1:]], axis=1)
    lc_o = LineCollection(segs_o, cmap=cm.Oranges, norm=BLUES_NORM,
                          linewidth=1.8, zorder=3)
    lc_o.set_array(t3_sig[:-1])
    ax_sig.add_collection(lc_o)
    ax_sig.plot([], [], "-", color=cm.Oranges(0.7), linewidth=1.8,
                label="tb3 opinion")
else:
    print("Warning: opinion not found in tb3 data")

ax_sig.axvline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5, zorder=2)
ax_sig.set_xlabel("time (s)", fontsize=11)
ax_sig.set_ylabel("value", fontsize=11)
ax_sig.set_title("tb3 Cooperation Estimate of tb6 & Opinion", fontsize=12)
ax_sig.set_ylim(-1.1, 1.1)
ax_sig.set_xlim(0, PLOT_DURATION)
ax_sig.legend().set_visible(False)
ax_sig.grid(True, linestyle="--", alpha=0.4)

# ── save static PNG (no time cursor) ──────────────────────────────────────────
plt.tight_layout()
static_path = os.path.join(PLOT_DIR, "tb3_tb6_figure.png")
fig.savefig(static_path, dpi=150)
print(f"Saved: {static_path}")

# Add time cursor for animated GIF only
time_cursor = ax_sig.axvline(x=t3_sig_plot[0], color="black", linewidth=1.2,
                              linestyle="--", alpha=0.7, zorder=10)

# ── animation ─────────────────────────────────────────────────────────────────
N_FRAMES = 150
t_grid   = np.linspace(t_cross2, t_plot_end, N_FRAMES)

def init():
    dot3.set_data([], [])
    dot6.set_data([], [])
    time_cursor.set_xdata([0])
    return dot3, dot6, time_cursor

def update(frame_idx):
    t_now = t_grid[frame_idx]
    dot3.set_data([np.interp(t_now, t3_plot, x3_plot)], [np.interp(t_now, t3_plot, y3_plot)])
    dot6.set_data([np.interp(t_now, t6_plot, x6_plot)], [np.interp(t_now, t6_plot, y6_plot)])
    time_cursor.set_xdata([t_now - t_cross2])
    return dot3, dot6, time_cursor

anim = FuncAnimation(fig, update, frames=N_FRAMES, init_func=init,
                     interval=50, blit=True)
anim_path = os.path.join(PLOT_DIR, "tb3_tb6_figure_animated.gif")
anim.save(anim_path, writer="pillow", fps=8)
print(f"Saved: {anim_path}")
