#!/usr/bin/env python3
"""
Two-panel figure (version 2) for nod_cooperation_test_14.

Top panel   : 4–5 disk snapshots per agent (evenly spaced in time), velocity arrows,
              outline-only intermediate disks, filled start/end disks.
              Wider aspect ratio to suit the mostly-horizontal motion.
Bottom panel: Cooperation estimate & opinion with shaded background (green/red by sign),
              vertical dashed lines at key narrative moments.

Caption (self-contained):
  The NOD-COOPERATION agent (green) initially advances toward its goal while the
  NON-COOPERATIVE agent (red) crosses its path. As the cooperative agent accumulates
  evidence of non-cooperation (cooperation estimate falling below zero), it infers
  that the oncoming agent will not yield and updates its opinion strongly negative,
  commanding a full stop to let the non-cooperative agent pass. Once the non-cooperative
  agent has cleared the intersection, the cooperation estimate recovers above zero and
  the opinion returns to neutral, allowing the cooperative agent to resume its journey.

Outputs:
    tb3_tb6_figure_version_2.png
    tb3_tb6_figure_version_2_animated.gif
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, FancyArrow

# ── paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRIAL_DIR  = os.path.join(SCRIPT_DIR, "..", "data",
                           "4_agents_0_humans_2_rogue", "nod_cooperation_test_14")
PLOT_DIR   = os.path.join(SCRIPT_DIR, "..", "plots",
                           "4_agents_0_humans_2_rogue", "nod_cooperation_test_14")
os.makedirs(PLOT_DIR, exist_ok=True)

# ── load data ──────────────────────────────────────────────────────────────────
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

# COM offset
x_off = np.mean([df["x"].values[0] for df in [df1, df2, df3, df6]])
y_off = np.mean([df["y"].values[0] for df in [df1, df2, df3, df6]])

BOUNDARY     = 1.5
ROBOT_RADIUS = 0.17

def clip_at_boundary(df):
    x     = df["x"].values - x_off
    y     = df["y"].values - y_off
    t     = df["t"].values
    speed = df["target_speed"].values if "target_speed" in df.columns else np.ones(len(t))
    dist  = np.sqrt((x - x[0])**2 + (y - y[0])**2)
    moved = np.where(dist > MOVE_THRESHOLD)[0]
    si    = moved[0] if len(moved) > 0 else 0
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

t_end_abs = max(t3_abs[-1], t6_abs[-1])
df3_sig   = df3[df3["t"] <= t_end_abs].reset_index(drop=True)
t3_sig    = df3_sig["t"].values - t0

# Second zero-crossing of cooperation
coop_vals_full = df3_sig["p_coop_tb6"].values if "p_coop_tb6" in df3_sig.columns else np.array([])
t_cross2 = 0.0
if len(coop_vals_full) > 0:
    signs     = np.sign(coop_vals_full)
    crossings = np.where(np.diff(signs) != 0)[0]
    if len(crossings) >= 2:
        i1 = crossings[1]; i2 = i1 + 1
        dv = coop_vals_full[i2] - coop_vals_full[i1]
        t_cross2 = t3_sig[i1] + (0 - coop_vals_full[i1]) / dv * (t3_sig[i2] - t3_sig[i1]) if dv != 0 else t3_sig[i1]
    elif len(crossings) == 1:
        i1 = crossings[0]; i2 = i1 + 1
        dv = coop_vals_full[i2] - coop_vals_full[i1]
        t_cross2 = t3_sig[i1] + (0 - coop_vals_full[i1]) / dv * (t3_sig[i2] - t3_sig[i1]) if dv != 0 else t3_sig[i1]

PLOT_DURATION = 10.0
t_plot_end    = t_cross2 + PLOT_DURATION

mask3 = (t3 >= t_cross2) & (t3 <= t_plot_end)
mask6 = (t6 >= t_cross2) & (t6 <= t_plot_end)
x3_plot, y3_plot, t3_plot = x3[mask3], y3[mask3], t3[mask3]
x6_plot, y6_plot, t6_plot = x6[mask6], y6[mask6], t6[mask6]

# ── colours ────────────────────────────────────────────────────────────────────
COLOR_TB3  = "#1a9641"   # vivid green
COLOR_TB6  = "#d7191c"   # vivid red
CMAP_BLUES = cm.Blues
CMAP_OPN   = cm.Purples
BLUES_NORM = mcolors.Normalize(vmin=0, vmax=t_max)

# ── figure layout ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(7, 6.5))
gs  = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1, 0.25], hspace=0.25)
gs.update(top=0.98, bottom=0.09, left=0.10, right=0.98)
ax_traj = fig.add_subplot(gs[0])
ax_sig  = fig.add_subplot(gs[1])

# ══ TOP: trajectories with disk snapshots + velocity arrows ═══════════════════
ARROW_SCALE = 0.35

def draw_agent(ax, x, y, t_arr, color, snap_times, labeled_times=None, right_labeled_times=None):
    """Draw trajectory, disk snapshots, velocity arrows.
    snap_times: list of (t, is_start, is_end, label_str or None)
    labeled_times: set of t values to annotate with a time label on the plot.
    """
    # trajectory line
    ax.plot(x, y, "-", color=color, linewidth=1.5, alpha=0.35, zorder=3)

    # per-point velocity
    dt    = np.diff(t_arr)
    dt    = np.where(dt > 0, dt, 1e-6)
    vx_r  = np.append(np.diff(x) / dt, 0)
    vy_r  = np.append(np.diff(y) / dt, 0)
    spd_r = np.sqrt(vx_r**2 + vy_r**2)

    for (t_s, is_start, is_end) in snap_times:
        sx  = float(np.interp(t_s, t_arr, x))
        sy  = float(np.interp(t_s, t_arr, y))
        svx = float(np.interp(t_s, t_arr, vx_r))
        svy = float(np.interp(t_s, t_arr, vy_r))
        ssp = float(np.interp(t_s, t_arr, spd_r))

        if is_start or is_end:
            ax.add_patch(plt.Circle((sx, sy), ROBOT_RADIUS,
                                    color=color, alpha=0.85, zorder=4))
        else:
            ax.add_patch(plt.Circle((sx, sy), ROBOT_RADIUS,
                                    color=color, fill=False,
                                    linewidth=1.8, alpha=0.8, zorder=4))

        if ssp > 0.01:
            norm_v = np.sqrt(svx**2 + svy**2)
            dx_ = svx / norm_v * ARROW_SCALE * ssp
            dy_ = svy / norm_v * ARROW_SCALE * ssp
            ax.annotate("", xy=(sx + dx_, sy + dy_), xytext=(sx, sy),
                        arrowprops=dict(arrowstyle="-|>", color=color,
                                        lw=1.5, mutation_scale=10),
                        zorder=6)

        # time label if requested
        if labeled_times and t_s in labeled_times:
            t_label = t_s - t_cross2   # shift to plot time
            if color == COLOR_TB3:
                if right_labeled_times and t_s in right_labeled_times:
                    ax.text(sx + ROBOT_RADIUS + 0.06, sy, f"t={t_label:.0f}s",
                            ha="left", va="center", fontsize=9, color=color,
                            fontweight="bold", zorder=8)
                else:
                    ax.text(sx - ROBOT_RADIUS - 0.06, sy, f"t={t_label:.0f}s",
                            ha="right", va="center", fontsize=9, color=color,
                            fontweight="bold", zorder=8)
            else:
                # red labels above
                ax.text(sx, sy + ROBOT_RADIUS + 0.08, f"t={t_label:.0f}s",
                        ha="center", va="bottom", fontsize=9, color=color,
                        fontweight="bold", zorder=8)

    # end X
    ex, ey = float(np.interp(t_arr[-1], t_arr, x)), float(np.interp(t_arr[-1], t_arr, y))
    r = ROBOT_RADIUS * 0.65
    for sgn in [(1, 1), (1, -1)]:
        ax.plot([ex - r*sgn[0], ex + r*sgn[0]],
                [ey + r*sgn[1], ey - r*sgn[1]],
                color="white", linewidth=1.8, zorder=7)

# Shared snapshot times for both agents
t4_abs = t_cross2 + 2.0
t6_abs = t_cross2 + 6.0

# Green (tb3): start + hollow at t=4,6 + end
green_snaps = [
    (t3_plot[0],  True,  False),
    (t4_abs,      False, False),
    (t6_abs,      False, False),
    (t3_plot[-1], False, True),
]
draw_agent(ax_traj, x3_plot, y3_plot, t3_plot, COLOR_TB3,
           green_snaps, labeled_times={t3_plot[0], t4_abs, t6_abs, t3_plot[-1]},
           right_labeled_times={t6_abs})

# Red (tb6): start + hollow at t=4,6 + end
red_snaps = [
    (t6_plot[0],  True,  False),
    (t4_abs,      False, False),
    (t6_abs,      False, False),
    (t6_plot[-1], False, True),
]
draw_agent(ax_traj, x6_plot, y6_plot, t6_plot, COLOR_TB6,
           red_snaps, labeled_times={t6_plot[0], t4_abs, t6_abs, t6_plot[-1]})

ax_traj.set_aspect("equal")
ax_traj.set_xlim(-2.3, 0.6)
ax_traj.set_ylim(-1.8, 0.7)
ax_traj.set_xlabel("x (m)", fontsize=11)
ax_traj.set_ylabel("y (m)", fontsize=11)
ax_traj.set_title("")
ax_traj.tick_params(labelsize=9)
ax_traj.grid(True, linestyle="--", alpha=0.4)
ax_traj.legend(
    handles=[Patch(facecolor=COLOR_TB3, label="NOD-COOPERATION"),
             Patch(facecolor=COLOR_TB6, label="NON-COOPERATIVE")],
    loc="lower left", fontsize=9, framealpha=0.85)

dot3, = ax_traj.plot([], [], "o", color=COLOR_TB3, markersize=10, zorder=8)
dot6, = ax_traj.plot([], [], "o", color=COLOR_TB6, markersize=10, zorder=8)

# ══ BOTTOM: signals ═══════════════════════════════════════════════════════════
coop_vals = df3_sig["p_coop_tb6"].values if "p_coop_tb6" in df3_sig.columns else None
t3_sig_plot = t3_sig - t_cross2

# Background shading by cooperation estimate sign (disabled)
# if coop_vals is not None:
#     t_w  = t3_sig_plot
#     c_w  = coop_vals
#     win  = (t_w >= 0) & (t_w <= PLOT_DURATION)
#     t_sh = t_w[win]
#     c_sh = c_w[win]
#     for i in range(len(t_sh) - 1):
#         shade = "#d7191c" if c_sh[i] < 0 else "#1a9641"
#         ax_sig.axvspan(t_sh[i], t_sh[i+1], alpha=0.06, color=shade, zorder=0)

# Cooperation estimate line
if coop_vals is not None:
    ax_sig.plot(t3_sig_plot, coop_vals, "-", color=CMAP_BLUES(0.8),
                linewidth=1.8, zorder=3, label="Cooperation Estimate")

# Opinion line
if "opinion" in df3_sig.columns:
    opn_vals = df3_sig["opinion"].values
    ax_sig.plot(t3_sig_plot, opn_vals, "-", color=CMAP_OPN(0.7),
                linewidth=1.8, zorder=3, label="Speed Opinion")

# Find first time cooperation drops below 0 in the plot window
t_detects = 2.0  # fallback
if coop_vals is not None:
    win_idx = np.where((t3_sig_plot >= 0) & (t3_sig_plot <= PLOT_DURATION))[0]
    if len(win_idx) > 1:
        c_win = coop_vals[win_idx]
        t_win = t3_sig_plot[win_idx]
        for k in range(1, len(c_win)):
            if c_win[k-1] >= 0 and c_win[k] < 0:
                dv = c_win[k] - c_win[k-1]
                t_detects = float(t_win[k-1] + (0 - c_win[k-1]) / dv * (t_win[k] - t_win[k-1])) if dv != 0 else float(t_win[k-1])
                break

# Find first time opinion drops below 0 after t_detects
t_yields = 6.4  # fallback
if "opinion" in df3_sig.columns:
    opn_full = df3_sig["opinion"].values
    win_idx_o = np.where((t3_sig_plot >= t_detects) & (t3_sig_plot <= PLOT_DURATION))[0]
    if len(win_idx_o) > 1:
        o_win = opn_full[win_idx_o]
        t_win_o = t3_sig_plot[win_idx_o]
        for k in range(1, len(o_win)):
            if o_win[k-1] >= 0 and o_win[k] < 0:
                dv = o_win[k] - o_win[k-1]
                t_yields = float(t_win_o[k-1] + (0 - o_win[k-1]) / dv * (t_win_o[k] - t_win_o[k-1])) if dv != 0 else float(t_win_o[k-1])
                break

# Key moment vertical lines  (t, label, va, ha, x_offset, y_pos)
key_moments = [
    (t_detects, "Detects\nnon-coop.", "bottom", "right", -0.1, 0.03),
    (t_yields,  "Yields",             "top",    "left",   0.1, 0.97),
    (8.0,       "Stopped,\nRed passes", "center", "left", 0.1, 0.5),
]
for t_k, label, va, ha, x_off, y_pos in key_moments:
    ax_sig.axvline(t_k, color="dimgray", linewidth=1.0, linestyle="--", alpha=0.7, zorder=4)
    ax_sig.text(t_k + x_off, y_pos, label, transform=ax_sig.get_xaxis_transform(),
                fontsize=9, va=va, ha=ha, color="black",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))

ax_sig.axhline(0, color="gray", linewidth=0.7, linestyle="--", alpha=0.5, zorder=2)
ax_sig.set_xlabel("Time (s)", fontsize=11)
ax_sig.set_ylabel("Value", fontsize=11)
ax_sig.set_title("")
ax_sig.set_ylim(-1.1, 1.1)
ax_sig.set_xlim(0, PLOT_DURATION)
ax_sig.tick_params(labelsize=9)
ax_sig.legend(loc="lower left", fontsize=9, framealpha=0.85)
ax_sig.grid(True, linestyle="--", alpha=0.4)

# ── save static PNG ────────────────────────────────────────────────────────────
static_path = os.path.join(PLOT_DIR, "tb3_tb6_figure_version_2.png")
fig.savefig(static_path, dpi=150)
print(f"Saved: {static_path}")

# ── animation ─────────────────────────────────────────────────────────────────
time_cursor = ax_sig.axvline(x=0, color="black", linewidth=1.2,
                              linestyle="-", alpha=0.8, zorder=10)

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
anim_path = os.path.join(PLOT_DIR, "tb3_tb6_figure_version_2_animated.gif")
anim.save(anim_path, writer="pillow", fps=8)
print(f"Saved: {anim_path}")
