import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "6_agents_0_humans_3_rogue" / "nod_cooperation_test_4"
PLOT_DIR = Path(__file__).parent.parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)

D_SAFE = 0.4
ROBOTS = ["tb1", "tb2", "tb3", "tb5", "tb6", "tb9"]
NOD    = ["tb3", "tb5", "tb9"]
ROGUE  = ["tb1", "tb2", "tb6"]

COLORS = {"tb1": "#d62728", "tb2": "#ff7f0e", "tb3": "#1f77b4",
          "tb5": "#2ca02c", "tb6": "#9467bd", "tb9": "#8c564b"}
NOD_LS   = "-"
ROGUE_LS = "--"

def n(s):
    return np.array(s, dtype=float)

dfs = {}
for r in ROBOTS:
    df = pd.read_csv(DATA_DIR / r / f"{r}_data.csv")
    df["t"] = n(df["t"]) - n(df["t"])[0]
    dfs[r] = df

def dist(df_ego, r_other):
    dx = n(df_ego[f"x_{r_other}"]) - n(df_ego["x"])
    dy = n(df_ego[f"y_{r_other}"]) - n(df_ego["y"])
    return np.sqrt(dx**2 + dy**2)

fig = plt.figure(figsize=(16, 12))
fig.suptitle("nod_cooperation_test_4  —  Rogues: tb1, tb2, tb6  |  NOD: tb3, tb5, tb9",
             fontsize=13, fontweight="bold")

gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35)

# ── Row 0: one p_coop subplot per NOD agent ───────────────────────────────────
for col_i, nod in enumerate(NOD):
    ax = fig.add_subplot(gs[0, col_i])
    df = dfs[nod]
    t  = n(df["t"])
    for rogue in ROGUE:
        col = f"p_coop_{rogue}"
        if col in df.columns:
            vals = n(pd.to_numeric(df[col], errors="coerce"))
            ax.plot(t, vals, color=COLORS[rogue], lw=1.4, label=f"→{rogue}")
    ax.axhline(0, color="k", lw=0.8, linestyle=":")
    ax.set_title(f"p_coop  [{nod} — NOD]", fontsize=10)
    ax.set_xlabel("time (s)"); ax.set_ylabel("p_coop")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.0, 1.1)

# ── Row 1: distances from each NOD agent to each rogue ───────────────────────
for col_i, nod in enumerate(NOD):
    ax = fig.add_subplot(gs[1, col_i])
    df = dfs[nod]
    t  = n(df["t"])
    for rogue in ROGUE:
        if f"x_{rogue}" in df.columns:
            d = dist(df, rogue)
            ax.plot(t, d, color=COLORS[rogue], lw=1.4, label=f"↔{rogue}")
    ax.axhline(D_SAFE, color="red", linestyle=":", lw=1.5, label="D_SAFE")
    ax.set_title(f"Distance  [{nod} to rogues]", fontsize=10)
    ax.set_xlabel("time (s)"); ax.set_ylabel("dist (m)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_ylim(bottom=0)

# ── Row 2 left: trajectories ─────────────────────────────────────────────────
ax = fig.add_subplot(gs[2, 0:2])
for r, df in dfs.items():
    ls = ROGUE_LS if r in ROGUE else NOD_LS
    lw = 1.8 if r in ROGUE else 1.4
    x, y = n(df["x"]), n(df["y"])
    ax.plot(x, y, color=COLORS[r], linestyle=ls, linewidth=lw,
            label=f"{r} ({'R' if r in ROGUE else 'N'})")
    ax.plot(x[0],  y[0],  "o", color=COLORS[r], ms=7)
    ax.plot(x[-1], y[-1], "s", color=COLORS[r], ms=7)

# collision zones (all NOD–rogue pairs)
for nod in NOD:
    for rogue in ROGUE:
        if f"x_{rogue}" in dfs[nod].columns:
            d = dist(dfs[nod], rogue)
            mask = d < D_SAFE
            if mask.any():
                ax.scatter(n(dfs[nod]["x"])[mask], n(dfs[nod]["y"])[mask],
                           s=20, color="red", zorder=5, alpha=0.6)

ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
ax.set_title("Trajectories  (○ start, □ end,  red = collision,  dashed = rogue)")
ax.legend(fontsize=8, ncol=2); ax.set_aspect("equal")
ax.grid(True, alpha=0.3)
ax.axhline(0, color="k", lw=0.4); ax.axvline(0, color="k", lw=0.4)

# ── Row 2 right: NOD target speeds ───────────────────────────────────────────
ax = fig.add_subplot(gs[2, 2])
for r in NOD:
    ax.plot(n(dfs[r]["t"]), n(dfs[r]["target_speed"]),
            color=COLORS[r], lw=1.4, label=r)
ax.axhline(0.35, color="gray", linestyle="--", lw=1.0, label="v_nominal")
ax.set_xlabel("time (s)"); ax.set_ylabel("target speed (m/s)")
ax.set_title("NOD agent speeds")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

out = PLOT_DIR / "nod_cooperation_test_4.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
