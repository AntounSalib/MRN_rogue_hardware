import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "6_agents_0_humans_3_rogue" / "nod_cooperation_test_1"
PLOT_DIR = Path(__file__).parent.parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)

D_SAFE = 0.4
ROBOTS = ["tb1", "tb2", "tb3", "tb5", "tb6", "tb9"]
NOD    = ["tb1", "tb3", "tb9"]
ROGUE  = ["tb2", "tb5", "tb6"]
COLORS = {"tb1": "#1f77b4", "tb2": "#d62728", "tb3": "#2ca02c",
          "tb5": "#ff7f0e", "tb6": "#9467bd", "tb9": "#8c564b"}

def n(s):
    """Convert Series (or array) to numpy."""
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

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("nod_cooperation_test_1  —  6 agents, 3 rogue", fontsize=13, fontweight="bold")

# ── 1. Trajectories ───────────────────────────────────────────────────────────
ax = axes[0, 0]
for r, df in dfs.items():
    ls = "--" if r in ROGUE else "-"
    x, y = n(df["x"]), n(df["y"])
    ax.plot(x, y, color=COLORS[r], linestyle=ls, linewidth=1.4, label=r)
    ax.plot(x[0],  y[0],  "o", color=COLORS[r], ms=7)
    ax.plot(x[-1], y[-1], "s", color=COLORS[r], ms=7)

for ego, other in [("tb1","tb2"), ("tb9","tb5")]:
    df = dfs[ego]
    d  = dist(df, other)
    mask = d < D_SAFE
    if mask.any():
        ax.scatter(n(df["x"])[mask], n(df["y"])[mask], s=18, color="red", zorder=5, alpha=0.5)

ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
ax.set_title("Trajectories  (○ start, □ end,  red = collision zone)")
ax.legend(fontsize=8, ncol=2)
ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
ax.axhline(0, color="k", lw=0.4); ax.axvline(0, color="k", lw=0.4)

# ── 2. Inter-robot distances ──────────────────────────────────────────────────
ax = axes[0, 1]
for ego, other, ls in [("tb1","tb2","-"), ("tb9","tb5","-"), ("tb1","tb6","--"), ("tb3","tb6","--")]:
    col_x = f"x_{other}"
    if col_x in dfs[ego].columns:
        d = dist(dfs[ego], other)
        ax.plot(n(dfs[ego]["t"]), d, color=COLORS[ego], linestyle=ls,
                lw=1.4, label=f"{ego}↔{other}")

ax.axhline(D_SAFE, color="red", linestyle=":", lw=1.5, label=f"D_SAFE={D_SAFE}m")
ax.set_xlabel("time (s)"); ax.set_ylabel("distance (m)")
ax.set_title("Inter-robot distances")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_ylim(bottom=0)

# ── 3. p_coop: NOD → rogue ────────────────────────────────────────────────────
ax = axes[1, 0]
for ego, other, ls in [("tb1","tb2","-"), ("tb9","tb5","-"), ("tb1","tb6","--"), ("tb3","tb2","--")]:
    col = f"p_coop_{other}"
    if col in dfs[ego].columns:
        vals = n(pd.to_numeric(dfs[ego][col], errors="coerce"))
        ax.plot(n(dfs[ego]["t"]), vals, color=COLORS[ego], linestyle=ls,
                lw=1.4, label=f"{ego}→{other}")

ax.axhline(0.0, color="k", linestyle=":", lw=1.0, label="threshold = 0")
ax.set_xlabel("time (s)"); ax.set_ylabel("p_coop")
ax.set_title("Cooperation detection (NOD → rogue)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# ── 4. NOD speeds ─────────────────────────────────────────────────────────────
ax = axes[1, 1]
for r in NOD:
    ax.plot(n(dfs[r]["t"]), n(dfs[r]["target_speed"]), color=COLORS[r], lw=1.4, label=r)

for ego, other, t_c in [("tb1","tb2",16.0), ("tb9","tb5",13.6)]:
    ax.axvline(t_c, color=COLORS[ego], linestyle=":", lw=1.2, label=f"{ego}↔{other} breach")

ax.axhline(0.35, color="gray", linestyle="--", lw=1.0, label="v_nominal")
ax.set_xlabel("time (s)"); ax.set_ylabel("target speed (m/s)")
ax.set_title("NOD agent speeds  (dotted = collision onset)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout()
out = PLOT_DIR / "nod_cooperation_test_1.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
