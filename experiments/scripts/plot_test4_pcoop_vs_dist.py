import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "6_agents_0_humans_3_rogue" / "nod_cooperation_test_4"
PLOT_DIR = Path(__file__).parent.parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)

D_SAFE = 0.4
NOD   = ["tb3", "tb5", "tb9"]
ROGUE = ["tb1", "tb2", "tb6"]
COLORS = {"tb1": "#d62728", "tb2": "#ff7f0e", "tb6": "#9467bd"}

def n(s):
    return np.array(pd.to_numeric(s, errors="coerce"), dtype=float)

dfs = {}
for r in NOD:
    df = pd.read_csv(DATA_DIR / r / f"{r}_data.csv")
    df["t"] = n(df["t"]) - n(df["t"])[0]
    dfs[r] = df

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
fig.suptitle("p_coop vs distance to rogue  (test_4)\n"
             "colour = rogue agent,  darker = later in trial",
             fontsize=12, fontweight="bold")

for ax, nod in zip(axes, NOD):
    df = dfs[nod]
    t  = n(df["t"])
    t_norm = (t - t.min()) / (t.max() - t.min())   # 0 → 1 over trial

    for rogue in ROGUE:
        if f"x_{rogue}" not in df.columns:
            continue
        dx   = n(df[f"x_{rogue}"]) - n(df["x"])
        dy   = n(df[f"y_{rogue}"]) - n(df["y"])
        dist = np.sqrt(dx**2 + dy**2)
        pcoop = n(df[f"p_coop_{rogue}"])

        mask = np.isfinite(dist) & np.isfinite(pcoop)
        d, p, t_n = dist[mask], pcoop[mask], t_norm[mask]

        # scatter: alpha encodes time (lighter = early, darker = later)
        base = np.array(plt.matplotlib.colors.to_rgb(COLORS[rogue]))
        colors = np.outer(t_n, base) + np.outer(1 - t_n, np.ones(3) * 0.85)
        colors = np.clip(colors, 0, 1)

        ax.scatter(d, p, c=colors, s=6, zorder=3)
        # invisible proxy for legend
        ax.scatter([], [], color=COLORS[rogue], s=20, label=rogue)

    ax.axhline(0, color="k", lw=1.0, linestyle="--", label="detection threshold")
    ax.axvline(D_SAFE, color="red", lw=1.0, linestyle=":", label="D_SAFE")
    ax.set_xlabel("distance to rogue (m)")
    ax.set_ylabel("p_coop" if nod == "tb3" else "")
    ax.set_title(f"{nod}  [NOD]")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.05, 1.1)
    ax.invert_xaxis()   # closer on the right → natural "approaching" read

# shared x-axis label note
fig.text(0.5, 0.01, "← closer                                further →",
         ha="center", fontsize=9, color="gray")

plt.tight_layout(rect=[0, 0.04, 1, 1])
out = PLOT_DIR / "nod_cooperation_test_4_pcoop_vs_dist.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
