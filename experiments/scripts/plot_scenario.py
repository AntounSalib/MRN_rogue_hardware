"""Preview: python3 experiments/scripts/plot_scenario.py turning_intersection."""
import argparse
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from scenarios import FIELD_BOUNDS, FIELD_CORNERS, INTERSECTION_CENTERS, LANE_OFFSET, EXTRA_WEST_LANE_Y, EXTRA_SOUTH_LANE_X, SCENARIO_NAMES, get_scenario


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", choices=SCENARIO_NAMES)
    args = parser.parse_args()
    layout = get_scenario(args.scenario)
    fig, ax = plt.subplots(figsize=(9, 8))
    xmin, xmax, ymin, ymax = FIELD_BOUNDS
    if layout.maneuvers:
        cx, cy = INTERSECTION_CENTERS[0]
        # Include the added west/south approach lanes in the same junction.
        left, right = cx - LANE_OFFSET - 0.3, EXTRA_SOUTH_LANE_X + 0.3
        bottom, top = EXTRA_WEST_LANE_Y - 0.3, cy + LANE_OFFSET + 0.3
        ax.add_patch(Rectangle((xmin, bottom), xmax - xmin, top - bottom, color="#eeeeee", zorder=0))
        ax.add_patch(Rectangle((left, ymin), right - left, ymax - ymin, color="#eeeeee", zorder=0))
        for divider in (cy, (EXTRA_WEST_LANE_Y + cy - LANE_OFFSET) / 2):
            for a, b in (((xmin, divider), (left, divider)), ((right, divider), (xmax, divider))):
                ax.plot(*zip(a, b), color="#888888", linestyle="--", linewidth=1)
        for divider in (cx, (EXTRA_SOUTH_LANE_X + cx + LANE_OFFSET) / 2):
            for a, b in (((divider, ymin), (divider, bottom)), ((divider, top), (divider, ymax))):
                ax.plot(*zip(a, b), color="#888888", linestyle="--", linewidth=1)
        ax.add_patch(Rectangle((left, bottom), right - left, top - bottom,
                              fill=False, edgecolor="#aaaaaa", linestyle=":"))
    polygon = FIELD_CORNERS + (FIELD_CORNERS[0],)
    ax.plot(*zip(*polygon), color="#999999", linestyle=":", label="Recorded room")
    ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                           fill=False, edgecolor="#333333", linestyle="--"))
    for i, (robot, start) in enumerate(layout.start_positions.items()):
        path = (start[:2],) + layout.waypoints[robot] + (layout.goal_positions[robot],)
        color = plt.get_cmap("tab10")(i)
        movements = " → ".join(layout.maneuvers.get(robot, ()))
        ax.plot(*zip(*path), color=color, linewidth=2, label=robot + (": " + movements if movements else ""))
        ax.scatter(*start[:2], color=color, s=60, marker="o", zorder=5)
        ax.scatter(*layout.goal_positions[robot], color=color, s=80, marker="x", zorder=5)
        ax.annotate(robot, start[:2], xytext=(5, 5), textcoords="offset points", color=color)
        ax.arrow(start[0], start[1], 0.2 * math.cos(start[2]),
                 0.2 * math.sin(start[2]), width=0.015, color=color)
    ax.set(aspect="equal", xlabel="Vicon x (m)", ylabel="Vicon y (m)",
           title=args.scenario + " — circles: starts, crosses: goals (inside field)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=9)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    output = ROOT / "experiments" / "plots" / "scenarios" / (args.scenario + ".png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
