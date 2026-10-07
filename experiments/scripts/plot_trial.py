#!/usr/bin/env python3
"""Plot saved trajectories, decision (opinion), and current speed without ROS.

Example: python3 experiments/scripts/plot_trial.py experiments/data/test_SR/R0
Defaults to the newest trial and its latest session (timestamp gaps >5 s).
"""
import argparse
import ast
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from plot_data import read_plot_csv, smooth_plot_signal


ROOT = Path(__file__).resolve().parents[1]


def cooperation_config(trial):
    """Read the recorded flag and threshold without executing saved Python."""
    path = trial / "constants_copy.py"
    if not path.exists():
        return False, 0.0
    tree = ast.parse(path.read_text())
    config = next((node for node in tree.body
                   if isinstance(node, ast.ClassDef) and node.name == "NodConfig"), None)
    if config is None:
        return False, 0.0
    cooperation = next((node for node in config.body
                        if isinstance(node, ast.ClassDef) and node.name == "cooperation"), None)
    values = {}
    if cooperation is not None:
        for node in cooperation.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in ("COOPERATION_LAYER_ON", "COOPERATION_THRESHOLD"):
                        values[target.id] = ast.literal_eval(node.value)
    return bool(values.get("COOPERATION_LAYER_ON", False)), float(values.get("COOPERATION_THRESHOLD", 0.0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial", nargs="?", type=Path, help="folder containing per-robot folders")
    parser.add_argument("--robots", nargs="+", help="robot names to display")
    parser.add_argument("--session", type=int, default=-1,
                        help="session index, starting at 0; default -1 selects latest")
    parser.add_argument("--gap", type=float, default=5,
                        help="timestamp gap in seconds defining separate sessions (default 5)")
    parser.add_argument("--show", action="store_true", help="also open an interactive plot window")
    args = parser.parse_args()
    if args.gap <= 0:
        parser.error("--gap must be positive")
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    trial = args.trial
    if trial is None:
        candidates = list((ROOT / "data").glob("*/*/*/*_data.csv"))
        if not candidates:
            parser.error("No saved CSV files found under experiments/data")
        trial = max(candidates, key=lambda p: p.stat().st_mtime).parents[1]
    trial = trial.resolve()
    coop_enabled, coop_threshold = cooperation_config(trial)
    data = {}
    for file in sorted(trial.glob("*/*_data.csv")):
        name = file.parent.name
        frame = read_plot_csv(file, usecols=lambda col: col in
                            {"t", "x", "y", "opinion", "current_speed", "target_speed"}
                            or (coop_enabled and col.startswith("p_coop_") and not col.startswith("p_coop_att_")))
        if not {"t", "x", "y"}.issubset(frame.columns):
            continue
        frame = frame.apply(pd.to_numeric, errors="coerce")
        # Keep missing positions as line breaks after tracking cleanup.
        frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["t"])
        if not frame.empty:
            data[name] = frame.sort_values("t").drop_duplicates("t", keep="last")
    if not data:
        parser.error("No usable position data in " + str(trial))

    # Use one shared clock and session window across robots.
    times = np.unique(np.concatenate([df.t.to_numpy() for df in data.values()]))
    splits = np.flatnonzero(np.diff(times) > args.gap) + 1
    sessions = np.split(times, splits)
    try:
        window = sessions[args.session]
    except IndexError:
        parser.error("Session index out of range; available: 0 to %d" % (len(sessions) - 1))
    index = args.session % len(sessions)
    print("Trial:", trial)
    print("Found %d session(s) using gaps > %g s; plotting session %d, duration %.2f s."
          % (len(sessions), args.gap, index, window[-1] - window[0]))
    print("Session timestamps: %.6f to %.6f" % (window[0], window[-1]))
    if args.robots:
        missing = set(args.robots) - set(data)
        if missing:
            parser.error("No data for: " + ", ".join(sorted(missing)))
    data = {name: df[(df.t >= window[0]) & (df.t <= window[-1])].copy()
            for name, df in data.items() if not args.robots or name in args.robots}
    data = {name: df for name, df in data.items() if not df.empty}
    if not data:
        parser.error("Selected robots have no data in this session")

    from matplotlib.collections import LineCollection

    figures = {name: plt.subplots(figsize=(7, 5.5))
               for name in ('decision', 'trajectory', 'speed')}
    opinion, trajectory, speed = (figures[name][1] for name in ('decision', 'trajectory', 'speed'))
    trajectory.set(xlabel="x (m)", ylabel="y (m)")
    trajectory.set_aspect("equal", adjustable="datalim")
    speed.set(xlabel="Time (s)", ylabel="Speed (m/s)")
    opinion.set(xlabel="Time (s)", ylabel="Decision (z)")
    for i, (name, df) in enumerate(data.items()):
        color = np.asarray(plt.get_cmap("tab10")(i % 10))
        t = df.t.to_numpy() - window[0]
        x, y = df.x.to_numpy(), df.y.to_numpy()
        positions = np.column_stack((x, y))
        segments = np.stack((positions[:-1], positions[1:]), axis=1)
        valid_segments = np.isfinite(segments).all(axis=(1, 2)) & (np.diff(t) > 0) & (np.diff(t) <= 0.5)
        step_distance = np.hypot(np.diff(x), np.diff(y))
        step_speed = np.divide(step_distance, np.diff(t), out=np.full(len(step_distance), np.inf),
                               where=np.diff(t) > 0)
        valid_segments &= step_speed <= 0.8
        # Fade by distance travelled so a long stationary tail cannot wash out
        # the moving part of the trajectory. Never connect tracking gaps.
        distances = np.where(valid_segments, step_distance, 0)
        progress = np.cumsum(distances) / max(np.sum(distances), 1e-12)
        colors = np.tile(color, (len(segments), 1))
        colors[:, :3] = 1 - (1 - color[:3]) * (0.25 + 0.75 * progress[:, None])
        trajectory.add_collection(LineCollection(segments[valid_segments], colors=colors[valid_segments],
                                                  linewidths=2.5, capstyle='round'))
        trajectory.plot([], [], color=color, linewidth=2.5, label=name)
        valid = np.flatnonzero(np.isfinite(x) & np.isfinite(y))
        if len(valid):
            trajectory.plot(x[valid[0]], y[valid[0]], "o", color=1 - (1 - color) * 0.4, markersize=6)
            trajectory.plot(x[valid[-1]], y[valid[-1]], "x", color=color, markersize=8, markeredgewidth=2)
        measured = (df.current_speed.to_numpy() if "current_speed" in df
                    else np.full(len(t), np.nan))
        if np.isfinite(measured).any():
            speed.plot(t, measured, color=color, linewidth=2, label=name)
        if len(t) >= 2 and not np.isfinite(measured).all():
            estimated = np.hypot(np.diff(x), np.diff(y)) / np.diff(t)
            estimated[~valid_segments] = np.nan
            estimated = smooth_plot_signal(estimated, (t[1:] + t[:-1]) / 2,
                                           preserve_zero=True, jump_threshold=0.12)
            estimated[np.isfinite(measured[1:]) & np.isfinite(measured[:-1])] = np.nan
            speed.plot((t[1:] + t[:-1]) / 2, estimated, ":", color=color,
                       linewidth=2, label=name + " estimate")
        if "opinion" in df:
            opinion.plot(t, df.opinion.to_numpy(), color=color, linewidth=2, label=name)
        if coop_enabled:
            columns = [col for col in df if col.startswith("p_coop_")
                       and not col.startswith("p_coop_att_") and np.isfinite(df[col]).any()]
            if columns:
                fig, ax = plt.subplots(figsize=(9, 5.5))
                for col in columns:
                    ax.plot(t, df[col].to_numpy(), linewidth=1.5, label=col[len("p_coop_"):])
                ax.axhline(coop_threshold, color="gray", linestyle="--", linewidth=1,
                           label="Classification threshold")
                ax.set(xlabel="Time (s)", ylabel="Cooperation score", title=name + " — per-neighbor cooperation",
                       ylim=(-1.05, 1.05))
                ax.legend(fontsize=9, ncol=3)
                figures["cooperation_" + name] = (fig, ax)
        print("  %s: %d samples" % (name, len(df)))
    trajectory.autoscale_view()
    opinion.axhline(0, color="gray", linewidth=0.7, alpha=0.5)
    speed.set_ylim(bottom=0)
    for fig, ax in figures.values():
        ax.tick_params(axis='both', labelsize=16)
        ax.xaxis.label.set_size(18)
        ax.yaxis.label.set_size(18)
        ax.grid(alpha=0.2)
        fig.tight_layout()
    # One shared color key, exported separately from the three data figures.
    from matplotlib.lines import Line2D
    handles, labels = trajectory.get_legend_handles_labels()
    if any(line.get_label().endswith(' estimate') and np.isfinite(line.get_ydata()).any()
           for line in speed.lines):
        handles.append(Line2D([], [], color='gray', linestyle=':', linewidth=2))
        labels.append('Speed estimate')
    legend_fig, legend_ax = plt.subplots(figsize=(7, 1.4))
    legend_ax.axis('off')
    legend_ax.legend(handles, labels, loc='center', ncol=3, fontsize=16,
                     frameon=False, handlelength=2.5)
    legend_fig.tight_layout()
    figures['legend'] = (legend_fig, legend_ax)
    out = ROOT / "plots" / trial.parent.name / trial.name
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / ("plot_filter_session_%d.json" % index)
    report_path.write_text(json.dumps({"scope": "source CSVs, filtered separately per session",
                                       "robots": {name: df.attrs.get('plot_filter', {}) for name, df in data.items()}}, indent=2))
    for name, (fig, ax) in figures.items():
        for extension in ("png", "pdf"):
            path = out / ("%s_session_%d.%s" % (name, index, extension))
            fig.savefig(path, dpi=200, bbox_inches="tight")
            print("Saved:", path)
    if args.show:
        plt.show()
    for fig, ax in figures.values():
        plt.close(fig)


if __name__ == "__main__":
    main()
