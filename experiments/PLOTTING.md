All saved-data plotting scripts in `experiments/scripts` use the shared
`plot_data.read_plot_csv()` reader. Tracking cleanup is enabled by default.
The reader returns a filtered copy for plotting and never modifies raw CSVs.

```bash
python3 experiments/scripts/plot_trial.py
# Or select a trial explicitly:
python3 experiments/scripts/plot_trial.py experiments/data/final_SR/R_orth
```

The filter checks isolated spatial jumps against neighboring positions and
generous speed limits (0.8 m/s for robots, 4 m/s for human tracks). Measured
`current_speed` receives a separate positive-spike filter. Drops to zero,
ordinary turns, sharp sustained speed changes, and commanded
`target_speed` remain intact. Neighbor position columns use the same spatial
filter for distance/trajectory plots.

Only missing samples bracketed by valid measurements no more than 0.5 s
apart are interpolated, using their actual timestamps. Longer losses and
missing endpoints remain NaN instead of inventing motion. Recordings are
filtered separately across restarts/nonincreasing timestamps or gaps over
5 s. This handles isolated tracking glitches; it does not reconstruct
persistent identity swaps between robots.

Speed spike rejection also checks short upward bursts against the local
median and speed inferred from recorded displacement, before smoothing.

Measured speed and opinion receive plotting-only smoothing: a
nine-sample median for speed (five for opinion) followed by a triangular timestamp-weighted average
within 0.75 s on either side. Smoothing never bridges missing samples, time
gaps over 0.5 s, or sharp steps (0.12 m/s for speed, 0.3 for opinion).
Zero-speed samples remain exactly zero and separate smoothing intervals.
Sustained steps are detected after median filtering so isolated jitter no
longer disables smoothing. Where recorded `target_speed` is exactly zero,
filtered measured speed below 0.03 m/s and recorded opinion magnitude below
0.03 are displayed as zero. Larger decision tails remain visible; the
filter does not extrapolate their decay or append unrecorded samples.

`plot_trial.py` exports three independent figures with no titles:
`decision_session_N`, `trajectory_session_N`, and `speed_session_N`, each
as PNG and PDF. Tick labels are 16 pt and axis labels are 18 pt.
The three data figures omit legends. One shared `legend_session_N` figure
contains the robot color key and is saved separately as PNG and PDF.

When the trial's saved `constants_copy.py` has `COOPERATION_LAYER_ON = True`,
the script also saves `cooperation_tbX_session_N.png` and `.pdf` for each
robot with recorded pairwise scores. These show each neighbor's `p_coop_*`
metric and the saved classification threshold; cooperation attention is a
separate metric and is not included. The flag comes from the recorded trial,
so changing today's controller settings does not change which past trials qualify.

If position-derived speeds are drawn, the shared legend also includes their
dotted line style.
Trajectories fade from light at the start to the full robot color at the end,
according to distance travelled. Circles mark starts and crosses mark ends.
Tracking gaps and steps implying speeds above 0.8 m/s are not connected.
Measured speed is lightly filtered; dotted lines indicate position-based
estimates where measured speed is missing.

The reader prints correction counts. `plot_trial.py` saves
`plot_filter_session_N.json` beside its PNG/PDF.
The report covers the source CSVs, with sessions filtered independently.

For new plotting scripts, import `read_plot_csv` from `plot_data` rather
than reading robot recordings directly with pandas.

Goal stops, boundary stops, and heading-alignment pivots continue recording
at the control rate until the node is shut down. These rows have a zero
linear `target_speed`; `current_speed` remains the actual Vicon measurement
and can show braking or tracking noise. After goal/boundary stops, NOD opinion
and attention follow the existing free-flow decay toward zero. Temporary
heading-alignment pivots hold the opinion. In the NOD speed mapping, opinion
zero corresponds to nominal speed; terminal stops keep their separate zero
speed override while opinion relaxes. Older logs that ended before the
stop have no measured stopping tail; plots do not add synthetic zero samples.
Previously recorded flat opinion tails remain unchanged in the raw data.
