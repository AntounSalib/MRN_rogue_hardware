Run from the package directory to preview a layout without ROS or robot motion:

```bash
python3 src/scenarios.py orthogonal
python3 src/scenarios.py nearly_antipodal
python3 src/scenarios.py turn_left
python3 src/scenarios.py turn_right
python3 src/scenarios.py turning_intersection
```

The layouts use `tb1, tb9, tb6, tb3, tb2, tb5`, matching the current active
robots. Positions are in Vicon metres; start yaw is in radians.

| Scenario | Nominal route |
| --- | --- |
| `orthogonal` | Three eastbound robots cross three southbound robots at 90°. |
| `orth_1` | Three eastbound and three southbound routes, with staggered starts for simultaneous paired crossings; perimeter goals and 1.2 m lane gaps. |
| `orth_2` | Three westbound and three northbound routes with simultaneous paired crossings; vertical lanes shift 0.3 m east and horizontal lanes shift 0.05 m north relative to `orth_1`. |
| `mix_lanes` | Four alternating east/west lanes crossed by two diagonal streams, producing nine interior crossings. |
| `nearly_antipodal` | Six straight routes through `(0, 0.7)`, with opposing headings 160° apart, ending near the far field edge. |
| `antipodal_1` | Same geometry as `nearly_antipodal`, with a numbered name for experiments. |
| `antipodal_2` | Common crossing shifts to `(0.15, 0.9)`; start rays are `15°, 75°, 135°, 220°, 280°, 340°`, with opposing headings 155° apart. |
| `turning_intersection` | One shared four-way intersection; two-way roads with left/right/straight traffic. |
| `turn_left`, `turn_right` | Aliases for the same mixed turning-intersection layout, preserving existing commands. |

Select a numbered variant using the same hardware wrapper. All variants use
the same six robots and automatically position them before running:

```bash
roslaunch MRN_rogue_hardware hardware_scenario.launch scenario:=orth_1
roslaunch MRN_rogue_hardware hardware_scenario.launch scenario:=orth_2
roslaunch MRN_rogue_hardware hardware_scenario.launch scenario:=antipodal_1
roslaunch MRN_rogue_hardware hardware_scenario.launch scenario:=antipodal_2
```

Run one launch at a time. Preview any variant using
`python3 experiments/scripts/plot_scenario.py orth_2` (replace the name).
The original `orthogonal` and `nearly_antipodal` names retain their geometry.
Choose a distinct `TRIAL_SEED` in `src/constants.py` for each recorded variant
to keep the recordings in separate folders.

`orth_1` and `orth_2` stagger start positions inward from the inset perimeter
so every robot has a conflict at equal constant nominal speed and a common
start time. The pairs are `tb1/tb5`, `tb9/tb2`, and `tb6/tb3`; each pair has
equal travel distance to its crossing. All nine horizontal/vertical path
crossings remain inside the routes, and goals retain their perimeter positions.
At 0.15 m/s, paired crossing times are approximately 19.19, 12.01, and
4.83 seconds for `orth_1`, and 5.22, 12.92, and 19.89 seconds for `orth_2`
(in the pair order above). Changing individual speeds or avoidance responses
changes these crossing times.

`mix_lanes` assigns `tb1` eastbound at `y=-0.4`, `tb9` westbound at
`y=0.35`, `tb6` eastbound at `y=1.1`, and `tb3` westbound at `y=1.85`.
`tb2` travels northeast at 63.4° and `tb5` southeast at -63.4°;
their diagonals meet at `(0, 0.8)` and each crosses all four horizontal
lanes. All six follow straight paths without turn waypoints. Starts and goals
use the inset perimeter; the diagonal slopes keep stopped robots clear of
the horizontal routes.

```bash
python3 experiments/scripts/plot_scenario.py mix_lanes
roslaunch MRN_rogue_hardware hardware_scenario.launch scenario:=mix_lanes
```

The mixed layout is a single Manhattan-style four-way intersection centered
at `(-0.15, 1.1)`, with right-hand traffic and extra incoming lanes from the west
and south. The standard east/west lane centers are `y=0.5` and `y=1.7`;
north/south lane centers are `x=0.45` and `x=-0.75`. Opposing lanes are 1.2 m
apart. The added eastbound lane is at `y=-0.45` (0.95 m from the other
eastbound lane), and the added northbound lane is at `x=1.65` (1.2 m apart).
The west lane spacing fits the tracked field while leaving stopped robots
clear of the other routes.

All six robots have independent approach and exit lanes. Each route crosses
another stream; the right-turning robot crosses the added eastbound stream
before turning, so it participates in the experiment. Each robot turns at
most once. Left turns use a 1.5 m radius and right turns a 0.3 m radius;
tangent entry/exit segments and a 0.2 m lookahead smooth steering. The
lookahead approach follows `MRN_software/GenericIntersection`. Crossings
remain for NOD to resolve through speed modulation.

| Robot | Entry lane | Movement | Exit lane |
| --- | --- | --- | --- |
| tb1 | West, y=0.5 | Left | North, x=0.45 |
| tb9 | East, y=1.7 | Straight | West, y=1.7 |
| tb6 | North, x=-0.75 | Straight | South, x=-0.75 |
| tb3 | West, y=-0.45 | Straight | East, y=-0.45 |
| tb2 | South, x=1.65 | Straight | North, x=1.65 |
| tb5 | South, x=0.45 | Right | East, y=0.5 |

Starts and goals sit near the inset field perimeter. No robots queue behind
one another or merge into the same outgoing lane.

Preview the full routes:

```bash
python3 experiments/scripts/plot_scenario.py turning_intersection
```

Launch with `scenario:=turning_intersection` or your existing `scenario:=turn_left`.

Launch once to position the robots and run the experiment automatically:

```bash
roslaunch MRN_rogue_hardware hardware_scenario.launch scenario:=orthogonal
```

The robots first move from their actual Vicon poses to the scenario starts
and align their yaw. Each waits until all six are within 0.15 m of their
starts and 0.02 rad of their initial headings, then the experiment begins.
No reset flag or relaunch is needed; `RESET_TO_START` in `constants.py` does
not control this scenario workflow. Robots without a Vicon pose remain
stopped during positioning. Positioning can drive back into the field from
the previous trial's exit points; the field-boundary stop applies during
the experiment, after positioning is complete. Restarting this launch repeats
the positioning and experiment sequence.

Replace `orthogonal` in the command to select another layout. This wrapper
launches all six robots on hardware and clears the previous readiness/start
barrier. The controller uses explicit goals, advances along the route's
intermediate points, and stops at an inset final goal. It pivots when the target
heading differs by more than 45° before driving the next leg. Collision
avoidance can change the actual trajectory and arrival timing.

Final goals are placed 0.5 m inside the first field edge along the last leg,
so robots finish near the perimeter while staying in the tracked area.
The controller stops within 0.15 m of the final goal and stays stopped even
if the measured pose subsequently jitters. This final-goal stop applies only
after intermediate waypoints have been traversed; the rectangle boundary
check remains a fallback. Relaunching resets the completed state and repeats
the automatic positioning and experiment sequence.

Starts, goals and turn samples are checked for at least 0.5 m clearance from
the existing controller rectangle (`x ∈ [-2.85, 2.85]`, `y ∈ [-1.8, 3.2]`)
and the unscaled room polygon recorded in `sim_room_boundaries.py`. Nominal
segments stay within this inset region; validation does not guarantee clearance
for every avoidance manoeuvre or reliable Vicon tracking. Start and final-goal positions have
at least 0.8 m pairwise separation. Gazebo's current room has a separate 2×
scale; the wrapper above is for hardware.

Nearly antipodal and turning starts and goals sit 0.5 m
inside the first field edge along their placement rays.
Placement respects both the controller rectangle
and the recorded room polygon, so the slanted room edge may determine the
start before the rectangular limit does. Nearly antipodal start rays are at
`0°, 60°, 120°, 200°, 260°, 320°` around `(0, 0.7)`. Each robot faces that
common crossing and continues straight through it toward its inset goal.
The 20° offset is applied to one half of the starts, so all routes intersect
at the same point without any exactly head-on pair. Turning curves remain in
the interior. This geometry allows robots to yield by adjusting speed;
controller convergence and hardware clearance still require trial validation.

Set `TRIAL_ID` / `TRIAL_SEED` in `src/constants.py` before each recorded trial
as usual. Agent types and speeds still come from that file. Launching
`launch_selected.launch` without a scenario uses its existing constants-based
start positions and boundary-projected goals.

`NodConfig.kin.ENABLE_DRIFT_CORRECTION = False` disables heading correction
and heading-alignment pivots during straight NOD/rogue experiment runs
(orthogonal and antipodal layouts, plus the legacy straight run). These runs
command zero angular velocity. Automatic start positioning and turning-layout
path steering remain active; ORCA/MPC steering is unchanged. Set the flag to
`True` before restarting the controllers to restore straight-run correction.

To inspect or adapt a layout from Python:

```python
from scenarios import get_scenario  # with src on PYTHONPATH

layout = get_scenario("turn_left")
starts = layout.start_positions  # robot -> (x, y, yaw)
goals = layout.goal_positions    # robot -> (x, y)
corners = layout.waypoints       # robot -> tuple of intermediate (x, y)
```

Edit `get_scenario()` in `src/scenarios.py` to adjust geometry; validation
runs whenever a scenario is loaded. An optional six-ID `robots` argument
remaps the Python layout's slots; hardware launch participants must match it.
