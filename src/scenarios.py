"""Hardware experiment layouts, in Vicon metres and radians.

Preview: python3 src/scenarios.py orthogonal
Use from Python: scenario = get_scenario("nearly_antipodal")
Hardware launch: roslaunch MRN_rogue_hardware hardware_scenario.launch scenario:=orthogonal
"""
import argparse
import json
import math
from dataclasses import dataclass, field
from itertools import combinations


# Existing controller stop limits: xmin, xmax, ymin, ymax.
FIELD_BOUNDS = (-2.85, 2.85, -1.8, 3.2)
# Unscaled room corners recorded in sim_room_boundaries.py (clockwise).
# Gazebo currently scales these by 2; hardware layouts use the unscaled field.
FIELD_CORNERS = ((-3.27, -1.23), (-3.1, 3.32), (2.69, 2.73), (2.83, -2.02))
BOUNDARY_MARGIN = 0.5
GOAL_TOLERANCE = 0.15
MIN_SEPARATION = 0.8
DEFAULT_ROBOTS = ("tb1", "tb9", "tb6", "tb3", "tb2", "tb5")
TURNING_SCENARIOS = ("turning_intersection", "turn_left", "turn_right")
ORTHOGONAL_SCENARIOS = ("orthogonal", "orth_1", "orth_2")
ANTIPODAL_SCENARIOS = ("nearly_antipodal", "antipodal_1", "antipodal_2")
SCENARIO_NAMES = ORTHOGONAL_SCENARIOS + ANTIPODAL_SCENARIOS + TURNING_SCENARIOS + ("mix_lanes",)
TURNING_LANE_GAP = 1.2
INTERSECTION_CENTERS = ((-0.15, 1.1),)
LANE_OFFSET = TURNING_LANE_GAP / 2
TURN_REACH = 0.9
EXTRA_WEST_LANE_Y = -0.45
EXTRA_SOUTH_LANE_X = 1.65
ORTHOGONAL_LANE_GAP = 1.2
PATH_LOOKAHEAD = 0.2
PATH_REACH_TOLERANCE = 0.08
ANTIPODAL_CROSSING = (0.0, 0.7)
ANTIPODAL_OFFSET_DEG = 20.0
# Shift one half of the starts, rather than rotating each robot's goal.
# Opposing headings are 160 degrees apart; all straight routes share a point.
ANTIPODAL_START_ANGLES_DEG = (0.0, 60.0, 120.0,
                              180.0 + ANTIPODAL_OFFSET_DEG,
                              240.0 + ANTIPODAL_OFFSET_DEG,
                              300.0 + ANTIPODAL_OFFSET_DEG)
ANTIPODAL_LAYOUTS = {
    "nearly_antipodal": (ANTIPODAL_CROSSING, ANTIPODAL_START_ANGLES_DEG),
    "antipodal_1": (ANTIPODAL_CROSSING, ANTIPODAL_START_ANGLES_DEG),
    # Move the common crossing and rotate/skew the six approaches.
    "antipodal_2": ((0.15, 0.9), (15.0, 75.0, 135.0, 220.0, 280.0, 340.0)),
}
# Four alternating east/west streams and two oblique crossing streams.
# The steeper diagonals leave their perimeter goals clear of horizontal lanes.
MIX_LANES_ROUTES = (((0.0, -0.4), (1, 0)),
                    ((0.0, 0.35), (-1, 0)),
                    ((0.0, 1.1), (1, 0)),
                    ((0.0, 1.85), (-1, 0)),
                    ((0.0, 0.8), (1, 2)),
                    ((0.0, 0.8), (1, -2)))


@dataclass
class Scenario:
    name: str
    start_positions: dict  # robot -> (x, y, initial yaw)
    goal_positions: dict   # robot -> (x, y)
    waypoints: dict        # robot -> tuple of intermediate (x, y) positions
    maneuvers: dict = field(default_factory=dict)  # robot -> movements at successive junctions
    path_lookahead: float = 0.0

    @property
    def active_robots(self):
        return set(self.start_positions)


def inside_field(position, margin=BOUNDARY_MARGIN):
    """Check clearance from both controller limits and the recorded room edges."""
    x, y = position[:2]
    xmin, xmax, ymin, ymax = FIELD_BOUNDS
    if not (xmin + margin <= x <= xmax - margin
            and ymin + margin <= y <= ymax - margin):
        return False
    for i, (ax, ay) in enumerate(FIELD_CORNERS):
        bx, by = FIELD_CORNERS[(i + 1) % len(FIELD_CORNERS)]
        # Clockwise polygon: interior lies to the right of each edge.
        clearance = -((bx - ax) * (y - ay) - (by - ay) * (x - ax)) / math.hypot(bx - ax, by - ay)
        if clearance < margin:
            return False
    return True


def inset_goal(origin, target):
    """Continue along the final leg to an endpoint inside the tracked field."""
    return near_boundary(origin[:2], (target[0] - origin[0], target[1] - origin[1]))


def near_boundary(origin, direction):
    """Place an endpoint 0.5 m inside the first field edge along a ray.

    Respect both the controller rectangle and the recorded room polygon.
    """
    x, y = origin
    dx, dy = direction
    if not inside_field(origin) or math.hypot(dx, dy) < 1e-12:
        raise ValueError("Boundary placement needs an interior origin and a direction")
    # A tiny extra inset keeps floating-point roundoff inside the margin.
    margin = BOUNDARY_MARGIN + 1e-9
    xmin, xmax, ymin, ymax = FIELD_BOUNDS
    candidates = []
    if abs(dx) > 1e-12:
        edge = xmax - margin if dx > 0 else xmin + margin
        candidates.append((edge - x) / dx)
    if abs(dy) > 1e-12:
        edge = ymax - margin if dy > 0 else ymin + margin
        candidates.append((edge - y) / dy)
    for i, (ax, ay) in enumerate(FIELD_CORNERS):
        bx, by = FIELD_CORNERS[(i + 1) % len(FIELD_CORNERS)]
        length = math.hypot(bx - ax, by - ay)
        clearance = -((bx - ax) * (y - ay) - (by - ay) * (x - ax)) / length
        change = -((bx - ax) * dy - (by - ay) * dx) / length
        if change < -1e-12:
            candidates.append((clearance - margin) / -change)
    scale = min(t for t in candidates if t >= 0)
    return (x + scale * dx, y + scale * dy)


def validate_scenario(scenario):
    if not scenario.start_positions or scenario.active_robots != set(scenario.goal_positions):
        raise ValueError("Every participating robot must have a start and a goal")
    for name, start in scenario.start_positions.items():
        goal = scenario.goal_positions[name]
        if len(start) != 3 or len(goal) != 2 or not all(math.isfinite(v) for v in (*start, *goal)):
            raise ValueError("{} needs a finite (x, y, yaw) start and (x, y) goal".format(name))
        if not inside_field(start) or not inside_field(goal):
            raise ValueError("{} start/goal violates the {} m field margin".format(name, BOUNDARY_MARGIN))
        if math.hypot(goal[0] - start[0], goal[1] - start[1]) < 0.2:
            raise ValueError("{} start and goal are too close".format(name))
        for point in scenario.waypoints.get(name, ()):
            if len(point) != 2 or not all(math.isfinite(v) for v in point) or not inside_field(point):
                raise ValueError("{} waypoint violates the field margin".format(name))
    for positions in (scenario.start_positions, scenario.goal_positions):
        for a, b in combinations(positions, 2):
            if math.hypot(positions[a][0] - positions[b][0], positions[a][1] - positions[b][1]) < MIN_SEPARATION:
                raise ValueError("{} and {} need at least {} m separation at endpoints".format(a, b, MIN_SEPARATION))
    # Starts, intermediate vertices and goals remain inside both convex
    # field regions, so every nominal segment does as well.
    return scenario


def intersection_leg(center, direction, maneuver):
    """One movement through a two-way, two-lane four-way intersection."""
    cx, cy = center
    dx, dy = direction
    start = (cx + LANE_OFFSET * dy - TURN_REACH * dx,
             cy - LANE_OFFSET * dx - TURN_REACH * dy)
    if maneuver == 'straight':
        return (start, (cx + LANE_OFFSET * dy + TURN_REACH * dx,
                        cy - LANE_OFFSET * dx + TURN_REACH * dy)), direction
    sign = 1 if maneuver == "left" else -1
    out = (-sign * dy, sign * dx)
    radius = TURN_REACH + sign * LANE_OFFSET
    arc_center = (start[0] - sign * radius * dy, start[1] + sign * radius * dx)
    angle = math.atan2(start[1] - arc_center[1], start[0] - arc_center[0])
    steps = math.ceil((math.pi / 2) * radius / 0.12)
    points = tuple((arc_center[0] + radius * math.cos(angle + sign * math.pi * i / (2 * steps)),
                    arc_center[1] + radius * math.sin(angle + sign * math.pi * i / (2 * steps)))
                   for i in range(steps + 1))
    return points, out


def turning_intersection(name, robots):
    """One shared intersection with six independent approach/exit lanes.

    West and south each have one additional incoming lane. No robot queues
    behind another, and no pair shares an outgoing lane. The extra west
    stream crosses the right-turn approach, preventing an isolated turn.
    """
    # incoming direction, movement, optional additional lane coordinate
    assignments = (((1, 0), "left", None),
                   ((-1, 0), "straight", None),
                   ((0, -1), "straight", None),
                   ((1, 0), "straight", EXTRA_WEST_LANE_Y),
                   ((0, 1), "straight", EXTRA_SOUTH_LANE_X),
                   ((0, 1), "right", None))
    starts, goals, waypoints, maneuvers = {}, {}, {}, {}
    cx, cy = INTERSECTION_CENTERS[0]
    for robot, (direction, movement, extra_lane) in zip(robots, assignments):
        dx, dy = direction
        lane_origin = (cx + LANE_OFFSET * dy, cy - LANE_OFFSET * dx)
        if extra_lane is not None:
            lane_origin = (cx, extra_lane) if dx else (extra_lane, cy)
        start = near_boundary(lane_origin, (-dx, -dy))
        starts[robot] = (*start, math.atan2(dy, dx))
        if extra_lane is None:
            route, direction = intersection_leg(INTERSECTION_CENTERS[0], direction, movement)
        else:
            route = ()
        last = route[-1] if route else start
        goal = inset_goal(last, (last[0] + direction[0], last[1] + direction[1]))
        goals[robot] = goal
        waypoints[robot] = tuple(route)
        maneuvers[robot] = (movement,)
    return validate_scenario(Scenario(name, starts, goals, waypoints, maneuvers, PATH_LOOKAHEAD))


def path_tracking_target(position, path, index, lookahead=PATH_LOOKAHEAD):
    """Advance reached/passed vertices and look ahead along the polyline.

    Follows MRN_software/GenericIntersection/nod/agent.py's projection and
    lookahead approach. `index` counts targets after the initial start.
    """
    x, y = position
    while index < len(path) - 2:
        ax, ay = path[index]
        bx, by = path[index + 1]
        dx, dy = bx - ax, by - ay
        length = math.hypot(dx, dy)
        near = math.hypot(bx - x, by - y) < PATH_REACH_TOLERANCE
        passed = length > 1e-12 and ((x - ax) * dx + (y - ay) * dy) / length >= length - PATH_REACH_TOLERANCE
        if not (near or passed):
            break
        index += 1
    ax, ay = path[index]
    bx, by = path[index + 1]
    dx, dy = bx - ax, by - ay
    length_squared = dx * dx + dy * dy
    fraction = max(0, min(1, ((x - ax) * dx + (y - ay) * dy) / length_squared)) if length_squared > 1e-12 else 1
    current = (ax + fraction * dx, ay + fraction * dy)
    remaining = lookahead
    for target in path[index + 1:]:
        dx, dy = target[0] - current[0], target[1] - current[1]
        length = math.hypot(dx, dy)
        if length > 1e-12 and remaining <= length:
            return index, (current[0] + remaining * dx / length, current[1] + remaining * dy / length)
        remaining -= length
        current = target
    return index, path[-1]


def get_scenario(name, robots=DEFAULT_ROBOTS):
    """Return a fresh six-robot layout; optional IDs map to slots in this order.

    orthogonal: three eastbound routes cross three southbound routes at 90°.
    nearly_antipodal: six straight routes through (0, 0.7); opposing routes
    differ from head-on by 20 degrees, with starts near the boundary.
    orth_1/2: orthogonal streams with staggered starts giving every robot
    a simultaneous paired crossing at equal speed; variant 2 reverses travel
    directions and shifts lane positions.
    antipodal_1/2: common-point crossings with different centers/approach angles.
    mix_lanes: four alternating east/west lanes crossed by northeast- and
    southeast-bound diagonals; nine crossings spread through the interior.
    turning_intersection (also turn_left/right): one shared four-way
    intersection with two-way roads and mixed left/right/straight movements.
    All endpoints have at least 0.5 m field clearance; six independent
    turning approaches avoid same-lane following and shared exit slots.
    """
    if name not in SCENARIO_NAMES:
        raise ValueError("Unknown scenario {!r}; choose {}".format(name, ", ".join(SCENARIO_NAMES)))
    robots = tuple(robots)
    if len(robots) != 6 or len(set(robots)) != 6:
        raise ValueError("These layouts require six distinct robot IDs")
    if name in TURNING_SCENARIOS:
        return turning_intersection(name, robots)
    starts, goals, waypoints = {}, {}, {}
    for i, robot in enumerate(robots):
        if name == "mix_lanes":
            origin, direction = MIX_LANES_ROUTES[i]
            x, y = near_boundary(origin, (-direction[0], -direction[1]))
            gx, gy = x + direction[0], y + direction[1]
        elif name in ANTIPODAL_SCENARIOS:
            crossing, angles = ANTIPODAL_LAYOUTS[name]
            angle = math.radians(angles[i])
            x, y = near_boundary(crossing, (math.cos(angle), math.sin(angle)))
            gx, gy = crossing
        elif name in ("orth_1", "orth_2"):
            reverse = name == "orth_2"
            if i < 3:
                origin = (0.0, (0.75 if reverse else 0.7) + (i - 1) * ORTHOGONAL_LANE_GAP)
                direction = (-1, 0) if reverse else (1, 0)
            else:
                origin = ((0.3 if reverse else 0.0) + (i - 4) * ORTHOGONAL_LANE_GAP, 0.7)
                direction = (0, 1) if reverse else (0, -1)
            x, y = near_boundary(origin, (-direction[0], -direction[1]))
            gx, gy = x + direction[0], y + direction[1]
        elif i < 3:
            x, y = -2.0, 0.7 + (i - 1) * ORTHOGONAL_LANE_GAP
            gx, gy = 2.0, y
        else:
            x, y = (i - 4) * ORTHOGONAL_LANE_GAP, 2.2
            gx, gy = x, -0.8
        points = ()
        first_target = points[0] if points else (gx, gy)
        yaw = math.atan2(first_target[1] - y, first_target[0] - x)
        yaw = math.atan2(math.sin(yaw), math.cos(yaw))
        starts[robot] = (x, y, yaw)
        goals[robot] = inset_goal(points[-1] if points else (x, y), (gx, gy))
        waypoints[robot] = points
    if name in ("orth_1", "orth_2"):
        # Pair opposite lane orders: tb1/tb5, tb9/tb2, tb6/tb3.
        # Equal remaining distances mean equal crossing times at any shared
        # constant speed. Move only inward from the inset boundary, keeping
        # all nine geometric crossings and the original perimeter goals.
        for horizontal, vertical in zip(robots[:3], reversed(robots[3:])):
            hx, hy, hyaw = starts[horizontal]
            vx, vy, vyaw = starts[vertical]
            distance = min(abs(vx - hx), abs(hy - vy))
            starts[horizontal] = (vx - math.cos(hyaw) * distance, hy, hyaw)
            starts[vertical] = (vx, hy - math.sin(vyaw) * distance, vyaw)
    return validate_scenario(Scenario(name, starts, goals, waypoints))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", choices=SCENARIO_NAMES)
    args = parser.parse_args()
    layout = get_scenario(args.scenario)
    print(json.dumps({"scenario": layout.name, "start_positions": layout.start_positions,
                      "goal_positions": layout.goal_positions, "waypoints": layout.waypoints,
                      "maneuvers": layout.maneuvers}, indent=2))
