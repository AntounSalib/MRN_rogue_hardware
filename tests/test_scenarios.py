"""Geometry and controller route regressions, runnable without ROS."""
import ast
import math
import sys
import time
import unittest
import numpy as np
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from constants import D_SAFE, NodConfig
from neighbors import solve_ray_intersection, sensed_neighbors
from nod_controller import NodController
from scenarios import (ANTIPODAL_CROSSING, ANTIPODAL_OFFSET_DEG,
                       BOUNDARY_MARGIN, DEFAULT_ROBOTS, FIELD_BOUNDS, FIELD_CORNERS, GOAL_TOLERANCE,
                       INTERSECTION_CENTERS, TURNING_LANE_GAP, ORTHOGONAL_LANE_GAP,
                       ANTIPODAL_LAYOUTS, ANTIPODAL_SCENARIOS,
                       SCENARIO_NAMES, get_scenario, inside_field, near_boundary,
                       inset_goal, path_tracking_target, validate_scenario)


def controller_methods():
    # Load the actual route/run methods without importing ROS or initializing
    # hardware. Tests below only exercise the early waypoint/stop branches.
    path = Path(__file__).resolve().parents[1] / "src" / "turtlebot.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Turtlebot")
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef)
                and n.name in ("_update_scenario_goal", "_prepare_scenario", "_run_stationary",
                               "_experiment_steering_enabled", "_heading_command", "run")]
    namespace = dict(math=math, time=time, FIELD_BOUNDS=FIELD_BOUNDS, NodConfig=NodConfig,
                     GOAL_TOLERANCE=GOAL_TOLERANCE,
                     path_tracking_target=path_tracking_target, sensed_neighbors=sensed_neighbors,
                     np=np, ROGUE_AGENTS={}, ROGUE_SPEEDS={}, ORCA_AGENTS={}, ORCA_DD_AGENTS={}, MPC_CBF_AGENTS={})
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["Turtlebot"]


Controller = controller_methods()


class ScenarioTests(unittest.TestCase):
    def test_entire_nominal_routes_stay_inside_tracking_margin(self):
        for name in SCENARIO_NAMES:
            layout = get_scenario(name)
            for robot, start in layout.start_positions.items():
                route = (start[:2],) + layout.waypoints[robot] + (layout.goal_positions[robot],)
                self.assertTrue(inside_field(start))
                for a, b in zip(route, route[1:]):
                    for i in range(101):
                        t = i / 100
                        self.assertTrue(inside_field((a[0] * (1 - t) + b[0] * t,
                                                      a[1] * (1 - t) + b[1] * t)), (name, robot, t))
                self.assertTrue(inside_field(layout.goal_positions[robot]))

    def test_final_goals_are_near_boundary_with_tracking_clearance(self):
        for target in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            goal = inset_goal((0, 0), target)
            self.assertTrue(inside_field(goal))
            self.assertFalse(inside_field(goal, margin=BOUNDARY_MARGIN + 0.01))

    def test_parallel_lane_centers_have_larger_spacing(self):
        self.assertAlmostEqual(TURNING_LANE_GAP, 1.2)
        turning = get_scenario('turning_intersection')
        starts = turning.start_positions
        self.assertAlmostEqual(starts['tb9'][1] - starts['tb1'][1], 1.2)
        self.assertAlmostEqual(starts['tb2'][0] - starts['tb5'][0], 1.2)
        self.assertAlmostEqual(starts['tb1'][1] - starts['tb3'][1], 0.95)
        layout = get_scenario('orthogonal')
        for names, coordinate in ((DEFAULT_ROBOTS[:3], 1), (DEFAULT_ROBOTS[3:], 0)):
            lanes = sorted(layout.start_positions[name][coordinate] for name in names)
            self.assertGreaterEqual(min(b - a for a, b in zip(lanes, lanes[1:])), ORTHOGONAL_LANE_GAP - 1e-9)

    def test_orthogonal_variants_cross_nine_times_and_keep_lane_spacing(self):
        for name, headings in (('orth_1', (0, -math.pi / 2)),
                               ('orth_2', (math.pi, math.pi / 2))):
            layout = get_scenario(name)
            for names, coordinate, heading in ((DEFAULT_ROBOTS[:3], 1, headings[0]),
                                                (DEFAULT_ROBOTS[3:], 0, headings[1])):
                lanes = sorted(layout.start_positions[r][coordinate] for r in names)
                for a, b in zip(lanes, lanes[1:]):
                    self.assertAlmostEqual(b - a, ORTHOGONAL_LANE_GAP)
                for r in names:
                    self.assertAlmostEqual(layout.start_positions[r][2], heading)
            for horizontal in DEFAULT_ROBOTS[:3]:
                for vertical in DEFAULT_ROBOTS[3:]:
                    agents = [dict(position=layout.start_positions[r][:2],
                                   heading=layout.start_positions[r][2], velocity=(0, 0))
                              for r in (horizontal, vertical)]
                    s, t, _, _ = solve_ray_intersection(*agents)
                    for r, distance in ((horizontal, s), (vertical, t)):
                        self.assertGreater(distance, 0)
                        self.assertLess(distance, math.dist(layout.start_positions[r][:2], layout.goal_positions[r]))
        for first, second in (('orth_1', 'orth_2'), ('antipodal_1', 'antipodal_2')):
            self.assertNotEqual(get_scenario(first).start_positions, get_scenario(second).start_positions)
            self.assertNotEqual(get_scenario(first).goal_positions, get_scenario(second).goal_positions)

    def test_orthogonal_variants_conflict_at_nominal_speed_for_every_robot(self):
        speed = NodConfig.kin.V_NOMINAL
        for name in ('orth_1', 'orth_2'):
            layout = get_scenario(name)
            conflicted = set()
            for horizontal in DEFAULT_ROBOTS[:3]:
                for vertical in DEFAULT_ROBOTS[3:]:
                    agents = [dict(position=layout.start_positions[r][:2],
                                   heading=layout.start_positions[r][2], velocity=(0, 0))
                              for r in (horizontal, vertical)]
                    s, t, ei, ej = solve_ray_intersection(*agents)
                    # Closest approach while both robots still traverse their
                    # routes, including final-goal stopping tolerance.
                    delta = np.array(agents[1]['position']) - agents[0]['position']
                    relative_velocity = speed * (ej - ei)
                    time_limit = min((math.dist(layout.start_positions[r][:2], layout.goal_positions[r])
                                      - GOAL_TOLERANCE) / speed for r in (horizontal, vertical))
                    closest_time = np.clip(-np.dot(delta, relative_velocity)
                                           / np.dot(relative_velocity, relative_velocity), 0, time_limit)
                    separation = np.linalg.norm(delta + closest_time * relative_velocity)
                    if separation < D_SAFE:
                        self.assertGreater(closest_time, 0)
                        self.assertLess(closest_time, time_limit)
                        conflicted.update((horizontal, vertical))
            self.assertEqual(conflicted, layout.active_robots, name)
            for horizontal, vertical in zip(DEFAULT_ROBOTS[:3], reversed(DEFAULT_ROBOTS[3:])):
                hx, hy, _ = layout.start_positions[horizontal]
                vx, vy, _ = layout.start_positions[vertical]
                self.assertAlmostEqual(abs(vx - hx) / speed, abs(hy - vy) / speed)

    def test_mixed_turns_have_lane_aligned_entry_and_exit_headings(self):
        for name in ("turning_intersection", "turn_left", "turn_right"):
            layout = get_scenario(name)
            self.assertEqual(set(m for turns in layout.maneuvers.values() for m in turns),
                             {"left", "right", "straight"})
            self.assertEqual(set(round(start[2], 5) for start in layout.start_positions.values()),
                             {0.0, round(math.pi, 5), round(-math.pi / 2, 5), round(math.pi / 2, 5)})
            for robot, (x, y, yaw) in layout.start_positions.items():
                path = ((x, y),) + layout.waypoints[robot] + (layout.goal_positions[robot],)
                angles = [math.atan2(b[1] - a[1], b[0] - a[0]) for a, b in zip(path, path[1:])]
                self.assertAlmostEqual(yaw, angles[0])
                changes = [math.atan2(math.sin(b - a), math.cos(b - a))
                           for a, b in zip(angles, angles[1:])]
                expected = sum({"left": 1, "right": -1, "straight": 0}[m]
                               for m in layout.maneuvers[robot]) * math.pi / 2
                self.assertAlmostEqual(sum(changes), expected)
                self.assertLess(max((abs(v) for v in changes), default=0), math.radians(25))
                self.assertEqual(len(layout.maneuvers[robot]), 1)

    def test_mix_lanes_has_four_horizontal_routes_and_nine_interior_crossings(self):
        layout = get_scenario('mix_lanes')
        for robot in DEFAULT_ROBOTS[:4]:
            self.assertAlmostEqual(layout.start_positions[robot][1], layout.goal_positions[robot][1])
        for robot in DEFAULT_ROBOTS:
            self.assertEqual(layout.waypoints[robot], ())
        pairs = [(horizontal, diagonal) for horizontal in DEFAULT_ROBOTS[:4]
                 for diagonal in DEFAULT_ROBOTS[4:]] + [DEFAULT_ROBOTS[4:]]
        for a, b in pairs:
            agents = [dict(position=layout.start_positions[r][:2],
                           heading=layout.start_positions[r][2], velocity=(0, 0))
                      for r in (a, b)]
            s, t, _, _ = solve_ray_intersection(*agents)
            for robot, distance in ((a, s), (b, t)):
                self.assertGreater(distance, 0)
                self.assertLess(distance, math.dist(layout.start_positions[robot][:2],
                                                   layout.goal_positions[robot]))
        # A finished robot must leave space for every other nominal route.
        for robot in DEFAULT_ROBOTS:
            ax, ay = layout.start_positions[robot][:2]
            bx, by = layout.goal_positions[robot]
            dx, dy = bx - ax, by - ay
            for other in layout.active_robots - {robot}:
                px, py = layout.goal_positions[other]
                fraction = max(0, min(1, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
                distance = math.hypot(px - ax - fraction * dx, py - ay - fraction * dy)
                self.assertGreater(distance, D_SAFE + GOAL_TOLERANCE, (robot, other))

    def test_turning_routes_share_one_intersection_and_keep_opposing_lanes_apart(self):
        layout = get_scenario('turning_intersection')
        routes = {r: (layout.start_positions[r][:2],) + layout.waypoints[r] + (layout.goal_positions[r],)
                  for r in layout.active_robots}
        movements = [m[0] for m in layout.maneuvers.values()]
        self.assertEqual([movements.count(m) for m in ('left', 'right', 'straight')], [1, 1, 4])
        self.assertEqual(len(INTERSECTION_CENTERS), 1)
        segments = {r: list(zip(path, path[1:])) for r, path in routes.items()}
        def point_distance(point, a, b):
            dx, dy = b[0] - a[0], b[1] - a[1]
            t = max(0, min(1, ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / (dx * dx + dy * dy)))
            return math.hypot(point[0] - a[0] - t * dx, point[1] - a[1] - t * dy)
        for robot, path in routes.items():
            for a, b in zip(path, path[1:]):
                dx, dy = b[0] - a[0], b[1] - a[1]
                for other in layout.active_robots - {robot}:
                    for c, d in segments[other]:
                        ex, ey = d[0] - c[0], d[1] - c[1]
                        cosine = (dx * ex + dy * ey) / (math.hypot(dx, dy) * math.hypot(ex, ey))
                        if cosine < math.cos(math.radians(160)):
                            determinant = dx * ey - dy * ex
                            crossing = False
                            if abs(determinant) > 1e-12:
                                tx, ty = c[0] - a[0], c[1] - a[1]
                                u = (tx * ey - ty * ex) / determinant
                                v = (tx * dy - ty * dx) / determinant
                                crossing = 0 <= u <= 1 and 0 <= v <= 1
                            gap = 0 if crossing else min(point_distance(a, c, d), point_distance(b, c, d),
                                                         point_distance(c, a, b), point_distance(d, a, b))
                            self.assertGreater(gap, D_SAFE + GOAL_TOLERANCE, (robot, other))
                    px, py = layout.goal_positions[other]
                    fraction = max(0, min(1, ((px - a[0]) * dx + (py - a[1]) * dy) / (dx * dx + dy * dy)))
                    distance = math.hypot(px - a[0] - fraction * dx, py - a[1] - fraction * dy)
                    self.assertGreater(distance, D_SAFE + GOAL_TOLERANCE, (robot, other))

    def test_six_independent_lanes_and_no_isolated_movements(self):
        layout = get_scenario('turning_intersection')
        paths = {r: (layout.start_positions[r][:2],) + layout.waypoints[r] + (layout.goal_positions[r],)
                 for r in layout.active_robots}
        approach_lanes, exit_lanes = set(), set()
        for robot, path in paths.items():
            for lanes, a, b in ((approach_lanes, path[0], path[1]),
                                (exit_lanes, path[-2], path[-1])):
                horizontal = abs(b[1] - a[1]) < 1e-9
                lane = ('horizontal', round(a[1], 6)) if horizontal else ('vertical', round(a[0], 6))
                self.assertNotIn(lane, lanes, (robot, lane))
                lanes.add(lane)
            crossings = []
            for other, other_path in paths.items():
                if other == robot:
                    continue
                for a, b in zip(path, path[1:]):
                    dx, dy = b[0] - a[0], b[1] - a[1]
                    for c, d in zip(other_path, other_path[1:]):
                        ex, ey = d[0] - c[0], d[1] - c[1]
                        det = dx * ey - dy * ex
                        if abs(det) < math.sin(math.radians(20)) * math.hypot(dx, dy) * math.hypot(ex, ey):
                            continue
                        tx, ty = c[0] - a[0], c[1] - a[1]
                        u = (tx * ey - ty * ex) / det
                        v = (tx * dy - ty * dx) / det
                        if 0 <= u <= 1 and 0 <= v <= 1:
                            crossings.append(other)
            self.assertTrue(crossings, (robot, 'isolated movement'))

    def test_lookahead_handles_passed_vertices_and_begins_turn_gradually(self):
        layout = get_scenario("turning_intersection")
        path = (layout.start_positions['tb1'][:2],) + layout.waypoints['tb1'] + (layout.goal_positions['tb1'],)
        corner = path[1]
        position = (corner[0] - 0.05, corner[1])
        index, target = path_tracking_target(position, path, 0)
        self.assertGreaterEqual(index, 1)
        angle = math.atan2(target[1] - position[1], target[0] - position[0])
        self.assertGreater(angle, 0)
        self.assertLess(angle, math.pi / 4)

    def test_antipodal_routes_all_cross_same_point_without_head_on_pairs(self):
        for name in ANTIPODAL_SCENARIOS:
            layout = get_scenario(name)
            agents = {}
            (cx, cy), _ = ANTIPODAL_LAYOUTS[name]
            for robot, (x, y, yaw) in layout.start_positions.items():
                gx, gy = layout.goal_positions[robot]
                self.assertAlmostEqual((cx - x) * (gy - y) - (cy - y) * (gx - x), 0)
                self.assertGreater(math.hypot(gx - x, gy - y), math.hypot(cx - x, cy - y))
                self.assertAlmostEqual(yaw, math.atan2(cy - y, cx - x))
                self.assertEqual(layout.waypoints[robot], ())
                agents[robot] = dict(position=(x, y), heading=yaw, velocity=(0, 0))
            # Use the actual neighbor solver to verify every pair sees the same
            # forward crossing, rather than parallel/opposing collinear rays.
            for i, a in enumerate(DEFAULT_ROBOTS):
                for b in DEFAULT_ROBOTS[i + 1:]:
                    result = solve_ray_intersection(agents[a], agents[b])
                    self.assertIsNotNone(result, (a, b))
                    s, t, ei, _ = result
                    self.assertGreater(s, 0)
                    self.assertGreater(t, 0)
                    self.assertAlmostEqual(agents[a]['position'][0] + s * ei[0], cx)
                    self.assertAlmostEqual(agents[a]['position'][1] + s * ei[1], cy)
            for i in range(3):
                a, b = DEFAULT_ROBOTS[i], DEFAULT_ROBOTS[i + 3]
                delta = agents[a]['heading'] - agents[b]['heading']
                angle = abs(math.atan2(math.sin(delta), math.cos(delta)))
                self.assertAlmostEqual(angle, math.radians(155 if name == "antipodal_2" else 180 - ANTIPODAL_OFFSET_DEG))

    def test_antipodal_crossing_has_a_feasible_speed_only_sequence(self):
        # A conservative witness: one robot traverses its straight route,
        # while the others wait at starts or at their boundary stops.
        for name in ANTIPODAL_SCENARIOS:
            layout = get_scenario(name)
            exits = layout.goal_positions
            for a, start in layout.start_positions.items():
                x, y = start[:2]
                dx, dy = exits[a][0] - x, exits[a][1] - y
                for b in layout.active_robots - {a}:
                    for parked in (layout.start_positions[b][:2], exits[b]):
                        t = ((parked[0] - x) * dx + (parked[1] - y) * dy) / (dx * dx + dy * dy)
                        t = max(0, min(1, t))
                        separation = math.hypot(parked[0] - x - t * dx, parked[1] - y - t * dy)
                        self.assertGreater(separation, D_SAFE, (a, b, parked))

    def test_antipodal_starts_and_orthogonal_goals_are_near_field_edges(self):
        xmin, xmax, ymin, ymax = FIELD_BOUNDS
        for name in ANTIPODAL_SCENARIOS + ("orth_1", "orth_2"):
            layout = get_scenario(name)
            positions = layout.start_positions if name in ANTIPODAL_SCENARIOS else layout.goal_positions
            for robot, position in positions.items():
                x, y = position[:2]
                clearances = [x - xmin, xmax - x, y - ymin, ymax - y]
                for i, (ax, ay) in enumerate(FIELD_CORNERS):
                    bx, by = FIELD_CORNERS[(i + 1) % len(FIELD_CORNERS)]
                    clearances.append(-((bx - ax) * (y - ay) - (by - ay) * (x - ax))
                                      / math.hypot(bx - ax, by - ay))
                self.assertAlmostEqual(min(clearances), BOUNDARY_MARGIN, places=7,
                                       msg=(name, robot))

    def test_invalid_and_overlapping_layouts_are_rejected(self):
        with self.assertRaises(ValueError):
            get_scenario("typo")
        with self.assertRaises(ValueError):
            get_scenario("orthogonal", ("tb1",) * 6)
        layout = get_scenario("orthogonal")
        layout.goal_positions["tb1"] = (2.84, 0)
        with self.assertRaises(ValueError):
            validate_scenario(layout)
        layout = get_scenario("orthogonal")
        layout.start_positions["tb1"] = layout.start_positions["tb9"]
        with self.assertRaises(ValueError):
            validate_scenario(layout)

    def make_controller(self, name="turn_left"):
        layout = get_scenario(name)
        tb = Controller()
        start = layout.start_positions["tb1"]
        tb.info = dict(position=list(start[:2]), heading=start[2], velocity=(0.0, 0.0))
        tb.neighbors = {}
        tb.nod_controller = NodController('tb1', time.monotonic())
        tb.nod_controller.z = 0.7
        tb.nod_controller.u = 0.2
        tb.nod_controller.time_step = 0.1
        tb.nod_controller.update_opinion = Mock(return_value=0.2)
        tb._get_v_commanded = Mock(return_value=0.2)
        tb.data_saver = SimpleNamespace(save_data=Mock())
        tb.scenario_route = layout.waypoints["tb1"] + (layout.goal_positions["tb1"],)
        tb.scenario_waypoint_index = 0
        tb.scenario_path_lookahead = layout.path_lookahead
        tb.heading_error_integral = 3.0
        tb.reset_to_start = False
        tb.scenario_preparing = False
        tb.scenario_finished = False
        tb.goal_position = None
        tb.move = Mock()
        tb.rate = SimpleNamespace(sleep=Mock())
        tb.robot_name = "tb1"
        tb.start_positions = layout.start_positions
        tb.active_robots = layout.active_robots
        tb.simulation_on = False
        tb.prev_time = 1.0
        tb._run_reset = Mock()
        return tb

    def test_automatic_positioning_waits_for_all_robots_then_releases(self):
        tb = self.make_controller()
        tb.scenario_preparing = True
        params = {}
        ros = SimpleNamespace(get_param=lambda key, default=None: params.get(key, default),
                              set_param=lambda key, value: params.__setitem__(key, value),
                              has_param=lambda key: key in params, loginfo=Mock())
        with patch.dict(Controller.run.__globals__, rospy=ros):
            # Arbitrary placement goes through positioning, not the route.
            tb.info['position'] = (0, 0)
            tb.run()
            tb._run_reset.assert_called_once()
            self.assertTrue(tb.scenario_preparing)
            self.assertIsNone(tb.goal_position)
            # Reaching our start does not release the other robots early.
            tb.info['position'] = tb.start_positions['tb1'][:2]
            tb.run()
            self.assertTrue(tb.scenario_preparing)
            self.assertNotIn('/start_time', params)
            for robot in tb.active_robots:
                params['/hardware_experiment/positioned/' + robot] = True
            tb.run()
            self.assertFalse(tb.scenario_preparing)
            self.assertTrue(params['/hardware_experiment/started'])
            self.assertIn('/start_time', params)
            # A slower robot still releases after the first robot departs.
            params['/hardware_experiment/positioned/tb1'] = False
            tb.scenario_preparing = True
            tb.run()
            self.assertFalse(tb.scenario_preparing)

    def test_positioning_waits_for_vicon_and_heading_alignment(self):
        tb = self.make_controller()
        tb.scenario_preparing = True
        params = {}
        ros = SimpleNamespace(get_param=lambda key, default=None: params.get(key, default),
                              set_param=lambda key, value: params.__setitem__(key, value))
        with patch.dict(Controller.run.__globals__, rospy=ros):
            tb.prev_time = None
            tb.run()
            tb.move.assert_called_once_with(0, 0)
            tb._run_reset.assert_not_called()
            tb.prev_time = 1.0
            tb.info['heading'] = math.pi / 2
            tb.run()
            tb._run_reset.assert_called_once()
            self.assertFalse(params['/hardware_experiment/positioned/tb1'])

    def test_restart_returns_from_previous_trial_exit(self):
        for name in SCENARIO_NAMES:
            layout = get_scenario(name)
            for robot in layout.active_robots:
                with self.subTest(scenario=name, robot=robot):
                    tb = self.make_controller(name)
                    tb.robot_name = robot
                    tb.scenario_preparing = True
                    tb.info['position'] = layout.goal_positions[robot]
                    params = {}
                    ros = SimpleNamespace(
                        get_param=lambda key, default=None: params.get(key, default),
                        set_param=lambda key, value: params.__setitem__(key, value))
                    with patch.dict(Controller.run.__globals__, rospy=ros):
                        tb.run()
                    tb._run_reset.assert_called_once()
                    self.assertTrue(tb.scenario_preparing)
                    self.assertFalse(params['/hardware_experiment/positioned/' + robot])
                    self.assertIsNone(tb.goal_position)

    def test_large_heading_error_pivots_toward_intersection_lane(self):
        tb = self.make_controller("turning_intersection")
        tb.info["position"] = tb.scenario_route[0]
        tb.info["heading"] = math.pi / 2
        tb.run()
        self.assertEqual(tb.scenario_waypoint_index, 1)
        self.assertEqual(tb.heading_error_integral, 0)
        linear, angular = tb.move.call_args.args
        self.assertEqual(linear, 0)
        self.assertLess(angular, 0)

    def test_goal_heading_tracks_position_and_inset_goal_stops(self):
        tb = self.make_controller("orthogonal")
        tb.info["position"] = (-1, 0)
        tb._update_scenario_goal()
        gx, gy = tb.scenario_route[-1]
        self.assertAlmostEqual(tb.goal_heading, math.atan2(gy, gx + 1))
        tb.info["position"] = tb.scenario_route[-1]
        tb.run()
        tb.move.assert_called_once_with(0, 0)
        self.assertTrue(tb.scenario_finished)

    def test_inset_arrival_stays_stopped_despite_pose_noise(self):
        for name in SCENARIO_NAMES:
            tb = self.make_controller(name)
            tb.scenario_waypoint_index = len(tb.scenario_route) - 1
            gx, gy = tb.scenario_route[-1]
            tb.info['position'] = (gx - GOAL_TOLERANCE / 2, gy)
            tb.run()
            self.assertTrue(tb.scenario_finished, name)
            tb.move.assert_called_once_with(0, 0)
            tb.info['position'] = (gx - 0.3, gy)
            tb.run()
            self.assertEqual(tb.move.call_count, 2)
            tb.move.assert_called_with(0, 0)

    def test_old_interior_goal_does_not_stop_run(self):
        tb = self.make_controller("orthogonal")
        tb.info['position'] = (2.0, -0.4)  # Previously the final goal.
        tb.info['heading'] = math.pi / 2
        tb.run()
        linear, angular = tb.move.call_args.args
        self.assertGreater(linear, 0)
        self.assertEqual(angular, 0)  # Drift correction is disabled during the straight run.

    def test_straight_nod_and_rogue_runs_disable_drift_correction(self):
        for name in ('orthogonal', 'orth_1', 'orth_2', 'antipodal_1', 'antipodal_2', 'mix_lanes'):
            for rogue in (False, True):
                with self.subTest(scenario=name, rogue=rogue):
                    tb = self.make_controller(name)
                    tb.info['heading'] += math.pi / 2
                    with patch.dict(Controller.run.__globals__, ROGUE_AGENTS={'tb1'} if rogue else {}):
                        tb.run()
                    linear, angular = tb.move.call_args.args
                    self.assertGreater(linear, 0)
                    self.assertEqual(angular, 0)
                    self.assertEqual(tb.heading_error_integral, 0)

    def test_turning_steering_stays_active_and_drift_can_be_reenabled(self):
        tb = self.make_controller('turning_intersection')
        tb._update_scenario_goal()
        tb.info['heading'] += 0.1
        tb.heading_error_integral = 0
        self.assertLess(tb._heading_command(0.1), 0)
        tb = self.make_controller('orth_1')
        tb._update_scenario_goal()
        tb.info['heading'] += 0.1
        tb.heading_error_integral = 0
        with patch.object(NodConfig.kin, 'ENABLE_DRIFT_CORRECTION', True):
            self.assertLess(tb._heading_command(0.1), 0)
        self.assertEqual(tb._heading_command(0.1), 0)

    def test_boundary_stop_precedes_route_following(self):
        tb = self.make_controller()
        tb.info["position"] = (3, 0)
        tb.run()
        tb.move.assert_called_once_with(0, 0)
        self.assertEqual(tb.scenario_waypoint_index, 0)

    def test_arrival_logs_braking_and_rest_without_fabricating_zero_measurements(self):
        tb = self.make_controller('orthogonal')
        tb.scenario_waypoint_index = len(tb.scenario_route) - 1
        tb.info['position'] = tb.scenario_route[-1]
        recorded = []
        tb.nod_controller.current_time = 0.0
        tb.data_saver.save_data.side_effect = lambda info, neighbors, sensed, nod, target: recorded.append(
            (math.hypot(*info['velocity']), target, nod.z))
        with patch.object(time, 'monotonic', side_effect=(0.1, 0.2, 0.3)):
            for speed in (0.2, 0.03, 0.0):
                tb.info['velocity'] = (speed, 0.0)
                tb.run()
                tb.move.assert_called_with(0, 0)
                self.assertEqual(tb.target_speed, 0.0)
                self.assertEqual(tb.info['velocity'], (speed, 0.0))
        np.testing.assert_allclose(np.array(recorded)[:, :2], [(0.2, 0), (0.03, 0), (0, 0)])
        rate = NodConfig.dynamics.OPINION_DECAY / NodConfig.dynamics.TAU_Z_RELAX
        np.testing.assert_allclose(np.array(recorded)[:, 2], 0.7 * np.exp(-rate * np.array([0.1, 0.2, 0.3])), rtol=1e-4)
        self.assertEqual(tb.rate.sleep.call_count, 3)

    def test_boundary_goal_and_pivot_log_zero_linear_target(self):
        for reason in ('boundary', 'legacy_goal', 'pivot'):
            with self.subTest(reason=reason):
                tb = self.make_controller()
                if reason == 'boundary':
                    tb.info['position'] = (3, 0)
                elif reason == 'legacy_goal':
                    tb.scenario_route = ()
                    tb.goal_position = tb.info['position']
                else:
                    tb.info['position'] = tb.scenario_route[0]
                    tb.info['heading'] = math.pi / 2
                tb.run()
                tb.data_saver.save_data.assert_called_once()
                self.assertEqual(tb.data_saver.save_data.call_args.args[-1], 0.0)
                if reason == 'pivot':
                    self.assertEqual(tb.nod_controller.z, 0.7)
                else:
                    self.assertGreater(tb.nod_controller.z, 0)
                    self.assertLess(tb.nod_controller.z, 0.7)


if __name__ == "__main__":
    unittest.main()
