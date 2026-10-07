"""Controller regressions runnable without ROS: python3 -m unittest discover -s tests."""
import math
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from constants import NodConfig
from neighbors import arrival_times_to_disk, conflicting_neighbors, solve_ray_intersection, tca_and_rmin
from nod_controller import NodController


def agent(name, position, heading, speed=0.35):
    return dict(name=name, position=position, heading=heading,
                velocity=[speed * math.cos(heading), speed * math.sin(heading)])


class NodRegressionTests(unittest.TestCase):
    def test_terminal_stop_relaxes_both_opinion_signs_without_generating_speed(self):
        for initial in (-0.7, 0.7):
            ctrl = NodController('ego', 0.0)
            ctrl.z, ctrl.u = initial, 0.5
            previous = abs(ctrl.z)
            for step in range(1, 51):
                self.assertIsNone(ctrl.relax_opinion(step * 0.1))
                self.assertLess(abs(ctrl.z), previous)
                self.assertGreater(ctrl.z * initial, 0)
                previous = abs(ctrl.z)
            rate = NodConfig.dynamics.OPINION_DECAY / NodConfig.dynamics.TAU_Z_RELAX
            self.assertAlmostEqual(ctrl.z, initial * math.exp(-5 * rate), delta=1e-5)

    def test_heading_rays_are_unit_length_even_with_velocity_noise(self):
        ego = agent("ego", [-1, 0], 0, 0.1)
        other = agent("other", [0, -1], math.pi / 2, 0.1)
        ego["velocity"] = [0.1, 0.03]
        s, t, ei, ej = solve_ray_intersection(ego, other)
        np.testing.assert_allclose([s, t], [1, 1])
        self.assertAlmostEqual(np.linalg.norm(ei), 1)
        self.assertAlmostEqual(np.linalg.norm(ej), 1)
        self.assertAlmostEqual(abs(np.linalg.det([ei, ej])), 1)

    def test_velocity_only_rays_are_normalized(self):
        ego = agent("ego", [-1, 0], 0, 0.1)
        other = agent("other", [0, -1], math.pi / 2, 0.1)
        del ego["heading"], other["heading"]
        np.testing.assert_allclose(solve_ray_intersection(ego, other)[:2], [1, 1])

    def test_shallow_crossing_uses_expanded_zone(self):
        ego = agent("ego", [0.5 * NodConfig.neighbors.R_OCC, 0], 0)
        other = agent("other", [-math.cos(math.pi / 6), -0.5], math.pi / 6)
        ti, tj, _, inside_i, _ = arrival_times_to_disk(ego, other)
        self.assertEqual(ti, 0)
        self.assertTrue(inside_i)
        self.assertAlmostEqual(tj, (1 - NodConfig.neighbors.R_OCC / math.sin(math.pi / 6)) / 0.35)
        self.assertIn("other", conflicting_neighbors(ego, {"other": other}))

    def test_parallel_rays_have_no_disk_arrival(self):
        ego = agent("ego", [0, 0], 0)
        other = agent("other", [1, 0], 0)
        self.assertEqual(arrival_times_to_disk(ego, other), (None, None, None, False, False))

    def test_receding_agents_use_present_clearance(self):
        ego = agent("ego", [0, 0], math.pi)
        other = agent("other", [1, 0], 0)
        _, _, t_star, d_min = tca_and_rmin(ego, other, False, False)
        self.assertEqual(t_star, 0)
        self.assertEqual(d_min, 1)

    def test_pressure_urgency_and_gate_match_crossing_equations(self):
        ego = agent("ego", [-0.8, 0], 0)
        other = agent("other", [0, -1.6], math.pi / 2)
        ctrl = NodController("ego", 0)
        pressures, gates, _ = ctrl._compute_pressure_and_gates(ego, {"other": other}, {"other"})
        a, b, _, d_min = tca_and_rmin(ego, other, False, False)
        t_star = ctrl._softmax(0, -b / (a + 0.01), NodConfig.pressure.TAU_SOFT_URGENCY)
        expected = expit(2 * (NodConfig.pressure.DMIN_CLEAR - d_min))
        expected *= expit(2 * ((0.8 + NodConfig.neighbors.R_OCC) / 0.35 - t_star))
        expected *= expit(2 * ((1.6 + NodConfig.neighbors.R_OCC) / 0.35 - t_star))
        self.assertAlmostEqual(pressures[0], expected)
        self.assertGreater(gates[0], 0)  # earlier ego goes
        _, reverse_gates, _ = ctrl._compute_pressure_and_gates(other, {"ego": ego}, {"ego"})
        self.assertLess(reverse_gates[0], 0)  # later robot yields

    def test_tilt_does_not_reverse_high_pressure_go_drive(self):
        ctrl = NodController("ego", 0)
        drive, _ = ctrl._aggregate([0.99], [1.0], [1.0])
        self.assertAlmostEqual(drive, 0.99)

    def test_tilt_prioritizes_yield_in_multiple_conflicts(self):
        ctrl = NodController("ego", 0)
        drive, _ = ctrl._aggregate([0.9, 0.9], [1.0, -1.0], [1.0, 1.0])
        self.assertLess(drive, -0.89)

    def test_attention_preserves_integrated_state(self):
        ctrl = NodController("ego", 0)
        with patch.object(NodConfig.dynamics, "K_U", 0), patch.object(NodConfig.dynamics, "USE_ATT_DYNAMICS", True):
            _, u = ctrl._integrate_fast(0, 0, -0.5, 0.5, 0.1)
            expected = NodConfig.dynamics.U_0 * (1 - math.exp(-0.1 / NodConfig.dynamics.TIMING_TAU_U_RELAX))
            self.assertAlmostEqual(u, expected, delta=1e-4)

    def test_free_flow_decay_uses_elapsed_time(self):
        ctrl = NodController("ego", 10)
        ctrl.z, ctrl.u = 0.5, 1.0
        with patch.object(NodConfig.dynamics, "USE_ATT_DYNAMICS", True):
            ctrl.update_opinion(agent("ego", [0, 0], 0), {}, 10.1)
        self.assertAlmostEqual(ctrl.z, 0.5 * math.exp(-0.1 / NodConfig.dynamics.TAU_Z_RELAX), delta=1e-4)
        self.assertAlmostEqual(ctrl.u, math.exp(-0.1 / NodConfig.dynamics.TIMING_TAU_U_RELAX), delta=1e-4)

    def test_zero_horizon_and_disabled_attention(self):
        ctrl = NodController("ego", 0)
        self.assertEqual(ctrl._integrate_fast(0.5, 1, None, None, 0), (0.5, 1))
        with patch.object(NodConfig.dynamics, "USE_ATT_DYNAMICS", False):
            self.assertEqual(ctrl._free_flow(0.5, 1)[1], 0)

    def test_elapsed_time_is_bounded_and_nonnegative(self):
        ctrl = NodController("ego", 0)
        ego = agent("ego", [0, 0], 0)
        ctrl.update_opinion(ego, {}, 10)
        self.assertEqual(ctrl.time_step, 0.1)
        ctrl.update_opinion(ego, {}, 9)
        self.assertEqual(ctrl.time_step, 0)



if __name__ == "__main__":
    unittest.main()
