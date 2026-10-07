"""Plot filtering preserves real motion and never changes raw input."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments' / 'scripts'))
from plot_data import clean_plot_data


class PlotDataTests(unittest.TestCase):
    def frame(self):
        t = np.arange(30) * 0.1
        return pd.DataFrame(dict(t=t, x=0.25 * t, y=np.zeros(len(t)),
                                 current_speed=np.full(len(t), 0.25), target_speed=np.full(len(t), 0.35)))

    def test_tracking_jump_and_speed_spike_are_repaired(self):
        raw = self.frame()
        raw.loc[12, ['x', 'y']] = [-2, 2]
        raw.loc[13, 'current_speed'] = 26
        before = raw.copy(deep=True)
        cleaned = clean_plot_data(raw)
        self.assertAlmostEqual(cleaned.loc[12, 'x'], 0.3)
        self.assertAlmostEqual(cleaned.loc[12, 'y'], 0)
        self.assertAlmostEqual(cleaned.loc[13, 'current_speed'], 0.25)
        pd.testing.assert_frame_equal(raw, before)
        pd.testing.assert_series_equal(cleaned.target_speed, raw.target_speed)

    def test_real_stops_turns_and_rogue_speed_remain(self):
        raw = self.frame()
        raw.loc[10:14, 'current_speed'] = 0
        raw.loc[20:, 'current_speed'] = 0.55
        raw.loc[15:, 'x'] = raw.loc[15, 'x']
        raw.loc[15:, 'y'] = (raw.loc[15:, 't'] - raw.loc[15, 't']) * 0.25
        cleaned = clean_plot_data(raw)
        pd.testing.assert_frame_equal(cleaned, raw)

    def test_long_losses_and_restart_boundaries_are_not_interpolated(self):
        raw = self.frame()
        raw.loc[10:20, ['x', 'y', 'current_speed']] = np.nan
        cleaned = clean_plot_data(raw)
        self.assertTrue(cleaned.loc[10:20, 'x'].isna().all())
        raw = self.frame()
        raw.loc[15:, 't'] += 10
        raw.loc[15:, 'x'] += 5
        cleaned = clean_plot_data(raw)
        pd.testing.assert_frame_equal(cleaned, raw)

    def test_metadata_is_unchanged(self):
        raw = pd.DataFrame(dict(robot=['tb1'], agent_type=['NOD']))
        pd.testing.assert_frame_equal(clean_plot_data(raw), raw)

    def test_light_speed_and_decision_jitter_is_reduced_without_changing_raw(self):
        raw = self.frame()
        jitter = 0.025 * np.where(np.arange(len(raw)) % 2, 1, -1)
        raw['current_speed'] += jitter
        raw['opinion'] = 0.4 + 2 * jitter
        before = raw.copy(deep=True)
        cleaned = clean_plot_data(raw)
        for column in ('current_speed', 'opinion'):
            self.assertLess(cleaned[column].iloc[3:-3].std(), raw[column].iloc[3:-3].std() * 0.6)
        pd.testing.assert_frame_equal(raw, before)
        pd.testing.assert_series_equal(cleaned.target_speed, raw.target_speed)

    def test_smoothing_preserves_stop_samples_and_missing_decisions(self):
        raw = self.frame()
        raw['current_speed'] += 0.02 * np.sin(np.arange(len(raw)))
        raw.loc[10:14, 'current_speed'] = 0
        raw['opinion'] = 0.3
        raw.loc[10:14, 'opinion'] = np.nan
        raw.loc[15:, 'opinion'] = -0.3
        cleaned = clean_plot_data(raw)
        self.assertTrue((cleaned.loc[10:14, 'current_speed'] == 0).all())
        self.assertTrue(cleaned.loc[10:14, 'opinion'].isna().all())
        np.testing.assert_allclose(cleaned.loc[:9, 'opinion'], 0.3)
        np.testing.assert_allclose(cleaned.loc[15:, 'opinion'], -0.3)

    def test_stopped_display_noise_floor_does_not_erase_large_decision_tails(self):
        raw = self.frame()
        raw['target_speed'] = 0
        raw['current_speed'] = 0.005
        raw['opinion'] = 0.004
        clean = clean_plot_data(raw)
        self.assertTrue((clean.current_speed == 0).all())
        self.assertTrue((clean.opinion == 0).all())
        raw['opinion'] = 0.4
        np.testing.assert_allclose(clean_plot_data(raw).opinion, 0.4)
        raw['target_speed'] = 0.1
        raw['opinion'] = 0.004
        clean = clean_plot_data(raw)
        np.testing.assert_allclose(clean.current_speed, 0.005)
        np.testing.assert_allclose(clean.opinion, 0.004)

    def test_median_filter_removes_isolated_jitter_before_step_detection(self):
        raw = self.frame()
        raw.loc[15, 'current_speed'] = 0.39
        clean = clean_plot_data(raw)
        np.testing.assert_allclose(clean.current_speed.iloc[12:19], 0.25)


if __name__ == '__main__':
    unittest.main()
