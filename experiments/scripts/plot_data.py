"""Shared plot-only tracking cleanup. Raw CSVs and live controllers stay intact.

Reject isolated spatial jumps and positive measured-speed spikes. Interpolate
only short, bracketed gaps (<=0.5 s); leave longer/missing-end gaps as NaN.
Filter each recording session separately, never across a restart or time gap.
Lightly smooth measured speed and opinion after tracking outlier removal.
"""
import numpy as np
import pandas as pd

MAX_REPAIR_GAP = 0.5
SESSION_GAP = 5.0
SMOOTH_HALF_WINDOW = 0.75
DECISION_ZERO_TOLERANCE = 0.03
STOPPED_SPEED_TOLERANCE = 0.03


def smooth_plot_signal(values, times, preserve_zero=False, jump_threshold=None, median_samples=5):
    """Median/triangular smoothing within 0.75 s, for display only.

    Keep missing samples, stops, sharp transitions, and recording gaps intact.
    Use timestamps rather than a fixed number of samples for the average.
    """
    values = np.asarray(values, dtype=float)
    times = np.asarray(times, dtype=float)
    result = values.copy()
    valid = np.isfinite(values) & np.isfinite(times)
    if preserve_zero:
        valid &= values > 0
    cuts = (np.diff(times) <= 0) | (np.diff(times) > MAX_REPAIR_GAP)
    cuts |= ~valid[:-1] | ~valid[1:]
    for indices in np.split(np.arange(len(values)), np.flatnonzero(cuts) + 1):
        if len(indices) < 3 or not valid[indices].all():
            continue
        median = pd.Series(values[indices]).rolling(median_samples, center=True, min_periods=1).median().to_numpy()
        # Detect sustained steps after the median removes isolated jitters.
        # Raw jitter previously split the filter into tiny unsmoothed pieces.
        steps = (np.flatnonzero(np.abs(np.diff(median)) > jump_threshold) + 1
                 if jump_threshold is not None else [])
        for local in np.split(np.arange(len(indices)), steps):
            t = times[indices[local]]
            signal = median[local]
            for i, stamp in enumerate(t):
                left, right = np.searchsorted(t, [stamp - SMOOTH_HALF_WINDOW, stamp + SMOOTH_HALF_WINDOW])
                weights = np.maximum(0, 1 - np.abs(t[left:right] - stamp) / SMOOTH_HALF_WINDOW)
                result[indices[local[i]]] = np.average(signal[left:right], weights=weights)
    return result


def _repair_short_gaps(values, times):
    result = np.asarray(values, dtype=float).copy()
    missing = ~np.isfinite(result)
    starts = np.flatnonzero(missing & ~np.r_[False, missing[:-1]])
    ends = np.flatnonzero(missing & ~np.r_[missing[1:], False])
    for first, last in zip(starts, ends):
        left, right = first - 1, last + 1
        if left >= 0 and right < len(result) and 0 < times[right] - times[left] <= MAX_REPAIR_GAP:
            result[first:right] = np.interp(times[first:right], times[[left, right]], result[[left, right]])
    return result


def _spatial_outliers(x, y, times, speed_limit):
    mx = pd.Series(x).rolling(3, center=True, min_periods=2).median().to_numpy()
    my = pd.Series(y).rolling(3, center=True, min_periods=2).median().to_numpy()
    residual = np.hypot(x - mx, y - my)
    local = pd.Series(residual).rolling(9, center=True, min_periods=2).median().to_numpy()
    dt = np.diff(times)
    step_speed = np.divide(np.hypot(np.diff(x), np.diff(y)), dt,
                           out=np.zeros(len(dt)), where=dt > 0)
    fast = np.r_[False, step_speed > speed_limit] | np.r_[step_speed > speed_limit, False]
    return fast & (residual > np.maximum(0.08, 6 * 1.4826 * local))


def _speed_outliers(values, limit):
    s = pd.Series(values)
    median = s.rolling(11, center=True, min_periods=2).median()
    deviation = (s - median).abs().rolling(11, center=True, min_periods=2).median()
    # Keep drops to zero: these may be the actual speed-modulation response.
    return ((s < 0) | (s > limit) | ((s - median) > np.maximum(0.08, 4 * 1.4826 * deviation))).to_numpy()


def clean_plot_data(frame, robot=True):
    if not {'t', 'x', 'y'}.issubset(frame.columns):
        return frame.copy()
    result = frame.copy()
    times = pd.to_numeric(result['t'], errors='coerce').to_numpy(dtype=float)
    boundaries = np.flatnonzero((np.diff(times) > SESSION_GAP) | (np.diff(times) <= 0)
                               | ~np.isfinite(times[1:]) | ~np.isfinite(times[:-1])) + 1
    report = dict(position_outliers=0, speed_outliers=0, unfilled_position_samples=0)
    pairs = [('x', 'y', 0.8 if robot else 4.0)]
    pairs += [(col, 'y_' + col[2:], 0.8 if col[2:].startswith('tb') else 4.0)
              for col in result.columns if col.startswith('x_') and 'y_' + col[2:] in result]
    for indices in np.split(np.arange(len(result)), boundaries):
        if not len(indices):
            continue
        t = times[indices]
        for xcol, ycol, limit in pairs:
            x = pd.to_numeric(result.iloc[indices][xcol], errors='coerce').to_numpy(dtype=float)
            y = pd.to_numeric(result.iloc[indices][ycol], errors='coerce').to_numpy(dtype=float)
            bad = _spatial_outliers(x, y, t, limit) if len(indices) >= 3 else np.zeros(len(indices), dtype=bool)
            bad |= ~np.isfinite(x) | ~np.isfinite(y)
            report['position_outliers'] += int(bad.sum())
            x[bad], y[bad] = np.nan, np.nan
            x, y = _repair_short_gaps(x, t), _repair_short_gaps(y, t)
            result.iloc[indices, result.columns.get_loc(xcol)] = x
            result.iloc[indices, result.columns.get_loc(ycol)] = y
            if xcol == 'x':
                report['unfilled_position_samples'] += int((~np.isfinite(x) | ~np.isfinite(y)).sum())
        if 'current_speed' in result:
            values = pd.to_numeric(result.iloc[indices]['current_speed'], errors='coerce').to_numpy(dtype=float)
            bad = _speed_outliers(values, 0.8 if robot else 4.0)
            if len(indices) >= 3:
                px = result.iloc[indices]['x'].to_numpy(dtype=float)
                py = result.iloc[indices]['y'].to_numpy(dtype=float)
                dt = np.diff(t)
                steps = np.divide(np.hypot(np.diff(px), np.diff(py)), dt,
                                  out=np.full(len(dt), np.nan), where=(dt > 0) & (dt <= MAX_REPAIR_GAP))
                reference = (np.r_[steps[0], steps] + np.r_[steps, steps[-1]]) / 2
                baseline = pd.Series(values).rolling(15, center=True, min_periods=2).median().to_numpy()
                # Catch short bursts of derivative noise that survive a
                # median filter, using recorded displacement as a cross-check.
                bad |= (values > baseline + 0.08) & (values > np.maximum(0.12, 1.8 * reference + 0.04))
            report['speed_outliers'] += int(bad.sum())
            values[bad] = np.nan
            result.iloc[indices, result.columns.get_loc('current_speed')] = _repair_short_gaps(values, t)
            repaired = result.iloc[indices]['current_speed'].to_numpy(dtype=float)
            result.iloc[indices, result.columns.get_loc('current_speed')] = smooth_plot_signal(
                repaired, t, preserve_zero=True, jump_threshold=0.12, median_samples=9)
            if 'target_speed' in result:
                target = pd.to_numeric(result.iloc[indices]['target_speed'], errors='coerce').to_numpy(dtype=float)
                speed = result.iloc[indices]['current_speed'].to_numpy(dtype=float)
                near_rest = (target == 0) & (speed < STOPPED_SPEED_TOLERANCE)
                speed[near_rest] = 0.0
                result.iloc[indices, result.columns.get_loc('current_speed')] = speed
        if 'opinion' in result:
            opinion = pd.to_numeric(result.iloc[indices]['opinion'], errors='coerce').to_numpy(dtype=float)
            smoothed = smooth_plot_signal(opinion, t, jump_threshold=0.3)
            if 'target_speed' in result:
                target = pd.to_numeric(result.iloc[indices]['target_speed'], errors='coerce').to_numpy(dtype=float)
                # Use the recorded decision for this deadband so centered
                # smoothing cannot lift a settled endpoint away from zero.
                smoothed[(target == 0) & (np.abs(opinion) < DECISION_ZERO_TOLERANCE)] = 0.0
            result.iloc[indices, result.columns.get_loc('opinion')] = smoothed
    report.update(smoothing_half_window_s=SMOOTH_HALF_WINDOW, speed_median_samples=9, decision_median_samples=5,
                  stopped_speed_display_threshold=STOPPED_SPEED_TOLERANCE,
                  stopped_decision_display_threshold=DECISION_ZERO_TOLERANCE)
    result.attrs['plot_filter'] = report
    return result


def read_plot_csv(path, **kwargs):
    """Drop-in CSV reader for all saved-data plotters; never writes a CSV."""
    frame = clean_plot_data(pd.read_csv(path, **kwargs), robot=str(path).split('/')[-1].startswith('tb'))
    report = frame.attrs.get('plot_filter', {})
    if report.get('position_outliers', 0) or report.get('speed_outliers', 0):
        print('Plot-only tracking filter:', path, report)
    return frame
