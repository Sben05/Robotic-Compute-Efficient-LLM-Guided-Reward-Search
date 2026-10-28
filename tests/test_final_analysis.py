"""Unit tests for the small statistical/IO helpers in final_analysis.py: the Wilson interval, Holm
correction, and the curve-file readers that every H1-H5 computation is built from."""
import json

import numpy as np
import pytest

import final_analysis as F


# ---------------------------------------------------------------- wilson

def test_wilson_guards_against_zero_denominator():
    assert F.wilson(0, 0) == [None, None]


def test_wilson_interval_contains_the_point_estimate_and_is_bounded():
    for k, n in [(5, 20), (1, 4), (19, 20), (0, 10), (10, 10)]:
        lo, hi = F.wilson(k, n)
        # k=0/k=n are edge cases where the formula's floating-point arithmetic can land an epsilon outside
        # [0, 1] (e.g. -2.8e-17 instead of exactly 0); allow that much slack, no more.
        assert -1e-9 <= lo <= (k / n) <= hi <= 1.0 + 1e-9


def test_wilson_interval_narrows_as_n_grows_at_the_same_proportion():
    lo_small, hi_small = F.wilson(1, 4)     # 25%, n=4
    lo_big, hi_big = F.wilson(25, 100)      # 25%, n=100
    assert (hi_big - lo_big) < (hi_small - lo_small)


def test_wilson_matches_a_hand_computed_value():
    # z=1.96, k=5, n=20 -> p=0.25; d=1+1.96^2/20=1.19208; center=(0.25+1.96^2/40)/d=0.29027;
    # h=1.96*sqrt(0.25*0.75/20+1.96^2/1600)/d=0.17843 -> [0.11184, 0.46870]
    lo, hi = F.wilson(5, 20)
    assert lo == pytest.approx(0.1118, abs=0.001)
    assert hi == pytest.approx(0.4687, abs=0.001)


# ---------------------------------------------------------------- holm

def test_holm_step_down_matches_a_hand_worked_example():
    ps = {"a": 0.01, "b": 0.04, "c": 0.03}
    adj = F.holm(ps)
    assert adj["a"] == pytest.approx(0.03)
    assert adj["c"] == pytest.approx(0.06)
    assert adj["b"] == pytest.approx(0.06)


def test_holm_adjusted_p_never_decreases_in_sorted_order():
    ps = {"a": 0.001, "b": 0.5, "c": 0.2, "d": 0.3, "e": 0.01}
    adj = F.holm(ps)
    ordered = sorted(ps, key=lambda n: ps[n])
    vals = [adj[n] for n in ordered]
    assert vals == sorted(vals)  # Holm's defining property: monotonic non-decreasing after the step-down max


def test_holm_caps_every_adjusted_p_at_one():
    ps = {"a": 0.9, "b": 0.95, "c": 0.99}
    adj = F.holm(ps)
    assert all(v <= 1.0 for v in adj.values())


# ---------------------------------------------------------------- curve / peak

def test_curve_returns_none_for_a_missing_file(tmp_path):
    assert F.curve(tmp_path / "does_not_exist.json") is None


def test_curve_returns_none_for_a_crashed_run_with_no_scores(tmp_path):
    f = tmp_path / "curves.json"
    f.write_text(json.dumps({"consecutive_successes": []}))
    assert F.curve(f) is None


def test_curve_loads_the_task_score_series(tmp_path):
    f = tmp_path / "curves.json"
    f.write_text(json.dumps({"consecutive_successes": [1.0, 5.0, 3.0]}))
    y = F.curve(f)
    assert np.allclose(y, [1.0, 5.0, 3.0])


def test_peak_is_the_max_over_the_whole_curve_by_default(tmp_path):
    f = tmp_path / "curves.json"
    f.write_text(json.dumps({"consecutive_successes": [1.0, 9.0, 3.0, 0.5]}))
    assert F.peak(f) == pytest.approx(9.0)


def test_peak_respects_the_early_checkpoint_k(tmp_path):
    # the peak score reached by iteration 500 can differ from the eventual full-run peak -- this is exactly
    # the distinction H1's "screen at 500, keep training only the leader" protocol depends on.
    f = tmp_path / "curves.json"
    f.write_text(json.dumps({"consecutive_successes": [1.0, 9.0, 3.0, 0.5, 20.0]}))
    assert F.peak(f, k=1) == pytest.approx(1.0)   # only the first iteration is visible
    assert F.peak(f, k=2) == pytest.approx(9.0)   # the early peak is already visible by iteration 2
    assert F.peak(f) == pytest.approx(20.0)       # the full-run peak (no k) sees the later, higher value too


def test_peak_returns_none_when_curve_is_none(tmp_path):
    assert F.peak(tmp_path / "missing.json") is None
