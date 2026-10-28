"""Unit tests for pruning/sim_core.py: the causal World/Library primitives every allocation rule builds on.
No GPU, network or real result files -- all curves here are synthetic numpy arrays of the real length T."""
import numpy as np
import pytest

from sim_core import C0, Library, World, calibrate_warp, run_policy, warp
from data import T


def ramp(v0, v1, t=T):
    """A simple monotonically increasing curve from v0 to v1 over t iterations."""
    return np.linspace(v0, v1, t)


def bump(peak_val, peak_at, final_val, t=T):
    """Rises to peak_val at iteration peak_at, then decays to final_val -- final < peak, so running-best and
    raw differ, the case every policy must handle correctly."""
    up = np.linspace(0, peak_val, peak_at)
    down = np.linspace(peak_val, final_val, t - peak_at)
    return np.concatenate([up, down])


# ---------------------------------------------------------------- World

def test_world_advance_is_monotonic_and_clamped():
    w = World(np.stack([ramp(0, 10), ramp(0, 20)]))
    w.advance(0, 100)
    w.advance(0, 50)  # going "backwards" must never un-train a candidate
    assert w.prog[0] == 100
    w.advance(0, T + 500)  # can never exceed the curve length
    assert w.prog[0] == T


def test_world_at_is_running_best_not_raw_value():
    w = World(np.stack([bump(peak_val=10, peak_at=100, final_val=2)]))
    w.advance(0, 200)
    # at iteration 200 the raw curve has already decayed, but at() must report the best-so-far (10), not the dip
    assert w.at(0, 200) == pytest.approx(10.0)
    assert w.at(0, 100) == pytest.approx(10.0)
    assert w.at(0, 50) < 10.0  # before the peak, running best is still below it


def test_world_at_requires_training_to_have_reached_t():
    w = World(np.stack([ramp(0, 10)]))
    w.advance(0, 50)
    with pytest.raises(AssertionError):
        w.at(0, 51)  # asking for a point not yet trained must fail loudly, not silently return something


def test_world_cost_sums_progress_across_candidates():
    w = World(np.stack([ramp(0, 1), ramp(0, 1), ramp(0, 1)]))
    w.advance(0, 500)
    w.advance(1, 300)
    assert w.cost == 800
    w.extra = 50
    assert w.cost == 850


def test_world_finish_trains_to_completion_and_returns_final_running_best():
    y = bump(peak_val=9, peak_at=10, final_val=1)
    w = World(np.stack([y]))
    val = w.finish(0)
    assert w.prog[0] == T
    assert val == pytest.approx(9.0)  # running-best at T, not the final (decayed) raw value


def test_world_truth_is_each_candidates_peak():
    w = World(np.stack([ramp(0, 5), bump(peak_val=8, peak_at=50, final_val=3)]))
    truth = w.truth()
    assert truth[0] == pytest.approx(5.0)
    assert truth[1] == pytest.approx(8.0)


# ---------------------------------------------------------------- Library

def test_library_ratio_matches_final_over_running_best():
    # one curve that ramps from 1 to 10: running-best at t is itself (monotone), final is 10.
    cm = ramp(1, 10)
    lib = Library(np.stack([cm]), np.stack([np.full(T, 5.0)]))
    expected_t100 = 10.0 / max(cm[99], C0)
    assert lib.R[0, 99] == pytest.approx(expected_t100)
    assert lib.final[0] == pytest.approx(10.0)
    assert lib.human[0] == pytest.approx(5.0)  # mean of the human curves, here a single constant curve


def test_library_gaps_is_final_minus_value_at_t():
    cm = ramp(0, 10)
    lib = Library(np.stack([cm]), np.stack([np.zeros(T)]))
    g = lib.gaps(500)
    assert g[0] == pytest.approx(10.0 - cm[499])


# ---------------------------------------------------------------- warp / calibrate_warp

def test_warp_identity_scale_is_a_no_op():
    y = ramp(0, 42)
    assert np.allclose(warp(y, 1.0), y)


def test_warp_faster_pulls_the_curve_earlier():
    y = ramp(0, 10)
    fast = warp(y, 2.0)  # s>1: same shape reached in half the time
    assert fast[T // 4] > y[T // 4]  # a quarter of the way through, the sped-up curve is further along


def test_calibrate_warp_recovers_a_known_time_dilation():
    base = bump(peak_val=10, peak_at=600, final_val=4, t=T)
    true_s = 1.4
    dilated = warp(base, true_s)
    s_hat, err = calibrate_warp(base, dilated)
    assert s_hat == pytest.approx(true_s, rel=0.05)
    assert err < 0.5  # recovered fit should be tight, not just "the closest grid point by luck"


# ---------------------------------------------------------------- run_policy

def test_run_policy_perfect_pick_retains_everything():
    def pick_the_best_at_500(w, lib, rng):
        for i in range(w.N):
            w.advance(i, 500)
        return int(np.argmax([w.at(i, 500) for i in range(w.N)]))

    # the leader at 500 is also the global best -> retained must be exactly 1.0
    curves = np.stack([ramp(0, 5), ramp(0, 20), ramp(0, 3)])
    w = World(curves)
    lib = Library(curves, curves[:1])
    r = run_policy(pick_the_best_at_500, w, lib, np.random.default_rng(0))
    assert r["retained"] == pytest.approx(1.0)
    assert r["hit"] == 1.0
    assert r["regret"] == pytest.approx(0.0)


def test_run_policy_wrong_pick_is_penalised_in_retained_and_regret():
    def always_pick_first(w, lib, rng):
        for i in range(w.N):
            w.advance(i, 500)
        return 0

    curves = np.stack([ramp(0, 3), ramp(0, 20)])  # candidate 0 is clearly worse than candidate 1
    w = World(curves)
    lib = Library(curves, curves[:1])
    r = run_policy(always_pick_first, w, lib, np.random.default_rng(0))
    assert r["retained"] == pytest.approx(3.0 / 20.0)
    assert r["hit"] == 0.0
    assert r["regret"] == pytest.approx(20.0 - 3.0)


def test_run_policy_resets_world_progress_between_calls():
    # a World handed to run_policy twice must not carry over progress from the first call
    curves = np.stack([ramp(0, 5), ramp(0, 9)])
    w = World(curves)
    lib = Library(curves, curves[:1])

    def advance_cand1_then_pick_cand0(w, lib, rng):
        w.advance(1, 200)
        return 0

    run_policy(advance_cand1_then_pick_cand0, w, lib, np.random.default_rng(0))
    assert list(w.prog) == [T, 200]  # candidate 0 was finish()ed (selected); candidate 1 stopped where the policy left it

    def pick_cand0_only(w, lib, rng):
        return 0  # never touches candidate 1 this time

    run_policy(pick_cand0_only, w, lib, np.random.default_rng(0))
    # if run_policy had NOT reset prog to 0 first, candidate 1 would still show >= 200 from the previous call
    assert list(w.prog) == [T, 0]
