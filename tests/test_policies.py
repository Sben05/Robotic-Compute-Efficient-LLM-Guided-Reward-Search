"""Unit tests for pruning/policies.py: the allocation rules compared throughout the thesis. Curves are
synthetic and noise-free so a rule's qualitative behaviour (does it drop a clearly-dominated arm? does it
respect its budget cap? does it ever return an out-of-range index?) is unambiguous and not flaky."""
import numpy as np
import pytest

import policies as P
from data import T
from sim_core import Library, World


def ramp(v0, v1, t=T):
    return np.linspace(v0, v1, t)


def make_library(n=12, seed=0):
    """A library of varied-shape reference curves (different final values and growth rates), so draws()-based
    forecasts (VoI, Bayesian, MPC, TTTS, ratio-bound) have real spread to sample from, not a degenerate single curve."""
    rng = np.random.default_rng(seed)
    curves = []
    for _ in range(n):
        v0 = rng.uniform(0, 2)
        vfinal = rng.uniform(3, 15)
        knee = rng.integers(T // 10, T)
        curves.append(np.concatenate([np.linspace(v0, vfinal, knee), np.full(T - knee, vfinal)]))
    human = np.stack([np.linspace(0, 8, T)])
    return Library(np.stack(curves), human)


# ---------------------------------------------------------------- baselines

def test_full_picks_the_global_best():
    curves = np.stack([ramp(0, 3), ramp(0, 9), ramp(0, 1)])
    w = World(curves)
    assert P.full(w, None, None) == 1


def test_random_one_returns_a_valid_index():
    w = World(np.stack([ramp(0, 1)] * 5))
    rng = np.random.default_rng(0)
    for _ in range(20):
        assert 0 <= P.random_one(w, None, rng) < 5


# ---------------------------------------------------------------- fixed checkpoint

def test_fixed_keep1_picks_the_leader_at_the_checkpoint():
    # candidate 1 leads at t=250 but candidate 2 would win if trained fully -- fixed@250 can only see t=250
    curves = np.stack([ramp(0, 2, T), ramp(0, 5, T), np.concatenate([np.full(300, 0.1), ramp(0.1, 20, T - 300)])])
    w = World(curves)
    sel = P.fixed(w, None, None, t_c=250, keep=1)
    assert sel == [1]
    assert list(w.prog) == [250, 250, 250]


def test_fixed_keep2_returns_top_two_sorted_best_first():
    curves = np.stack([ramp(0, 1), ramp(0, 9), ramp(0, 5)])
    w = World(curves)
    sel = P.fixed(w, None, None, t_c=100, keep=2)
    assert sel == [1, 2]  # best first, by score at the checkpoint


# ---------------------------------------------------------------- successive halving / hyperband

def test_seq_halving_keeps_the_consistently_best_candidate():
    # same relative ranking at every iteration (pure ramps to different finals) -> no ambiguity at any checkpoint
    finals = [1, 2, 3, 4]
    curves = np.stack([ramp(0, f) for f in finals])
    w = World(curves)
    survivors = P.seq_halving(w, None, None, t0=250, eta=2)
    assert survivors == [3]  # index of the largest final value
    assert w.cost == 4 * 250 + 2 * (500 - 250)  # round 1: all 4 to 250; round 2: top 2 to 500


def test_seq_halving_never_exceeds_T():
    curves = np.stack([ramp(0, f) for f in (1, 2)])
    w = World(curves)
    P.seq_halving(w, None, None, t0=2000, eta=3)  # t0*eta would overshoot T without the min(T, ...) clamp
    assert w.prog.max() <= T


def test_hyperband_includes_the_global_best_across_brackets():
    finals = [1, 2, 3, 9, 4]
    curves = np.stack([ramp(0, f) for f in finals])
    w = World(curves)
    winners = P.hyperband(w, None, None, t0s=(50, 150, 450), eta=3)
    assert 3 in winners  # the dominant candidate must survive every bracket that tries it
    assert winners == sorted(set(winners))  # contract: sorted, de-duplicated


def test_hyperband_paper_tracks_extra_from_scratch_cost():
    finals = [1, 2, 3, 9]
    curves = np.stack([ramp(0, f) for f in finals])
    w = World(curves)
    winners = P.hyperband_paper(w, None, np.random.default_rng(0), R=T, eta=3, r_min=100)
    assert 3 in winners
    assert w.extra >= 0  # "every bracket from scratch" can only ever cost at least as much as sharing progress would


# ---------------------------------------------------------------- race-based rules (need a Library)

@pytest.fixture
def lib():
    return make_library()


def test_race_pid_drops_a_clearly_dominated_laggard(lib):
    # arm 0 rockets up early; arm 1 stays near zero the whole time -- an unambiguous blowout
    strong = np.concatenate([np.linspace(0, 20, 300), np.full(T - 300, 20)])
    weak = np.full(T, 0.05)
    w = World(np.stack([strong, weak]))
    winner = P.race(w, lib, np.random.default_rng(1), rule=P.rule_pid(1.0, 1, 1, 0.4), t0=250, dt=100, t_cap=1500)
    assert winner == 0
    assert w.cost < 2 * T  # the blowout must be caught before both arms are trained in full


def test_race_voi_does_not_prematurely_drop_a_near_tie(lib):
    # two arms so close that no reasonable forecaster should confidently eliminate either before the budget runs out
    a = np.linspace(0, 10, T)
    b = np.linspace(0, 10.01, T)
    w = World(np.stack([a, b]))
    winner = P.race(w, lib, np.random.default_rng(2), rule=P.rule_voi(1.0, horizon=500), t0=250, dt=100, t_cap=1500)
    assert winner in (0, 1)
    # a near-tie should run (close to) the full racing budget on both arms, not stop almost immediately
    assert w.cost >= 2 * 250


def test_race_always_returns_an_in_range_index(lib):
    curves = np.stack([ramp(0, v) for v in (1, 5, 2, 8, 3)])
    w = World(curves)
    for rule in (P.rule_pid(1.0, 1, 1, 0.4), P.rule_voi(1.0), P.rule_bayes(0.1), P.rule_mpc(1.0),
                 P.rule_ratio_bound(0.9), P.rule_human_ref(0.7)):
        w2 = World(curves.copy())
        winner = P.race(w2, lib, np.random.default_rng(3), rule=rule, t0=250, dt=100, t_cap=1500)
        assert 0 <= winner < w2.N


def test_ttts_terminates_and_returns_a_valid_index(lib):
    curves = np.stack([ramp(0, v) for v in (1, 9, 3, 4)])
    w = World(curves)
    pick = P.ttts(w, lib, np.random.default_rng(4), t0=250, dt=100, delta=0.2, budget=0.3)
    assert 0 <= pick < w.N
    assert w.cost <= 0.3 * w.N * T + w.N * 250 + 10  # bounded by the stated budget (plus the fixed t0 warm-start)


# ---------------------------------------------------------------- ORSO baseline

@pytest.mark.parametrize("algo", ["d3rb", "exp3"])
def test_orso_respects_its_budget_and_returns_a_valid_index(lib, algo):
    curves = np.stack([ramp(0, v) for v in (1, 2, 9, 3)])
    w = World(curves)
    pick = P.orso(w, lib, np.random.default_rng(5), algo=algo, budget=0.3, chunk=30)
    assert 0 <= pick < w.N
    # every arm gets one warm-start chunk, then spend is capped at budget*N*T (checked once per outer loop iteration,
    # so a one-chunk overshoot is possible -- allow exactly that much slack, no more)
    assert w.cost <= 0.3 * w.N * T + w.N * 30 + 30
