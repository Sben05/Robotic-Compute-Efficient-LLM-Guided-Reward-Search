"""Causal simulator for compute-allocation policies over finished PPO learning curves.

A World holds N candidate curves but only exposes the prefix each candidate has been trained for, so a policy
cannot peek ahead. Score = running-best consecutive_successes (the environment metric, as upstream Eureka).
Cost = PPO iterations run; the selected candidate is always trained to T=3,000 and counted.
"""
import itertools

import numpy as np

from data import T, load_all

C0 = 1.0  # floor for ratio-based forecasts (scores below 1 are indistinguishable from a cold start)


def warp(y, s):
    """Seed-noise replicate: same learning curve run faster (s>1) or slower (s<1)."""
    x = np.arange(T) * s
    return np.interp(x, np.arange(T), y)


def calibrate_warp(human0, human1):
    """Time-scale s minimizing |human1(t) - human0(s t)| over the full run (single seed pair)."""
    grid = np.exp(np.linspace(-0.6, 0.6, 241))
    err = [np.mean(np.abs(human1 - warp(human0, s))) for s in grid]
    return float(grid[int(np.argmin(err))]), float(min(err))


class Library:
    """Previously finished runs: used only for priors (ratios, human reference), never the candidates being screened."""

    def __init__(self, cms, human_cm):
        self.cm = np.asarray(cms)                                         # (L, T) running-best curves
        self.final = self.cm[:, -1]
        self.R = self.final[:, None] / np.maximum(self.cm, C0)             # (L, T) final / max(cm(t), C0)
        self.human = np.mean(human_cm, axis=0)                            # reference curve (mean of seeds)

    def gaps(self, t):
        return self.final - self.cm[:, t - 1]


class World:
    def __init__(self, raw):
        self._raw = raw
        self._cm = np.maximum.accumulate(raw, axis=1)
        self.N = len(raw)
        self.prog = np.zeros(self.N, int)
        self.extra = 0  # iterations spent on from-scratch retraining (non-resumable policies, e.g. paper Hyperband)

    def advance(self, i, to):
        self.prog[i] = max(self.prog[i], min(int(to), T))

    def cm(self, i):            # running best, causal
        return self._cm[i, : self.prog[i]]

    def at(self, i, t=None):    # running best at iteration t (must be trained)
        t = self.prog[i] if t is None else t
        assert t <= self.prog[i] and t >= 1
        return self._cm[i, t - 1]

    @property
    def cost(self):
        return int(self.prog.sum()) + self.extra

    def finish(self, i):
        self.advance(i, T)
        return self._cm[i, -1]

    def truth(self):
        return self._cm[:, -1]


def make_worlds(curves, human, sigma_s, n_arms, n_rep, rng, max_subsets=None, lib_mode="rest", pool_draws=None):
    """Yield (World, Library, ids). Arms = every n_arms-subset of the LLM curves (or, with pool_draws=K, K random draws of
    n_arms curves WITH replacement, each repeat getting its own seed warp). Library = the human curves plus, when
    lib_mode == "rest", every LLM curve not among the arms; lib_mode == "human" gives the human curves only."""
    names = list(curves)
    if pool_draws:
        subsets = [tuple(rng.integers(len(names), size=n_arms)) for _ in range(pool_draws)]
    else:
        subsets = list(itertools.combinations(range(len(names)), n_arms))
        if max_subsets and len(subsets) > max_subsets:
            subsets = [subsets[k] for k in rng.choice(len(subsets), max_subsets, replace=False)]
    hum_cm = [np.maximum.accumulate(h) for h in human.values()]
    for sub in subsets:
        lib_ids = [k for k in range(len(names)) if k not in sub] if lib_mode == "rest" else []
        lib = Library([np.maximum.accumulate(curves[names[k]]) for k in lib_ids] + hum_cm, hum_cm)
        for _ in range(n_rep):
            raws = np.stack([warp(curves[names[k]], float(np.exp(sigma_s * rng.standard_normal()))) if sigma_s > 0
                             else curves[names[k]] for k in sub])
            yield World(raws), lib, [names[k] for k in sub]


def run_policy(policy, world, lib, rng, **kw):
    world.prog[:] = 0      # every policy starts from untrained candidates
    world.extra = 0
    sel = policy(world, lib, rng, **kw)
    sel = [sel] if np.isscalar(sel) else list(sel)
    finals = [world.finish(i) for i in sel]       # train every selected candidate to completion
    pick = sel[int(np.argmax(finals))]
    tr = world.truth()
    return {"cost": world.cost / (world.N * T), "overhead": (world.cost - T) / world.N, "retained": float(tr[pick] / tr.max()), "hit": float(pick == np.argmax(tr)),
            "regret": float(tr.max() - tr[pick])}


def evaluate(policy, worlds, seed=0, **kw):
    rng = np.random.default_rng(seed)
    rows = [run_policy(policy, w, lib, rng, **kw) for w, lib, _ in worlds]
    a = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    return {"overhead_its_per_arm": float(a["overhead"].mean()), "compute": float(a["cost"].mean()), "saved": float(1 - a["cost"].mean()), "retained": float(a["retained"].mean()),
            "ret_p05": float(np.percentile(a["retained"], 5)), "ret_min": float(a["retained"].min()),
            "frac_below_95": float((a["retained"] < 0.95).mean()), "hit": float(a["hit"].mean()),
            "regret": float(a["regret"].mean()), "n_worlds": len(rows)}
