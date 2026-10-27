"""Allocation policies. Each sees only World.cm(i)/at(i) (prefixes), the Library, and its own parameters.
Returns the candidate (or list of candidates) to be trained to completion."""
import numpy as np

from data import T
from sim_core import C0


# ---------- baselines ----------
def full(W, lib, rng):
    return int(np.argmax([W.finish(i) for i in range(W.N)]))


def random_one(W, lib, rng):
    return int(rng.integers(W.N))


def fixed(W, lib, rng, t_c=500, keep=1):
    for i in range(W.N):
        W.advance(i, t_c)
    order = np.argsort([-W.at(i, t_c) for i in range(W.N)])
    return [int(i) for i in order[:keep]]


def seq_halving(W, lib, rng, t0=250, eta=2):
    """Successive/sequential halving: budget per survivor multiplies by eta each round; keep top 1/eta."""
    active, t = list(range(W.N)), t0
    while len(active) > 1:
        for i in active:
            W.advance(i, t)
        active.sort(key=lambda i: -W.at(i, min(t, W.prog[i])))
        active = active[: max(1, int(np.ceil(len(active) / eta)))]
        t = min(T, t * eta)
        if t >= T and len(active) > 1:
            break
    return active


def hyperband(W, lib, rng, t0s=(50, 150, 450), eta=3):
    """Resumable Hyperband on a fixed candidate set: brackets of successive halving that start at different budgets,
    sharing training progress (a candidate trained by one bracket is not retrained by another)."""
    finalists = []
    for t0 in t0s:
        finalists += seq_halving(W, lib, rng, t0=t0, eta=eta)
    return sorted(set(finalists))


# ---------- racing skeleton: train survivors in epochs of dt, a rule drops arms ----------
def race(W, lib, rng, rule, t0=250, dt=100, t_cap=1500):
    active, t = list(range(W.N)), t0
    for i in active:
        W.advance(i, t)
    state = {"hist": {i: [] for i in active}, "I": {i: 0.0 for i in active}}
    while len(active) > 1 and t < t_cap:
        drop = rule(W, lib, rng, active, t, state)
        keep = [i for i in active if i not in drop]
        if not keep:
            keep = [max(active, key=lambda i: W.at(i, t))]
        active = keep
        t += dt
        for i in active:
            W.advance(i, t)
    return max(active, key=lambda i: W.at(i, min(t, W.prog[i])))


def rule_human_ref(rho):
    """Absolute threshold: drop an arm whose peak is below rho * (human reward's peak at the same iteration)."""
    def rule(W, lib, rng, active, t, st):
        thr = rho * lib.human[t - 1]
        return {i for i in active if W.at(i, t) < thr}
    return rule


def rule_ratio_bound(q):
    """Safe-ish elimination: the leader's final >= its current peak (peaks never decrease); drop an arm whose
    optimistic final (current peak x q-quantile of library final/peak ratios at t) cannot exceed it."""
    def rule(W, lib, rng, active, t, st):
        rq = np.quantile(lib.R[:, t - 1], q)
        lead = max(W.at(i, t) for i in active)
        return {i for i in active if max(W.at(i, t), C0) * rq < lead and W.at(i, t) < lead}
    return rule


def rule_pid(kp, ki, kd, theta, dt=100):
    """Control signal on each arm's relative lag behind the leader: e = (leader - peak)/max(leader, 1).
    u = kp*e + ki*sum(e)*dt/1000 + kd*de/dt*1000. Drop when u > theta."""
    def rule(W, lib, rng, active, t, st):
        lead = max(W.at(i, t) for i in active)
        drop = set()
        for i in active:
            e = (lead - W.at(i, t)) / max(lead, 1.0)
            h = st["hist"][i]
            de = (e - h[-1]) / dt * 1000 if h else 0.0
            st["I"][i] += e * dt / 1000
            h.append(e)
            if kp * e + ki * st["I"][i] + kd * de > theta and e > 0:
                drop.add(i)
        return drop
    return rule


def draws(W, lib, rng, idx, M=96):
    """Posterior draws of each arm's final peak from library ratios (analog bootstrap): final = max(peak,C0) * R[l, t]."""
    base = np.array([max(W.at(i), C0) for i in idx])
    t_idx = np.array([W.prog[i] - 1 for i in idx])
    ell = rng.integers(lib.R.shape[0], size=(M, len(idx)))
    d = base[None, :] * lib.R[ell, t_idx[None, :]]
    return np.maximum(d, np.array([W.at(i) for i in idx])[None, :])


def rule_bayes(delta, M=96):
    """Uncertainty-based pruning: drop an arm whose posterior probability of being the best final is < delta."""
    def rule(W, lib, rng, active, t, st):
        d = draws(W, lib, rng, active, M)
        p = np.bincount(d.argmax(1), minlength=len(active)) / M
        return {i for i, pi in zip(active, p) if pi < delta}
    return rule


def rule_mpc(lam, horizon=500, M=96):
    """Receding-horizon planner. For each candidate plan 'keep the top-m arms for another `horizon` iterations',
    predict J(m) = E[best final among kept] - lam * (m-1) * horizon / T with the library forecast model; apply the best
    plan's first step (drop the rest), re-plan next epoch."""
    def rule(W, lib, rng, active, t, st):
        d = draws(W, lib, rng, active, M)
        order = np.argsort(-np.median(d, axis=0))
        J = [d[:, order[:m]].max(1).mean() - lam * (m - 1) * horizon / T for m in range(1, len(active) + 1)]
        m = int(np.argmax(J)) + 1
        return {active[k] for k in order[m:]}
    return rule


def ttts(W, lib, rng, t0=250, dt=100, delta=0.1, budget=0.3, M=96):
    """Best-arm identification by top-two Thompson sampling on library-forecast posteriors.
    Asynchronous: each step trains one arm for dt iterations. Stops when P(best) > 1-delta or the budget is spent."""
    idx = list(range(W.N))
    for i in idx:
        W.advance(i, t0)
    while W.cost < budget * W.N * T:
        d = draws(W, lib, rng, idx, M)
        p = np.bincount(d.argmax(1), minlength=W.N) / M
        if p.max() >= 1 - delta:
            break
        lead = rng.integers(M)
        a = int(d[lead].argmax())
        if rng.random() < 0.5:
            pick = a
        else:
            pick = a
            for _ in range(50):
                pick = int(d[rng.integers(M)].argmax())
                if pick != a:
                    break
        if W.prog[pick] >= T:
            break
        W.advance(pick, W.prog[pick] + dt)
    d = draws(W, lib, rng, idx, 256)
    return int(np.bincount(d.argmax(1), minlength=W.N).argmax())


def hyperband_paper(W, lib, rng, R=T, eta=3, r_min=37):
    """Hyperband as in Li et al. (JMLR 2018, Alg. 1): brackets s = s_max..0, each samples n configs and runs successive halving
    from r = R*eta^-s up to R, every bracket from scratch. Here the 'configurations' are the N given rewards, so a bracket asking
    for n >= N configs gets all N (with N <= 12 every bracket but the last does): HB then reduces to several SH runs with
    different starting budgets, each paid from scratch."""
    s_max = int(np.floor(np.log(R / r_min) / np.log(eta) + 1e-9))
    B = (s_max + 1) * R
    spent, winners = 0, []
    for s in range(s_max, -1, -1):
        n = int(np.ceil(B / R / (s + 1) * eta ** s))
        r = R * eta ** (-s)
        active = list(rng.permutation(W.N)[: min(n, W.N)])
        prev = 0
        for i in range(s + 1):
            r_i = int(round(min(R, r * eta ** i)))
            for c in active:
                W.advance(c, r_i)
            spent += len(active) * (r_i - prev)
            prev = r_i
            n_keep = max(1, int(np.floor(n * eta ** (-(i + 1)))))
            active = sorted(active, key=lambda c: -W.at(c, r_i))[: min(n_keep, len(active))] if i < s else active
        winners += [int(c) for c in active]
    W.extra = spent - int(W.prog.sum())
    return sorted(set(winners))


def rule_voi(lam, horizon=500, M=96):
    """Value of information: keep training a non-leader only if its expected improvement over the leader's final
    (library-forecast posterior) exceeds the price of `horizon` more iterations, lam * horizon / T points."""
    def rule(W, lib, rng, active, t, st):
        d = draws(W, lib, rng, active, M)
        lead = int(np.argmax(np.median(d, axis=0)))
        return {i for k, i in enumerate(active)
                if k != lead and np.maximum(d[:, k] - d[:, lead], 0).mean() < lam * horizon / T}
    return rule
