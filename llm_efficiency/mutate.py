"""E3: programmatic ('local') reward mutations: zero LLM tokens. Operators act on named float constants.
  scale_const   : multiply 1-3 random named float constants by a log-uniform factor in [0.5, 2]
  desaturate    : feedback rule. If the feedback shows a tanh-shaped component with mean ~> 0.85 of its max, widen that
                  tanh's temperature by x2.5 (the edit both Round-2 LLM samples made to the Round-1 winner)
Validity = TorchScript compiles, outputs finite, reward not constant, on synthetic Ant states (CPU, no simulator).
This measures validity and reward-shape change only. It does NOT show a mutated reward trains better."""
import importlib.util, json, re, sys, tempfile, uuid
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from common import *

HEADER = "from typing import Dict, Tuple\nimport torch\n\n\n"
FLOAT_ASSIGN = re.compile(r"^(?P<ind>\s+)(?P<name>[A-Za-z_]\w*)(?P<ann>\s*:\s*float)?\s*=\s*(?P<val>-?\d+\.\d*(?:e-?\d+)?|-?\d+e-?\d+)(?P<rest>\s*(#.*)?)$")
SIG_DEFAULT = re.compile(r"(?P<name>[A-Za-z_]\w*)\s*:\s*float\s*=\s*(?P<val>-?\d+\.?\d*(?:e-?\d+)?)")
TMP = Path(tempfile.mkdtemp(prefix="mut_"))


def named_consts(src):
    out = []
    for i, line in enumerate(src.split("\n")):
        m = FLOAT_ASSIGN.match(line)
        if m and float(m["val"]) != 0.0:
            out.append((i, m["name"], float(m["val"])))
    return out


def set_const(src, line_idx, name, new):
    lines = src.split("\n")
    m = FLOAT_ASSIGN.match(lines[line_idx])
    lines[line_idx] = f"{m['ind']}{name}{m['ann'] or ''} = {new:.6g}{m['rest']}"
    return "\n".join(lines)


def scale_const(src, rng, k=None):
    cs = named_consts(src)
    if not cs:
        return None, []
    k = k or int(rng.integers(1, 4))
    pick = rng.choice(len(cs), size=min(k, len(cs)), replace=False)
    log = []
    for j in pick:
        i, name, v = cs[j]
        f = float(np.exp(rng.uniform(np.log(0.5), np.log(2))))
        src = set_const(src, i, name, v * f)
        log.append((name, v, v * f))
    return src, log


def tanh_temps(src):
    """names of temperature variables used as torch.tanh(<expr> / NAME)"""
    return sorted(set(re.findall(r"torch\.tanh\([^()]*?/\s*([A-Za-z_]\w*)\s*\)", src)))


def desaturate(src, factor=2.5):
    cs = {n: (i, v) for i, n, v in named_consts(src)}
    log = []
    for t in tanh_temps(src):
        if t in cs:
            i, v = cs[t]
            src = set_const(src, i, t, v * factor)
            log.append((t, v, v * factor))
    return (src if log else None), log


def load_fn(src):
    p = TMP / f"m_{uuid.uuid4().hex}.py"
    p.write_text(HEADER + src)
    spec = importlib.util.spec_from_file_location(p.stem, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.compute_reward


def states(n=2048, speed=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)
    rs = torch.zeros(n, 13)
    rs[:, 2] = 0.55 + 0.1 * r(n)
    tilt = 0.25 * r(n, 3)
    q = torch.cat([tilt, torch.ones(n, 1)], 1)
    rs[:, 3:7] = q / q.norm(dim=1, keepdim=True)  # xyzw
    rs[:, 7] = speed if speed is not None else torch.rand(n, generator=g) * 12 - 1
    rs[:, 8:10] = 0.3 * r(n, 2)
    rs[:, 10:13] = r(n, 3)
    up = torch.cat([tilt[:, :2], torch.ones(n, 1)], 1)
    lo, hi = -0.7 * torch.ones(8), 0.7 * torch.ones(8)
    hv = torch.zeros(n, 3); hv[:, 0] = 1
    return dict(root_states=rs, actions=2 * torch.rand(n, 8, generator=g) - 1, dof_vel=5 * r(n, 8), dof_pos=0.5 * r(n, 8),
                up_vec=up / up.norm(dim=1, keepdim=True), heading_vec=hv, targets=torch.tensor([1000., 0, 0]).expand(n, 3).clone(),
                dof_limits_lower=lo, dof_limits_upper=hi, dt=1 / 60, potentials=torch.zeros(n), prev_potentials=torch.zeros(n))


def run(fn, st):
    names = [a.name for a in fn.schema.arguments]
    return fn(**{k: st[k] for k in names if k in st})[0]


def evaluate(src, st):
    try:
        fn = load_fn(src)
        r = run(fn, st)
    except Exception as e:  # compile / runtime failure
        return None, f"{type(e).__name__}"
    if not torch.isfinite(r).all():
        return None, "nonfinite"
    if r.std() < 1e-8:
        return None, "constant"
    return r, ""


def speed_slope_ratio(src):
    """(R(10)-R(6))/(R(4)-R(0)): ~0 if the reward is flat in speed above 6 m/s (saturated)"""
    vals = {}
    fn = load_fn(src)
    for v in (0, 4, 6, 10):
        r = run(fn, states(512, speed=float(v), seed=1))
        vals[v] = r.mean().item()
    lo = (vals[4] - vals[0]) / 4
    return ((vals[10] - vals[6]) / 4) / lo if abs(lo) > 1e-9 else float("nan")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    llm, _ = load_curves()
    st = states()
    res = {}
    print(f"{'reward':6s} {'peak':>6s} {'#const':>6s} {'tanhT':>14s} {'valid':>7s} {'rank-corr':>9s}   slope hi/lo: parent -> desaturated")
    for k in LLM:
        src = (ROOT / "results" / LLM[k][0] / "env_rewardonly.py").read_text()
        base, why = evaluate(src, st)
        if base is None:
            print(k, "parent failed in harness:", why); res[k] = {"parent_failed": why}; continue
        ok, rc, fails = 0, [], {}
        N = 200
        for _ in range(N):
            m, log = scale_const(src, rng)
            if m is None:
                break
            r, why = evaluate(m, st)
            if r is None:
                fails[why] = fails.get(why, 0) + 1
            else:
                ok += 1; rc.append(spearmanr(base.numpy(), r.numpy())[0])
        d, dlog = desaturate(src)
        slope0 = speed_slope_ratio(src)
        slope1 = speed_slope_ratio(d) if d else None
        res[k] = {"peak": float(llm[k].max()), "n_named_consts": len(named_consts(src)), "tanh_temps": tanh_temps(src),
                  "scale_valid_frac": ok / N, "scale_fail_reasons": fails, "scale_rank_corr_mean": float(np.mean(rc)) if rc else None,
                  "scale_rank_corr_min": float(np.min(rc)) if rc else None, "slope_parent": slope0, "slope_desaturated": slope1, "desat_edit": dlog}
        print(f"{k:6s} {llm[k].max():6.2f} {len(named_consts(src)):6d} {','.join(tanh_temps(src)) or '-':>14s} {ok/N:7.2f} "
              f"{(np.mean(rc) if rc else float('nan')):9.3f}   {slope0:6.2f} -> {('%.2f' % slope1) if slope1 is not None else '  n/a'}  {dlog}")
    (OUT / "mutation_probe.json").write_text(json.dumps(res, indent=1, default=float))
