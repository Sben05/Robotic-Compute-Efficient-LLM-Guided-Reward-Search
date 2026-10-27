"""A-5: live confirmation of a frozen pruning rule. Candidates train CONCURRENTLY and continuously (no resume); a controller pauses
each process (SIGSTOP) when its logged training reaches the rule's next checkpoint, applies the frozen rule to exactly the prefixes
seen so far, kills (SIGKILL) the candidates the rule drops and continues (SIGCONT) the rest. The selected candidate trains to 3,000.
Afterwards the same rule is replayed on the recorded prefixes; live and replay decisions must match.

  python live_prune.py OUTDIR SEED "RULE LABEL" LIB.npz SRC_DIR...
  python live_prune.py pack BATCH.json      (runner: jobs = [[OUTDIR, SEED, RULE, LIB, SRC...]])
LIVE_SMOKE=1 stops after the decision (machinery test; the pick is not trained on).
RULE LABEL is a config label from pruning/configs.py (as frozen in results/pruning/frozen_rules.json).
LIB.npz: library curves (running best, leave-reward-out: no curve of these rewards) + human curves, built on the Mac.
"""
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "pruning"))
import eureka_ab as e  # noqa: E402
import policies as P  # noqa: E402
from configs import build  # noqa: E402
from sim_core import Library, World, run_policy  # noqa: E402

T = 3000


class Arm:
    def __init__(self, out, src, seed, k):
        self.dir = out / f"arm{k}_{src.name}"
        self.dir.mkdir(parents=True, exist_ok=True)
        gen = e.load_json(src / "generation.json")
        base = gen["validation"]["suffix"]
        suffix = base + f"P{seed}"
        self.task = "Ant" + suffix
        for f in ("generation.json", "env_generated.py", "env_rewardonly.py"):
            (self.dir / f).write_text((src / f).read_text())
        (self.dir / "source.txt").write_text(str(src) + "\n")
        from utils.create_task import create_task
        create_task(str(e.ISAAC), "Ant", "ant", suffix)
        self.created = [e.ISAAC / f"cfg/task/{self.task}.yaml", e.ISAAC / f"cfg/train/{self.task}PPO.yaml",
                        e.ISAAC / f"tasks/ant{suffix.lower()}.py"]
        self.created[2].write_text((src / "env_generated.py").read_text().replace("Ant" + base, self.task))
        cmd = [sys.executable, "-u", str(e.ISAAC / "train.py"), "hydra/output=subprocess", f"task={self.task}", "wandb_activate=False",
               "wandb_entity=", "wandb_project=", "headless=True", "capture_video=False", "force_render=False",
               f"max_iterations={T}", f"seed={seed}", f"num_envs={e.NUM_ENVS}"]
        self.log = open(self.dir / "train.log", "w")
        self.log.write("Command: " + " ".join(cmd) + "\n"); self.log.flush()
        env = dict(os.environ, PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""))  # ninja for gymtorch
        self.proc = subprocess.Popen(cmd, cwd=str(self.dir), stdout=self.log, stderr=subprocess.STDOUT, env=env)
        self.tb, self.curve, self.target, self.paused, self.killed, self.t0 = None, [], 0, False, False, time.time()
        self.events = []

    def refresh(self):
        if self.tb is None:
            m = re.search(r"^Tensorboard Directory: (.*)$", (self.dir / "train.log").read_text(), re.M)
            if not m:
                return
            self.tb = m.group(1).strip()
        from utils.file_utils import load_tensorboard_logs
        try:
            self.curve = list(load_tensorboard_logs(self.tb).get("consecutive_successes", []))
        except Exception:
            pass

    def signal(self, sig, what):
        if self.proc.poll() is None:
            self.proc.send_signal(sig)
        self.events.append({"t": time.time(), "what": what, "iters_logged": len(self.curve)})

    def cleanup(self):
        for p in self.created:
            p.unlink(missing_ok=True)


class LiveWorld(World):
    """World API over live processes: advance() sets a target and resumes; at()/cm() block until the target is logged."""

    def __init__(self, arms):
        self.arms = arms
        self.N = len(arms)
        self.prog = np.zeros(self.N, int)
        self.extra = 0
        self.lock = threading.Lock()
        threading.Thread(target=self._watch, daemon=True).start()

    def _watch(self):
        while True:
            for a in self.arms:
                if a.killed or a.proc.poll() is not None:
                    continue
                a.refresh()
                if not a.paused and a.target < T and len(a.curve) >= a.target:
                    a.signal(signal.SIGSTOP, f"pause at target {a.target}"); a.paused = True
            time.sleep(1.0)

    def advance(self, i, to):
        to = min(int(to), T)
        a = self.arms[i]
        if a.killed or to <= a.target:
            return
        a.target = to
        self.prog[i] = to
        if a.paused:
            a.paused = False
            a.signal(signal.SIGCONT, f"continue to {to}")

    def _wait(self, i, t):
        a = self.arms[i]
        while len(a.curve) < t:
            if a.proc.poll() is not None:
                a.refresh()
                if len(a.curve) < t:
                    raise RuntimeError(f"arm {i} exited at {len(a.curve)} < {t}")
            time.sleep(0.5)

    def cm(self, i):
        self._wait(i, self.prog[i])
        return np.maximum.accumulate(np.asarray(self.arms[i].curve[: self.prog[i]], float))

    def at(self, i, t=None):
        t = self.prog[i] if t is None else t
        assert 1 <= t <= self.prog[i]
        self._wait(i, t)
        return float(np.max(self.arms[i].curve[:t]))

    def finish(self, i):
        if os.environ.get("LIVE_SMOKE"):  # smoke test: stop after the decision instead of training the pick to 3,000
            self.kill(i, "smoke test: selected, not finished")
            return float(np.max(self.arms[i].curve)) if self.arms[i].curve else -1e9
        self.advance(i, T)
        a = self.arms[i]
        a.proc.wait()
        a.refresh()
        return float(np.max(a.curve)) if a.curve else -1e9

    def kill(self, i, why):
        a = self.arms[i]
        if not a.killed and a.proc.poll() is None:
            a.signal(signal.SIGKILL, f"KILL ({why}) at {len(a.curve)} logged")
            a.proc.wait()
        a.killed = True

    def truth(self):
        return np.array([max(a.curve) if a.curve else -1e9 for a in self.arms])


def wrap_rule(kw, W):
    """Kill dropped arms the moment a race rule drops them."""
    if "rule" not in kw:
        return kw
    inner = kw["rule"]

    def rule(Wd, lib, rng, active, t, st):
        drop = inner(Wd, lib, rng, active, t, st)
        for i in drop:
            W.kill(i, f"rule dropped at checkpoint {t}")
        return drop
    return dict(kw, rule=rule)


def main():
    out, seed, label, libf = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
    srcs = [Path(s) for s in sys.argv[5:]]
    out.mkdir(parents=True, exist_ok=True)
    cfg = {lab: (pol, kw) for fam, lab, pol, kw in build()}
    pol, kw = cfg[label]
    z = np.load(libf)
    lib = Library(list(z["lib"]), list(z["human"]))
    t_start = time.time()
    arms = [Arm(out, s, seed, k) for k, s in enumerate(srcs)]
    W = LiveWorld(arms)
    rng = np.random.default_rng(11)
    try:
        sel = pol(W, lib, rng, **wrap_rule(kw, W))
        sel = [sel] if np.isscalar(sel) else list(sel)
        for i in range(W.N):
            if i not in sel:
                W.kill(i, "not selected")
        finals = {i: W.finish(i) for i in sel}
        pick = max(finals, key=finals.get)
    finally:
        for a in arms:
            if a.proc.poll() is None:
                a.proc.kill()
            a.cleanup()
            a.log.close()
    wall = time.time() - t_start
    for a in arms:
        a.refresh()
    iters = [len(a.curve) for a in arms]
    # replay: same rule on the recorded prefixes (NaN beyond what was trained -> any peek past it would surface as an error)
    raw = np.full((len(arms), T), np.nan)
    for k, a in enumerate(arms):
        raw[k, : len(a.curve)] = a.curve
    Wr = World(np.where(np.isnan(raw), -1e9, raw))
    if os.environ.get("LIVE_SMOKE"):
        Wr.finish = lambda i: float(np.max(Wr._cm[i, : max(1, Wr.prog[i])]))
    sel_r = pol(Wr, lib, np.random.default_rng(11), **kw)
    sel_r = [sel_r] if np.isscalar(sel_r) else list(sel_r)
    pick_r = max(sel_r, key=lambda i: Wr.finish(i))
    res = {"label": "actual run (live pruning, processes killed)", "rule": label, "seed": seed, "sources": [str(s) for s in srcs],
           "pick_live": int(pick), "pick_replay": int(pick_r), "selected_live": [int(i) for i in sel], "selected_replay": [int(i) for i in sel_r],
           "replay_matches_live": bool(sorted(sel) == sorted(sel_r) and pick == pick_r and list(Wr.prog) == list(W.prog)),
           "replay_within_recorded_prefixes": bool(all(Wr.prog[k] <= iters[k] for k in range(len(arms)))),
           "replay_prog": [int(x) for x in Wr.prog], "live_targets": [int(x) for x in W.prog],
           "iterations_logged_per_arm": iters, "total_iterations": int(sum(iters)), "train_all_iterations": len(arms) * T,
           "compute_fraction": sum(iters) / (len(arms) * T), "pick_peak": float(max(arms[pick].curve)), "wall_seconds": wall,
           "events": {a.dir.name: a.events for a in arms}}
    for a in arms:
        e.save_json(a.dir / "curves_partial.json", {"consecutive_successes": a.curve})
    e.save_json(out / "live_prune_result.json", res)
    print(json.dumps({k: v for k, v in res.items() if k != "events"}, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "pack":  # runner entry point: the batch's single job is this script's argument list
        sys.argv = [sys.argv[0]] + [str(x) for x in json.loads(Path(sys.argv[2]).read_text())[0]]
    main()
