"""Experiments E-F: live early stopping (Mini-Eureka with real 500-iteration stops) and multi-seed reproduction.

Reuses eureka_ab.py for prompts, extraction, validation, feedback and task creation. What is new here:

  * Every sampled reward trains for only STOP (500) PPO iterations. Within a round the candidate with the
    highest consecutive_successes reached in those 500 iterations is kept; the rest are stopped for good.
  * Round 2 is written from the kept candidate's 500-iteration training log (upstream feedback format, with
    epoch_freq 50 instead of 300). Nothing from a longer run is used.
  * The best candidate over both rounds (same 500-iteration score) resumes from its checkpoint to 3,000 iterations.
  * Audit (not part of the live search): the stopped candidates are resumed from their 500-iteration checkpoints to 3,000,
    to measure what the stop decisions cost.

A resumed run restores weights, observation/value normalisation and optimiser state (this rl_games fork does not restore
the epoch counter, so the continuation runs FULL-STOP more epochs). A control (human reward, 500 + 2,500) checks the splice.

Subcommands (run by eureka.sbatch):
  trial T N1 N2          full live pipeline for trial T (idempotent), then continue the winner, then audit
  run CDIR SRC SEED PHASE [S1 TOTAL]   one training phase (s1 = first S1 iterations, cont = resume to TOTAL); SRC=human or a candidate dir
  pack JOBS.json         run a list of [cdir, src, seed, phase] jobs, LIVE_WORKERS at a time on this GPU
  seeds S                Experiment F: fixed reward set (results/exp_f_seeds/rewards.json) on seed S, same live protocol
  smoke                  short end-to-end check of the machinery (results/_smoke)
"""
import concurrent.futures as cf
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import eureka_ab as e

STOP, FULL = 500, e.ITERATIONS
BASE = e.ROOT / "results/exp_e_live"
WORKERS = int(os.environ.get("LIVE_WORKERS", "4"))
RULE = "keep the one candidate with the highest consecutive_successes within the first %d iterations; stop the rest" % STOP


def set_trial(t):
    e.OUT = BASE / ("trial%s" % t)
    e.suffix_for = lambda name: "GPTMLT%s" % t + name.replace("round", "R").replace("/cand", "C")
    return e.OUT


# ---------------------------------------------------------------- one training phase

def launch(src, cdir, seed, iters, log, ckpt=None):
    """Run upstream train.py for a reward (src = candidate dir, or None for the human reward) in cdir."""
    cdir.mkdir(parents=True, exist_ok=True)
    if not (cdir / "run_info.txt").exists():
        e.run_info(cdir / "run_info.txt")
    created = []
    if src is None:
        task = "Ant"
    else:
        gen = e.load_json(src / "generation.json")
        base = gen["validation"]["suffix"]
        suffix = base + ("S%d" % seed if seed else "")
        task = "Ant" + suffix
        code = (src / "env_generated.py").read_text().replace("Ant" + base, task)
        for f in ("generation.json", "env_generated.py", "env_rewardonly.py"):
            if src != cdir:
                (cdir / f).write_text((src / f).read_text())
        from utils.create_task import create_task
        create_task(str(e.ISAAC), "Ant", "ant", suffix)
        created = [e.ISAAC / f"cfg/task/{task}.yaml", e.ISAAC / f"cfg/train/{task}PPO.yaml",
                   e.ISAAC / f"tasks/ant{suffix.lower()}.py"]
        created[2].write_text(code)
    cmd = [sys.executable, "-u", str(e.ISAAC / "train.py"), "hydra/output=subprocess", f"task={task}",
           "wandb_activate=False", "wandb_entity=", "wandb_project=", "headless=True", "capture_video=False",
           "force_render=False", f"max_iterations={iters}", f"seed={seed}", f"num_envs={e.NUM_ENVS}"]
    if ckpt:
        cmd.append(f"checkpoint={ckpt}")
    started = time.time()
    try:
        with open(cdir / log, "w") as f:
            f.write("Command: " + " ".join(cmd) + "\n")
            f.flush()
            code = subprocess.run(cmd, cwd=str(cdir), stdout=f, stderr=subprocess.STDOUT).returncode
    finally:
        for path in created:
            path.unlink(missing_ok=True)
    return code, time.time() - started


def summarize(cdir, log, seed, iters, code, seconds, curves_name, result_name):
    import numpy as np
    from utils.file_utils import load_tensorboard_logs
    text = (cdir / log).read_text()
    result = {"trained": True, "exit_code": code, "training_seconds": seconds, "seed": seed,
              "concurrent_runs_on_gpu": int(os.environ.get("PACKED_RUNS", "1")),
              "num_envs": e.NUM_ENVS, "max_iterations": iters, "traceback": e.filter_traceback(text)}
    tb = re.search(r"^Tensorboard Directory: (.*)$", text, re.M)
    if tb:
        logs = load_tensorboard_logs(tb.group(1).strip())
        curves = {k: v for k, v in logs.items() if "/" not in k}
        e.save_json(cdir / curves_name, curves)
        cs = curves.get("consecutive_successes", [])
        result.update(iterations_logged=len(cs), peak_task_score=max(cs) if cs else None,
                      final_task_score=cs[-1] if cs else None,
                      peak_iteration=int(np.argmax(cs)) + 1 if cs else None)
        ckpts = sorted(Path(tb.group(1).strip()).parent.glob("nn/last_*_ep_*.pth"))
        if ckpts:
            result["checkpoint"] = str(ckpts[-1].relative_to(cdir))
    e.save_json(cdir / result_name, result)
    return result


def phase(cdir, src, seed, name, s1=STOP, total=FULL):
    """s1: train from scratch for s1 iterations. cont: resume that run's checkpoint up to `total` iterations."""
    if name == "s1":
        if (cdir / "result.json").exists():
            print(cdir, "stage 1 already done")
            return
        if src is not None:
            gen = e.load_json(src / "generation.json")
            if not gen["validation"]["valid"]:
                cdir.mkdir(parents=True, exist_ok=True)
                e.save_json(cdir / "result.json", {"trained": False, "reason": gen["validation"].get("error")})
                return
        code, sec = launch(src, cdir, seed, s1, "train.log")
        summarize(cdir, "train.log", seed, s1, code, sec, "curves.json", "result.json")
        return
    if (cdir / "result_full.json").exists():
        print(cdir, "already continued")
        return
    r1 = e.load_json(cdir / "result.json")
    done = r1.get("iterations_logged")
    if not (r1.get("trained") and r1.get("exit_code") == 0 and r1.get("checkpoint") and done):
        print(cdir, "no usable checkpoint; not continued")
        return
    code, sec = launch(src, cdir, seed, total - done, "train_cont.log", ckpt=cdir / r1["checkpoint"])
    rc = summarize(cdir, "train_cont.log", seed, total - done, code, sec, "curves_cont.json", "result_cont.json")
    a, b = e.load_json(cdir / "curves.json"), e.load_json(cdir / "curves_cont.json")
    full = {k: a[k] + b[k] for k in a if k in b and len(a[k]) == done}
    e.save_json(cdir / "curves_full.json", full)
    cs = full.get("consecutive_successes", [])
    import numpy as np
    e.save_json(cdir / "result_full.json", {
        "stage1_iterations": done, "continued_iterations": rc.get("iterations_logged"), "iterations_logged": len(cs),
        "exit_code": code, "peak_task_score": max(cs) if cs else None, "final_task_score": cs[-1] if cs else None,
        "peak_iteration": int(np.argmax(cs)) + 1 if cs else None, "score_at_stop": cs[done - 1] if len(cs) >= done else None,
        "training_seconds_stage1": r1.get("training_seconds"), "training_seconds_cont": sec,
        "traceback": rc.get("traceback")})


def pack(jobs):
    """jobs: [cdir, src or 'human', seed, phase(, s1, total)] run as separate processes, WORKERS at a time."""
    env = dict(os.environ, PACKED_RUNS=str(min(WORKERS, len(jobs))))
    def go(j):
        return subprocess.run([sys.executable, "-u", __file__, "run"] + [str(x) for x in j], env=env).returncode
    with cf.ThreadPoolExecutor(WORKERS) as ex:
        codes = list(ex.map(go, jobs))
    print("pack exit codes:", codes, flush=True)


# ---------------------------------------------------------------- live pipeline

def generate_parallel(round_id, n):
    """e.generate with the LLM calls issued concurrently (same requests, same validation, same files)."""
    p = e.prompt_parts()
    rdir = e.OUT / ("round%d" % round_id)
    rdir.mkdir(parents=True, exist_ok=True)
    messages, _ = e.round_messages(round_id, p)
    todo = [i for i in range(n) if not (rdir / ("cand%d" % i) / "generation.json").exists()]
    with cf.ThreadPoolExecutor(max(1, len(todo))) as ex:
        replies = iter(list(ex.map(lambda _: e.llm(messages), todo)))
    real = e.llm
    e.llm = lambda _messages: next(replies)
    try:
        e.generate(round_id, n)
    finally:
        e.llm = real


def scores(rdir, k=STOP):
    """Best consecutive_successes within the first k iterations; None for candidates that did not train to k."""
    out = {}
    for c in sorted(rdir.glob("cand*")):
        res = e.load_json(c / "result.json") if (c / "result.json").exists() else {}
        cur = e.load_json(c / "curves.json").get("consecutive_successes", []) if (c / "curves.json").exists() else []
        ok = res.get("trained") and res.get("exit_code") == 0 and len(cur) >= k
        out[c.name] = max(cur[:k]) if ok else None
    return out


def run_round(r, n):
    rdir = e.OUT / ("round%d" % r)
    sel = rdir / "screen.json"
    if sel.exists():
        return e.load_json(sel)
    generate_parallel(r, n)
    valid = [c for c in sorted(rdir.glob("cand*")) if e.load_json(c / "generation.json")["validation"]["valid"]]
    pack([[c, c, 0, "s1"] for c in valid])
    sc = scores(rdir)
    live = {k: v for k, v in sc.items() if v is not None}
    if not live:
        sys.exit("round %d: no candidate trained to %d iterations (upstream would repeat the round)" % (r, STOP))
    kept = max(live, key=live.get)
    out = {"rule": RULE, "round": r, "scores_at_stop": sc, "kept": kept,
           "stopped": [k for k in live if k != kept], "failed_or_invalid": [k for k, v in sc.items() if v is None],
           "iterations_spent": STOP * len(live)}
    e.save_json(sel, out)
    return out


def trial(t, n1, n2):
    out = set_trial(t)
    rounds = [run_round(1, n1), run_round(2, n2)]
    best = {}
    for s in rounds:
        for k, v in s["scores_at_stop"].items():
            if v is not None:
                best["round%d/%s" % (s["round"], k)] = v
    winner = max(best, key=best.get)
    e.save_json(out / "winner.json", {"rule": "highest 500-iteration score across both rounds", "winner": winner,
                                      "score_at_stop": best[winner], "all": best})
    pack([[out / winner, out / winner, 0, "cont"]])
    audit = [[out / k, out / k, 0, "cont"] for k in best if k != winner]
    pack(audit)  # audit only: resumes the candidates that the live search had already stopped


SEEDS = e.ROOT / "results/exp_f_seeds"


def seeds(s):
    """Experiment F: retrain a fixed reward set on seed s with the same live protocol.
    All rewards (and the human reward) train 500 iterations; the best LLM reward by 500-iteration score is
    the one the live search would keep and is continued first; the rest are stopped-then-resumed audit runs."""
    rewards = e.load_json(SEEDS / "rewards.json")
    root = SEEDS / ("seed%d" % s)
    src = {k: "human" if v == "human" else str(e.ROOT / "results" / v) for k, v in rewards.items()}
    pack([[root / k, v, s, "s1"] for k, v in src.items()])
    sc = scores_of(root, [k for k in src if k != "human"])
    live = {k: v for k, v in sc.items() if v is not None}
    kept = max(live, key=live.get)
    e.save_json(root / "screen.json", {"rule": RULE, "seed": s, "scores_at_stop": sc, "kept": kept,
                                       "human_at_stop": scores_of(root, ["human"])["human"]})
    order = [kept] + [k for k in src if k != kept]
    pack([[root / k, src[k], s, "cont"] for k in order])


def scores_of(root, keys, k=STOP):
    out = {}
    for key in keys:
        c = root / key
        res = e.load_json(c / "result.json") if (c / "result.json").exists() else {}
        cur = e.load_json(c / "curves.json").get("consecutive_successes", []) if (c / "curves.json").exists() else []
        out[key] = max(cur[:k]) if res.get("trained") and res.get("exit_code") == 0 and len(cur) >= k else None
    return out


def smoke():
    root = e.ROOT / "results/_smoke"
    src = e.ROOT / "results/exp_b_minimax/round2/cand0"
    c0 = root / "round1/cand0"
    c0.mkdir(parents=True, exist_ok=True)
    for f in ("generation.json", "env_generated.py", "env_rewardonly.py"):
        (c0 / f).write_text((src / f).read_text())
    jobs = [[root / "human", "human", 0, "s1", 20, 30], [c0, c0, 0, "s1", 20, 30],
            [root / "seed1", c0, 1, "s1", 20, 30]]
    pack(jobs)
    pack([[j[0], j[1], j[2], "cont", 20, 30] for j in jobs])
    for j in jobs:
        print(j[0].name, json.dumps(e.load_json(j[0] / "result_full.json")))
    e.OUT = root
    msgs, sel = e.feedback(root / "round1", e.prompt_parts(), [{"role": "system", "content": "s"}])
    print(json.dumps(sel), "\n", msgs[-1]["content"][:1500])


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "trial":
        trial(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    elif cmd == "run":
        cdir, src = Path(sys.argv[2]), sys.argv[3]
        extra = [int(x) for x in sys.argv[6:8]]
        phase(cdir, None if src == "human" else Path(src), int(sys.argv[4]), sys.argv[5], *extra)
    elif cmd == "pack":
        pack(json.loads(Path(sys.argv[2]).read_text()))
    elif cmd == "seeds":
        seeds(int(sys.argv[2]))
    elif cmd == "smoke":
        smoke()
