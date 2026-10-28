"""Fresh matched A/B batching-quality replication (STATS_PLAN_AMENDMENT2.md), Ant/MiniMax.

Group A (indep): 36 one-reward calls. Group B (batch4): 9 four-reward calls (36 rewards). 3 blocks of
(12 indep + 3 batch4), each block's calls issued through one shared thread pool so both arms share the
same wall-clock window. No candidate from exp_h_llm_trial / exp_b / exp_d / exp_e is reused.

  python batch_quality.py generate BLOCK     (Mac; BLOCK in 0,1,2; needs EUREKA_ENDPOINT=vercel; idempotent)
  python batch_quality.py build              (GPU box; validates every reply -> candidate dirs, screen_jobs.json)
  python batch_quality.py screen             (GPU box; packs the 500-it screens, LIVE_WORKERS at a time)
  python batch_quality.py select_resume K    (GPU box; top-K per arm by 500-it score -> resumes to 3,000)
"""
import concurrent.futures as cf
import json
import os
import sys
import time
from pathlib import Path

import eureka_ab as e
import gen_task as g
import live as L

OUT = e.ROOT / "results/exp_l_batch_quality"
TASK = "ant"
PER_BLOCK_INDEP, PER_BLOCK_BATCH = 12, 3  # 3 blocks -> 36 indep calls, 9 batch4 calls (36 rewards each arm)
g.OUT = OUT  # gen_task.build()/prompts() write under our tree, not exp_i_generalize


def generate_block(block):
    assert os.environ.get("EUREKA_ENDPOINT") == "vercel", "set EUREKA_ENDPOINT=vercel (only key present)"
    msgs, _ = g.prompts(TASK)
    batch_msgs = [msgs[0], {"role": "user", "content": msgs[1]["content"] + "\n\n" + g.BATCH}]
    d_i, d_b = OUT / TASK / "indep", OUT / TASK / "batch4"
    d_i.mkdir(parents=True, exist_ok=True)
    d_b.mkdir(parents=True, exist_ok=True)
    e.save_json(d_i / "messages.json", msgs)
    e.save_json(d_b / "messages.json", batch_msgs)
    todo = [(d_i / f"reply{i}.json", msgs) for i in range(block * PER_BLOCK_INDEP, (block + 1) * PER_BLOCK_INDEP)]
    todo += [(d_b / f"reply{i}.json", batch_msgs) for i in range(block * PER_BLOCK_BATCH, (block + 1) * PER_BLOCK_BATCH)]
    todo = [(f, m) for f, m in todo if not f.exists()]
    if not todo:
        print(f"block {block}: all calls already exist")
        return

    def one(f, m):
        payload, raw, attempt, errors, seconds = e.llm(m)
        rec = {"block": block, "attempts": attempt, "errors": errors, "seconds": seconds, "request": payload, "submitted_at": time.time()}
        if raw:
            rec["response"] = json.loads(raw)
        e.save_json(f, rec)
        print(f.relative_to(OUT), "ok" if raw else "FAILED", flush=True)
    with cf.ThreadPoolExecutor(min(8, len(todo))) as ex:
        list(ex.map(lambda fm: one(*fm), todo))
    print(f"block {block}: {len(todo)} new calls issued")


def build():
    g.build(TASK)  # writes OUT/ant/{indep,batch4}/call*/, build_summary.json, valid_candidates.json
    valid = json.loads((OUT / TASK / "valid_candidates.json").read_text())
    jobs = [[c, c, 0, "s1", 500, 3000] for c in valid]
    e.save_json(OUT / "screen_jobs.json", jobs)
    print(f"{len(valid)} valid candidates -> screen_jobs.json")


def screen():
    jobs = json.loads((OUT / "screen_jobs.json").read_text())
    print(f"packing {len(jobs)} 500-it screens, LIVE_WORKERS={os.environ.get('LIVE_WORKERS', '4')}")
    L.pack(jobs)


def select_resume(k):
    scored = []
    for arm in ("indep", "batch4"):
        for c in sorted((OUT / TASK / arm).glob("call*")):
            r = c / "result.json"
            if not r.exists():
                continue
            res = json.loads(r.read_text())
            if res.get("trained") and res.get("peak_task_score") is not None:
                scored.append((arm, str(c), res["peak_task_score"]))
    picks = []
    for arm in ("indep", "batch4"):
        top = sorted([s for s in scored if s[0] == arm], key=lambda x: -x[2])[:k]
        picks += top
        print(arm, [(Path(p).name, round(s, 2)) for _, p, s in top])
    jobs = [[p, p, 0, "cont"] for _, p, _ in picks]
    e.save_json(OUT / "resume_jobs.json", jobs)
    print(f"{len(jobs)} finalists -> resume_jobs.json")
    L.pack(jobs)


def run_gpu(k):
    """One sbatch job: screen every valid candidate at 500 its, then resume the top K per arm to 3,000. Idempotent
    (pack()/phase() skip anything with an existing result.json), so a requeue after a time-limit kill just resumes."""
    screen()
    select_resume(k)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "generate":
        generate_block(int(sys.argv[2]))
    elif cmd == "build":
        build()
    elif cmd == "screen":
        screen()
    elif cmd == "select_resume":
        select_resume(int(sys.argv[2]))
    elif cmd == "run_gpu":
        run_gpu(int(sys.argv[2]) if len(sys.argv) > 2 else 4)
    else:
        sys.exit("unknown subcommand: " + cmd)
