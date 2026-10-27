"""LLM-as-judge from early training logs (pre-registered in results/pruning/llm_judge_prereg.md before any call).

For each held-out candidate set (Ant, seeds 7-9, all 35 four-reward subsets of the 7 strong rewards = 105 sets) the LLM sees
the task description, each candidate's reward code, and its first 500 PPO iterations of logs (task score and every reward
component, sampled every 50 iterations, Eureka feedback style), in a shuffled order with neutral labels. It must name the
candidate that will reach the highest task score after 3,000 iterations. The pick is then 'finished'; compute equals fixed@500
(4 x 500 + 2,500 iterations), so the fair comparison is against fixed@500 at equal compute.

  python llm_judge.py run      (MiniMax M3.1 Flash via Vercel; needs EUREKA_ENDPOINT=vercel EUREKA_MAX_TOKENS=32768)
  python llm_judge.py score    (results/pruning/llm_judge.json)
"""
import concurrent.futures as cf
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np

import eureka_ab as e

R = e.ROOT / "results"
OUT = R / "pruning/llm_judge"
SEEDS = (7, 8, 9)
KEYS = {"A_round2c0_10.93": "exp_b_minimax/round2/cand0", "B_round2c1_10.75": "exp_b_minimax/round2/cand1",
        "C_pool_c3_10.73": "exp_d_pool_minimax/round1/cand3", "D_live2_r2c2_10.43": "exp_e_live/trial2/round2/cand2",
        "E_live2_r1c3_9.62": "exp_e_live/trial2/round1/cand3", "F_live2_r2c3_9.61_winner": "exp_e_live/trial2/round2/cand3",
        "G_live1_r2c0_9.29_winner": "exp_e_live/trial1/round2/cand0"}
SKIP = {"gt_reward", "episode_lengths"}
SYSTEM = ("You are an expert in reinforcement learning and reward design. Several candidate reward functions for the same task are "
          "each being used to train a PPO policy (identical hyperparameters, 3,000 iterations planned). You see each reward's code and "
          "its first 500 iterations of training logs. Predict which candidate will reach the HIGHEST task_score (the environment's own "
          "forward-progress metric, not the candidate's own reward) by iteration 3,000.")


def curves(key, seed):
    return json.loads((R / f"exp_f_seeds/seed{seed}/{key}/curves_full.json").read_text())


def block(label, key, seed):
    code = (R / KEYS[key] / "env_rewardonly.py").read_text()
    c = curves(key, seed)
    lines = []
    for m, v in c.items():
        if m in SKIP or len(v) < 500:
            continue
        name = "task_score" if m == "consecutive_successes" else ("candidate_reward_total" if m == "gpt_reward" else m)
        vals = [f"{x:.2f}" for x in v[:500][::50]]
        lines.append(f"{name}: {vals}, max {max(v[:500]):.2f}")
    return f"### Candidate {label}\n```python\n{code}```\nTraining log, iterations 1-500 (every 50 iterations):\n" + "\n".join(lines)


def sets():
    out = []
    for s in SEEDS:
        for sub in itertools.combinations(sorted(KEYS), 4):
            order = list(np.random.default_rng(hash((s, sub)) % 2 ** 32).permutation(sub))
            out.append({"seed": s, "order": order, "id": f"s{s}_" + "_".join(k[0] for k in sub)})
    return out


def prompt(st):
    desc = "make the ant run forward as fast as possible"
    blocks = "\n\n".join(block(f"C{i + 1}", k, st["seed"]) for i, k in enumerate(st["order"]))
    user = (f"Task: {desc} (Isaac Gym Ant).\n\n{blocks}\n\nWhich candidate will reach the highest task_score by iteration 3,000? "
            f"Reply with a JSON object {{\"ranking\": [best, ..., worst]}} using the labels C1-C{len(st['order'])}.")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    todo = [st for st in sets() if not (OUT / f"{st['id']}.json").exists()]

    def one(st):
        msgs = prompt(st)
        payload, raw, attempt, errors, seconds = e.llm(msgs)
        rec = {"set": st, "attempts": attempt, "errors": errors, "seconds": seconds}
        if raw:
            body = json.loads(raw)
            rec["usage"] = body.get("usage")
            rec["text"] = ((body.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        e.save_json(OUT / f"{st['id']}.json", rec)
        return rec
    with cf.ThreadPoolExecutor(8) as ex:
        for r in ex.map(one, todo):
            print(r["set"]["id"], (r.get("usage") or {}).get("completion_tokens"), flush=True)


def score():
    sys.path.insert(0, str(e.ROOT / "pruning"))
    recs, usd = [], 0.0
    for f in sorted(OUT.glob("*.json")):
        r = json.loads(f.read_text())
        u = r.get("usage") or {}
        usd += u.get("prompt_tokens", 0) * 0.15e-6 + u.get("completion_tokens", 0) * 0.6e-6
        st, txt = r["set"], r.get("text") or ""
        labs = re.findall(r"C(\d)", txt[txt.rfind("ranking"):] if "ranking" in txt else txt)
        pick = st["order"][int(labs[0]) - 1] if labs and 1 <= int(labs[0]) <= 4 else None
        fin = {k: max(curves(k, st["seed"])["consecutive_successes"]) for k in st["order"]}
        at500 = {k: max(curves(k, st["seed"])["consecutive_successes"][:500]) for k in st["order"]}
        best = max(fin.values())
        fx = max(at500, key=at500.get)
        recs.append({"seed": st["seed"], "parsed": pick is not None,
                     "llm_retained": fin[pick] / best if pick else fin[fx] / best,  # unparsable reply -> fall back to fixed@500
                     "llm_hit": float(pick is not None and fin[pick] == best), "fixed_retained": fin[fx] / best, "fixed_hit": float(fin[fx] == best)})
    d = np.array([x["llm_retained"] - x["fixed_retained"] for x in recs])
    by = {}
    for x, v in zip(recs, d):
        by.setdefault(x["seed"], []).append(v)
    rng = np.random.default_rng(0)
    bs = np.array([np.concatenate([rng.choice(by[s], len(by[s])) for s in rng.choice(list(by), len(by))]).mean() for _ in range(10000)])
    out = {"label": "exploratory pre-registered test (actual LLM calls on retrospective held-out curves)", "n_sets": len(recs),
           "parsed": sum(x["parsed"] for x in recs), "usd": usd,
           "llm_retained": float(np.mean([x["llm_retained"] for x in recs])), "llm_hit": float(np.mean([x["llm_hit"] for x in recs])),
           "fixed500_retained": float(np.mean([x["fixed_retained"] for x in recs])), "fixed500_hit": float(np.mean([x["fixed_hit"] for x in recs])),
           "diff_retained": float(d.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "p_one_sided": float((bs <= 0).mean())}
    json.dump(out, open(R / "pruning/llm_judge.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"run": run, "score": score}[sys.argv[1]]()
