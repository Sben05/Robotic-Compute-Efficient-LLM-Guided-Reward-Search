"""Blind LLM pre-ranking of the 16-sample pool (exploratory; our method, not upstream Eureka).

The arm's LLM sees only the task context and the reward code (shuffled, anonymised) and predicts which
rewards will train best. Run before any pool training results exist; the ranking is saved and
evaluated later against the actual 3,000-iteration task scores.
"""
import json
import random
import re

import yaml

import eureka_ab as e
import pool

OUT = e.OUT / "llm_ranking"  # e.OUT is the pool directory once pool is imported
POOL = [c for d in pool.POOL_DIRS for c in sorted((e.ROOT / "results" / d).glob("cand*"))]
REPEATS = 3

SYSTEM = ("You are an expert in reinforcement learning reward design. You will be shown several candidate "
          "reward functions for the same task. Each will be used, unchanged, to train a PPO policy for 3000 "
          "iterations with identical hyperparameters. Predict which rewards will produce the best task "
          "performance, where task performance is measured by the environment's own forward-progress "
          "metric, not by the candidate reward itself.")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    p = e.prompt_parts()
    cands = [c for c in POOL if e.load_json(c / "generation.json")["validation"]["valid"]]
    order = list(range(len(cands)))
    random.Random(0).shuffle(order)
    labels = {"R%02d" % k: str(cands[i].relative_to(e.ROOT / "results")) for k, i in enumerate(order)}
    blocks = ["### %s\n```python\n%s```" % (lab, (e.ROOT / "results" / path / "env_rewardonly.py").read_text())
              for lab, path in labels.items()]
    user = ("Task: %s.\n\nEnvironment code:\n%s\n\nCandidate reward functions:\n\n%s\n\n"
            "Rank ALL %d candidates from best to worst expected task performance. Reply with only a JSON "
            "list of their labels, best first." % (yaml.safe_load((e.EUREKA / "cfg/env/ant.yaml").read_text())["description"],
                                                   p["obs"], "\n\n".join(blocks), len(labels)))
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    e.save_json(OUT / "messages.json", messages)
    e.save_json(OUT / "labels.json", labels)
    rankings = []
    for r in range(REPEATS):
        payload, raw, attempt, errors, seconds = e.llm(messages)
        rec = {"attempts": attempt, "errors": errors, "seconds": seconds}
        if raw:
            text, finish, usage, used = e.parse_response(raw)
            text = text or ""
            rec.update(usage=usage, model_used=used, finish_reason=finish, text=text)
            found = re.findall(r"R\d\d", text)
            ranking = list(dict.fromkeys(x for x in found if x in labels))
            rec["ranking"] = ranking
            if len(ranking) == len(labels):
                rankings.append(ranking)
        e.save_json(OUT / ("response%d.json" % r), rec)
    # Mean rank across complete rankings (Borda); ties broken by label.
    mean_rank = {lab: sum(rk.index(lab) for rk in rankings) / len(rankings) for lab in labels} if rankings else {}
    consensus = sorted(mean_rank, key=lambda lab: (mean_rank[lab], lab))
    e.save_json(OUT / "ranking.json", {"complete_rankings": len(rankings), "mean_rank": mean_rank,
                                       "consensus": [labels[lab] for lab in consensus]})
    print(json.dumps([labels[lab] for lab in consensus], indent=1))


if __name__ == "__main__":
    main()
