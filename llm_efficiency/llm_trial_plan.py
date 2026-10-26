"""PREPARED, NOT RUN. Controlled real-LLM comparison for batching / diversity / reasoning limits. Writes request payloads and a
worst-case budget; spends nothing unless --run is passed together with --max-usd and a key file. Baseline arm A reuses the 9 existing
independent round-1 samples (usage logged), so it costs 0 new calls.

  python llm_trial_plan.py                          # dry run: writes results/llm_efficiency/trial_plan/*.json, prints worst-case cost
  python llm_trial_plan.py --run --max-usd 0.25 --key-file ~/.config/msthesis/vercel_ai_gateway_key   # only after approval
"""
import argparse, json, time, urllib.request
from common import ROOT, OUT

P_IN, P_OUT = 0.15e-6, 0.60e-6
MAX_TOK = 32768  # the Vercel route needs >16k (16k gave empty reasoning-only replies, per project notes)
STRATS = [  # hand-written from the archetypes seen in the 9 independent samples
    "speed-centric: a smooth forward-velocity term, widened so it does not saturate at running speeds, with small regularisers",
    "posture-gated: forward velocity multiplied by an uprightness gate so tumbling cannot be rewarded",
    "progress-based: reward the per-step change in distance to the target (potential difference) plus survival terms",
    "exponential shaping: exp(-error) of the gap between current and a target forward speed, plus heading alignment",
]
base = json.loads((ROOT / "results/exp_b_minimax/round1/cand0/request.json").read_text())
MSG = base["messages"]


def with_user_suffix(text):
    m = [dict(x) for x in MSG]
    m[-1]["content"] = m[-1]["content"] + "\n\n" + text
    return m


ARMS = {}
ARMS["B_batch4"] = [with_user_suffix("Write FOUR different reward functions in this one reply, each in its own ```python block, each named compute_reward_1 ... compute_reward_4.")] * 3
ARMS["C_batch4_diverse"] = [with_user_suffix("Write FOUR reward functions in this one reply, each in its own ```python block, named compute_reward_1 ... compute_reward_4, following these four different strategies in order:\n" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(STRATS)))] * 3
ARMS["D_indep_strategy"] = [with_user_suffix(f"Strategy hint for this reward: {s}.") for s in STRATS] * 3
ARMS["E_indep_short_reasoning"] = [with_user_suffix("Think briefly; answer with only the code.")] * 4  # + reasoning_effort param if the endpoint accepts one (to be checked with 1 call)


def payload(messages):
    return {"model": "minimax/minimax-m3.1-flash", "messages": messages, "temperature": 1.0, "n": 1, "max_tokens": MAX_TOK}


def worst_case(msgs):
    p_tok = sum(len(x["content"]) for x in msgs) / 3.8
    return p_tok * P_IN + MAX_TOK * P_OUT


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true"); ap.add_argument("--max-usd", type=float, default=0.0); ap.add_argument("--key-file")
    a = ap.parse_args()
    (OUT / "trial_plan").mkdir(exist_ok=True)
    plan, total, ncalls = {}, 0.0, 0
    for arm, reqs in ARMS.items():
        wc = sum(worst_case(m) for m in reqs)
        plan[arm] = dict(calls=len(reqs), worst_case_usd=wc, typical_usd=len(reqs) * 0.0045 * (3 if "batch" in arm else 1))
        total += wc; ncalls += len(reqs)
        (OUT / "trial_plan" / f"{arm}.json").write_text(json.dumps([payload(m) for m in reqs], indent=1))
        print(f"{arm:26s} calls {len(reqs):2d}  worst-case ${wc:.3f}  typical ~${plan[arm]['typical_usd']:.3f}")
    print(f"TOTAL calls {ncalls}  worst-case ${total:.2f}  typical ~${sum(v['typical_usd'] for v in plan.values()):.2f}")
    (OUT / "trial_plan" / "budget.json").write_text(json.dumps(plan, indent=1))
    if a.run:
        assert a.key_file and a.max_usd > 0, "--run needs --key-file and --max-usd"  # spend is checked after every call
        key, spent = open(a.key_file).read().strip(), 0.0
        for arm, reqs in ARMS.items():
            for i, m in enumerate(reqs):
                t0 = time.time()
                req = urllib.request.Request("https://ai-gateway.vercel.sh/v1/chat/completions", data=json.dumps(payload(m)).encode(),
                                             headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
                raw = json.loads(urllib.request.urlopen(req, timeout=900).read().decode())
                u = raw["usage"]; spent += u["prompt_tokens"] * P_IN + u["completion_tokens"] * P_OUT
                (OUT / "trial_plan" / f"result_{arm}_{i}.json").write_text(json.dumps(dict(seconds=time.time() - t0, response=raw)))
                if spent > a.max_usd:
                    raise SystemExit(f"budget exceeded: ${spent:.3f}")
