"""E1: where do the LLM tokens/seconds/dollars go, and how much could batching or feedback compression save?
Measured: the 11 real MiniMax generations (usage fields). Prices: MiniMax M3.1 Flash list price (see report)."""
import json

import numpy as np
from scipy.stats import spearmanr

from common import *

P_IN, P_OUT, P_CACHE = 0.15e-6, 0.60e-6, 0.015e-6  # $/token, MiniMax M3.1 Flash list price
GPU_USD_HR = [1.0, 2.5, 4.0]  # assumed A100 rental range; Quest itself is allocation-based, not billed

llm, human = load_curves()
rows = []
for k, (rel, kind) in LLM.items():
    g = generation(k)
    u = g["usage"]
    rows.append(dict(name=k, kind=kind, prompt=u["prompt_tokens"], cached=u["prompt_tokens_details"]["cached_tokens"],
                     comp=u["completion_tokens"], reason=u["completion_tokens_details"]["reasoning_tokens"],
                     code_chars=len(g["response_text"]), sec=g["generation_seconds"], attempts=g["request_attempts"],
                     peak=float(llm[k].max())))
for r in rows:
    r["answer"] = r["comp"] - r["reason"]
    r["usd"] = (r["prompt"] - r["cached"]) * P_IN + r["cached"] * P_CACHE + r["comp"] * P_OUT
    r["usd_nocache"] = r["prompt"] * P_IN + r["comp"] * P_OUT

def m(key, kind=None):
    return float(np.mean([r[key] for r in rows if kind in (None, r["kind"])]))

print(f"{'cand':5s} {'kind':8s} {'prompt':>6s} {'compl':>6s} {'reason':>6s} {'answer':>6s} {'reason%':>7s} {'usd':>8s} {'sec':>6s} {'peak':>6s}")
for r in rows:
    print(f"{r['name']:5s} {r['kind']:8s} {r['prompt']:6d} {r['comp']:6d} {r['reason']:6d} {r['answer']:6d} "
          f"{100*r['reason']/r['comp']:6.1f}% {r['usd']:8.5f} {r['sec']:6.0f} {r['peak']:6.2f}")

fresh = [r for r in rows if r["kind"] == "fresh"]
out = {"per_call": rows,
       "mean_fresh": {k: m(k, "fresh") for k in ["prompt", "comp", "reason", "answer", "usd", "usd_nocache"]},
       "mean_refined": {k: m(k, "refined") for k in ["prompt", "comp", "reason", "answer", "usd", "usd_nocache"]}}
out["reasoning_share"] = m("reason") / m("comp")
out["prompt_share_of_tokens_fresh"] = m("prompt", "fresh") / (m("prompt", "fresh") + m("comp", "fresh"))
out["prompt_share_of_usd_fresh"] = (m("prompt", "fresh") * P_IN) / (m("prompt", "fresh") * P_IN + m("comp", "fresh") * P_OUT)

# does more thinking buy a better reward? (n = 9 fresh samples; descriptive only)
rho, p = spearmanr([r["reason"] for r in fresh], [r["peak"] for r in fresh])
out["reasoning_vs_peak_fresh"] = {"spearman": rho, "p": p, "n": len(fresh)}
print(f"\nreasoning tokens vs PPO peak, fresh n={len(fresh)}: Spearman {rho:.2f} (p={p:.2f})")

# PPO cost per candidate, in dollars and seconds
for hr in GPU_USD_HR:
    out[f"usd_per_ppo_iter_at_{hr}"] = SEC_PER_ITER_SOLO * hr / 3600
sec500 = 500 * SEC_PER_ITER_SOLO
print(f"\nPPO: {SEC_PER_ITER_SOLO:.3f} s/it solo on A100 -> 500 its = {sec500:.0f} s, 3000 its = {3000*SEC_PER_ITER_SOLO:.0f} s")
usd_call = out["mean_fresh"]["usd"]
print(f"LLM call (fresh, mean): ${usd_call:.4f}  (${out['mean_fresh']['usd_nocache']:.4f} with no prompt caching)")
ratio = {}
for hr in GPU_USD_HR:
    usd500 = sec500 * hr / 3600
    ratio[hr] = {"usd_500it_screen": usd500, "screen_over_call": usd500 / usd_call, "call_in_ppo_iters": usd_call / (SEC_PER_ITER_SOLO * hr / 3600)}
    print(f"  A100 @ ${hr}/h: 500-it screen = ${usd500:.3f} = {usd500/usd_call:.0f} LLM calls; 1 call = {ratio[hr]['call_in_ppo_iters']:.0f} PPO iterations")
out["call_vs_ppo"] = ratio
out["latency_sec"] = {"mean_nim_all": m("sec"), "median_nim_all": float(np.median([r['sec'] for r in rows])),
                      "median_no_retry": float(np.median([r['sec'] for r in rows if r['attempts'] == 1]))}
print(f"NIM latency per call: median {out['latency_sec']['median_nim_all']:.0f}s; no-retry median {out['latency_sec']['median_no_retry']:.0f}s "
      f"(={out['latency_sec']['median_no_retry']/SEC_PER_ITER_SOLO:.0f} PPO iterations of wall-clock)")

# --- batching model: what can be saved?  cost(call) = prompt + reasoning + answer, in $.
# Scenario params for a call producing K rewards:
#   independent : K x (prompt + reasoning + answer)
#   n=K sampling: prompt once (cached anyway), K x (reasoning + answer)
#   joint call  : prompt once, reasoning grows by factor g(K) (alpha=0: all 4 share one reasoning pass; alpha=1: reasoning scales with K), answer K x
rp, rr, ra = out["mean_fresh"]["prompt"], out["mean_fresh"]["reason"], out["mean_fresh"]["answer"]
def usd(p, c): return (p * P_IN + c * P_OUT)
K = 4
base = K * usd(rp, rr + ra)
scen = {"4 independent calls": base,
        "n=4 sampling (shared prompt)": usd(rp, K * (rr + ra)),
        "1 joint call, reasoning x1.0 (optimistic)": usd(rp, rr + K * ra),
        "1 joint call, reasoning x2.0": usd(rp, 2 * rr + K * ra),
        "1 joint call, reasoning x4.0 (no saving)": usd(rp, K * rr + K * ra)}
out["batching_model"] = {k: {"usd": v, "vs_independent": v / base} for k, v in scen.items()}
print("\nBatching model, 4 rewards (token cost only; uses measured mean fresh call):")
for k, v in scen.items():
    print(f"  {k:46s} ${v:.4f}  ({100*v/base:5.1f}% of independent)")
out["batching_note"] = ("Prompt is only %.0f%% of token cost; 'n=4' saves at most that. Savings beyond ~12%% require the model to "
                        "reuse one reasoning pass for several rewards, which is untested." % (100 * out["prompt_share_of_usd_fresh"]))

# --- feedback compression: how much of the round-2 prompt is feedback?
msgs = json.loads((ROOT / "results/exp_b_minimax/round2/messages.json").read_text())
m1 = json.loads((ROOT / "results/exp_b_minimax/round1/messages.json").read_text()) if (ROOT / "results/exp_b_minimax/round1/messages.json").exists() else None
r1_chars = sum(len(x["content"]) for x in (m1 or msgs)[:2])
cpt = r1_chars / rows[0]["prompt"]  # chars per token calibrated on round-1 prompt (system+user = 1215 tokens)
parts = {x["role"] + str(i): len(x["content"]) for i, x in enumerate(msgs)}
tok = {k: v / cpt for k, v in parts.items()}
fb = msgs[3]["content"]
out["round2_prompt_breakdown_tokens_est"] = tok
out["chars_per_token"] = cpt
print(f"\nRound-2 prompt split (est. from {cpt:.2f} chars/token, total measured {rows[-1]['prompt']}):")
for k, v in tok.items():
    print(f"  {k:10s} ~{v:5.0f} tokens")
# compression experiment: keep only last/peak/slope per component instead of 10-point trajectory + max/mean/min
import re
comp_lines = [l for l in fb.split("\n") if re.match(r"^\w+: \[", l)]
def compress(l):
    name = l.split(":")[0]
    vals = [float(x) for x in re.findall(r"'(-?[\d.]+)'", l)]
    return f"{name}: start {vals[0]:.2f}, end {vals[-1]:.2f}, peak {max(vals):.2f}"
comp_c = "\n".join(compress(l) for l in comp_lines)
saved_chars = sum(len(l) for l in comp_lines) - len(comp_c)
out["feedback_compression"] = {"n_component_lines": len(comp_lines), "chars_saved": saved_chars,
                               "tokens_saved_est": saved_chars / cpt, "usd_saved_per_call": saved_chars / cpt * P_IN,
                               "fraction_of_call_usd": saved_chars / cpt * P_IN / out["mean_refined"]["usd_nocache"]}
print(f"feedback lines {len(comp_lines)}; trajectory->start/end/peak saves ~{saved_chars/cpt:.0f} prompt tokens "
      f"= ${saved_chars/cpt*P_IN:.5f}/call = {100*out['feedback_compression']['fraction_of_call_usd']:.2f}% of a refinement call")
(OUT / "token_accounting.json").write_text(json.dumps(out, indent=1, default=float))
