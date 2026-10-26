# Reducing LLM calls, tokens and cost in reward search: feasibility

Branch `llm-efficiency` (own worktree). Offline only: **no GPU job, no PPO, no LLM/API call, nothing on Quest, no other
agent's file modified.** Inputs are the finished results already in `results/` (11 real MiniMax generations with token
usage, 11 + 2 human PPO curves). Code: `llm_efficiency/`; outputs: `results/llm_efficiency/`.

Written for: you, as the thesis author deciding whether this is a second contribution.

## 1. ELI5

- **The LLM is the cheap part.** One reward-writing call costs about **half a cent** ($0.0044 at MiniMax M3.1 Flash list price).
  One 500-iteration PPO screen costs about **$0.03-0.13** (an A100 at $1-4/h), i.e. **7-29 LLM calls**. In the whole 3+2
  search the LLM is **1-4% of the spend**. Cutting LLM calls by 75% would save ~1-3% of the bill.
- **Batching cannot save much, because the cost is thinking, not prompting.** 88% of every completion is hidden reasoning
  (~6,500 of ~7,250 tokens); the prompt is 14% of tokens and **4% of dollars**. Asking for 4 rewards per call only helps if
  the model reuses one reasoning pass for all four. Nobody has measured that for this model, so I prepared the test.
- **Independent samples are already diverse.** Among the 9 independent samples there are no near-duplicates (code
  distance 0.44-0.68), so "force different strategies" has no duplicates to remove.
- **The one thing that clearly mattered was feedback, not sampling.** Round 1 (fresh samples) reached 4.5-7.6 in 3 tries; the
  two feedback-based rewrites hit 10.9 and 10.75. What changed? Mostly one number: the speed term `tanh(v/4)` was saturated
  (feedback showed it pinned at 0.91), and both rewrites widened the temperature (4 -> 10 and 4 -> 6). A 5-line rule does
  that edit with **zero LLM tokens**. Whether that alone gives the gain is untested; it costs 3 short GPU runs to find out.
- **A smart "generate or train?" controller does not beat simple rules on this data.** In simulation on the real curves it
  matches a "keep generating until one reaches the human level" rule within ~0.1-0.2 points, and only when LLM calls are
  expensive relative to PPO, which they are not here. I would not build the thesis on it.

## 2. Methods compared (what was measured and what was not)

Proxy ≠ PPO evidence. "Measured" below means measured on the existing data or an offline simulation of it.

| Strategy | What I did | Result | Status |
| --- | --- | --- | --- |
| **Batching 4 rewards/call** | Token model on the 11 real calls (`token_accounting.py`) | `n=4` / shared prompt saves ≤3% of dollars. A joint call saves 46-67% **only if** reasoning grows ≤2x for 4 rewards; at 4x it saves 3%. | Model only. **Needs a real call to settle** (arm B). |
| **Diversity-aware batching** | Code distance, recipe count, 36 pairs (`diversity_analysis.py`) | Mean pairwise code distance 0.55 (min 0.44); 0 identical pairs; 8 distinct (inputs, transforms) recipes among 9 samples (only p3 and p5 share one). Code distance does not predict score gap (Spearman 0.10; nearest code neighbour is no closer in score: 3.51 vs 3.70). | Duplicates are not a problem at T=1.0. Whether *forced* diversity raises the hit-rate is untested (arms C, D). |
| **Adaptive generation** (2 first, more if weak) | Simulation, 84 library/test splits, seed-noise sigma 0 / 0.35 / 0.7 | See §2.2. Beats a fixed N at matched cost by ~0.1-0.2 points of ~9; the gain grows as LLM calls get dearer. | Simulation on 9 curves. |
| **Local mutations** | `mutate.py`: random constant scaling and a feedback-triggered "desaturate" rule, run on synthetic Ant states (CPU torch) | 9/11 rewards have named constants; of 1,800 random x[0.5, 2] mutants **100% compile and are finite**, but they barely move the reward (state-rank correlation with the parent 0.986-1.000). The desaturate rule turns the Round-1 winner's `tanh` temperature 4 -> 10 (**exactly Round-2 cand 0's value**) and lifts its flatness ratio (reward gain 6->10 m/s vs 0->4 m/s) from 0.11 to 0.59; the Round-2 rewards sit at 0.94 and 0.78. The two best pool rewards (p1, p3) use inline literals, so a named-constant operator cannot touch them. | Validity and shape only. **No evidence yet that a mutant trains better.** |
| **Feedback compression** | Re-tokenised the Round-2 prompt (calibrated 3.79 chars/token) | The feedback message (8-line component table plus instructions) is ~900 of the 3,127 tokens. Replacing each 10-point trajectory by start/end/peak saves ~190 tokens = **0.55% of a refinement call**. Compression can matter only through *quality* or *reasoning length*, not input cost. | Cost: negligible. Quality: untested (arm F below). |
| **Joint generate-vs-train** | `joint_alloc_sim.py` | §2.2. | Simulation. |
| **Reasoning budget** (not in the brief) | Spearman of reasoning tokens vs PPO peak, 9 fresh samples | -0.50 (p=0.17): more thinking did not buy a better reward; the best sample (p3, 10.73) used 4,781 reasoning tokens vs ~8,000 for several weak ones. MiniMax M3.1 Flash advertises "tunable thinking depth". | Because 88% of cost is reasoning, a thinking cap is the **only** lever that could cut tokens by more than a few percent. Untested (arm E). |

### 2.1 Cost structure (`token_accounting.py`, real data)

| | Fresh sample (n=9) | Feedback rewrite (n=2) |
| --- | ---: | ---: |
| Prompt tokens | 1,215 | 3,127 |
| Completion tokens (of which reasoning) | 7,255 (6,498, 90%) | 7,992 (6,401, 80%) |
| Visible answer (the code) | 757 | 1,591 |
| Cost at $0.15 in / $0.60 out per M | **$0.0044** | $0.0050 |

PPO: 0.235 s/iteration alone on an A100 (704 s per 3,000) -> a 500-it screen is 117 s. At $1 / 2.5 / 4 per A100-hour one
call equals **68 / 27 / 17 PPO iterations**. The 3+2 search is 5 calls (~$0.022) plus 7,500 screened iterations
($0.49 / $1.22 / $1.96): LLM share **4.3% / 1.8% / 1.1%**. Latency is the other cost: median 282 s per call on the NIM endpoint
without retries (563 s with), which is ~1,200 PPO-iterations of wall-clock if calls are serial. That is an endpoint property
(rate limits, 504 retries), not a token property. Quest bills no dollars, so wall-clock and queue time are the real
currencies there. Price source: MiniMax M3.1 Flash list price $0.15 / $0.60 / $0.015 cached per M tokens (press and gateway
pages; I did not verify it on MiniMax's own pricing page).

### 2.2 Generate or train? Simulation on real curves (`joint_alloc_sim.py`, `defaults_compare.py`, `frontier_gap.py`)

Setup. A "reward" is a curve drawn from 3 held-out fresh curves with a lognormal time-warp as seed noise. Policies see only
prefixes plus a library of the other 6 fresh and 2 human curves, never the curve they are scored on. All C(9,3) = 84 splits
are enumerated. Cost = PPO iterations + `c_gen` x calls; the score is the selected reward's true 3,000-it peak (selected
reward is always finished and counted). Policies: **fixed(N, s)** (generate N, screen at s, finish best; the thesis default
is N=4, s=500); **target(θ, s)** (generate one at a time, stop when the best screened peak ≥ θ x human peak at s; max 12);
**joint(λ, s)**, an expected-improvement-per-cost rule that at each step compares "generate" (E[(Y_prior - m)+] / (c_gen + s))
with "train candidate k another s iterations" and stops when the best ratio falls below λ (m = the leader's median forecast).

Matched-cost gap to the fixed-N frontier, averaged over budgets of 4k-12k iterations (`frontier_gap.json`; every family's
frontier is optimistic because its config is chosen on the same worlds, so read the differences):

| LLM call = | target-stop | joint EI/cost | note |
| ---: | ---: | ---: | --- |
| 0 its (free) | +0.01 to +0.08 | -0.11 to -0.02 | |
| **40 its (A100 dollars)** | **+0.02 to +0.10** | **-0.11 to 0.00** | realistic for dollars: no gain |
| 250 its | +0.04 to +0.18 | -0.05 to +0.09 | |
| 1,000 its (serial slow endpoint) | +0.11 to +0.27 | +0.10 to +0.22 | gains appear when calls are slow/expensive |

(ranges over sigma = 0, 0.35, 0.7; the fixed frontier sits at 8.3-8.9, so the gaps are 1-3%). Against the thesis default
fixed(4, 500) (score 8.41, 4,500 iterations), paired over splits (±SE over 84 overlapping splits): stop-at-human-level
buys +0.50±0.05 but at +64% cost; the cost-matched views above are the fair comparison. The pre-specified joint settings
(λ = 1 point per 1,500 its) lie on the same curve as everything else; they gain +0.1 to +0.35 for cost changes of -10% to +26%, i.e. no free lunch. The only visible advantage of an adaptive rule is that it can *stop
early when it got lucky* (target-stop uses 4.8-9.8 calls, fixed uses exactly N).

The cost ratio that decides everything is **PPO-iterations per LLM call**. On this task it is 17-68 (dollars), so more
candidates are almost free relative to training them and the question is the cheap screen, not the call count.

## 3. Novelty (with prior work found)

Searches were a handful of web queries, not a survey; items marked (unverified) I did not open.

- **Eureka** (Ma et al.): fixed K samples x 5 iterations (16 x 5 per run, 5 runs); no cost-aware stopping. **REvolve**
  (arXiv 2406.01309): evolutionary reward search with human feedback; fixed generation budget. **RF-Agent** (arXiv 2602.23876):
  MCTS over reward functions with LLM actions (mutation, crossover, path reasoning); caps at 80 sampled rewards, trains each
  with PPO, no per-run early stopping, no token accounting in the part I read.
- **ORSO** (Zhang, Hong, Pacchiano, Agrawal; ICLR 2025, arXiv 2410.13837): bandit/model-selection over a candidate reward
  set (D3RB works best), "up to 8x" less compute. Its candidate set is generated up front and in the large-set experiment
  **fixed**; resampling exists as an implementation mechanism but its trigger is a fixed schedule (resample after ≥100
  selection steps), not a cost-based rule. No LLM token costs reported.
- **EvoNav** (arXiv 2605.11859): three-stage multi-fidelity filtering of LLM rewards (rule-based score -> 8k-step proxy
  policy -> full training), 74% GPU cost cut vs CARD, but **fixed schedules (N=8, G=10/16/3), more LLM tokens (720K vs 72K)**,
  no adaptive call count.
- **Infinite-armed best-arm identification** already poses exactly "add a new arm or re-evaluate an old one": **D-TTTS**
  (Shang, Kaufmann, Valko, AutoML workshop 2019) for hyperparameter tuning; Hyperband and Carpentier-Valko / Berry et al.
  infinite-armed bandits (unverified). So the algorithmic skeleton of a joint rule is **not new**.
- **Adaptive LLM sampling**: AdaBoN (arXiv 2505.12050) and "Adaptive Inference-Time Compute" (arXiv 2410.02725) allocate best-of-N
  samples by estimated marginal value. Same idea, different task.
- **Auto-Robotist**: I found only a 2026 paper (arXiv 2605.25832) on *morphology* search with skill memory, not reward design;
  I could not find a reward-design paper by that name.
- **ERFSL** (arXiv 2605.19259, student abstract): splits reward-code generation from reward-weight search with a flexible-step
  weight searcher, avg 5.2 adjustments: the nearest relative of the "programmatic edits after the LLM" idea.
- From the earlier controller study (not re-checked): early-termination of LLM candidates (arXiv 2404.01617), RHyVE
  (arXiv 2604.28056, reward ranks unreliable at low competence).

**Where I think novelty is real, ranked by how much I trust it:**

1. **A measured cost model for LLM-guided reward search** (tokens, reasoning share, dollars and wall-clock vs PPO) and the
   conclusion that the LLM is 1-4% of spend on Ant. Small but concrete, and nobody I found reports it; it also tells readers
   *when* LLM-side savings matter (fast simulators, expensive frontier models: the PPO-iterations-per-call ratio).
2. **Diagnostic-driven programmatic edits replacing the reflection call** (e.g. "saturated tanh component -> widen its
   temperature"). The saturation mechanism is real in the data and the rule reproduces the LLM's edit exactly, but the
   claim "it reproduces the *gain*" is unproven. The idea sits close to ERFSL's weight search; the new part would be
   deriving the edit from Eureka's component statistics, with the LLM called only when no rule fires.
3. **A three-way decision (fresh sample / refine the best / train more).** The data hint that refinement is the high-yield
   action: P(peak ≥ 10) is 1/9 for fresh vs 2/2 for refined (Fisher one-sided p = 0.055; 95% CIs 0-0.48 vs 0.16-1.00). That is two
   samples from one parent. If a larger test confirms it, the interesting decision is *when* to refine rather than fresh-sample.
   This is the part of "joint allocation" with headroom; the generate-vs-train part has almost none here.
4. **Not recommended as a headline: a generate-vs-train controller.** The skeleton is D-TTTS/ORSO territory and my
   simulation shows no robust gain over target-stop or fixed N on 9 curves.

## 4. Recommendation for the thesis

Make contribution 2 a **cost-model-plus-feedback-efficiency study**, not a new allocation algorithm:

1. Report the cost structure (§2.1) and the PPO-per-call ratio as a design rule: on Ant, spend the effort on screening
   (contribution 1), not on trimming LLM calls.
2. Test whether feedback can be made cheaper and at least as good: **(a)** programmatic desaturation vs LLM refinement,
   **(b)** compressed/targeted feedback vs the full Eureka prompt, **(c)** thinking-depth cap. These are the only levers that
   move the numbers that matter (quality per round and the 88% reasoning share).
3. Use adaptive stopping ("generate until a screened candidate reaches the human level", with a max) as the practical
   controller; it is one parameter and was the best simulated family. Do not build a joint EI controller unless step 2 shows
   refinement/fresh yields differ enough to justify a three-way rule.
4. Keep every claim at the level "one seed, small n". If the thesis claim is "cheaper at equal quality", it needs PPO runs,
   not these proxies.

## 5. Next experiments and resources

Nothing below has been run. Approval needed for anything touching an API key or Quest; the A100 hours are the real cost
(LLM spend is cents).

| # | Experiment | LLM calls ($) | GPU | Decision it informs |
| --- | --- | --- | --- | --- |
| N1 | **Desaturation ablation.** Retrain the Round-1 winner (r1c1) with only `forward_speed_temperature` 4 -> 10 (the rule's edit) and 4 -> 6; plus Round-2 cand 0 with the temperature put back to 4. Seeds 0 and 1, 3,000 its (500 as a screen) | 0 | 6 runs x 704 s solo; ~1 h packed on 1 A100 | Does the free edit reproduce the gain? Settles contribution 2.2. |
| N2 | **Batching / diversity / reasoning cap** (`llm_efficiency/llm_trial_plan.py`, payloads already generated, dry-run only). Arms: B batch-4 (3 calls), C batch-4 with assigned strategies (3), D 4 independent calls with strategy hints (12), E short-reasoning (4) vs baseline A = the existing 9 samples (0 new calls). Record tokens, reasoning share, latency, validity | 22 calls: **~$0.15 typical, $0.44 worst case** | Optional: 500-it screens on the ~40 valid rewards: ~40 x 117 s = 80 min solo, ~30 min packed | Does joint generation reuse reasoning (cost) and does forcing diversity raise yield (quality)? |
| N3 | **Refine vs fresh vs targeted feedback.** From one fixed parent (the best of 4 at 500 its), 10 refinement samples each with (i) full Eureka feedback, (ii) compressed/targeted feedback; plus 10 new fresh samples. Screen all at 500 its | 30 calls, ~$0.2 | 30 x 117 s = 1 h solo, ~25 min packed | The 3-way decision and feedback compression; n=10 per arm can detect hit rates like 10% vs 60%, not small effects. |
| N4 | Only if N3 shows a gap: replay target-stop and a refine-vs-fresh rule *live* on the existing `live.py` machinery | ~50 calls | 2-3 h on 1 A100 per seed | Whether the simulated rule survives real feedback; defer until the live early-stopping agent is done. |

Needed from you before N2/N3: approval for the API spend (a key file path; the script refuses to run without `--run`, a key
and `--max-usd`, and stops once the cap is hit), and agreement to use the Vercel route (needs `max_tokens` ≥ 32768 per the
project notes; 16k returned empty reasoning-only replies). N1 needs only Quest GPU time and `eureka_ab.py`-style training
of three edited reward files, which I did not create so as not to touch other agents' paths.

## 6. Limitations

- **9 fresh curves, 1 seed each.** Seed noise is a time-warp assumption (sigma 0 / 0.35 / 0.7 all reported). The 84 library/test
  splits overlap, so SEs are optimistic. Frontiers pick configs on the same worlds (optimistic for all families alike).
- **Round-2 data are 2 samples from one parent**; the refine-vs-fresh contrast is a hint, not a result.
- **Scripts do not generate new rewards**: nothing here says a mutated or compressed-feedback reward is better; only that it is
  valid, cheap, and mimics an edit the LLM made.
- **Mutation tests use synthetic states**, not simulator rollouts; "flatness ratio" is the controller study's idea applied to
  mutants, with the same small-n caveat.
- **Prices and latencies**: list prices from secondary pages; NIM latency includes rate-limit retries and may not apply to
  the Vercel route. The "A100 $/h" figures are assumptions (Quest charges no dollars).
- Simulation ignores invalid/failed generations (0 of the 11 recorded were invalid), which would raise the effective cost of
  a call.

## Reproduce

```
python3 -m venv .venv-eff && .venv-eff/bin/pip install numpy scipy matplotlib torch
bash llm_efficiency/run_all.sh        # ~10 min CPU total; writes results/llm_efficiency/
```
