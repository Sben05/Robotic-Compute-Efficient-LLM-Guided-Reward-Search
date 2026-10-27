# Using the LLM more efficiently when it writes RL rewards

Tags: **[actual]** real LLM calls / real PPO runs, **[simulation]** curve-based simulation. p-values only for the pre-registered
H2-H4 (STATS_PLAN.md, Holm across H1-H5). Prices: MiniMax M3.1 Flash and Qwen-3-32B $0.15/$0.60, Gemini 2.5 Flash $0.30/$2.50
per million input/output tokens. All new calls tonight: MiniMax 50 calls ($0.47), Gemini 10 ($0.18), Qwen 10 ($0.04).

## 1. Batching: ask for four rewards in one call (H2 significant, H3 inconclusive)

**H2 - cost [actual].** Completion tokens per *valid* reward, one-per-call vs four-per-call, MiniMax, 4 tasks:

| Task | one per call | four per call | ratio | validity (1/call vs 4/call) |
| --- | ---: | ---: | ---: | --- |
| Ant | 12,154 | 4,530 | 2.7× | 17/17 vs 12/12 |
| Cartpole | 5,048 | 1,347 | 3.7× | 6/8 vs 8/8 |
| BallBalance | 17,148 | 6,658 | 2.6× | 8/8 vs 8/8 |
| Humanoid | 18,374 | 3,346 | 5.5× | 5/6 vs 8/8 |

Batching is cheaper on all 4 tasks; stratified permutation test p < 0.0001 (Holm p < 0.0001). **Significant.**
(Sign test 4/4, p = 0.0625, as expected with only 4 tasks; the permutation test is the pre-registered decision test.)
Why it works: ~90% of every completion is hidden reasoning, and one reasoning pass is shared by four functions.

**H3 - quality [actual].** Scores normalised by the same-seed human score (Ant: 500-it screens; other tasks: full training).
Mean difference batch − one-per-call = **−0.023 [95% CI −0.120, +0.073]**; the pre-registered non-inferiority margin is −0.10,
p = 0.064 (Holm 0.128). **Inconclusive**: the CI crosses the margin. Per task: Ant equal (0.58 vs 0.57), BallBalance batch
better (1.00 vs 0.90), Cartpole both at the ceiling, Humanoid batch worse (0.12 vs 0.33, n = 7 vs 4).

## 2. Every generation strategy tested on Ant (MiniMax) [actual]

500-iteration screen of every valid reward on seeds 0 and 1 (human reward at 500 its: 8.01 seed 0, 6.99 seed 1).

| Arm | Calls | Valid | Tokens / valid | $ / valid | Best 500-it (s0 / s1) | Median 500-it (s0 / s1) | Seed corr. |
| --- | ---: | ---: | ---: | ---: | --- | --- | ---: |
| A one per call, no hint (baseline) | 17 | 17 | 12,154 | 0.0075 | 7.29 / - | 5.48 / - | - |
| **B four per call** | 3 | 12 | **4,530** | **0.0028** | **8.39** / 6.21 | 4.49 / 4.88 | 0.65 |
| C four per call, four named strategies | 3 | 8 | 7,340 | 0.0045 | 7.00 / 7.21 | 5.52 / 5.36 | 0.88 |
| D one per call + strategy hint | 12 | 7 | 38,704 | 0.0235 | 5.99 / 6.68 | 3.43 / 4.04 | 0.94 |
| E one per call, "think briefly" | 4 | 4 | 11,270 | 0.0069 | 4.33 / 4.30 | 1.00 / 0.17 | 0.17 |

The 6 best trial rewards (all from batched arms B and C) resumed to 3,000 iterations [actual, resume protocol]:
11.11, 10.42, 10.14, 9.28, 8.82, 8.50. `B_batch4/call2_f1` (11.11) is MiniMax's best single-seed reward (human reward 10.48 on the same seed); see §3 for Qwen's 11.32.

Failed hypotheses: strategy hints (D) made rewards worse and 5× more expensive per valid reward (2 replies hit the 32k-token
cap); asking for brief reasoning (E) produced much weaker rewards; naming four diverse strategies (C) reduced validity
(8/12) without raising the best score.

## 3. Other LLMs (robustness, exploratory) [actual, Ant, seed 0]

Same prompts, same two arms, 10 calls each. DeepSeek V4.1 Flash and Qwen 3.8 Flash were refused by the free gateway tier, so
Gemini 2.5 Flash and Qwen-3-32B were used (substitution recorded in STATS_PLAN.md before any call).

| Model, arm | Calls | Valid | Tokens / valid | $ / valid | Best 500-it | Median 500-it |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Gemini 2.5 Flash, one per call | 8 | 5 | 8,410 | 0.0216 | 5.18 | 1.27 |
| Gemini 2.5 Flash, four per call | 2 | 4 of 7 | 6,792 | 0.0172 | 2.07 | 1.88 |
| Qwen-3-32B, one per call | 8 | 8 | 6,074 | 0.0038 | 7.16 | 2.50 |
| Qwen-3-32B, four per call | 2 | 7 of 8 | 1,833 | 0.0011 | 4.07 | 2.32 |

Batching is again cheaper per valid reward for both models (1.2× Gemini, 3.3× Qwen). At the 500-iteration screen both models'
rewards are weaker on average than MiniMax's (best 5.2 and 7.2 vs 8.4; medians 1.3-2.5 vs ~4.5-5.5), and for both the best
reward came from the one-per-call arm.

Top 4 per model (by 500-it score) resumed to 3,000 iterations [actual, seed 0, resume protocol]:

| Model | one per call | four per call |
| --- | --- | --- |
| Gemini 2.5 Flash | 6.90, 3.18 | 6.07, 4.42 |
| Qwen-3-32B | **11.32**, 9.81, 6.74 | 5.41 |

**Qwen-3-32B's best reward (11.32) is the highest single-seed reward measured in the project**, above MiniMax's best (11.11)
and the human reward (10.48 on seed 0). One seed only, so treat it as a lead, not a result: a weaker model on average can still
produce the single best reward, which again argues for screening many candidates cheaply.

## 4. A zero-token edit instead of an LLM refinement call (H4, not significant after correction) [actual]

The LLM's round-2 refinement of reward r1c1 mainly widened the forward-speed tanh (temperature 4 → 10). Applying only that one
edit, no LLM call (straight 3,000 its, seeds 0-3):

| r1c1 variant | seed 0 | seed 1 | seed 2 | seed 3 | mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| unchanged (τ = 4) | 7.55 | 7.49 | 7.27 | 7.96 | 7.57 |
| τ = 10 (the edit) | 8.95 | 9.69 | 9.80 | 7.71 | 9.04 |

+1.47 points, Welch one-sided p = 0.025, **Holm p = 0.075: not significant after correction**. The pre-registered reading
(τ = 10 mean ≥ 9.0 → the edit explains most of the gain) is met, narrowly. Context: the LLM-refined reward r2c0 averaged 9.15
over 9 seeds, so the one-line edit recovers about the whole multi-seed refinement gain; τ = 6 gave 10.16 / 8.88 (2 seeds), and
reverting r2c0 to τ = 4 gave 8.97 / 9.26.

## 5. Live search with 500-iteration feedback [actual, n = 2 trials]

Writing round 2 from the best round-1 reward's 500-iteration log (not a full run) worked mechanically in both trials
(16/16 valid) and cut PPO compute 73%. Round-2 rewards averaged 8.52 vs 8.33 for round 1 (no no-feedback control): this
does not show that feedback improved quality.

## 6. The LLM as a judge of early training (pre-registered, exploratory) [actual LLM calls on held-out curves]

MiniMax was shown each candidate's reward code and its first 500 iterations of logs (task score and every reward component)
and asked which would score highest after 3,000 iterations (`llm_judge.py`, pre-registered in
`results/pruning/llm_judge_prereg.md` before any call). Held-out seeds 7-9, 105 four-reward sets, shuffled labels, same compute
as fixed@500.

| | Best reward kept | Winner found |
| --- | ---: | ---: |
| fixed@500 (keep the leader at 500) | 0.975 | 56% |
| LLM judge (code + 500-it logs) | 0.987 | 74% |

Difference +0.013 [95% CI −0.001, +0.032], p = 0.057: **promising but not significant** under the pre-registered threshold;
with each seed as one observation (the 105 sets reuse 7 rewards, so they are not independent) p = 0.17. This test is closed;
it will not be extended on the same curves.
All 105 replies were parseable; cost $0.74. VoI and PID reach 0.997 on these seeds with ~12% more compute. The earlier blind
ranking (code only, no logs) failed badly, so the training logs are what make the LLM useful as a judge.

## 7. Generate another reward or keep training? [simulation]

Measured exchange rate: one solo PPO iteration costs $0.0000473 on the rented A100 (0.251 s at $0.68/h), so one MiniMax call
costs as much as ~160 (one per call) or ~60 (per reward when batched) PPO iterations; wall-clock latency (~2 min) is worth
~500 iterations. In the existing allocation simulation (`LLM_EFFICIENCY_FEASIBILITY.md` §2.2, 84 splits of real curves), at
40-250 iterations per call adaptive generate-vs-train rules match but do not beat fixed "generate 4, screen at 500" at equal
cost (gaps −0.11 to +0.18 points); gains only appear when calls are slow or expensive (≥ 1,000 iterations per call).
Conclusion: at these prices the lever is the cheap screen, not the number of calls.

## Prior work

Batched/multi-sample generation, reasoning-budget control and programmatic reward mutation all have prior work (Eureka's
own sampling, evolutionary reward search such as Text2Reward/EvoPrompt-style mutation, reasoning-effort controls). No
novelty is claimed; the contribution is the measured cost/quality trade-off on this task family.
