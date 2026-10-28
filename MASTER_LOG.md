# Master log: the complete record (26-27 Oct 2026)

Everything ideated, built, run and found: setup, robots, LLMs, search rounds, every experiment with its results, the
literature, the statistics (including the corrections), the infrastructure, incidents, costs and next steps. Times are CDT
(UTC−5) unless marked. Commit hashes refer to `main`. Evidence tags: **[actual]** = real training runs or LLM calls as
described; **[held-out]** = rules frozen beforehand, replayed on curves never used for tuning; **[simulation]** = curve-based
simulation; **[exploratory]** = no significance claim.

Contents: 1 Summary · 2 Setup (robots, LLMs, PPO, rounds) · 3 Literature and novelty · 4 Chronology · 5 Every experiment and
its results · 6 Statistics · 7 Ideas and their fate · 8 Infrastructure and incidents · 9 Costs · 10 Open items · 11 Files

---

## 1. Summary

**Question.** Eureka-style systems have an LLM write reward functions for a robot and judge each candidate by training a PPO
policy with it. Training dominates the cost. Two hypotheses: **(1) stop weak candidates early without losing good ones;
(2) use the LLM more cheaply.**

**Headline findings.**

| Finding | Evidence | Status |
| --- | --- | --- |
| Early stopping saves most of the PPO compute | live searches: 73% fewer iterations; adaptive rules use ~42% of training-all | solid [actual, held-out] |
| VoI (value-of-information forecasting) beats "keep the leader at 500" | 0.997 vs 0.972 of the best reward kept, winner 86% vs 59%, 5 held-out seeds; one live run at 32% compute | consistent on every seed; significant as pre-registered, **not** under the stricter seed-level test (p = 0.041, Holm 0.21) |
| A simple PID rule is about as good as VoI | ties VoI on the clean held-out seeds; transfers to Humanoid where an Ant-tuned VoI fails | solid [held-out] |
| VoI and PID match or beat the closest published method (ORSO, ICLR 2025, rebuilt) | VoI +0.7% kept (p = 0.012) with 5% less compute; PID ties with 6% less | [held-out, post hoc] |
| Forecasting rules need history from the same robot | Humanoid: VoI 81% with Ant history → 97% with Humanoid history | [exploratory] |
| Asking the LLM for 4 rewards per call is much cheaper | 2.6-5.5× fewer tokens per usable reward, 4/4 tasks (MiniMax); also cheaper for Gemini and Qwen | descriptive; original permutation p-value invalid (see AUDIT_REPORT.md) |
| Batched rewards are as good | −2% [−12%, +7%] | inconclusive |
| One constant edit replaces an LLM refinement | 7.57 → 9.04 (4 seeds each) | not significant after correction |
| The best reward found beats the human reward | +0.86 over 6 seeds | not significant |
| Single-seed winners are unreliable | the original "best" reward (10.93) averages 9.15 on 8 other seeds; human reward 7.9-10.6 by seed | solid [actual] |
| An LLM reading early training logs can judge rewards | winner 74% vs 56% at equal compute | promising, not significant (p = 0.057; 0.17 seed-level) |
| Reward-structure signals, strategy hints, short reasoning, generate-vs-train juggling, published Hyperband | none helped | negative results |

**Framing agreed for the thesis:** no new algorithm is supported by the data. The contribution is a rigorous, pre-registered,
held-out, multi-seed and live evaluation of compute allocation for LLM reward search, compared against the closest published
method, with clear positive findings, scoped claims and documented negative results.

---

## 2. Setup

### 2.1 Robots / tasks (Isaac Gym Preview 4, IsaacGymEnvs as shipped with Eureka @9eee428)

| Task | Goal (Eureka description) | PPO iterations | Envs | Score used ("task score", `consecutive_successes`) | Role in the study |
| --- | --- | ---: | ---: | --- | --- |
| **Ant** (quadruped) | run forward as fast as possible | 3,000 | 4,096 | forward progress (higher = faster) | main task: all searches, seeds, pruning tests |
| **Cartpole** | keep the pole upright | 100 | 512 | episode length (max ~500) | generalisation (saturated) |
| **BallBalance** | keep the ball on the table | 250 | 4,096 | episode length (max ~500) | generalisation (saturated) |
| **Humanoid** | run forward as fast as possible | 1,000 | 4,096 | forward progress | generalisation (hard) |

Score of a run = its peak task score over training (as upstream Eureka). Quadcopter was considered and skipped (negative score
scale breaks ratio-based rules).

### 2.2 LLMs

| Model | Access | Used for | Calls this session |
| --- | --- | --- | ---: |
| **MiniMax M3.1 Flash** (main) | NVIDIA NIM earlier; Vercel AI Gateway this session; temperature 1.0; max_tokens 32,768 | every search, the arm trial, generalisation, the LLM judge | 50 + 105 |
| Google Gemini 2.5 Flash | Vercel gateway | second-LLM robustness (Ant) | 10 |
| Alibaba Qwen-3-32B | Vercel gateway | third-LLM robustness (Ant) | 10 |
| DeepSeek V4.1 Flash, Qwen 3.8 Flash | refused (free tier, HTTP 403, $0) | planned; substituted before any call | 0 |
| Earlier, removed before this session | Gemini (quota), gpt-oss-120b (weak), DeepSeek on NIM (hung), local Qwen3.8-27B | - | - |

Prompts: upstream Eureka (`initial_system`, `initial_user` with the environment's observation code, `policy_feedback`).
Rewards are TorchScript functions injected into the environment's `compute_reward`; invalid code (signature, inputs, TorchScript)
is rejected exactly as upstream.

### 2.3 PPO and hardware

Upstream `AntPPO` and task-default PPO configs, unchanged; seed varied explicitly. Hardware: Quest A100 (PCIe-40GB / SXM-80GB)
until 26 Oct evening, then Vast.ai A100-PCIE-40GB boxes 1 and 2 (box 3 rented, never trained). Identical software everywhere;
20+10-iteration smoke runs gave identical scores to the last digit on Quest, box 1 and box 2. Measured speed on Vast:
0.251 s/iteration solo; 3 concurrent runs ≈ 5.2-5.5 it/s total (saturated; 6 concurrent = 5.08).

### 2.4 Search structure ("rounds")

- **Round 1:** N independent LLM samples from the initial prompt.
- **Round 2:** N new samples written after seeing the best round-1 reward and its training statistics (Eureka reflection).
- **Original search (exp_b):** 3 + 2 samples, every candidate trained fully, round 2 from full-training feedback.
- **Live search (exp_e, this session):** 4 + 4 samples per trial, every candidate trained only 500 its; the leader is kept,
  round 2 is written from the leader's **500-iteration** log, the overall leader resumes to 3,000; stopped runs are resumed later
  only as an audit.
- Round-2 refinements were also probed by ablation (the "desaturation" constant).

---

## 3. Literature and novelty

| Work | What it does | Relation to this thesis |
| --- | --- | --- |
| [Eureka (Ma et al., ICLR 2024)](https://www.seas.upenn.edu/~dineshj/publication/ma-2024-eureka/ma-2024-eureka.pdf) | LLM writes reward code; candidates trained fully; reflection loop | the system whose compute we cut; its prompts are used unchanged |
| [ORSO (Zhang et al., ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/hash/6155cfb4e650d41167848fe2ead30ab8-Abstract-Conference.html) | online selection among LLM-generated rewards in Isaac Gym (Cartpole, BallBalance, Ant, Humanoid, ...): 1% training chunks allocated by bandit model selection (D3RB, Exp3, UCB, ETC) on the current task score; up to 8× compute reduction | **closest prior work**: adaptive allocation itself is not new. Rebuilt and used as a baseline (VoI ≥ ORSO, PID ≈ ORSO at lower compute). ORSO uses no forecasting, no component signals, single-seed selection |
| [Learning-curve extrapolation (Domhan et al., IJCAI 2015)](https://ijcai.org/proceedings/2015/487) | predict final performance from early curve; terminate weak runs | VoI/Bayesian rules are in this family |
| Successive halving (Jamieson & Talwalkar, AISTATS 2016); Hyperband (Li et al., JMLR 2018); ASHA | budget-doubling elimination | tested; SH did not beat fixed@500; published Hyperband collapses at ≤12 candidates (3× cost) |
| [PASHA (ICLR 2023)](https://openreview.net/forum?id=syfgJE6nFRW); [LC-PFN (NeurIPS 2023)](https://papers.nips.cc/paper_files/paper/2023/hash/3f1a5e8bfcc3005724d246abe454c1e5-Abstract-Conference.html); freeze-thaw BO; BOHB (Falkner et al., ICML 2018) | adaptive budgets, probabilistic learning-curve forecasting, Bayesian guidance + multi-fidelity | a VoI + PID or "trust the forecaster" hybrid would overlap with these |
| [Calibrated online BO (2024)](https://proceedings.mlr.press/v238/deshpande24a.html); [HiP-BO (2026)](https://www.sciencedirect.com/science/article/pii/S092523122602802X) | forecast calibration as evidence accrues; historical-task knowledge + learning curves | cold-start / library ideas overlap |
| [Early discarding after one epoch (2024)](https://arxiv.org/pdf/2404.04111); [one epoch for multi-fidelity HPO (2023)](https://arxiv.org/pdf/2307.15422) | simple fixed early discarding is hard to beat | the alternative our adaptive rules must beat (they do, modestly, on Ant) |
| [Batch prompting (Cheng et al., 2023)](https://arxiv.org/pdf/2301.08721); [BatchPrompt (ICLR 2024)](https://iclr.cc/virtual/2024/poster/19241) | several inputs per call, savings from the shared prompt | batching itself is known; our mechanism (shared hidden *reasoning* of reasoning models when generating several programs) and measurement for reward design are the new part |
| [LEARN-Opt (2025)](https://arxiv.org/pdf/2511.19355); [RHYVE (2026)](https://www.alphaxiv.org/abs/2604.28056.md) | automated reward design is high-variance; candidate-family-dependent warm-up | supports screening many candidates; no compute-allocation study |
| Reward-hacking detection ([survey-style study](https://arxiv.org/html/2507.05619v1), [gradient fingerprints](https://arxiv.org/pdf/2604.16242)) | proxy vs true objective divergence | inspired the reward-structure probe (which did not help) |
| LLM performance prediction for fine-tuning ([survey](https://dsa.hkust-gz.edu.cn/blog/2026/05/29/a-survey-of-performance-prediction-for-llm-fine-tuningfrom-early-stopping-to-decision-oriented-post-training/)) | predicting LLM fine-tuning outcomes | nearest to the LLM-judge idea; none found for RL reward search |

**Novelty conclusion.** Not new: adaptive allocation among LLM rewards (ORSO), learning-curve forecasting, PID-style racing,
batching prompts. Tested and not supported: reward-structure early signals, a VoI+PID hybrid as a needed fix (PID already ties
VoI; VoI's failure was the wrong library). Defensible contributions: the evaluation methodology and findings (held-out,
multi-seed, live, pre-registered, vs ORSO), the winner's-curse evidence against single-seed selection, the measured
batching savings for reasoning models, and the library/cold-start diagnosis.

---

## 4. Chronology

| When | Phase | What happened |
| --- | --- | --- |
| 26 Oct 13:50 | start | repo review: all screening had been retrospective; round 2 had used full-training feedback |
| 14:00-19:00 | Quest live search | `live.py` built; resume semantics found (epoch counter not restored; ~10-20-it dip after resume); 16k token cap gave empty LLM replies → 32k; live trials 1-2 ran (trial 3 never started); H100 lane tried; seed stage designed; Quest socket died ~18:36 |
| 18:00-19:00 | Vast audit and setup | read-only Quest audit of the exact stack; CMP 170HX NO-GO; SSH key; box 1 built (tkinter fix); smoke identical; seeds 1-3 launched 18:55 |
| 19:00-24:00 | maximise GPU | merge/commits; batch runner, verified backups, provenance, ledger; G1 ablation; pruning simulator on 27 curves; LLM arm trial (22 calls); boxes 2 and 3 (3 failed); concurrency benchmark; box 2 network outage → proxy; caps raised with user credit |
| 22:00-01:00 | defensible design | held-out design; **rules frozen 23:41 (`e868927`)**; live-rule criterion; live-pruning machinery + smoke; generalisation generation (28 calls); **STATS_PLAN 00:14 (`4f7c0a1`)**, amendment 1 (`0a48a11`), LLM substitution (`4bd08b2`) |
| 27 Oct 01:00-07:45 | runs + analysis | interim H1 → VoI chosen; **live pruning run** (seed 6); held-out seeds 7-9; generalisation; LLM-trial second seed; G1 extra seeds; Gemini/Qwen screens + resumes; final analysis + 5 docs; box 2 done 06:22, box 1 done 07:41 |
| 06:30-08:00 | novelty | Humanoid library test; literature (ORSO found); structure probe; seed-allocation probe; ORSO baseline; LLM judge (pre-registered) |
| ~08:30 | statistics correction | critique: sets reuse 7 rewards → conservative seed-level re-analysis; H1 no longer survives Holm under it; claims scoped |

---

## 5. Every experiment and its results

### 5.1 Before this session (context)

| ID | What | Result |
| --- | --- | --- |
| Exp 1 | upstream 500-it Ant baseline | peak 8.01 |
| exp_a | human reward, 3,000 its | seed 0: 10.48, seed 1: 10.33 |
| exp_b | 3 + 2 search, all trained fully | R1 6.59 / 7.55 / 4.46; R2 10.93 / 10.75 |
| exp_c | retrospective fixed@500 screening of exp_b | 50% saved, 100% kept |
| exp_d | 6 extra round-1 samples; blind LLM ranking from code; zero-training probe | pool 10.73 (p3), 9.41, 9.25, 6.95, 5.28, 0.01; LLM ranked the best reward last |

### 5.2 Live early-stopping search (exp_e) [actual]

| Trial | Round-1 scores at 500 (kept) | Round-2 scores at 500 (kept) | Winner at 3,000 | Best in trial (audit) |
| --- | --- | --- | --- | --- |
| 1 | 0.02, **6.53**, 6.05, 5.74 | **7.08**, 6.15, 6.45, 5.75 | 9.29 | 9.29 (same) |
| 2 | 4.68, 0.01, **6.54**, 6.11 | 3.80, 6.08, 6.49, **7.01** | 9.61 | 10.43 |

Compute 6,500 vs 24,000 iterations per trial (**73% saved**); kept candidate retained 0.83 / 1.00 / 0.94 / 0.92 of each round's
best; top-1 correct 1/4 rounds; **false prunes 4/12 (33%, Wilson 14-61%)**, each when the leader at 500 led by < 0.8;
all 16 round-2 samples written from 500-it feedback were valid; round-2 mean 8.52 vs round-1 8.33 (no control).

### 5.3 Seed replication (exp_f) [actual; seeds ≥ 1 use 500 + resume]

Peak task score per seed (8 rewards × 9 seeds; seed 6 used for the live pruning run):

| Reward | s0 | s1 | s2 | s3 | s4 | s5 | s7 | s8 | s9 | mean of s1-9 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| p3 (pool) | 10.73 | 10.97 | 10.96 | 11.04 | 10.45 | 9.59 | 9.72 | 11.26 | 10.75 | 10.59 |
| t2r2c2 (live) | 10.43 | 10.43 | 9.46 | 9.14 | 10.94 | 10.53 | 10.45 | 10.38 | 10.92 | 10.28 |
| r2c1 | 10.75 | 10.32 | 10.06 | 9.71 | 11.48 | 9.69 | 10.02 | 9.04 | 9.92 | 10.03 |
| human | 10.48 | 10.16 | 7.89 | 9.74 | 10.64 | 9.70 | 10.58 | 9.92 | 9.33 | 9.74 |
| t1r2c0 (live winner) | 9.29 | 9.94 | 9.17 | 8.84 | 10.36 | 9.43 | 10.31 | 8.94 | 9.07 | 9.51 |
| t2r1c3 | 9.62 | 10.60 | 8.98 | 10.44 | 9.18 | 9.88 | 7.88 | 7.31 | 9.11 | 9.17 |
| **r2c0 (original "best")** | **10.93** | 9.71 | 8.09 | 9.21 | 8.65 | 10.56 | 9.71 | 8.30 | 8.99 | **9.15** |
| t2r2c3 (live winner) | 9.61 | 9.13 | 9.18 | 8.45 | 9.46 | 8.65 | 9.09 | 8.02 | 9.97 | 9.00 |

Also: seed-1 replication of all 10 viable live-round candidates; the 500-it pick equals the 3,000-it winner on 6/9 seeds (mean
retained 0.977); rank stability between seeds is low (Spearman 0.1-0.8).

**Resume control** (same seed, 500 + resume vs uninterrupted 3,000; human reward): peaks 10.16 vs 10.33 (s1), 7.89 vs 7.95 (s2),
9.74 vs 10.03 (s3) → differences −0.18, −0.07, −0.30. Every resumed run dips briefly after the resume (to 1.3-2.3) and recovers;
the strict pre-registered "no reset" rule failed on 1/8 seeds (19.7% vs a 20% threshold). Resumed numbers are labelled.

### 5.4 Desaturation ablation, G1 / H4 [actual, straight 3,000]

| Variant | s0 | s1 | s2 | s3 | mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| r1c1 unchanged (τ = 4) | 7.55 | 7.49 | 7.27 | 7.96 | 7.57 |
| r1c1, τ 4 → 10 | 8.95 | 9.69 | 9.80 | 7.71 | 9.04 |
| r1c1, τ 4 → 6 | 10.16 | 8.88 | - | - | 9.52 |
| r2c0 reverted τ 10 → 4 | 8.97 | 9.26 | - | - | 9.12 (r2c0 itself: 9.15 mean over 8 new seeds) |

### 5.5 LLM generation strategies on Ant, MiniMax (exp_h) [actual]

| Arm | Calls | Valid | $ | Best 500-it (s0 / s1) | Median 500-it (s0 / s1) | Tokens per valid |
| --- | ---: | ---: | ---: | --- | --- | ---: |
| A one per call (17 earlier samples) | 17 | 17 | 0.127 | 7.29 / - | 5.48 / - | 12,154 |
| B four per call | 3 | 12 | 0.033 | **8.39** / 6.21 | 4.49 / 4.88 | **4,530** |
| C four per call, named strategies | 3 | 8 | 0.036 | 7.00 / 7.21 | 5.52 / 5.36 | 7,340 |
| D one per call + strategy hint | 12 | 7 | 0.165 | 5.99 / 6.68 | 3.43 / 4.04 | 38,704 |
| E one per call, "think briefly" | 4 | 4 | 0.028 | 4.33 / 4.30 | 1.00 / 0.17 | 11,270 |

Human at 500 its: 8.01 (s0), 6.99 (s1). Top 6 (all batched) resumed to 3,000: 11.11, 10.42, 10.14, 9.28, 8.82, 8.50.

### 5.6 Other LLMs on Ant (exp_k) [actual, seed 0]

| Model, arm | Calls | Valid | $ | Best / median 500-it | Top resumed to 3,000 |
| --- | ---: | ---: | ---: | --- | --- |
| Gemini 2.5 Flash, one per call | 8 | 5 | 0.108 | 5.18 / 1.27 | 6.90, 3.18 |
| Gemini 2.5 Flash, four per call | 2 | 4 of 7 | 0.069 | 2.07 / 1.88 | 6.07, 4.42 |
| Qwen-3-32B, one per call | 8 | 8 | 0.030 | 7.16 / 2.50 | **11.32**, 9.81, 6.74 |
| Qwen-3-32B, four per call | 2 | 7 of 8 | 0.008 | 4.07 / 2.32 | 5.41 |

Batching cheaper per valid reward for both (1.2×, 3.3×). Qwen's 11.32 is the best single-seed reward of the project.

### 5.7 Generalisation to other robots (exp_i) [actual]

| Task | Calls (1/call + 4/call) | Valid (1/call, 4/call) | Tokens per valid (1/call vs 4/call) | Runs | Human peak | LLM rewards vs human (mean, best) |
| --- | --- | --- | --- | ---: | --- | --- |
| Cartpole | 8 + 2 | 6/8, 8/8 | 5,048 vs 1,347 (3.7×) | 30 | 499 / 499 | 1.00 / 1.00 (saturated) |
| BallBalance | 8 + 2 | 8/8, 8/8 | 17,148 vs 6,658 (2.6×) | 34 | 499 / 499 | 0.90-1.00 / 1.00 (saturated) |
| Humanoid | 6 + 2 | 5/6, 8/8 | 18,374 vs 3,346 (5.5×) | 15 (2 crashed) | 7.13 / 5.03 | 1/call 0.33 / 0.76; 4/call 0.12 / 0.37 |

Cost $0.21. Two Humanoid rewards crashed at run time (wrong `Tensor` annotation on a float; out-of-range index) → failed candidates.

### 5.8 Pruning rules [held-out]

Frozen on TRAIN (seeds 0-3, 65 curves) before held-out data existed; candidate sets = same-seed groups of 4 from the 7 strong
rewards; libraries exclude every curve of the rewards being judged. Held-out seeds 4, 5, 7, 8, 9 → 175 sets.

| Rule | Compute | Kept [95% CI] | Winner |
| --- | ---: | ---: | ---: |
| random | 0.250 | 0.908 | 26% |
| **fixed@500 (reference)** | 0.375 | 0.972 [0.952, 0.991] | 59% |
| PID | 0.413 | 0.991 [0.979, 0.999] | 78% |
| MPC (receding horizon) | 0.418 | 0.995 | 83% |
| **VoI** | 0.425 | **0.997 [0.993, 0.999]** | **86%** |
| PD | 0.440 | 0.991 | 78% |
| Bayesian pruning | 0.453 | 0.996 | 85% |
| Thompson best-arm (bandit) | 0.454 | 0.996 | 87% |
| successive halving | 0.458 | 0.988 | 75% |
| ORSO-D3RB (rebuilt, post hoc) | 0.475 | 0.990 | 78% |
| PI | 0.476 | 0.997 | 87% |
| fixed@1000 | 0.500 | 0.992 | 81% |
| ratio-bound racing | 0.535 | 0.997 | 87% |
| Hyperband (resumable) | 0.547 | 0.992 | 87% |
| P | 0.560 | 0.997 | 87% |
| ORSO-Exp3 (rebuilt) | 0.617 | 0.980 | 68% |
| human-reference threshold | 0.615 | 0.997 | 87% |
| Hyperband as published | 3.055 | 1.000 | 100% |

Other pruning results: equal-compute fixed checkpoint (fixed@698) 0.981 → VoI still +0.015; clean seeds 7-9: VoI 0.997 vs fixed
0.975, PID 0.997; offline cross-validation on the 27-curve pool showed the same ordering; leave-one-round-out on the 6 real rounds:
VoI kept the best in 6/6, fixed@500 in 4/6.

**Live pruning confirmation (exp_j) [actual]:** seed 6, 8 rewards, VoI really paused/killed processes: 7,769 of 24,000 its
(32%), 6 runs killed at 250-850 its and 1 at 1,551, survivor r2c1 peaked 11.09; replay of the same rule on the recorded curves
made identical decisions (overshoot ≤ 6 its per arm).

**Transfer to other robots [held-out task]:** with the Ant library, on Humanoid VoI kept 0.814 and MPC 0.805 vs fixed@500 0.972,
PID-family 0.972 at 0.32 compute, successive halving 0.994; Cartpole/BallBalance every rule 1.000 (saturated).
**Humanoid with a Humanoid library (leave-one-out, 11 curves) [exploratory]:** VoI 0.814 → 0.971 (winner 75% → 95%), Bayesian
0.972 → 1.000, MPC 0.805 → 0.930; PID/fixed unchanged.

### 5.9 Novelty probes [exploratory, offline]

- **Reward-structure signals** (component trends, own-reward vs task-score gap), pair prediction split by reward identity
  (104 runs, 27 rewards): at 500 its 0.77 with structure vs 0.79 without; close pairs 0.71 vs 0.69 → no improvement.
- **Seeds vs horizon** (7 rewards × 9 seeds; truth = mean over held-out seeds): at full screening budget 2 seeds × 1,500 its finds
  the true best 55% vs 48% for 1 × 3,000; quality kept 0.967 vs 0.962; at small budgets no gain.
- **LLM judge** (pre-registered `1703d9c`): MiniMax shown code + first 500 its of logs per candidate, seeds 7-9, 105 sets, equal
  compute to fixed@500: kept 0.987 vs 0.975, winner 74% vs 56%; +0.013 [−0.001, +0.032], p = 0.057 (0.17 with seed as unit);
  all replies parsed; $0.74. Code alone (earlier) had failed badly. Test closed; not to be extended on the same curves.

### 5.10 Generate vs train [simulation]

From the llm-efficiency simulator (84 splits of real curves): at 40-250 PPO iterations per LLM call, adaptive generate-or-train
rules match but do not beat "generate 4, screen at 500" at equal cost (gaps −0.11 to +0.18); gains appear only at ≥ 1,000
iterations per call. Measured on Vast: one MiniMax call ≈ 160 iterations of A100 time (≈ 60 per reward when batched).

---

## 6. Statistics

**Pre-registrations (all committed before the governed data existed):** frozen rules `e868927`; live-rule criterion `b9c1fc2`;
`STATS_PLAN.md` `4f7c0a1` (H1-H5, one-sided, Holm, seed as the unit of noise); amendment 1 `0a48a11` (H1 on seeds 4, 5, 7, 8, 9,
run unconditionally; other-LLM study); substitution `4bd08b2`; LLM judge `1703d9c`.

**Original primary output (historical; H1 bootstrap and H2 permutation inference are qualified below and in `AUDIT_REPORT.md`):**

| | Effect | p | Holm p |
| --- | --- | ---: | ---: |
| H1 VoI > fixed@500 | +0.025 [0.006, 0.046] | 0.0019 | 0.0076 |
| H2 batching cheaper | 2.6-5.5× on 4/4 tasks | original <0.0001 (invalid test); task sign test 0.0625 | invalid |
| H3 batching not worse (−10% margin) | −0.023 [−0.120, +0.073] | 0.064 | 0.128 |
| H4 τ edit | +1.47 (n 4 vs 4) | 0.025 | 0.075 |
| H5 p3 > human | +0.86 (6 paired seeds) | 0.073 | 0.128 |

**Conservative re-analysis (after critique; `results/pruning/conservative_reanalysis.json`).** Held-out sets recombine the same
7 rewards, so they are not independent, and a bootstrap over 3-5 seeds is optimistic. With each seed as one observation:
H1 per-seed differences +0.057, +0.002, +0.029, 0.000, +0.037 → **p = 0.041 uncorrected, 0.21 after Holm (not significant)**;
dropping any one reward keeps it positive (+0.017 to +0.033); clean seeds 7-9 p = 0.095; PID vs fixed@500 p = 0.030; LLM judge
p = 0.17. **What may be claimed:** H2 as descriptive cost savings; H1 as a consistent improvement on this reward pool (never negative on any seed or
reward subset) but not a firm statistical win and not yet for new rewards; H3-H5 and the LLM judge not established.

---

## 7. Every idea considered, and its fate

| Idea | Status | Outcome |
| --- | --- | --- |
| Stop everyone at 500 (fixed screen) | live + held-out | saves 63-73%; misses close winners |
| Round 2 from 500-it feedback | live, 2 trials | works mechanically; quality gain unproven |
| Fixed checkpoints 100-1,500 | held-out | 1,000 keeps 0.992 at 0.50 compute |
| Successive halving | held-out | did not beat fixed@500 |
| Hyperband (resumable / published) | held-out | resumable ≈ others; published 3× cost |
| Threshold, ratio-bound racing | held-out | 0.997 at 0.54-0.62 compute |
| P / PI / PD / PID | held-out | PID 0.991 at 0.41; transfers to Humanoid |
| Bayesian, Thompson, MPC | held-out | 0.995-0.996 at 0.42-0.45 |
| **VoI** | held-out + live | **best on Ant**; needs same-task library |
| ORSO (bandit chunks) | rebuilt baseline | beaten by VoI, tied by PID at lower compute |
| VoI + PID hybrid / cold-start trust gate | discussed, not built | not justified by data |
| Reward-structure signals | probe | no improvement |
| More seeds instead of longer screens | probe | small gain at full budget |
| Batch 4 per call | 4 tasks, 3 LLMs | 2.6-5.5× cheaper |
| Named strategies / hints / "think briefly" | actual | no better or worse |
| Zero-token constant edit | actual, 4 seeds | +1.47, not significant after correction |
| LLM blind ranking (code only) | earlier | failed |
| LLM judge with early logs | pre-registered | promising, not significant |
| Generate-vs-train joint allocation | simulation | no gain at measured prices |
| Other LLMs | actual | weaker on average; Qwen best single reward 11.32 |
| Other robots | actual | easy tasks saturate; Humanoid LLM rewards weak |
| H100, CMP 170HX, 6 runs per GPU, more boxes | tried / audited | no gain or not worth it |

---

## 8. Infrastructure and incidents

**Code built this session:** `live.py` (live search, seeds, resume, packing), `live_task.py` (any task), `gen_task.py`
(generation for any task/LLM), `live_prune.py` (live pruning with SIGSTOP/SIGKILL + replay check), `pruning/heldout.py`
(frozen-rule held-out scoring), `pruning/run_pool_cv.py`, `orso_baseline.py`, `structure_probe.py`, `seed_allocation.py`,
`final_analysis.py`, `exploratory_analysis.py`, `llm_judge.py`, `analyze_live.py`, `analyze_seeds.py`; `vast/runner.sh`
(queue + spend cap), `backup.sh` (append-only, sha256-verified), `provenance.sh`, `ledger.py`, `setup_box.sh`,
`restart_runner.sh`. Every run keeps reward code, LLM request/response, raw `train.log` (first line = full command incl. seed),
Hydra config, tensorboard events, `run_info.txt`; final checkpoints in `~/MSThesis_checkpoints/`.

| Incident | Handling |
| --- | --- |
| 16k-token cap → empty replies | 32k cap via env var |
| Quest queue ~600 deep; SSH socket died | moved to Vast |
| `tkinter` missing on Ubuntu Python | apt package |
| shell pitfalls (zsh splitting, `pkill -f` self-match, exec bit, no `timeout`) | wrappers, PID kills, `chmod`, `setsid -f` |
| runner restarts mid-batch | runner waits for running packs; batches idempotent |
| box 3 pip stall; inter-box key blocked by safety check | user shut it off |
| box 2 port down | Vast proxy |
| live batch picked up late | moved to the other box |
| backup mismatches on files being written | manifest on settled files only |
| Humanoid reward crashes | counted as failed candidates |
| resume-control strict rule failed on 1 seed | reported; resumed numbers labelled |
| doc errors (run counts, H3 n, "best reward" attribution) | corrected in later commits |
| statistical over-confidence (overlapping sets) | conservative re-analysis; claims scoped |

---

## 9. Costs and compute

- PPO: 384 completed training phases (280 from scratch, 104 resumed), 451,000 iterations ≈ 3.02 × 10^10 environment steps,
  plus the live pruning run (`results/LEDGER.json`).
- Vast: ~$18.5 estimated (credit $21); Quest free. LLM this session: MiniMax $1.22 (trial 0.26 + generalisation 0.21 + judge
  0.74), Gemini $0.18, Qwen $0.04.

---

## 10. Open items

1. **Fresh-reward replication (~$3.5):** new batched LLM rounds, each reward trained fully once, rules frozen, a pre-registered
   reward-and-seed-level test (the only way to support H1 for new rewards). Rent an A100 PCIe 40GB, rebuild with `vast/setup_box.sh`.
2. More seeds for H4/H5 (~$3); a hard second task with an in-task library (Humanoid, ShadowHand); H3 with ~3× more rewards.
3. Any LLM-judge follow-up only as a new pre-registered replication on fresh rewards.
4. Rotate the API key that was pasted in chat. Destroy the Vast instances (all data is on the Mac).

---

## 11. Files

| Purpose | Files |
| --- | --- |
| Write-ups | `THESIS_EXPERIMENT_SUMMARY.md`, `PRUNING_COMPARISON.md`, `LLM_EFFICIENCY_RESULTS.md`, `GENERALIZATION_RESULTS.md`, `EXPERIMENT_MANIFEST.md`, `STATS_PLAN.md`, this file, earlier `REPORT.md`, `LLM_EFFICIENCY_FEASIBILITY.md` |
| Results | `results/exp_*` (per run), `results/pruning/`, `results/final/`, `results/LEDGER.json` |
| Provenance | `provenance/`, `backups/`, `vast/box_logs/`, `vast/queue_batches/` |
