# Master log: everything ideated, built, run and found (26-27 Oct 2026)

A chronological, complete record of the work done in this session: the ideas, the reasoning behind each decision, every
experiment, every failure and fix, the costs, and the results. Times are CDT (UTC−5) unless marked UTC. Commit hashes refer to
`main` in this repository. For the clean result write-ups see `THESIS_EXPERIMENT_SUMMARY.md` (start there),
`PRUNING_COMPARISON.md`, `LLM_EFFICIENCY_RESULTS.md`, `GENERALIZATION_RESULTS.md`, `EXPERIMENT_MANIFEST.md`, `STATS_PLAN.md`.

---

## 0. The research question in one paragraph

Eureka-style systems ask an LLM to write reward functions for a robot, then judge each candidate by training a PPO policy with
it in simulation. Training is the expensive part (one Ant run = 3,000 PPO iterations ≈ 12 GPU-minutes; a search trains many
candidates). The thesis asks: **(1) can we stop training weak candidates early without losing the good ones (PPO compute), and
(2) can we use the LLM more cheaply (generation cost)?** Everything below serves those two questions.

---

## 1. Where the project stood at the start of the session (26 Oct, ~13:50)

Already in the repo (from earlier work, on Northwestern's Quest cluster):

| Experiment | What it was | Key numbers |
| --- | --- | --- |
| Experiment 1 | Upstream 500-iteration Ant PPO baseline | peak 8.01 |
| exp_a | Human (hand-written) reward, 3,000 its, seeds 0-1 | 10.48 / 10.33 |
| exp_b | Mini-Eureka: 3 round-1 samples, then 2 round-2 samples written from full-training feedback (MiniMax M3.1 Flash) | R1 6.59 / 7.55 / 4.46; R2 10.93 / 10.75 |
| exp_c | Retrospective screening of exp_b: "stop all at 500, keep the leader" | saved 50%, kept 100% |
| exp_d | 6 extra round-1 samples, blind LLM pre-ranking of rewards (code only), a zero-training "probe" | LLM ranked the best reward last |

Two feasibility branches existed: `llm-efficiency` (token accounting, a generate-vs-train simulator, a prepared but unrun LLM
trial plan) and `controller-feasibility` (a causal pruning simulator with fixed, successive halving, Hyperband, PID, Bayesian,
MPC and Thompson-sampling rules, a MuJoCo Ant twin). An earlier local Qwen arm had been removed.

**The weakness I flagged on reading the repo:** all screening was retrospective. The exp_c "100% retained" came from replaying
finished curves, and exp_b's round 2 was written from *full*-training feedback, so the claim "you can stop at 500" had never
been tested live.

---

## 2. Phase 1 - live early stopping on Quest (26 Oct, 14:00-19:00)

**Request:** actually stop weak runs at 500 iterations and write the next round only from that partial feedback; if it works,
replicate the strongest rewards over seeds.

**Design (`live.py`):** per trial, 4 LLM samples → each trains 500 its → keep the best (peak score within 500) → round 2 (4 more
samples) is written from the kept run's 500-iteration log in upstream Eureka's feedback format → same stop rule → the overall
best resumes from its checkpoint to 3,000 → *audit*: the stopped runs are also resumed to 3,000 afterwards, only to measure
what stopping cost. Three trials were planned.

**Engineering facts discovered:**
- This rl_games fork restores weights, normalisation and optimiser on resume but **not the epoch counter**, so a "500 then
  continue" run is 500 + 2,500 epochs.
- Resuming causes a **brief score dip for ~10-20 iterations** (e.g. 7.06 → 2.8 → 7.1) that recovers; peaks are unaffected.
- **Token-cap bug:** through the Vercel gateway, MiniMax at `max_tokens=16384` returned empty replies (all hidden reasoning,
  `finish_reason=length`) 2 of 2 times. Raised to 32,768 (`EUREKA_MAX_TOKENS` env var added to `eureka_ab.py`); 4/4 replies,
  3/4 valid.
- Quest's GPU queue was ~600 jobs deep; estimated starts drifted between 16:17 and 01:28 next day. A short H100 smoke job was
  tried as a faster lane; it was no faster and was cancelled.

**What ran:** smoke test (resume + feedback machinery: passed), live trials 1 and 2 (2 rounds × 4 samples each, all 16 valid,
plus the 16-run audit). Trial 3 never started (Quest access was lost, see below).

**Results [actual]:**
- Live stopping used 6,500 instead of 24,000 iterations per trial: **73% saved**.
- Winners after 3,000 its: 9.29 (trial 1, the true best of its trial) and 9.61 (trial 2; the trial's best was 10.43).
- The kept candidate retained 0.83, 1.00, 0.94, 0.92 of each round's best; top-1 correct in 1 of 4 rounds.
- **False prunes: 4 of 12 stopped runs would have won** (33%); each time the leader at 500 was ahead by < 0.8 points.
- The earlier retrospective "100% retained" was partly luck of a small sample.

I also built the first seed stage (`live.py seeds`: 7 strongest rewards + human on new seeds, same 500 + resume protocol) and
a self-checking pinger (a recurring scheduled prompt) to watch jobs.

**Incident:** at ~18:36 the user's Quest SSH control socket died and password login was not possible from my side; Quest work
stopped there (later the user said Quest is gone).

---

## 3. Phase 2 - moving to rented GPUs (Vast.ai) (26 Oct, 18:00-19:00)

**Pre-purchase audit (read-only on Quest):** recorded the exact working stack (Python 3.8.20, torch 2.0.0+cu117 from pip,
Isaac Gym Preview 4, Eureka @9eee428, numpy 1.20.0, hydra 1.3.2; Quest GPU nodes A100 PCIe-40GB and SXM-80GB). Key findings:
no CUDA toolkit is needed (torch brings its own; `gymtorch` compiles as plain C++), training is headless (no Vulkan/GL),
the host driver matters more than the Docker image's CUDA version. Recommended an A100 PCIe 40GB with driver ≥ 515 and **NO-GO
on the CMP 170HX** (a mining card: 8 GB stock, modified for "64 GB", throttled FP32, narrow PCIe).

**Setup:** generated a dedicated SSH key; box 1 (A100 PCIe 40GB, driver 595) built from the Mac copies of Isaac Gym and Eureka,
venv with the exact Quest pip freeze. One portability gap: Ubuntu's Python lacked `tkinter` (imported by an upstream task file)
→ installed. **Smoke results identical to Quest to the last digit** (human peak 0.4689645767211914 etc.).

**Seed stage launched on box 1 at 18:55** (seeds 1-3 × 8 rewards, ~4.4 h).

---

## 4. Phase 3 - "maximise the GPU" (26 Oct, 19:00 - 27 Oct, 00:00)

The user asked for maximum GPU use, a single agent on `main`, and a long plan (pasted) covering pruning comparisons, LLM
efficiency and joint allocation.

**Housekeeping:** committed the live work (`2f946a8`), merged `llm-efficiency` (`a4dc5d9`); copied (not merged) the controller
simulator into `pruning/`. Commits are authored by the user only, no co-author trailer (user rule).

**GPU queue infrastructure (all in `vast/`):**
- `runner.sh`: a batch runner that takes the lowest-numbered JSON batch from `queue/`, runs it via `live.py pack` (3 runs per
  GPU), moves it to `done/`, and refuses any batch whose estimated cost would pass a spending cap. Later versions: combined
  spend over several boxes from `boxes.json`, never start a batch while another pack runs, per-batch training script.
- `backup.sh`: append-only pull of every result (raw `train.log`, tensorboard events, Hydra config, curves, reward code) to the
  Mac, verified against a sha256 manifest made on the box (only files unchanged for 5+ minutes; running logs are picked up
  later). Final checkpoints copied outside git to `~/MSThesis_checkpoints/`.
- `provenance.sh`: snapshot of code hashes (byte-identical to the repo), upstream-tree hashes, pip freeze, `nvidia-smi -q`,
  filtered environment (no credentials), queue state.
- `ledger.py`: running count of completed phases, PPO iterations, GPU-hours, reward-seed pairs.

**G1 desaturation ablation (idea from the llm-efficiency study):** the LLM's round-2 improvement of reward r1c1 seemed to be
mostly one constant, the forward-speed tanh temperature 4 → 10. Variants: r1c1 with τ=10, r1c1 with τ=6, r2c0 reverted to τ=4,
and r1c1 unchanged on seed 1; straight 3,000 its, seeds 0-1 (later seeds 2-3 for the key arms). Pre-registered reading:
τ=10 mean ≥ 9.0 → the edit explains most of the gain.

**Offline pruning simulator on the bigger pool:** extended with the 16 live curves, Hyperband exactly as in the paper, a VoI
(value-of-information) rule and a 1,500-it checkpoint; held-out cross-validation (tune on half the curves, score on the other
half) and leave-one-round-out on the 6 real rounds. Sanity check: the simulator replays the live rounds exactly (fixed@500
retained 0.83/1.00/0.94/0.92, as observed). First signal: forecasting rules keep ~98-99% vs ~95% for fixed@500 at slightly more
compute; paper Hyperband costs ~3× training everything.

**A1 / A10 tables:** false-prune rate 4/12 (above); no "slow starters" in 27 curves (500-it vs 3,000-it rank correlation 0.82
across the whole pool, but close calls within a round are coin flips).

**LLM trial (approved by the user, key provided in chat; advised rotation):** the prepared 22-call plan. Arms:
B = four rewards per call, C = four per call with four named strategies, D = one per call with a strategy hint,
E = one per call "think briefly"; A = the existing independent samples (no new calls). Cost **$0.262**. Valid rewards: B 12/12,
C 8/12, D 7/12 (2 replies hit the 32k cap), E 4/4. All 31 valid rewards got 500-it screens (later on a second seed too); the 6
best (all batched) were resumed to 3,000.

**More GPUs:**
- **Box 2** (same card/driver class): built in ~15 min, smoke identical; queue split.
- **Concurrency benchmark:** 6 runs per GPU = 5.08 it/s vs 3 runs = 5.2-5.5 it/s → keep 3 (the card is saturated; a single
  run alone ≈ 4.0 it/s, measured 0.251 s/it solo).
- **Box 3:** its pip install stalled on a slow network; copying box 2's environment across was blocked by a safety check
  (adding an SSH key between boxes); the user shut it off. No training ran; its ~1.1 h were folded into the spend estimate.
- **Box 2 outage:** its direct SSH port stopped answering at ~21:15; reached it through Vast's proxy (`ssh4.vast.ai`), no data
  lost, all scripts rerouted transparently.

Credit/caps over the night: $10 → +$5 (cap $13 → $14) → +$6 (cap $20).

---

## 5. Phase 4 - turning data into defensible results (26 Oct 22:00 - 27 Oct 01:00)

The user asked for actual, defensible results on both hypotheses, with held-out testing, live confirmation, and labels on
every number (actual run / retrospective held-out / simulation).

**Held-out pruning design (`pruning/heldout.py`):**
- One causal interface for every rule: a rule sees only the prefix each candidate has trained so far and decides keep/stop.
- TRAIN = all curves of seeds 0-3 (65 curves); TEST = seeds 4 and 5 (later 4, 5, 7, 8, 9) of the 8 strong rewards.
- Candidate sets = same-seed groups of 4 (as in a real round). Libraries never contain any curve of a reward in the set
  (leave-reward-out), so forecasts can't peek at other seeds of the same reward.
- **Rules frozen on TRAIN and committed before any held-out curve existed (`e868927`, 23:41).**

**Pre-registrations (all committed before the data they govern):**
- `results/pruning/best_rule_criterion.md` (`b9c1fc2`): which rule gets the live test.
- `STATS_PLAN.md` (`4f7c0a1`, 00:14): five primary hypotheses, tests, units (a PPO seed, never an iteration), Holm correction:
  H1 adaptive pruning > fixed@500; H2 batching cheaper per valid reward; H3 batching not worse (10% margin); H4 the one-constant
  edit recovers the refinement gain; H5 the best reward beats the human reward.
- Amendment 1 (`0a48a11`): H1's held-out set fixed to seeds 4, 5, 7, 8, 9, run unconditionally (to avoid optional stopping),
  plus an exploratory second/third-LLM study. Substitution (`4bd08b2`): DeepSeek and Qwen 3.8 Flash were refused on the free
  gateway tier (HTTP 403, $0) → Gemini 2.5 Flash and Qwen-3-32B.
- Swapped a planned "borderline candidate seeds" batch for 2 extra G1 seeds (same cost, more power for H4).

**Live pruning machinery (`live_prune.py`):** 8 candidates train concurrently and continuously; a controller reads each run's
tensorboard log, pauses it (SIGSTOP) at the rule's checkpoint, applies the frozen rule to exactly the logged prefix, kills
(SIGKILL) dropped runs, continues the rest; afterwards the same rule is replayed on the recorded curves. Smoke test (2 arms,
20-iteration checkpoint): pause/kill/replay matched. One bug fixed: child processes needed the venv on PATH for `ninja`.

**Generalisation study (LLM calls approved by the user):** `gen_task.py` (upstream prompts for any task; one-per-call vs
four-per-call arms) and `live_task.py` (training for any task). Cartpole 100 its, BallBalance 250 its, Humanoid 1,000 its;
28 calls, $0.21; 43 valid rewards; every valid reward + human trained to full length (Cartpole/BallBalance seeds 0-1,
Humanoid seed 0, human 0-1).

**Statistical honesty decisions:** the original analysis's resume control criterion compared against other seeds; replaced by
a same-seed comparison and added straight human runs on seeds 2-3 as controls.

---

## 6. Phase 5 - the night's runs and the final analysis (27 Oct 01:00-07:45)

All on Vast boxes 1-2, in batches (`vast/queue_batches/`). Key milestones:
- **Interim H1 (seeds 4-5) → rule selection by the pre-registered criterion: VoI (λ=1.0, first decision at 250 its)**
  (`90e1572`).
- **Live confirmation (seed 6, 8 rewards) [actual]:** VoI used 7,769 of 24,000 iterations (32%), killed 7 runs, kept r2c1
  (peak 11.09); replay decisions identical, overshoot ≤ 6 iterations per arm (`11eef5c`).
- Held-out seeds 7-9, the generalisation runs, the second-seed LLM-trial screens, G1 extra seeds, the solo calibration run,
  Gemini/Qwen screens and top-4 resumes.
- Two Humanoid LLM rewards crashed at run time (a wrong `Tensor` type annotation on a float; an out-of-range index): counted as
  failed candidates, as upstream Eureka would.
- `final_analysis.py` and `exploratory_analysis.py` were written and dry-run while the GPUs worked, so results came minutes
  after the last batch.

**Final pre-registered results (Holm across H1-H5):**

| | Result | Holm p | Verdict |
| --- | --- | ---: | --- |
| H1 | VoI 0.997 vs fixed@500 0.972 retained (+0.025 [0.006, 0.046]); winner 86% vs 59%; compute 0.42 vs 0.375; 175 held-out sets | 0.0076 | **significant** as pre-registered; **not** under the stricter seed-level test added later (p = 0.041, Holm 0.21) |
| H2 | four-per-call 2.6-5.5× fewer tokens per valid reward on 4/4 tasks | <0.0001 | **significant** |
| H3 | batch − one-per-call quality −0.023 [−0.12, +0.07] | 0.128 | inconclusive |
| H4 | τ 4→10: 7.57 → 9.04 (+1.47), n = 4 vs 4 | 0.075 | not significant after correction (pre-registered reading met) |
| H5 | p3 vs human: +0.86 over 6 paired seeds | 0.128 | not significant |

Extra H1 checks: vs a fixed checkpoint of equal compute (+0.015, p = 0.0099); on seeds 7-9 only, the seeds never used to choose
the rule (+0.022 [0.000, 0.039], p = 0.036).

**Exploratory results:** winner's curse (the original 10.93 reward averages 9.15 on 8 other seeds; human reward 7.9-10.6 by
seed); the 500-it pick equals the 3,000-it winner on 6 of 9 seeds; Cartpole/BallBalance saturate (every reward hits the ceiling)
so they can't test anything; on Humanoid all LLM rewards were far below the human reward; the Ant-tuned VoI did not transfer to
Humanoid (0.81 retained); generate-vs-train allocation (simulation, with measured prices: one LLM call ≈ 60-230 PPO iterations
of A100 time) does not beat "generate 4, screen at 500".

**Resume control:** same-seed split-vs-straight peaks differ by −0.18, −0.07, −0.30 (seeds 1-3), but the strict pre-registered
"no reset" check failed on 1 of 8 seeds (dip to 19.7% vs a 20% threshold). Reported as such; resumed numbers are labelled.

**Shutdown:** box 2 finished at 06:22 (backed up, provenance snapshot); box 1 finished at 07:41 (backed up, 1,859 files, 0
mismatches; home-folder logs archived to `vast/box_logs/box1`). The user is destroying both instances; nothing needed is lost
(box 2's packing console log could not be copied after it was stopped; its queue log is in its provenance snapshot).

---

## 7. Phase 6 - the novelty question (27 Oct 06:30-07:45)

**User concern:** VoI looked overfit and PID strong; could a VoI + PID hybrid be a novel algorithm?

1. **Diagnosis, not overfitting:** VoI forecasts from a *library* of past runs. On Humanoid it had only Ant runs. With a
   leave-one-out Humanoid library: VoI 0.814 → **0.971** retained (winner 75% → 95%), Bayesian 0.972 → 1.000; PID/fixed unchanged
   (they use no library). So forecasting rules need history from the same task; this is a cold-start problem.
2. **Literature check:** the closest prior work is **ORSO (Zhang et al., ICLR 2025)**: online selection among LLM-generated
   rewards in Isaac Gym (Cartpole, BallBalance, Ant, Humanoid, ...), training in 1% chunks, chosen by bandit model selection
   (D3RB, Exp3, UCB, ETC) on the current task score; up to 8× compute reduction. No forecasting, no reward-component signals.
   Related: learning-curve extrapolation (Domhan 2015), PASHA, LC-PFN, freeze-thaw BO, BOHB, calibrated BO, the 2024 finding that
   fixed early discarding is hard to beat, batch prompting (Cheng et al. 2023; BatchPrompt, ICLR 2024). Conclusion: "adaptive
   allocation" and "VoI + PID" are not new by themselves.
3. **Novelty probe 1: reward-structure signals (offline)** [exploratory]: do per-component trends and the gap between a
   run's own reward and the task score predict the winner better than the score? **No** (pair accuracy at 500 its: 0.77 with
   structure vs 0.79 without; split by reward identity, 104 runs, 27 rewards). Not pursued further (no feature fishing).
4. **Novelty probe 2: seeds vs horizon (offline):** at full screening budget, 2 seeds × 1,500 its finds the truly best reward
   (mean over held-out seeds) 55% vs 48% for 1 seed × 3,000, but retained quality barely changes (96.7% vs 96.2%). Small effect.
5. **ORSO baseline (post hoc, tuned on training seeds only):** on held-out seeds, ORSO-D3RB 0.990 retained at 0.475 compute;
   **VoI beats it (+0.007, p = 0.012) with 5% less compute; PID ties it with 6% less compute.**
6. **LLM-as-judge from early logs** (pre-registered `1703d9c` before any call, ~$1): can MiniMax pick the eventual winner from
   reward code + 500-it logs, better than fixed@500 at equal compute? LLM judge result (105 sets, all replies parsed, $0.74): at the same compute as fixed@500, the LLM's pick kept **0.987** of
   the best reward vs **0.975** for fixed@500 and found the winner **74% vs 56%** of the time; difference +0.013 [−0.001, +0.032],
   **p = 0.057, just missing the pre-registered p < 0.05: promising, not significant.** For context, VoI and PID keep 0.997 on
   the same seeds but spend ~0.42 compute vs 0.375. Contrast with the earlier blind ranking from code alone, which failed badly:
   the training logs are what make the LLM a useful judge. `results/pruning/llm_judge.json`.

**Where novelty stands (agreed framing):** no new algorithm is supported by the data. The contribution is a rigorous,
pre-registered, held-out, multi-seed and live evaluation of compute allocation for LLM reward search, with clear findings
(VoI and PID beat fixed stopping and a rebuilt ORSO; forecasting needs in-task history; single-seed selection is unreliable;
batched generation is much cheaper) and documented negative results.

---

### Statistical correction after a reviewer-style critique (27 Oct ~08:30)

The held-out "sets" recombine the same 7 rewards, so they are not independent. Re-analysed with each seed as one observation
plus drop-one-reward sensitivity (`results/pruning/conservative_reanalysis.json`): H1 p = 0.041 uncorrected (n = 5 seeds),
0.21 after Holm, so not significant under the stricter test though the direction is consistent (never negative per seed or
per dropped reward); clean seeds 7-9 p = 0.095; PID vs fixed@500 p = 0.030 uncorrected; LLM judge p = 0.17. Decisions: report
both analyses; scope H1 to this reward pool; do not extend the LLM-judge test on the same curves (optional stopping); spend
remaining credit on fresh rewards with a pre-registered reward-and-seed-level test.

## 8. Every idea considered, and what happened to it

| Idea | Status | Outcome |
| --- | --- | --- |
| Stop everyone at 500 its (fixed screen) | live + held-out | saves 63-73%, but misses close winners (33% false prunes live) |
| Round 2 from 500-it feedback | live, 2 trials | works mechanically; quality gain unproven (no control) |
| Fixed checkpoints 100-1,500 | held-out | 1,000 keeps 0.992 at 0.50 compute |
| Successive halving | held-out | did not beat fixed@500 |
| Hyperband (resumable / as published) | held-out | resumable ≈ others; as published costs 3× (collapses at ≤12 candidates) |
| Human-reference threshold, ratio-bound racing | held-out | keep ~0.997 at 0.54-0.62 compute |
| P / PI / PD / PID controllers | held-out | PID 0.991 at 0.41; PI 0.997 at 0.48; transfers to Humanoid |
| Bayesian pruning, Thompson best-arm, MPC | held-out | 0.995-0.996 at 0.42-0.45 |
| **VoI (expected improvement per cost)** | held-out + live | **best: 0.997 at 0.425; live 32% compute** |
| ORSO bandit allocation | rebuilt baseline | 0.990 at 0.475; beaten by VoI, tied by PID |
| VoI + PID hybrid / cold-start gate | discussed, not built | VoI's Humanoid failure explained by library; PID already ties VoI |
| Reward-structure (component, proxy-gap) signals | offline probe | no improvement |
| More seeds instead of longer screens | offline probe | small gain at full budget only |
| Batch 4 rewards per call | actual, 4 tasks, 3 LLMs | 2.6-5.5× cheaper (MiniMax), also cheaper for Gemini/Qwen |
| Four named strategies per call | actual | lower validity, no better best |
| Strategy hints, "think briefly" | actual | worse rewards (and hints 5× costlier) |
| Zero-token constant edit (desaturation) | actual, 4 seeds | +1.47 points; not significant after correction |
| LLM blind pre-ranking (code only) | earlier, actual | failed (best reward ranked last) |
| LLM judge with early logs | actual (pre-registered), 105 held-out sets | better than fixed@500 at equal compute (0.987 vs 0.975, winner 74% vs 56%), p = 0.057: not significant |
| Generate-vs-train joint allocation | simulation | no gain at measured prices |
| Other LLMs (Gemini 2.5 Flash, Qwen-3-32B) | actual | weaker on average; Qwen's best reward 11.32 = best of the project |
| Generalisation to Cartpole/BallBalance/Humanoid | actual | easy tasks saturate; Humanoid LLM rewards weak |
| H100 / CMP 170HX / 6 runs per GPU / more boxes | tested or audited | no gain, or not worth it |

---

## 9. Costs and compute

- **PPO:** 376 completed training phases (280 from scratch, 96 resumed), ~431,000 PPO iterations ≈ 2.8 × 10^10 environment
  steps (`results/LEDGER.json`), plus the live-pruning run and the final Gemini/Qwen resumes.
- **Vast:** ~$18.5 estimated from wall time across 3 boxes (credit $21). Quest compute was free.
- **LLM calls this session:** MiniMax 50 ($0.47) + LLM judge 105 ($0.74) + Gemini 10 ($0.18) + Qwen 10 ($0.04) + a few cent-level
  connectivity probes.

---

## 10. Incidents and how they were handled

| Incident | Handling |
| --- | --- |
| 16k-token cap → empty LLM replies | cap 32k via env var; truncations counted as failures |
| Quest queue ~600 deep | H100 lane tried; moved to Vast |
| Quest SSH socket died | stopped Quest work; Vast replaced it |
| `tkinter` missing on Vast | apt package; documented in setup script |
| zsh word-splitting / `pkill -f` matching its own SSH / missing exec bit / no `timeout` on macOS | wrappers, PID-based kills, `chmod`, `setsid -f` |
| Runner restarts mid-batch | runner v2 waits for any running pack; batches are idempotent |
| Box 3 pip stall; inter-box key blocked | user shut it off; cost folded into estimate |
| Box 2 direct port down | Vast proxy; wrapper rewrites host |
| Live-test batch picked up late by box 1 | moved to box 2 |
| Backup mismatches on files still being written | manifest restricted to settled files |
| Humanoid reward runtime crashes | counted as failed candidates |
| Resume control strict criterion failed on 1 seed | reported, resumed numbers labelled |
| Errors I corrected in docs | manifest run counts, H3 n, "best reward" attribution after Qwen's 11.32 |

---

## 11. Open items and next steps

1. LLM judge: concluded not significant (p = 0.057; 0.17 seed-level). Any follow-up must be a new pre-registered replication on fresh rewards, not an extension of this test.
2. A fresh from-scratch robustness test on new rewards (~$3.5 on a rented A100 rebuilt with `vast/setup_box.sh`): ~4 new batched
   rounds of 8 rewards trained fully once, all rules replayed frozen, to remove the "same 7 rewards" limitation.
3. More seeds for H4/H5 (~$3); a hard second task with real reward variation (Humanoid with more seeds and an in-task library, or
   ShadowHand); H3 with ~3× more rewards per arm.
4. Write-up framing: empirical/methodological contribution; cite ORSO, learning-curve extrapolation, PASHA, LC-PFN, BOHB, batch
   prompting.
5. Rotate the API key that was pasted in chat.

---

## 12. File index

| Purpose | Files |
| --- | --- |
| Live search, seeds, resume | `live.py`, `live_task.py`, `analyze_live.py`, `analyze_seeds.py` |
| Live pruning | `live_prune.py`, `results/exp_j_live_prune` |
| Generation | `eureka_ab.py`, `gen_task.py`, `llm_efficiency/` |
| Pruning simulator and tests | `pruning/` (`heldout.py`, `policies.py`, `configs.py`, `run_pool_cv.py`, `structure_probe.py`, `seed_allocation.py`, `orso_baseline.py`) |
| Analyses | `final_analysis.py`, `exploratory_analysis.py`, `llm_judge.py`, `results/final/` |
| GPU operations | `vast/` (runner, queue batches, backup, provenance, ledger, setup, logs) |
| Pre-registration | `STATS_PLAN.md`, `results/pruning/frozen_rules.json`, `best_rule_criterion.md`, `llm_judge_prereg.md` |
| Write-ups | `THESIS_EXPERIMENT_SUMMARY.md`, `PRUNING_COMPARISON.md`, `LLM_EFFICIENCY_RESULTS.md`, `GENERALIZATION_RESULTS.md`, `EXPERIMENT_MANIFEST.md`, this file |
