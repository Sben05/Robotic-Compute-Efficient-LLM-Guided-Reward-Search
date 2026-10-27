# Pre-registered statistical analysis plan

Written 2026-10-27 ~05:10 UTC, before held-out seeds 4-5, the LLM-trial seed-1 screens, the generalisation runs, the live pruning
confirmation and the extra G1 seeds exist. Pruning parameters were frozen earlier (results/pruning/frozen_rules.json, e868927);
the live-rule choice is fixed by results/pruning/best_rule_criterion.md.

Every number in the write-ups is tagged **actual run**, **retrospective (held-out curves)** or **simulation**.
Primary tests are one-sided at alpha = 0.05 with Holm correction across the five primary hypotheses. Everything else is
exploratory and reported with effect sizes and 95% CIs but no significance claims.

## Units of analysis

PPO seed noise is the dominant noise source (human reward peaks range 7.9-10.5 across seeds), so the independent unit is a
(task, seed) training run, never an iteration. Bootstrap CIs are hierarchical: resample (task, seed) clusters, then candidates or
candidate subsets within a cluster. Overlapping candidate subsets (worlds) are never treated as independent.

## Primary hypotheses

**H1. Adaptive pruning beats fixed-500 on held-out data at no more than half the compute.**
Rule = the one selected by best_rule_criterion.md. Outcome = per-world retained fraction (selected reward's 3,000-it peak / best
reward's peak). Test = paired difference (adaptive - fixed@500 keep-1) over held-out worlds: Ant seeds 4-5 (7 LLM rewards each,
all 4-subsets), with a hierarchical bootstrap over seeds; p = bootstrap fraction of mean difference <= 0.
Fairness check (reported alongside): the adaptive rule is also compared with the fixed checkpoint of EQUAL mean compute
(linear interpolation between fixed@500 and fixed@750/1000), because adaptive rules spend more than fixed-500.

**H2. Batched generation (4 rewards per call) costs less per valid reward than one reward per call.**
Outcome = completion tokens per valid reward. Units = tasks (Ant, Cartpole, BallBalance, Humanoid; Ant uses arm B vs the
independent samples with logged usage). Test = per-task ratio, one-sided sign test across the 4 tasks plus a permutation test on
per-call tokens/valid-reward pooled over tasks (task-stratified).

**H3. Batched generation is not worse in reward quality (non-inferiority).**
Outcome = 500-it score (Ant) or full-training peak (other tasks), each divided by the same task-and-seed human reward's score.
Non-inferiority margin = 10% of the human score. Test = one-sided bootstrap (task-stratified) that the mean difference
(batch - independent) > -0.10.

**H4. A zero-token edit (forward-speed temperature 4 -> 10) recovers the LLM refinement gain on r1c1.**
Outcome = 3,000-it peak; arms r1c1 unmodified vs r1c1 tau=10, seeds 0-3 (n = 4 each). Test = one-sided Welch t-test.
Pre-registered reading (from the G1 plan): tau=10 mean >= 9.0 means the edit explains most of the gain.

**H5. The best reward found (p3) outperforms the human reward across seeds.**
Outcome = 3,000-it peak, seeds 0-5 (n = 6 each, same seeds). Test = one-sided paired t-test over seeds (paired by seed).

## Amendment 1 (2026-10-27 ~05:20 UTC, before any seed-4/5 result was seen; seeds 4-5 still training)

- H1's held-out set is extended, unconditionally, to Ant seeds 4, 5, 7, 8 and 9 (same 8 rewards, same 500 + resume protocol).
  Seeds 7-9 are run whatever seeds 4-5 show, so there is no optional stopping. Seed 6 is reserved for the live confirmation.
  The H1 test uses all five held-out seeds; a seeds-4-5-only result is reported as interim.
- New exploratory robustness study (no significance claims): the Ant generation comparison (one reward per call vs four per call)
  repeated with two other LLMs, DeepSeek V4.1 Flash and Qwen 3.8 Flash (Vercel AI Gateway, same prompts, temperature 1.0):
  8 independent calls + 2 batched calls each, 500-it screens of every valid reward, top 4 per LLM resumed to 3,000.
  Reported per LLM next to MiniMax; H2/H3 stay defined on the MiniMax data.

## Secondary / exploratory (no significance claims)

Live early stopping (n = 2 search trials, 4 rounds): descriptive. Round-2 vs round-1 quality from 500-it feedback: descriptive
(no-feedback control absent). False-prune rate of fixed-500 (A1): rate with Wilson 95% CI. Rank stability across seeds: Spearman
per seed pair. Generalisation of the frozen Ant pruning rules to Cartpole / BallBalance / Humanoid: per-task retained/compute
with cluster-bootstrap CIs (time rescaled to fraction of training, scores scaled so the human peak = 10.48). Joint
generate-vs-train allocation: simulation only. Live pruning confirmation: one actual run, replay equality reported.

## Known limits that no test here removes

One LLM (MiniMax M3.1 Flash), one PPO configuration, few independent LLM search trials, and Isaac Gym tasks only.
