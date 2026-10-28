# Amendment 2 (2026-10-28, before any call of this batch exists)

Fresh-reward replication for H2/H3 (MASTER_LOG open item 1), prompted by two reviews of the existing work: (a) a
consultant review noting the existing H3 CI [-12%, +7.3%] does not establish comparable quality, and recommending a
matched, time-interleaved A/B generation trial plus a diversity check and an equal-budget comparison; (b) an audit
finding that the existing H2 permutation test mixes calls with structurally different output counts (1 vs up to 4)
under relabelling, which is not a valid exchangeability assumption, and that `pruning/heldout.py` leaked each held-out
seed's own human curve into that seed's VoI library. Both are addressed below and in code, separately from this new
data collection.

## New data: Ant, MiniMax M3.1 Flash (Vercel gateway), same prompts/temperature/max_tokens as every other Ant arm

- **Group A (indep):** 36 calls, one reward per call.
- **Group B (batch4):** 9 calls, four rewards per call (same "write FOUR ... compute_reward_1..4" suffix as the
  existing B arm) = 36 rewards.
- Issued in **3 blocks** of (12 indep calls + 3 batch4 calls); each block's calls (indep and batch4 together) are
  submitted through one shared thread pool so the two arms share the same wall-clock window per block (removes the
  across-day timing confound of the original A/B pilot, whose group-A calls were pulled from three earlier, separately
  timed experiments). No candidate from the original pilot (`exp_h_llm_trial`) or from `exp_b`/`exp_d`/`exp_e` is
  reused in this sample.
- This fresh sample is the **primary** dataset for H2 and H3 going forward. The original pilot (17 indep / 12 batch4
  valid rewards, non-matched in time) is kept and reported only as the earlier, non-matched pilot.

## Metric and margin (unchanged from the original pre-registration)

- H3 outcome: 500-iteration peak score, normalised by the same-seed human reward's 500-iteration peak. Non-inferiority
  margin: batch4 mean - indep mean > -0.10. Test: task-stratified bootstrap (scipy percentile, as coded in
  `final_analysis.py:h3`), rerun on this sample only.
- H2 outcome: completion tokens per valid reward. **Primary test = the task-level sign test** (unit = task, matching
  STATS_PLAN's declared unit of analysis), since Ant is now a single task in this sample: report the per-call token
  ratio and its 95% bootstrap CI (resampling calls within arm) instead of a cross-task sign test. The permutation test
  is reported only as a descriptive secondary number, explicitly labelled as anti-conservative per the heterogeneity
  issue above, not as a significance claim.

## New checks (exploratory, no significance claim fixed in advance)

- **Diversity:** for every batch-of-4 reply, pairwise Jaccard similarity over the reward's identifier/operator/term
  set (AST-based, no embeddings needed) between its 4 outputs; compare the within-call distribution to the
  between-call distribution (4 reward functions drawn from different Group A calls at random, same sample size).
  Lower within-call similarity spread than between-call would be the finding of interest; this is descriptive.
- **Equal-budget comparison:** Quest GPU time is unmetered (no $ cost), so "equal budget" here is operationalised as
  **equal total PPO iterations** (the actually scarce, trained resource), not dollars. LLM token cost is reported
  separately (USD, same pricing as every other arm). For each arm: iterations spent generating + screening + training
  finalists, and the best finalist's 3,000-iteration peak at matched iteration spend across arms.

## Screening and finalist protocol (same live-search protocol as the rest of the thesis)

500-iteration screen, seed 0, for every valid candidate in both arms. Top 4 per arm (8 total) resume to 3,000
iterations (same checkpoint-resume mechanism as `live.py`). If time allows, finalists are replicated on additional
seeds using the same 500+resume protocol as `exp_f_seeds`, reported with per-seed means, not single-seed peaks.

## Corrections to existing results (tracked separately, not blocking this new run)

1. `pruning/heldout.py`: a held-out seed's own human curve must be excluded from that seed's library (it is currently
   included via the unconditional `+ hum` in `worlds()`). H1 to be rescored after the fix; both numbers reported.
2. H2 headline in `MASTER_LOG.md` ("<0.0001") is the permutation test's output on the pilot data; it is to be relabelled
   as the sign test's p=0.0625 (not significant) for that pilot, pending this amendment's fresh sign-test result.
