#!/bin/bash
# Reproduce every number in LLM_EFFICIENCY_FEASIBILITY.md (CPU only, no network, no API key). Run from the repo root.
set -e
PY=.venv-eff/bin/python
cd llm_efficiency
$PY token_accounting.py
$PY diversity_analysis.py
$PY mutate.py
for s in 0.0 0.35 0.7; do $PY joint_alloc_sim.py $s 25 & done; wait
$PY summarize_alloc.py 0.35
for s in 0.0 0.35 0.7; do $PY defaults_compare.py $s 40 > ../results/llm_efficiency/defaults_compare_sig$s.txt & done; wait
$PY frontier_gap.py
$PY llm_trial_plan.py   # dry run only: writes payloads + worst-case budget, sends nothing
