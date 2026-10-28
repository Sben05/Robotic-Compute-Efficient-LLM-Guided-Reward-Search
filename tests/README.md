# Tests

Unit tests for the pure-Python code: the allocation-rule simulator (`pruning/sim_core.py`, `pruning/policies.py`,
`pruning/heldout.py`), the LLM generation/validation helpers (`eureka_ab.py`, `gen_task.py`), the statistics helpers
in `final_analysis.py`, and the per-task compute-accounting table in `vast/ledger.py`. No GPU, Isaac Gym, network
access or API key is needed -- everything here runs on synthetic data or small temp files.

```
pip install pytest numpy scipy pyyaml matplotlib   # matplotlib/pyyaml/scipy only for imports some modules do at load time
python3 -m pytest tests/ -q
```

What is **not** covered here, and why: the actual PPO training and LLM-call code paths (`live.py`'s `launch()`,
`eureka_ab.py`'s `train()`, `gen_task.py`'s `generate()`, every `llm()` call) need a real GPU/Isaac Gym install or a
paid API key, so they are exercised by the real experiments themselves (every run's `train.log`, `curves.json` and
`generation.json` is the record of that), not by this suite.

`test_heldout.py` is the regression test for this session's main finding: `pruning/heldout.py` (and, found during
the same pass, `pruning/orso_baseline.py` and `exploratory_analysis.py`'s `e4()`) were handing a held-out seed's own
human curve to that seed's own forecasting library. `worlds()` no longer filters by seed itself -- the contract is
that **callers** pass an already-filtered human dict -- so `test_worlds_passing_the_full_human_dict_does_leak_the_sentinel`
deliberately reproduces the old bug to document what it looked like, right next to the test that pins the fixed
behaviour.
