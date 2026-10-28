"""Test setup: make the repo root and pruning/ importable, the way every analysis script already does
with its own sys.path.insert. No GPU, network or Isaac Gym needed for anything under tests/ -- these cover
the pure-Python generation/validation/statistics/simulation code only."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "pruning"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
