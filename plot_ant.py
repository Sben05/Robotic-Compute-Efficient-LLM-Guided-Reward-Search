"""Plot an actual upstream Ant run; no simulator or training modifications."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("result_dir", type=Path)
args = parser.parse_args()
summaries = list(args.result_dir.glob("runs/*/summaries"))
if len(summaries) != 1:
    parser.error("Expected exactly one original runs/*/summaries directory.")
events = EventAccumulator(str(summaries[0]), size_guidance={"scalars": 0})
events.Reload()
if "rewards/iter" not in events.Tags().get("scalars", []):
    parser.error("No rewards/iter data found; cannot plot training performance.")
points = events.Scalars("rewards/iter")
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot([p.step for p in points], [p.value for p in points])
ax.set(xlabel="PPO iteration", ylabel="Mean episodic return (unscaled)",
       title="Original Ant PPO — seed 0")
ax.grid(alpha=0.25)
fig.tight_layout()
output = args.result_dir / "performance.png"
fig.savefig(output, dpi=150)
plt.close(fig)
last = points[-1]
print("Last logged PPO iteration:", last.step)
print("Final logged mean episodic return:", last.value)
print("Environment steps implied by that iteration:", last.step * 4096 * 16)
print("Confirm run completion and checkpoint frames before reporting final results.")
print("Plot:", output)
