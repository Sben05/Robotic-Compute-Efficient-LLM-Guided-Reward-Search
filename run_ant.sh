#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
    echo 'Blocked: original Isaac Gym Preview 4 needs compatible Linux x86_64/NVIDIA hardware.' >&2
    exit 1
fi
command -v nvidia-smi >/dev/null || { echo 'Blocked: nvidia-smi unavailable.' >&2; exit 1; }
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

project_dir="$(cd "$(dirname "$0")" && pwd)"
result_dir="$project_dir/results/ant_seed0_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$result_dir"
echo "Output: $result_dir"
cd "$result_dir"

{
    date -u
    uname -sm
    git -C "$project_dir/Eureka" rev-parse HEAD
    git -C "$project_dir/Eureka" status --short
    nvidia-smi
    python --version
    python -m pip freeze
    echo 'Command: python -u <Eureka>/isaacgymenvs/isaacgymenvs/train.py task=Ant train=AntPPO seed=0 num_envs=4096 max_iterations=500 headless=True hydra.run.dir=.'
    started_at=$SECONDS
    if python -u "$project_dir/Eureka/isaacgymenvs/isaacgymenvs/train.py" \
        task=Ant train=AntPPO seed=0 num_envs=4096 max_iterations=500 \
        headless=True hydra.run.dir=.; then
        training_status=0
    else
        training_status=$?
    fi
    echo "Training command elapsed seconds: $((SECONDS - started_at))"
    echo "Training exit code: $training_status"
    exit "$training_status"
} 2>&1 | tee run.log
python "$project_dir/plot_ant.py" "$result_dir" 2>&1 | tee -a run.log
