#!/usr/bin/env bash
# Generic launcher: sbatch [options] eureka.sbatch SCRIPT.py ARGS...
# Partition, GPU, time and dependencies are given at submission.
#SBATCH --account=e32695
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load git/2.37.2 mamba/24.3.0
eval "$(conda shell.bash hook)"
conda activate "$PWD/.env"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export MAX_JOBS="$SLURM_CPUS_PER_TASK"
python -u "$@"
