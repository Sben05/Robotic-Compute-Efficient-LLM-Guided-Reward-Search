#!/usr/bin/env bash
#SBATCH --job-name=ant-baseline-seed0
#SBATCH --account=e32695
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=quest-ant-%j.log

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load git/2.37.2
module load mamba/24.3.0
eval "$(conda shell.bash hook)"
conda activate "$PWD/.env"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export MAX_JOBS="$SLURM_CPUS_PER_TASK"
bash run_ant.sh
