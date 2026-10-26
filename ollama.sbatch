#!/usr/bin/env bash
# Local open-weight LLM on Quest: sbatch [opts] ollama.sbatch [SCRIPT.py ARGS...]
# Installs Ollama into /scratch (first run), starts a server on this node, pulls the model, then runs
# `python -u SCRIPT.py ARGS` with the server reachable at 127.0.0.1:11434. No args: install and pull only.
#SBATCH --account=e32695
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
OLLAMA_DIR=/scratch/$USER/ollama
export OLLAMA_MODELS=$OLLAMA_DIR/models OLLAMA_HOST=127.0.0.1:11434 OLLAMA_CONTEXT_LENGTH=32768
export OLLAMA_KEEP_ALIVE=24h OLLAMA_NUM_PARALLEL=${OLLAMA_NUM_PARALLEL:-1}
MODEL=${OLLAMA_MODEL:-qwen3.8:27b}
if [[ ! -x $OLLAMA_DIR/bin/ollama ]]; then
    mkdir -p "$OLLAMA_DIR"
    curl -fL --retry 3 https://github.com/ollama/ollama/releases/download/v0.40.11/ollama-linux-amd64.tar.zst \
        | unzstd | tar -x -C "$OLLAMA_DIR"
fi
"$OLLAMA_DIR/bin/ollama" serve > "$OLLAMA_DIR/serve-$SLURM_JOB_ID.log" 2>&1 &
server=$!
trap 'kill $server 2>/dev/null || true' EXIT
for _ in $(seq 120); do curl -sf http://127.0.0.1:11434/api/version >/dev/null && break; sleep 2; done
"$OLLAMA_DIR/bin/ollama" pull "$MODEL"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no GPU"
[[ $# -eq 0 ]] && exit 0
module purge
module load git/2.37.2 mamba/24.3.0
eval "$(conda shell.bash hook)"
conda activate "$PWD/.env"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
python -u "$@"
