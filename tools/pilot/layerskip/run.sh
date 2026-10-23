#!/bin/bash
# Run a layerskip script in the WSL uv venv (engine/PILOTS.md "Layer skipping").
#   wsl -e bash tools/pilot/layerskip/run.sh skip_study.py check
# Venv: uv venv ~/.venvs/layerskip --python 3.12; uv pip install torch --index-url
# https://download.pytorch.org/whl/cu128 "transformers==5.20.2" tokenizers numpy safetensors
# .pq: $LAYERSKIP_PQ, else ~/layerskip/cerno-4b-q4_0.pq (a local copy), else models/cerno-4b/.
set -e
export PATH="$HOME/.local/bin:$PATH"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"     # host threads <= 4 (lead 21:28 / 21:45: shared PC)
if [ -z "$LAYERSKIP_PQ" ] && [ -f "$HOME/layerskip/cerno-4b-q4_0.pq" ]; then
    export LAYERSKIP_PQ="$HOME/layerskip/cerno-4b-q4_0.pq"
fi
cd "$(dirname "$0")/../../.."
exec "$HOME/.venvs/layerskip/bin/python" -u "tools/pilot/layerskip/$@"
