#!/bin/bash
# Run a GPU script with the py-videocore6 checkout on the path (Pi side).
# usage (from ~/pinfer): bash tmp/gpu/gpurun.sh script.py [args...]
G="$HOME/pinfer/tmp/gpu"
export PYTHONPATH="$G/pyvc6/src:$G"
exec "$HOME/pinfer/venvs/gpu/bin/python" -X faulthandler "$G/$1" "${@:2}"
