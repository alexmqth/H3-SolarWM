# Source from bash or zsh: source /home/qma/work/GWM/env.sh
if [ -n "${BASH_VERSION:-}" ]; then
    GWM_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
elif [ -n "${ZSH_VERSION:-}" ]; then
    GWM_ROOT="$(cd -- "$(dirname -- "${(%):-%N}")" && pwd -P)"
else
    GWM_ROOT=/home/qma/work/GWM
fi
export GWM_ROOT
export GWM_MODEL_ROOT="$GWM_ROOT/models"
export VIRTUAL_ENV="$GWM_ROOT/.venvs/h3world"
export PATH="$VIRTUAL_ENV/bin:$PATH"
export DIFFSYNTH_ROOT="$GWM_ROOT/H3-World/DiffSynth-Studio-h3-v2"
export ABOT_DIFFSYNTH_ROOT="$DIFFSYNTH_ROOT"
export DIFFSYNTH_MODEL_BASE_PATH="$DIFFSYNTH_ROOT/models"
export HF_HOME="$GWM_ROOT/H3-World/.cache/hf"
export TORCHINDUCTOR_CACHE_DIR="$GWM_ROOT/H3-World/.cache/torchinductor"
export TRITON_CACHE_DIR="$GWM_ROOT/H3-World/.cache/triton"
export XDG_CACHE_HOME="$GWM_ROOT/H3-World/.cache/xdg"
