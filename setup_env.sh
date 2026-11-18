#!/bin/bash
# Prepare this recipe for Coding Assignment 4 on Delta (project bjdh):
#   1. the Python environment, built on the NVMe scratch of the cluster
#   2. a `dump` symlink to the shared, read-only copy of the dataset
#
#   ./setup_env.sh            # from the recipe directory
#
# Everything heavy goes to ENV_ROOT (default /work/nvme/bjdh/$USER/python_env):
# the pixi environment itself (pixi's `detached-environments` setting, stored in
# ./.pixi/config.toml) and the package cache (PIXI_CACHE_DIR). Environments on
# /work/hdd make `import torch` take tens of minutes, and node-local /tmp is not
# visible to batch jobs, so neither is used. The package list is pixi.toml
# (Python 3.13, ffmpeg, torch 2.15, torchaudio 2.11 and requirements.txt).
#
# The dataset is shared for the whole class at DATA_ROOT (default
# /work/nvme/bjdh/shared/coding4, a copy on the fast NVMe scratch; the original on
# /work/hdd/bjdh/shared/coding4 makes one training epoch I/O-bound):
# dump/raw/{train,dev,test}_{monolingual,bilingual}/data.json
# plus the audio under dump/raw/org and dump/raw/test_bilingual. data.json refers
# to the audio by paths relative to the recipe directory (dump/raw/...), so a
# `dump` symlink here is all that is needed; nothing is copied.
set -euo pipefail

RECIPE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ENV_ROOT=${ENV_ROOT:-/work/nvme/bjdh/${USER}/python_env}
DATA_ROOT=${DATA_ROOT:-/work/nvme/bjdh/shared/coding4}
export PIXI_CACHE_DIR=${PIXI_CACHE_DIR:-${ENV_ROOT}/pixi_cache}

mkdir -p "${ENV_ROOT}/pixi_envs" "${PIXI_CACHE_DIR}"

if ! command -v pixi >/dev/null 2>&1; then
    echo "pixi not found; installing to ~/.pixi/bin ..."
    curl -fsSL --retry 3 https://pixi.sh/install.sh | bash
    export PATH="${HOME}/.pixi/bin:${PATH}"
fi

cd "${RECIPE_DIR}"

# --- 1. Python environment -------------------------------------------------
pixi config set --local detached-environments "${ENV_ROOT}/pixi_envs"
echo "environment dir : ${ENV_ROOT}/pixi_envs"
echo "package cache   : ${PIXI_CACHE_DIR}"

pixi install
pixi run python -c "import torch, torchaudio, kaldiio, sentencepiece, jiwer, librosa; print('torch', torch.__version__, 'torchaudio', torchaudio.__version__, 'from', torch.__file__)"
pixi run ffmpeg -version | head -n 1

# --- 2. Dataset ------------------------------------------------------------
if [ ! -f "${DATA_ROOT}/dump/raw/train_bilingual/data.json" ]; then
    echo "ERROR: dataset not found under ${DATA_ROOT}/dump/raw (set DATA_ROOT to override)" >&2
    exit 1
fi
if [ -e dump ] && [ ! -L dump ]; then
    echo "ERROR: ./dump exists and is not a symlink; move it away and rerun" >&2
    exit 1
fi
ln -sfn "${DATA_ROOT}/dump" dump
echo "dataset         : dump -> $(readlink dump)"
for s in train_bilingual dev_bilingual test_bilingual train_monolingual dev_monolingual test_monolingual; do
    [ -f "dump/raw/${s}/data.json" ] || { echo "ERROR: dump/raw/${s}/data.json missing" >&2; exit 1; }
done

echo "Done. Enter the environment with: pixi shell   (or prefix commands with: pixi run)"
