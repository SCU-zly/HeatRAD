#!/usr/bin/env bash
# Launch HeatRAD training on 4× H20.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

source /home/zhouly/miniconda3/etc/profile.d/conda.sh
conda activate heatrad

cd "$HEATRAD_ROOT"

torchrun --standalone --nproc_per_node=4 \
    -m heatrad.train.trainer \
    "$@"
