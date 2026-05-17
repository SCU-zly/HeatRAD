# HeatRAD environment variables — H20 4-GPU profile.
# Source this file:  source /data/DaZhou_data/heatrad/scripts/env.sh

export RADGENOME_PATH=/data/DaZhou_data/RadGenome-ChestCT
export HEATRAD_ROOT=/data/DaZhou_data/heatrad

# h20_4gpu profile (4× NVIDIA H20-3e, sm_90, 141GB/card)
export HEATRAD_PROFILE=h20_4gpu
export CUDA_VISIBLE_DEVICES=2,3,4,5
export TORCH_CUDA_ARCH_LIST=9.0
export TOKENIZERS_PARALLELISM=false
# NCCL: H20-3e split-mode often needs P2P/IB disabled to avoid CUDA errors.
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=0
export NCCL_NVLS_ENABLE=0     # H20-3e split mode does not support NVLink-Sharp

# HF cache redirect to project to avoid bloating $HOME
export HF_HOME=$HEATRAD_ROOT/checkpoints/.hf_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
mkdir -p "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE"
