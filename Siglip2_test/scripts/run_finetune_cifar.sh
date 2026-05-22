#!/usr/bin/env bash
# Run SigLIP2 CIFAR pipeline test (same env assumptions as set_up.ipynb).
# Usage: conda activate siglip2 && ./scripts/run_finetune_cifar.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/big_vision"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TF_CPP_MIN_LOG_LEVEL=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
unset BV_JAX_INIT

export SIGLIP_DATASET_MODE=cifar10
export SIGLIP_DATASET_ROOT="$ROOT/datasets/cifar-10-batches-py"
export SIGLIP_CKPT_PATH="${SIGLIP_CKPT_PATH:-/tmp/siglip2_b16_224.npz}"
export SIGLIP_FREEZE_IMAGE=1
export SIGLIP_CIFAR_MULTI_PROMPT=1

WORKDIR="$ROOT/workdirs/siglip2_cifar_test"
mkdir -p "$WORKDIR"

python -m big_vision.trainers.proj.image_text.siglip \
  --config big_vision/configs/proj/image_text/siglip2_finetune_local.py:runlocal,res=224,variant=B/16,batch_size=16,total_steps=500,log_steps=25,ckpt_steps=250,freeze_image=True,seqlen=64 \
  --workdir "$WORKDIR"

echo "Done. Checkpoint: $WORKDIR/checkpoint.bv-LAST"
