#!/usr/bin/env bash
# ============================================================================
# Glassbox-100M Pure CUDA C++17 Turnkey GPU Pretraining Launcher
# Zero PyTorch! Compiles pure CUDA C++17 kernels and trains Glassbox-100M
# (100,682,496 exact params) on 100M tokens (~2.5m on H100 / ~8m on 4090)
# or 10B tokens (~4h on H100 / ~14h on 4090).
#
# Usage:
#   ./cuda/launch_gpu.sh --tokens 100M --dataset fineweb-edu
#   ./cuda/launch_gpu.sh --tokens 10B  --dataset fineweb-edu --pure-cuda
# ============================================================================

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TOKENS_SPEC="100M"
DATASET="fineweb-edu"
DATA_DIR="data/shards"
OUT_CKPT="checkpoints/glassbox_100m.gbx"
BATCH_SIZE=8
SEQ_LEN=1024
GRAD_ACCUM=8
PURE_CUDA=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tokens)     TOKENS_SPEC="$2"; shift 2 ;;
    --dataset)    DATASET="$2"; shift 2 ;;
    --data-dir)   DATA_DIR="$2"; shift 2 ;;
    --out)        OUT_CKPT="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --seq-len)    SEQ_LEN="$2"; shift 2 ;;
    --grad-accum) GRAD_ACCUM="$2"; shift 2 ;;
    --pure-cuda)  PURE_CUDA=1; shift 1 ;;
    *) echo "Unknown flag: $1"; exit 1 ;;
  esac
done

# Convert human token count (100M, 1B, 10B) to integer
case "$TOKENS_SPEC" in
  *B|*b) TOTAL_TOKENS=$(( ${TOKENS_SPEC%[Bb]} * 1000000000 )) ;;
  *M|*m) TOTAL_TOKENS=$(( ${TOKENS_SPEC%[Mm]} * 1000000 )) ;;
  *)     TOTAL_TOKENS="$TOKENS_SPEC" ;;
esac

# 1. Verify NVIDIA GPU & auto-detect compute capability
if ! command -v nvidia-smi >/dev/null 2>&1 || ! command -v nvcc >/dev/null 2>&1; then
  echo "[ERROR] nvidia-smi and nvcc are required to run the CUDA C++17 GPU trainer."
  echo "        On CPU-only machines, run:  make -C cuda cpu && ./cuda/glassbox_cpu_train"
  exit 1
fi

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
CC_RAW="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '.')"
ARCH="sm_${CC_RAW}"
echo "==========================================================================="
echo " Glassbox-100M Pure CUDA C++17 Launcher"
echo " Detected GPU : ${GPU_NAME} (${ARCH})"
echo " Target Budget: ${TOKENS_SPEC} (${TOTAL_TOKENS} tokens) on ${DATASET}"
echo "==========================================================================="

# 2. Build Pure CUDA C++17 Trainer
if [[ "$PURE_CUDA" -eq 1 ]]; then
  echo "[1/3] Compiling 100% Pure Custom CUDA C++17 Kernels (USE_CUBLAS=0, ${ARCH})..."
  make -C cuda cuda-pure ARCH="$ARCH"
else
  echo "[1/3] Compiling Pure CUDA C++17 Trainer with TF32 Tensor Cores (${ARCH})..."
  make -C cuda cuda ARCH="$ARCH" USE_CUBLAS=1
fi

# 3. Prepare .gbxtok Binary Token Shards if not present
if ! ls "$DATA_DIR"/train_*.gbxtok >/dev/null 2>&1; then
  echo "[2/3] Preparing ${TOKENS_SPEC} binary .gbxtok shards in ${DATA_DIR}..."
  python3 scripts/prepare_tokens.py \
    --dataset "$DATASET" \
    --target-tokens "$TOTAL_TOKENS" \
    --vocab-size 32768 \
    --out-dir "$DATA_DIR"
else
  echo "[2/3] Found existing .gbxtok shards in ${DATA_DIR}, skipping download."
fi

# 4. Launch Pure CUDA Pretraining
mkdir -p "$(dirname "$OUT_CKPT")"
echo "[3/3] Launching ./cuda/glassbox_cuda_train ..."
./cuda/glassbox_cuda_train \
  --preset 100m \
  --data-dir "$DATA_DIR" \
  --total-tokens "$TOTAL_TOKENS" \
  --batch-size "$BATCH_SIZE" \
  --seq-len "$SEQ_LEN" \
  --grad-accum "$GRAD_ACCUM" \
  --lr 6e-4 \
  --min-lr 6e-5 \
  --warmup-steps 100 \
  --out "$OUT_CKPT"

echo "==========================================================================="
echo " Training Complete! Checkpoint saved to: ${OUT_CKPT}"
echo " Load directly in Python Glassbox:"
echo "   from glassbox import load_gbx_checkpoint"
echo "   model, info = load_gbx_checkpoint('${OUT_CKPT}')"
echo "==========================================================================="
