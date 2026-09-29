#!/usr/bin/env bash
# =============================================================================
# run_train.sh —— SHAPE 正式训练启动脚本（服务器；被 gpu_grab / start_shape_grab.sh 调用）
#
# 用法（一般经 start_shape_grab.sh 调起；也可手动执行）:
#   bash run_train.sh <dataset> <src> <tgt> <gpu> <name>
# 例:
#   bash run_train.sh abdomen ABCT ABMR 0 ABCT2ABMR
#   bash run_train.sh mmwhs  CT   MR   0 CT2MR
#
# 约定:
#   - 在 SHAPE 的 conda 环境里执行（脚本里的 python 即该环境）
#   - DATASETS_ROOT = {ABDOMEN_2D, MMWHS_2D} 的【父目录】，可 export 覆盖
#   - DINO 权重默认在 <repo>/dinov3/ 下（--dino_ckpt dinov3）；如放在他处 export DINO_CKPT
#   - 可选环境变量（不改脚本调整）:
#       BATCH=64 NUM_WORKERS=12 NUM_DATA=8000 EPOCHS=200 WARMUP_EPOCHS=10
#       DINO_CKPT=dinov3 DINO_SIZE=s EXTRA_ARGS="..."（追加任意 train.py 参数）
#   - 论文参数: batch 64 / epochs 200 / AdamW lr 1e-4 / EMA 0.9 / HPE α=0.25
#               SAP θ=50 分位 / γ_unsup=1；四大模块开关（HFM/HPE/SAP/伪标签）全开
# =============================================================================
set -euo pipefail

DATASET="${1:?usage: run_train.sh <dataset:abdomen|mmwhs> <src:ABCT|ABMR|CT|MR> <tgt> <gpu> <name>}"
SRC="${2:?missing src mode}"
TGT="${3:?missing tgt mode}"
GPU="${4:-0}"
NAME="${5:-${SRC}2${TGT}}"

DATASETS_ROOT="${DATASETS_ROOT:-/lxm_4t/lulian/dataset}"
DATA_DIR="${DATASETS_ROOT}/${DATASET^^}_2D"

BATCH="${BATCH:-64}"
NUM_WORKERS="${NUM_WORKERS:-12}"
NUM_DATA="${NUM_DATA:-8000}"
EPOCHS="${EPOCHS:-200}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-20}"
DINO_CKPT="${DINO_CKPT:-dinov3}"
DINO_SIZE="${DINO_SIZE:-s}"

cd "$(dirname "$0")"

if [ ! -d "$DATA_DIR" ]; then
  echo "[run_train] ERROR: 数据目录不存在: $DATA_DIR" >&2
  echo "[run_train] 请检查 DATASETS_ROOT（应为 ABDOMEN_2D / MMWHS_2D 的父目录）" >&2
  exit 1
fi

# 与 gpu_grab 的 CUDA_VISIBLE_DEVICES 保持一致（手动执行时也生效）
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1

LOG="train_${NAME}.log"
echo "[run_train] dataset=$DATASET  $SRC -> $TGT  gpu=$GPU  name=$NAME"
echo "[run_train] data=$DATA_DIR"
echo "[run_train] batch=$BATCH  workers=$NUM_WORKERS  num_data=$NUM_DATA  epochs=$EPOCHS (warmup $WARMUP_EPOCHS)"
echo "[run_train] log -> $LOG"

python train.py \
  --stage unsup \
  --mode "$SRC" \
  --data_path "$DATA_DIR" \
  --checkpoint_name "$NAME" \
  --gpu "$GPU" \
  --batch_size "$BATCH" \
  --num_workers "$NUM_WORKERS" \
  --num_data "$NUM_DATA" \
  --epochs "$EPOCHS" \
  --warmup_epochs "$WARMUP_EPOCHS" \
  --dino_ckpt "$DINO_CKPT" \
  --dino_size "$DINO_SIZE" \
  --use_hfm --use_selector --use_refinement --use_pseudo_labels \
  ${EXTRA_ARGS:-} \
  2>&1 | tee -a "$LOG"
