#!/usr/bin/env bash
# =============================================================================
# run_test.sh —— SHAPE 目标域评测脚本（服务器）
#
# 用法:
#   bash run_test.sh <dataset> <src> <tgt> <gpu> <name>
# 例:
#   bash run_test.sh abdomen ABCT ABMR 0 ABCT2ABMR
#   bash run_test.sh mmwhs  CT   MR   0 CT2MR
#
# 说明:
#   - 从 checkpoints/<src>/<name>/ 自动选取最优 decoder 权重:
#       1) unet2D_best_model.pth                  （旧命名，兼容）
#       2) shape_best_model_<score>.pth           （UDA 训练产物，取分数最高）
#       3) best_model.pth                         （兜底）
#       4) ../shape_best_warmup/best_warmup.pth   （只跑完 warmup 时的兜底，会告警）
#   - 评估结果打印在 stdout（target Dice/ASD）
#   - 可选环境变量: BATCH=64 NUM_WORKERS=12 DINO_CKPT=dinov3 DINO_SIZE=s
#       * DINO_SIZE 命名与 train.py 不同: train 的 b 档在 test 里叫 s+（同为 vits16plus），
#         本脚本会自动映射 b -> s+
# =============================================================================
set -euo pipefail

DATASET="${1:?usage: run_test.sh <dataset:abdomen|mmwhs> <src:ABCT|ABMR|CT|MR> <tgt> <gpu> <name>}"
SRC="${2:?missing src mode}"
TGT="${3:?missing tgt mode}"
GPU="${4:-0}"
NAME="${5:-${SRC}2${TGT}}"

DATASETS_ROOT="${DATASETS_ROOT:-/lxm_4t/lulian/dataset}"
DATA_DIR="${DATASETS_ROOT}/${DATASET^^}_2D"

BATCH="${BATCH:-64}"
NUM_WORKERS="${NUM_WORKERS:-12}"
DINO_CKPT="${DINO_CKPT:-dinov3}"
DINO_SIZE="${DINO_SIZE:-s}"

# train.py 的 b 档 == test.py 的 s+ 档（均为 vits16plus）
if [ "$DINO_SIZE" = "b" ]; then
  DINO_SIZE="s+"
fi

cd "$(dirname "$0")"

if [ ! -d "$DATA_DIR" ]; then
  echo "[run_test] ERROR: 数据目录不存在: $DATA_DIR" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1

LOG="test_${NAME}.log"
echo "[run_test] dataset=$DATASET  $SRC -> $TGT  gpu=$GPU  name=$NAME"
echo "[run_test] data=$DATA_DIR  pth_dir=checkpoints/$SRC/$NAME  dino_size=$DINO_SIZE"
echo "[run_test] log -> $LOG"

python test.py \
  --mode "$SRC" \
  --data_path "$DATA_DIR" \
  --pth_path "$SRC/$NAME" \
  --gpu "$GPU" \
  --batch_size "$BATCH" \
  --num_workers "$NUM_WORKERS" \
  --dino_ckpt "$DINO_CKPT" \
  --dino_size "$DINO_SIZE" \
  2>&1 | tee -a "$LOG"
