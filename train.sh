#!/usr/bin/env bash
# TAPIR：冻结 BERT 前 5 轮，再训满 30 轮，按 valid 综合分保留最好一轮
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-/home/yangming/anaconda3/envs/pai/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p logs checkpoints
echo "[train.sh] TAPIR 30 epochs gpu=${CUDA_VISIBLE_DEVICES} $(date)"
"$PY" -u train_extended.py \
  --epochs 30 --patience 30 --batch_size 64 --seed 42 \
  --freeze_list 5 --run_name tapir30 --device cuda:0 \
  "$@"
echo "[train.sh] done $(date)"
