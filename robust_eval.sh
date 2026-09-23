#!/usr/bin/env bash
# 对照官方仓库 robust_eval.sh
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-/home/yangming/anaconda3/envs/pai/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

"$PY" -u eval_robust.py \
  --config_file configs/eval_mosi.yaml \
  --save_path logs
