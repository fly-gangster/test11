#!/usr/bin/env bash
set -e
export PYTHONUNBUFFERED=True
export CUDA_VISIBLE_DEVICES=0

python3 tools/eval_linemod.py \
  --dataset_root datasets/linemod/Linemod_preprocessed \
  --model        trained_models/linemod/posenet_latest.pth \
  --refine_model trained_models/linemod/posenet_best_0.0632.pth

