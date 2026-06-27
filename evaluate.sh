#!/usr/bin/env bash
# Evaluate a trained Graph-Q-SAT-family model on a test set (prints MRIR).
#
# Usage:
#   bash evaluate.sh <model_dir> <checkpoint> <test_path> [cap]
#     model_dir  : directory holding model.yaml + the checkpoint
#     checkpoint : e.g. model_50000.chkp
#     test_path  : a directory of .cnf files (with a METADATA file)
#     cap        : test-time decision budget (default 500)
set -e

MODEL_DIR="${1:?usage: evaluate.sh <model_dir> <checkpoint> <test_path> [cap]}"
CHECKPOINT="${2:?usage: evaluate.sh <model_dir> <checkpoint> <test_path> [cap]}"
TEST="${3:?usage: evaluate.sh <model_dir> <checkpoint> <test_path> [cap]}"
CAP="${4:-500}"

python3 evaluate.py \
  --env-name sat-v0 --core-steps -1 --eps-final 0.0 --no_restarts \
  --test_time_max_decisions_allowed "$CAP" \
  --eval-problems-paths "$TEST" \
  --model-dir "$MODEL_DIR" --model-checkpoint "$CHECKPOINT"
