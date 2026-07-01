#!/usr/bin/env bash
# Train a Graph-Q-SAT-family model with the published configuration.
#
# Usage:
#   bash train.sh <variant> <train_path> <val_path> [logdir]
#     variant  : graphqsat | gatqsat
#
# Hyperparameters below reproduce the original Graph-Q-SAT training exactly
# (verified against the released checkpoints): in particular the per-episode
# decision cap (train/test_time_max_decisions_allowed 500), penalty_size 0.1 and
# grad_clip 0.1 — omitting these (i.e. relying on argparse defaults) destabilises
# the DQN and the agent never learns. Auto-resumes from <logdir>/status.yaml.
set -e

VARIANT="${1:-graphqsat}"
TRAIN="${2:?usage: train.sh <variant> <train_path> <val_path> [logdir]}"
VAL="${3:?usage: train.sh <variant> <train_path> <val_path> [logdir]}"
LOGDIR="${4:-runs/$VARIANT}"

case "$VARIANT" in
  graphqsat) ATTN="" ;;
  gatqsat)   ATTN="--use_attention --heads 3" ;;
  *) echo "unknown variant '$VARIANT' (graphqsat|gatqsat)"; exit 1 ;;
esac

mkdir -p "$LOGDIR"

# Discard a checkpoint left by an OLD, mis-configured run so we retrain correctly.
# (Robust to stale Colab notebook cells: this script is always pulled fresh.)
if [ -f "$LOGDIR/status.yaml" ] && ! { \
      grep -qE '^[[:space:]]*train_time_max_decisions_allowed:[[:space:]]*500[[:space:]]*$' "$LOGDIR/status.yaml" && \
      grep -qE '^[[:space:]]*grad_clip:[[:space:]]*0\.1[[:space:]]*$' "$LOGDIR/status.yaml" && \
      grep -qE '^[[:space:]]*penalty_size:[[:space:]]*0\.1[[:space:]]*$' "$LOGDIR/status.yaml"; }; then
    echo ">>> discarding mis-configured checkpoint in $LOGDIR (will retrain from scratch)"
    rm -rf "$LOGDIR"; mkdir -p "$LOGDIR"
fi

RESUME=""
[ -f "$LOGDIR/status.yaml" ] && RESUME="--status-dict-path $LOGDIR/status.yaml"

python3 dqn.py \
  --logdir "$LOGDIR" $RESUME --env-name sat-v0 \
  --train-problems-paths "$TRAIN" \
  --eval-problems-paths "$VAL" \
  $ATTN \
  --lr 0.00002 --bsize 64 --buffer-size 20000 \
  --eps-init 1.0 --eps-final 0.01 --eps-decay-steps 30000 --gamma 0.99 \
  --batch-updates 50000 --history-len 1 --init-exploration-steps 5000 \
  --step-freq 4 --target-update-freq 10 --loss mse --opt adam \
  --save-freq 500 --grad_clip 0.1 --grad_clip_norm_type 2 \
  --eval-freq 1000 --eval-time-limit 3600 --core-steps 4 \
  --expert-exploration-prob 0.0 --priority_alpha 0.5 --priority_beta 0.5 \
  --e2v-aggregator sum --n_hidden 1 --hidden_size 64 \
  --decoder_v_out_size 32 --decoder_e_out_size 1 --decoder_g_out_size 1 \
  --encoder_v_out_size 32 --encoder_e_out_size 32 --encoder_g_out_size 32 \
  --core_v_out_size 64 --core_e_out_size 64 --core_g_out_size 32 \
  --activation relu --penalty_size 0.1 \
  --train_time_max_decisions_allowed 500 --test_time_max_decisions_allowed 500 \
  --no_max_cap_fill_buffer \
  --lr_scheduler_gamma 1 --lr_scheduler_frequency 3000 \
  --independent_block_layers 0
