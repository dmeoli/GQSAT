#!/usr/bin/env bash
# Trains a branching policy in the branch and bound of SMS++ on weighted
# MaxSAT (--env-name maxsat-v0, smspp/SMSppMaxSATEnv.py), with the same DQN
# and the same network as train.sh: the rows of the vertices have the 7
# columns of a weighted MaxSAT node (--bnb-features 1), OLL runs within 20
# calls of the SAT solver in each node (--bnb-max-iter 20), and past 500
# decisions of the policy the rule of the cores of SMS++ decides.
#
# usage: train_maxsat.sh <variant> <train_path> <val_path> [logdir]
#        variants as in train.sh; UPDATES, EXPLORE and EVAL override the
#        number of batch updates, of initial exploration steps and the
#        evaluation frequency (a short trial takes small ones); BC > 0
#        is a pretraining of BC batch updates on DEMO transitions of the
#        rule of the cores (--pretrain-bc-steps, to be run with UPDATES=0)
#        at the learning rate BCLR (0.001),
#        FROM and FROMCHKP the model directory and checkpoint to start
#        from (e.g. the pretrained one), EPSINIT the initial exploration
#        epsilon (1.0), IDX 1 to give the vertices the index of the
#        variable (--bnb-index-feature)
set -e

VARIANT="${1:-gatqsat}"
TRAIN="${2:?usage: train_maxsat.sh <variant> <train_path> <val_path> [logdir]}"
VAL="${3:?usage: train_maxsat.sh <variant> <train_path> <val_path> [logdir]}"
LOGDIR="${4:-runs/maxsat_$VARIANT}"
UPDATES="${UPDATES:-50000}"
EXPLORE="${EXPLORE:-5000}"
EVAL="${EVAL:-1000}"

AGG=sum; HIDDEN=64
case "$VARIANT" in
  graphqsat) ATTN="" ;;
  gatqsat)   ATTN="--use_attention --heads 3" ;;
  graphwide) ATTN=""; HIDDEN=104 ;;
  attnagg)   ATTN="--heads 3"; AGG=attention ;;
  *) echo "unknown variant '$VARIANT' (graphqsat|gatqsat|graphwide|attnagg)"; exit 1 ;;
esac
SEEDARG=""
[ -n "${SEED:-}" ] && SEEDARG="--seed $SEED"

mkdir -p "$LOGDIR"
RESUME=""
[ -f "$LOGDIR/status.yaml" ] && RESUME="--status-dict-path $LOGDIR/status.yaml"
EXTRA=""
[ -n "${BC:-}" ] && EXTRA="$EXTRA --pretrain-bc-steps $BC --bc-demo-transitions ${DEMO:-20000} --bc-lr ${BCLR:-0.001}"
[ -n "${FROM:-}" ] && EXTRA="$EXTRA --model-dir $FROM --model-checkpoint $FROMCHKP"

python3 -u dqn.py $EXTRA \
  --logdir "$LOGDIR" $RESUME --env-name maxsat-v0 \
  --bnb-features 1 --bnb-max-iter 20 --bnb-index-feature "${IDX:-0}" \
  --train-problems-paths "$TRAIN" \
  --eval-problems-paths "$VAL" \
  $ATTN $SEEDARG \
  --lr 0.00002 --bsize 64 --buffer-size 20000 \
  --eps-init "${EPSINIT:-1.0}" --eps-final 0.01 --eps-decay-steps 30000 --gamma 0.99 \
  --batch-updates "$UPDATES" --history-len 1 --init-exploration-steps "$EXPLORE" \
  --step-freq 4 --target-update-freq 10 --loss mse --opt adam \
  --save-freq 500 --grad_clip 0.1 --grad_clip_norm_type 2 \
  --eval-freq "$EVAL" --eval-time-limit 3600 --core-steps 4 \
  --expert-exploration-prob 0.0 --priority_alpha 0.5 --priority_beta 0.5 \
  --e2v-aggregator $AGG --n_hidden 1 --hidden_size $HIDDEN \
  --decoder_v_out_size 32 --decoder_e_out_size 1 --decoder_g_out_size 1 \
  --encoder_v_out_size 32 --encoder_e_out_size 32 --encoder_g_out_size 32 \
  --core_v_out_size 64 --core_e_out_size 64 --core_g_out_size 32 \
  --activation relu --penalty_size 0.1 \
  --train_time_max_decisions_allowed 500 --test_time_max_decisions_allowed 500 \
  --no_max_cap_fill_buffer \
  --lr_scheduler_gamma 1 --lr_scheduler_frequency 3000 \
  --independent_block_layers 0
