# Copyright 2019-2020 Nvidia Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

python3 dqn.py \
  --logdir log \
  --env-name sat-v0 \
  --train-problems-paths PATH_TO_TRAIN_DATA \
  --eval-problems-paths PATH_TO_VAL_DATA \
  --lr 0.00002 \
  --bsize 64 \
  --buffer-size 20000 \
  --eps-init 1.0 \
  --eps-final 0.01 \
  --gamma 0.99 \
  --eps-decay-steps 30000 \
  --batch-updates 50000 \
  --history-len 1 \
  --init-exploration-steps 5000 \
  --step-freq 4 \
  --target-update-freq 10 \
  --loss mse \
  --opt adam \
  --save-freq 500 \
  --grad-clip 1 \
  --grad-clip-norm-type 2 \
  --eval-freq 1000 \
  --eval-time-limit 3600 \
  --core-steps 4 \
  --expert-exploration-prob 0.0 \
  --priority-alpha 0.5 \
  --priority-beta 0.5 \
  --e2v-aggregator sum \
  --n-hidden 1 \
  --hidden-size 64 \
  --decoder-v-out-size 32 \
  --decoder-e-out-size 1 \
  --decoder-g-out-size 1 \
  --encoder-v-out-size 32 \
  --encoder-e-out-size 32 \
  --encoder-g-out-size 32 \
  --core-v-out-size 64 \
  --core-e-out-size 64 \
  --core-g-out-size 32 \
  --activation relu \
  --penalty-size 0.1 \
  --train-time-max-decisions-allowed 500 \
  --test-time-max-decisions-allowed 500 \
  --no-max-cap-fill-buffer \
  --lr-scheduler-gamma 1 \
  --lr-scheduler-frequency 3000 \
  --independent-block-layers 0
