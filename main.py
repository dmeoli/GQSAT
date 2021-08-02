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

import argparse
import os

import torch
import yaml

from dqn import DQN
from gqsat.utils import build_argparser, build_eval_argparser


def train(args):
    args.device = (
        torch.device("cpu")
        if args.no_cuda or not torch.cuda.is_available()
        else torch.device("cuda")
    )

    if args.status_dict_path:
        # training mode, resuming from the status dict

        # load the train status dict
        with open(args.status_dict_path, "r") as f:
            train_status = yaml.load(f, Loader=yaml.Loader)
        # swap the args
        args = train_status["args"]

        dqn = DQN(args, train_status)
    else:
        dqn = DQN(args)

    # train
    dqn.train()


def eval_runtime(eval_args):
    with open(os.path.join(eval_args.model_dir, "status.yaml"), "r") as f:
        train_status = yaml.load(f, Loader=yaml.Loader)

    dqn = DQN(eval_args, train_status, True)

    # evaluate run-time
    dqn.eval_runtime()

    # evaluate q-values
    print(dqn.eval_q_from_file(agg="mean"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluate",
        dest="to_train",
        help="evaluation mode",
        action="store_false"
    )
    parser.set_defaults(to_train=True)
    current_args, _ = parser.parse_known_args()

    if current_args.to_train:
        print("In training mode")
        parser = build_argparser()
        args = parser.parse_args()
        args.device = (
            torch.device("cpu")
            if args.no_cuda or not torch.cuda.is_available()
            else torch.device("cuda")
        )

        # training
        train(args)

    else:
        print("In evaluation mode")
        eval_parser = build_eval_argparser()
        eval_args, others = eval_parser.parse_known_args()

        # evaluation
        eval(eval_args)
