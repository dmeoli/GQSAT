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

import os
import time

import numpy as np
import torch
import yaml

from gqsat.agents import GraphAgent, RestrictedGraphAgent
from gqsat.models import SATModel
from gqsat.utils import build_eval_argparser, evaluate

if __name__ == "__main__":
    parser = build_eval_argparser()
    eval_args = parser.parse_args()

    # status.yaml (and the checkpoints) embed torch tensors saved on CUDA. When
    # running without a GPU, PyYAML's reconstruction hits torch's _load_from_bytes,
    # which ignores map_location, so force every storage to restore on CPU.
    if eval_args.no_cuda or not torch.cuda.is_available():
        import torch.serialization as _ts
        _orig_restore = _ts.default_restore_location
        _ts.default_restore_location = lambda storage, loc: _orig_restore(storage, "cpu")

    with open(os.path.join(eval_args.model_dir, "status.yaml"), "r") as f:
        train_status = yaml.load(f, Loader=yaml.Loader)
        args = train_status["args"]

    # use same args used for training and overwrite them with those asked for eval
    for k, v in vars(eval_args).items():
        setattr(args, k, v)

    args.device = (
        torch.device("cpu")
        if args.no_cuda or not torch.cuda.is_available()
        else torch.device("cuda")
    )
    net = SATModel.load_from_yaml(os.path.join(args.model_dir, "model.yaml")).to(
        args.device
    )

    # modify core steps for the eval as requested
    if args.core_steps != -1:
        # -1 if use the same as for training
        net.steps = args.core_steps

    state_dict = torch.load(
        os.path.join(args.model_dir, args.model_checkpoint), map_location=args.device
    )
    # bridge GATConv lin_src/lin_dst <-> lin naming across torch-geometric versions
    state_dict = SATModel.reconcile_gat_lin_keys(net, state_dict)
    net.load_state_dict(state_dict, strict=False)

    if (getattr(args, "release_after", 0) > 0 or getattr(args, "action_pool_size", 1) > 1
            or getattr(args, "warmstart_release", False)):
        agent = RestrictedGraphAgent(net, args)  # early-release / pool / Q warm-start
    else:
        agent = GraphAgent(net, args)

    st_time = time.time()
    _, _, scores, eval_metadata, _ = evaluate(agent, args)
    end_time = time.time()

    # print(
    #     f"Evaluation is over. It took {end_time - st_time} seconds for the whole procedure"
    # )
    print(
        f"total_eval_time\t{end_time - st_time}"
    )

    # with open("../eval_results.pkl", "wb") as f:
    #     pickle.dump(scores, f)

    for pset, pset_res in scores.items():
        res_list = [el for el in pset_res.values()]
        print(f"Results for\t{pset}")
        print(
            f"median_relative_score:\t{np.nanmedian(res_list)}\n"
            f"mean_relative_score:\t{np.mean(res_list)}\n"
            f"min_score:\t{np.min(res_list)}\n"
            f"max_score:\t{np.max(res_list)}"
        )
