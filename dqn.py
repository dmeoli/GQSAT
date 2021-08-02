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

import copy
import os
import pickle
import time
from collections import deque, defaultdict

import numpy as np
import torch
import yaml
from tensorboardX import SummaryWriter

from gqsat.agents import GraphAgent, MiniSATAgent
from gqsat.buffer import ReplayGraphBuffer
from gqsat.learners import GraphLearner
from gqsat.models import EncoderCoreDecoder, SatModel
from gqsat.utils import evaluate, make_env


def save_training_state(
        model,
        learner,
        episodes_done,
        transitions_seen,
        best_eval_so_far,
        args,
        in_eval_mode=False
):
    # save the model
    model_path = os.path.join(args.logdir, f"model_{learner.step_ctr}.chkp")
    torch.save(model.state_dict(), model_path)

    # save the experience replay
    buffer_path = os.path.join(args.logdir, "buffer.pkl")

    with open(buffer_path, "wb") as f:
        pickle.dump(learner.buffer, f)

    # save important parameters
    train_status = {
        "step_ctr": learner.step_ctr,
        "latest_model_name": model_path,
        "buffer_path": buffer_path,
        "args": args,
        "episodes_done": episodes_done,
        "logdir": args.logdir,
        "transitions_seen": transitions_seen,
        "optimizer_state_dict": learner.optimizer.state_dict(),
        "optimizer_class": type(learner.optimizer),
        "best_eval_so_far": best_eval_so_far,
        "scheduler_class": type(learner.lr_scheduler),
        "scheduler_state_dict": learner.lr_scheduler.state_dict(),
        "in_eval_mode": in_eval_mode,
    }
    status_path = os.path.join(args.logdir, "status.yaml")

    with open(status_path, "w") as f:
        yaml.dump(train_status, f, default_flow_style=False)

    return status_path


def get_annealed_eps(n_trans, args):
    if n_trans < args.init_exploration_steps:
        return args.eps_init
    if n_trans > args.eps_decay_steps:
        return args.eps_final
    else:
        assert n_trans - args.init_exploration_steps >= 0
        return (args.eps_init - args.eps_final) * (
                1 - (n_trans - args.init_exploration_steps) / args.eps_decay_steps
        ) + args.eps_final


def arg2activation(activ_str):
    if activ_str == "relu":
        return torch.nn.ReLU
    elif activ_str == "tanh":
        return torch.nn.Tanh
    elif activ_str == "leaky_relu":
        return torch.nn.LeakyReLU
    else:
        raise ValueError("Unknown activation function")


class DQN:
    """
    DQN object for setting up env, agent, learner
    Training happens in train() function
    For evaluation there are two modes:
    (1) runtime evaluation for the problems in eval_problems_paths happens in eval_runtime()
    (2) Q-value evaluation for the problems from directory happens in eval_q_from_file()
    (3) Q-value evaluation for the given graph happens in eval_q_from_graph
    """

    def __init__(self, args, train_status=None, eval=False):
        self.writer = SummaryWriter()
        self.env = None

        if train_status is not None:
            if not eval:
                self._init_from_status(args, train_status)
            else:
                self._init_for_eval(args, train_status)
        else:
            self._init_from_scratch(args)
        print(args.__str__())

    def _init_from_status(self, args, train_status):
        """
        Initialization for training from previously incomplete run
        :param args: arguments for training
        :param train_status: train status from status.yaml file from the previous run

        :returns: different self.object to be used in train() function
        """

        self.eval_resume_signal = train_status["in_eval_mode"]
        # load the model
        net = SatModel.load_from_yaml(os.path.join(args.logdir, "model.yaml")).to(
            args.device
        )
        net.load_state_dict(torch.load(train_status["latest_model_name"]))

        target_net = SatModel.load_from_yaml(
            os.path.join(args.logdir, "model.yaml")
        ).to(args.device)
        target_net.load_state_dict(net.state_dict())

        # load the buffer
        if train_status["buffer_path"] is not None:
            with open(train_status["buffer_path"], "rb") as f:
                self.buffer = pickle.load(f)
        else:
            self.buffer = None
        self.learner = GraphLearner(net, target_net, self.buffer, args)
        self.learner.step_ctr = train_status["step_ctr"]

        self.learner.optimizer = train_status["optimizer_class"](
            net.parameters(), lr=args.lr
        )
        self.learner.optimizer.load_state_dict(train_status["optimizer_state_dict"])
        self.learner.lr_scheduler = train_status["scheduler_class"](
            self.learner.optimizer, args.lr_scheduler_frequency, args.lr_scheduler_gamma
        )
        self.learner.lr_scheduler.load_state_dict(train_status["scheduler_state_dict"])

        # load misc training status params
        self.n_trans = train_status["transitions_seen"]
        self.ep = train_status["episodes_done"]

        self.env = make_env(args.train_problems_paths, args, test_mode=False)

        self.agent = GraphAgent(net, args)

        self.best_eval_so_far = train_status["best_eval_so_far"]

        self.args = args

    def _init_from_scratch(self, args):
        """
        Initialization for training from scratch
        :param args: arguments for training

        :returns: different self.object to be used in train() function
        """
        # training mode, learning from scratch or continuing learning from some previously trained model
        args.logdir = self.writer.logdir

        model_save_path = os.path.join(args.logdir, "model.yaml")
        self.best_eval_so_far = (
            {args.eval_problems_paths: -1}
            if not args.eval_separately_on_each
            else {k: -1 for k in args.eval_problems_paths.split(":")}
        )

        self.env = make_env(args.train_problems_paths, args, test_mode=False)
        if args.model_dir is not None:
            # load an existing model and continue training
            net = SatModel.load_from_yaml(
                os.path.join(args.model_dir, "model.yaml")
            ).to(args.device)
            net.load_state_dict(
                torch.load(os.path.join(args.model_dir, args.model_checkpoint))
            )
        else:
            # learning from scratch
            net = EncoderCoreDecoder(
                (self.env.vertex_in_size, self.env.edge_in_size, self.env.global_in_size),
                core_out_dims=(
                    args.core_v_out_size,
                    args.core_e_out_size,
                    args.core_e_out_size
                ),
                out_dims=(2, None, None),
                core_steps=args.core_steps,
                dec_out_dims=(
                    args.decoder_v_out_size,
                    args.decoder_e_out_size,
                    args.decoder_e_out_size
                ),
                encoder_out_dims=(
                    args.encoder_v_out_size,
                    args.encoder_e_out_size,
                    args.encoder_e_out_size
                ),
                save_name=model_save_path,
                e2v_agg=args.e2v_aggregator,
                n_hidden=args.n_hidden,
                hidden_size=args.hidden_size,
                activation=arg2activation(args.activation),
                independent_block_layers=args.independent_block_layers
            ).to(args.device)
        print(str(net))
        target_net = copy.deepcopy(net)

        self.buffer = ReplayGraphBuffer(args, args.buffer_size)
        self.agent = GraphAgent(net, args)

        self.n_trans = 0
        self.ep = 0
        self.learner = GraphLearner(net, target_net, self.buffer, args)
        self.eval_resume_signal = False
        self.args = args

    def _init_for_eval(self, args, train_status):
        """
        Initialization for evaluating on problems from a given directory
        :param args: arguments for evaluation
        :param train_status: training status from status.yaml file from the run
        """
        eval_args = copy.deepcopy(args)
        args = train_status["args"]

        # use same args used for training and overwrite them with those asked for eval
        for k, v in vars(eval_args).items():
            setattr(args, k, v)

        args.device = (
            torch.device("cpu")
            if args.no_cuda or not torch.cuda.is_available()
            else torch.device("cuda")
        )

        net = SatModel.load_from_yaml(os.path.join(args.model_dir, "model.yaml")).to(
            args.device
        )

        # modify core steps for the eval as requested
        if args.core_steps != -1:
            # -1 if use the same as for training
            net.steps = args.core_steps

        net.load_state_dict(
            torch.load(os.path.join(args.model_dir, args.model_checkpoint)), strict=False
        )

        self.agent = GraphAgent(net, args)
        self.agent.net.eval()

        self.args = args

    def set_problems(self, adj_mat_list):
        self.env = make_env(None, self.args, adj_mat_list)

    def train(self):
        """
        Training happens here.
        """
        while self.learner.step_ctr < self.args.batch_updates:

            ret = 0
            r = 0
            obs = self.env.reset(
                max_decisions_cap=self.args.train_time_max_decisions_allowed
            )
            done = self.env.is_solved

            if self.args.history_len > 1:
                raise NotImplementedError(
                    "History len greater than one is not implemented for graph nets."
                )
            hist_buffer = deque(maxlen=self.args.history_len)
            for _ in range(self.args.history_len):
                hist_buffer.append(obs)
            ep_step = 0

            save_flag = False

            while not done:
                annealed_eps = get_annealed_eps(self.n_trans, self.args)
                action = self.agent.act(hist_buffer, eps=annealed_eps)
                next_obs, r, done, _ = self.env.step(action)
                self.buffer.add_transition(obs, action, r, done)
                obs = next_obs

                hist_buffer.append(obs)
                ret += r

                if (not self.n_trans % self.args.step_freq) and (
                        self.buffer.ctr > max(self.args.init_exploration_steps, self.args.bsize + 1)
                        or self.buffer.full
                ):
                    step_info = self.learner.step()
                    if annealed_eps is not None:
                        step_info["annealed_eps"] = annealed_eps

                    # we increment the step_ctr in the learner.step(), that's why we need to do -1 in tensorboarding
                    # we do not need to do -1 in checking for frequency since 0 has already passed

                    if not self.learner.step_ctr % self.args.save_freq:
                        # save the exact model you evaluated and make another save after the episode ends
                        # to have proper transitions in the replay buffer to pickle
                        status_path = save_training_state(
                            self.agent.net,  # TODO : It was only net (but this should also be correct)
                            self.learner,
                            self.ep - 1,
                            self.n_trans,
                            self.best_eval_so_far,
                            self.args,
                            in_eval_mode=self.eval_resume_signal
                        )
                        save_flag = True
                    if (
                            self.args.env_name == "sat-v0" and not self.learner.step_ctr % self.args.eval_freq
                    ) or self.eval_resume_signal:
                        _, _, scores, _, self.eval_resume_signal = evaluate(
                            self.agent, self.args, include_train_set=False
                        )

                        for sc_key, sc_val in scores.items():
                            # list can be empty if we hit the time limit for eval
                            if len(sc_val) > 0:
                                res_vals = [el for el in sc_val.values()]
                                median_score = np.nanmedian(res_vals)
                                if (
                                        self.best_eval_so_far[sc_key] < median_score
                                        or self.best_eval_so_far[sc_key] == -1
                                ):
                                    self.best_eval_so_far[sc_key] = median_score
                                self.writer.add_scalar(
                                    f"data/median relative score: {sc_key}",
                                    np.nanmedian(res_vals),
                                    self.learner.step_ctr - 1
                                )
                                self.writer.add_scalar(
                                    f"data/mean relative score: {sc_key}",
                                    np.nanmean(res_vals),
                                    self.learner.step_ctr - 1
                                )
                                self.writer.add_scalar(
                                    f"data/max relative score: {sc_key}",
                                    np.nanmax(res_vals),
                                    self.learner.step_ctr - 1
                                )
                        for k, v in self.best_eval_so_far.items():
                            self.writer.add_scalar(k, v, self.learner.step_ctr - 1)

                    for k, v in step_info.items():
                        self.writer.add_scalar(k, v, self.learner.step_ctr - 1)

                    self.writer.add_scalar("data/num_episodes", self.ep, self.learner.step_ctr - 1)

                self.n_trans += 1
                ep_step += 1

            self.writer.add_scalar("data/ep_return", ret, self.learner.step_ctr - 1)
            self.writer.add_scalar("data/ep_steps", self.env.step_ctr, self.learner.step_ctr - 1)
            self.writer.add_scalar("data/ep_last_reward", r, self.learner.step_ctr - 1)
            print(f"Episode {self.ep + 1}: Return {ret}.")
            self.ep += 1

            if save_flag:
                status_path = save_training_state(
                    self.agent.net,  # TODO: Is agent net the same as net?
                    self.learner,
                    self.ep - 1,
                    self.n_trans,
                    self.best_eval_so_far,
                    self.args,
                    in_eval_mode=self.eval_resume_signal
                )
                save_flag = False

    def eval_runtime(self):
        """
        Evaluation on different problem sets to compare performance of RL solver.
        This function will directly use function available in gqsat/utils.py
        """
        st_time = time.time()
        _, _, scores, eval_metadata, _ = evaluate(self.agent, self.args)
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
            print(f"Results for {pset}")
            print(
                f"median_relative_score: {np.nanmedian(res_list)}, mean_relative_score: {np.mean(res_list)}"
            )

    def eval_q_for_agent_from_graph(self, adj_mat, use_minisat=False):
        """
        evaluate runtime for a given adj mat for the minisat or GQSat agent
        :param adj_mat: adjacency matrix for the problem
        :param use_minisat: uses minisat agent if true, else self.agent from the solver object
        """

        agent = MiniSATAgent() if use_minisat else self.agent
        env = make_env(None, self.args, [adj_mat])
        obs = env.reset(
            max_decisions_cap=self.args.train_time_max_decisions_allowed
        )
        done = env.is_solved
        if done:
            return 0
        q = 0
        with torch.no_grad():
            while not done:
                obs, r, done, _ = env.step(agent.act([obs]))
                q += r
        return q

    def eval_q_from_file(self, eval_problems_paths=None, agg="sum"):
        """
        Q-value evaluation of problems in eval_problems_paths.
        If eval_problems_paths is None, evaluation will happen in args.eval_problems_paths
        
        :param eval_problems_paths: dir(s) where problems are saved for evaluation
        :param agg: aggregation of q-values for a graph (either "sum" or "mean")
        
        :returns res_q: Dict of Dicts where structure of dict is as follows
                        res_q[eval_problem_path][problem_filename] = QValue
        """
        # if eval problems are not provided q value evaluation happens for the
        # problem sets in self.args.eval_problems_paths
        if not eval_problems_paths:
            eval_problems_paths = self.args.eval_problems_paths

        problem_sets = (
            [eval_problems_paths]
            if not self.args.eval_separately_on_each
            else [k for k in self.args.eval_problems_paths.split(":")]
        )

        res_q = defaultdict(dict)

        for pset in problem_sets:
            eval_env = make_env(pset, self.args, test_mode=True)
            q_scores = {}
            pr = 0
            with torch.no_grad():
                while eval_env.test_to != 0 or pr == 0:
                    obs = eval_env.reset(
                        max_decisions_cap=self.args.test_time_max_decisions_allowed
                    )
                    # TODO: This is broken since eval_q_from_graph is different now
                    q = self.eval_q_from_graph([obs], agg)

                    q_scores[eval_env.curr_problem] = q

                    pr += 1

            res_q[pset] = q_scores

        return res_q

    def eval_q_from_graph(self, adj_mat, agg="max", use_minisat=False):
        """
        Evaluation of q-value from the graph structure. This function directly calls forward pass for the agent.
        :param adj_mat: adjacency matrix for the problem
        :param agg: aggregation of q-values for a graph (either "sum" or "mean")
        :param use_minisat: Whether a run of minisat should be used to calculate the reward.

        :returns q: q-value for a given graph
        """

        env = make_env(None, self.args, [adj_mat])
        obs = env.reset(
            max_decisions_cap=self.args.train_time_max_decisions_allowed
        )
        if env.is_solved:
            return 0

        if use_minisat:
            # run the minisat agent to calculate the number of branches
            agent = MiniSATAgent()
            done = env.is_solved
            q = 0
            while not done:
                obs, r, done, _ = env.step(agent.act(obs))
                q += r
            return q

        q = self.agent.forward([obs])
        if agg == "sum":
            q = q.max(1).values.sum().cpu().item()
        elif agg == "mean":
            q = q.max(1).values.mean().cpu().item()
        elif agg == "max":
            q = q.flatten().max().cpu().item()
        elif agg == "expectation":
            flat_q = q.flatten()
            q = torch.sum(torch.softmax(flat_q, dim=0) * flat_q).cpu().item()
        else:
            raise ValueError(f"agg {agg} is not recognized")
        return q
