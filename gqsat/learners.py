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

import torch
from minisat.minisat.gym.MiniSATEnv import VAR_ID_IDX
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch_geometric.utils import scatter


class GraphLearner:

    def __init__(self, net, target, buffer, args):
        self.net = net
        self.target = target
        self.target.eval()

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=args.lr)
        self.lr_scheduler = StepLR(
            self.optimizer, args.lr_scheduler_frequency, args.lr_scheduler_gamma
        )

        if args.loss == "mse":
            self.loss = nn.MSELoss()
        elif args.loss == "huber":
            self.loss = nn.SmoothL1Loss()
        else:
            raise ValueError("Unknown Loss function.")

        self.bsize = args.bsize
        self.gamma = args.gamma
        self.buffer = buffer
        self.target_update_freq = args.target_update_freq
        self.step_ctr = 0
        self.grad_clip = args.grad_clip
        self.grad_clip_norm_type = args.grad_clip_norm_type
        self.device = args.device

    def get_qs(self, states):
        v_out, e_out, _ = self.net(
            x=states[0],
            edge_index=states[2],
            edge_attr=states[1],
            v_indices=states[4],
            e_indices=states[5],
            u=states[6]
        )
        return v_out[states[0][:, VAR_ID_IDX] == 1], states[3]

    def get_target_qs(self, states):
        v_out, e_out, _ = self.target(
            x=states[0],
            edge_index=states[2],
            edge_attr=states[1],
            v_indices=states[4],
            e_indices=states[5],
            u=states[6]
        )
        return v_out[states[0][:, VAR_ID_IDX] == 1].detach(), states[3]

    def expert_actions(self, var_feats, var_vertex_sizes):
        """The actions the rule of the cores of SMS++ may take, as a 0/1 mask
        over the flattened (vertices, 2) actions of a batch whose variable
        vertices have the rows var_feats of eMaxSAT: any variable of largest
        core score (column 3), the rule breaking the ties by an index the
        graph does not show unless the rows have the index of the variable as
        their last column (--bnb-index-feature), in which case the variable
        is the one of smallest index among them, as the rule takes; true first
        (action 2i) if it is true in the best solution (column 2 equal to 1)
        and false first (2i + 1) otherwise."""
        graph_of_var = torch.repeat_interleave(
            torch.arange(len(var_vertex_sizes), device=self.device),
            var_vertex_sizes)
        score = var_feats[:, 3]
        top = scatter(score, graph_of_var, dim=0, reduce='max')
        is_max = score >= top[graph_of_var] - 1e-6
        if var_feats.shape[1] == 8:
            index = torch.where(is_max, var_feats[:, 7],
                                torch.full_like(score, 2.0))
            first = scatter(index, graph_of_var, dim=0, reduce='min')
            is_max = is_max & (var_feats[:, 7] <= first[graph_of_var])
        true_first = var_feats[:, 2] == 1
        return torch.stack([is_max & true_first, is_max & ~true_first],
                           dim=1).flatten().float()

    def margin_loss(self, qs, var_vertex_sizes, a, margin, expert=None):
        """The large margin loss of DQfD on a batch whose actions are those of
        an expert: the mean over the graphs of max_b (Q(s,b) + margin
        [b not expert]) - max_{b expert} Q(s,b), qs being the (vertices, 2) Q
        of the batch; the expert actions are those of the 0/1 mask expert
        over the flattened actions if given, the actions a otherwise."""
        flat = qs.flatten()
        gather_idx = (var_vertex_sizes * qs.shape[1]).cumsum(0).roll(1)
        gather_idx[0] = 0
        graph_of = torch.repeat_interleave(
            torch.arange(len(var_vertex_sizes), device=self.device),
            var_vertex_sizes * qs.shape[1])
        if expert is None:
            expert = torch.zeros_like(flat)
            expert[gather_idx + a] = 1
        best = scatter(flat + margin * (1 - expert), graph_of, dim=0,
                       reduce='max')
        expert_q = scatter(torch.where(expert > 0, flat,
                                       torch.full_like(flat, -1e9)),
                           graph_of, dim=0, reduce='max')
        # the fraction of the graphs whose greedy action is an expert one
        with torch.no_grad():
            top = scatter(flat, graph_of, dim=0, reduce='max')
            self.last_margin_acc = (expert_q >= top).float().mean()
        return (best - expert_q).mean()

    def step(self, margin=None, margin_weight=1.0, expert_set=False,
             grad_clip=None):
        """One batch update; with a margin, the actions of the batch are those
        of an expert and the large margin loss is added to the TD one, the
        expert actions being all those the rule of the cores may take if
        expert_set [see expert_actions()]; grad_clip overrides --grad_clip."""
        s, a, r, s_next, nonterminals = self.buffer.sample(self.bsize)
        # calculate the targets first to optimize the GPU memory

        with torch.no_grad():
            target_qs, target_vertex_sizes = self.get_target_qs(s_next)
            idx_for_scatter = [
                [i] * el.item() * 2 for i, el in enumerate(target_vertex_sizes)
            ]
            idx_for_scatter = torch.tensor(
                [el for subl in idx_for_scatter for el in subl],
                dtype=torch.long,
                device=self.device
            ).flatten()
            target_qs = scatter(target_qs.flatten(), idx_for_scatter, dim=0, reduce='max')
            targets = r + nonterminals * self.gamma * target_qs

        self.net.train()
        qs, var_vertex_sizes = self.get_qs(s)
        # qs.shape[1] values per node (same num of actions per node)
        gather_idx = (var_vertex_sizes * qs.shape[1]).cumsum(0).roll(1)
        gather_idx[0] = 0

        all_qs = qs
        qs = qs.flatten()[gather_idx + a]

        loss = self.loss(qs, targets)
        m_loss = None
        if margin is not None:
            expert = None
            if expert_set:
                var_feats = s[0][s[0][:, VAR_ID_IDX] == 1]
                expert = self.expert_actions(var_feats, var_vertex_sizes)
            m_loss = self.margin_loss(all_qs, var_vertex_sizes, a, margin,
                                      expert)
            loss = loss + margin_weight * m_loss

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.net.parameters(),
            self.grad_clip if grad_clip is None else grad_clip,
            norm_type=self.grad_clip_norm_type
        )
        self.optimizer.step()

        if not self.step_ctr % self.target_update_freq:
            self.target.load_state_dict(self.net.state_dict())

        self.step_ctr += 1

        # I do not know a better solution for getting the lr from the scheduler.
        # This will fail for different lrs for different layers.
        lr_for_the_update = self.lr_scheduler.get_last_lr()[0]

        self.lr_scheduler.step()
        info = {}
        if m_loss is not None:
            info = {"margin_loss": m_loss.item(),
                    "margin_acc": self.last_margin_acc.item()}
        return {
            **info,
            "loss": loss.item(),
            "grad_norm": grad_norm,
            "lr": lr_for_the_update,
            "average_q": qs.mean(),
        }
