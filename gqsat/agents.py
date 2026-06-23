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
import numpy as np
from minisat.minisat.gym.MiniSATEnv import VAR_ID_IDX


class Agent:

    def act(self, state):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError


class MiniSATAgent(Agent):
    """Use MiniSAT agent to solve the problem"""

    def act(self, observation):
        return -1  # this will make GymSolver use VSIDS to make a decision

    def __str__(self):
        return "<MiniSAT Agent>"


class RandomAgent(Agent):
    """Uniformly sample the action space"""

    def __init__(self, action_space):
        super().__init__()
        self.action_space = action_space

    def act(self, observation):
        return self.action_space.sample()

    def __str__(self):
        return "<Random Agent>"


class GraphAgent:

    def __init__(self, net, args):
        self.net = net
        self.device = args.device
        self.debug = args.debug
        self.qs_buffer = []

    def forward(self, hist_buffer):
        self.net.eval()
        with torch.no_grad():
            vdata, edata, conn, udata = hist_buffer[0]
            vdata = torch.tensor(vdata, device=self.device)
            edata = torch.tensor(edata, device=self.device)
            udata = torch.tensor(udata, device=self.device)
            conn = torch.tensor(conn, device=self.device)
            vout, eout, _ = self.net(x=vdata, edge_index=conn, edge_attr=edata, u=udata)
            res = vout[vdata[:, VAR_ID_IDX] == 1]

            if self.debug:
                self.qs_buffer.append(res.flatten().cpu().numpy())
            return res

    def act(self, hist_buffer, eps=0):
        if np.random.random() < eps:
            vars_to_decide = np.where(hist_buffer[-1][0][:, VAR_ID_IDX] == 1)[0]
            acts = [a for v in vars_to_decide for a in (v * 2, v * 2 + 1)]
            return int(np.random.choice(acts))
        else:
            qs = self.forward(hist_buffer)
            return self.choose_actions(qs)

    def choose_actions(self, qs):
        return qs.flatten().argmax().item()


class RestrictedGraphAgent(GraphAgent):
    """Graph-Q-SAT restricted to the cold-start phase, with optional action pooling.

    Two engineering tricks from Shirokikh et al. (2023), *Machine Learning for SAT:
    Restricted Heuristics and New Graph Representations* (arXiv:2307.09141), aimed at
    the wall-clock cost of running the GNN at every CDCL decision:

    * **early release** (``release_after`` > 0): the model guides only the first
      ``release_after`` decisions, then control is handed back to MiniSat's VSIDS
      (action ``-1``). The learned heuristic matters most at the cold start, where
      VSIDS activities are still uninformative; VSIDS is far cheaper afterwards.
    * **action pool** (``action_pool_size`` > 1): one GNN forward yields the top-k
      actions, executed over the next steps without re-running the net. Pooled
      actions are cached as *original* decision ids and remapped to the rebuilt
      graph each step via the env's ``decision_to_var_mapping``, skipping variables
      that meanwhile got assigned.

    Falls back to plain Graph-Q-SAT behaviour when ``release_after == 0`` and
    ``action_pool_size == 1``.
    """

    wants_env = True  # the eval loop passes `env` to act() for the action pool

    def __init__(self, net, args):
        super().__init__(net, args)
        self.release_after = int(getattr(args, "release_after", 0) or 0)
        self.pool_size = max(1, int(getattr(args, "action_pool_size", 1) or 1))
        self.warmstart = bool(getattr(args, "warmstart_release", False))
        self.reset()

    def reset(self):
        self._decisions = 0
        self._pool = []  # cached original decision ids, best-first
        self._warmed = False

    def _apply_warmstart(self, hist_buffer, env):
        """Seed MiniSat's VSIDS activities from the Q-values of the root state
        (Shirokikh et al. 2023): activity(x) = -1 / max(Q(x,True), Q(x,False))."""
        qs = self.forward(hist_buffer).cpu().numpy()       # (n_decidable, 2)
        mapping = list(env.decision_to_var_mapping)
        max_q = qs.max(axis=1)
        orig_vars = [abs(mapping[2 * i]) - 1 for i in range(len(max_q))]
        acts = np.zeros((max(orig_vars) + 1) if orig_vars else 0, dtype=np.float64)
        for i, ov in enumerate(orig_vars):
            acts[ov] = -1.0 / min(float(max_q[i]), -1e-6)   # >0, larger for higher Q
        env.set_activities(acts)

    def _next_from_pool(self, mapping):
        """Best cached action whose variable is still decidable, as a current local
        action index; None when the pool holds no valid action."""
        inv = {orig: local for local, orig in enumerate(mapping)}
        while self._pool:
            orig = self._pool.pop(0)
            if orig in inv:
                return inv[orig]
        return None

    def act(self, hist_buffer, eps=0, env=None):
        if self.warmstart:
            if not self._warmed and env is not None:
                self._apply_warmstart(hist_buffer, env)
                self._warmed = True
            return -1  # one Q-forward to seed activities, then pure VSIDS

        if self.release_after and self._decisions >= self.release_after:
            return -1  # hand control back to MiniSat's VSIDS

        mapping = list(env.decision_to_var_mapping) if env is not None else None

        # reuse a still-valid pooled action without touching the GNN
        if self.pool_size > 1 and mapping is not None and self._pool:
            local = self._next_from_pool(mapping)
            if local is not None:
                self._decisions += 1
                return local

        flat = self.forward(hist_buffer).flatten()
        if self.pool_size > 1 and mapping is not None:
            order = torch.argsort(flat, descending=True).cpu().numpy()
            top = order[: self.pool_size]
            self._pool = [mapping[int(a)] for a in top]  # cache original ids
            local = self._next_from_pool(mapping)
            action = local if local is not None else int(top[0])
        else:
            action = int(flat.argmax().item())
        self._decisions += 1
        return action
