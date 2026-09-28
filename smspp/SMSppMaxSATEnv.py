"""The branch and bound of SMS++ on weighted MaxSAT instances, as an
environment with the interface of gym_sat_Env (minisat/minisat/gym/
MiniSATEnv.py), so that dqn.py and evaluate() work on it unchanged.

A state is the graph of the residual formula of the node to branch on
(SATResidualGraph in SMS++, SATBlock/include/SATSolver.h): the unfixed
variables, then the clauses no fixed variable satisfies, with the rows of
eMaxSAT (7 columns, the first being 1 for a variable) or of Graph-Q-SAT (2
columns). An action 2*i + p fixes the i-th variable vertex to true (p = 0)
or false (p = 1) as the first child; -1 leaves the choice to the rule of the
cores of SMS++. The reward is -penalty per node evaluated. The metadata of a
problem directory (METADATA, "name,steps,steps" per line) are the steps the
rule of the cores takes, which the normalized score divides by the steps
of the policy, as the one of Graph-Q-SAT divides those of MiniSat.
"""
import os
import random
import sys
from os import listdir
from os.path import join, realpath, split

import numpy as np

sys.path.insert(0, join(os.path.dirname(os.path.abspath(__file__)), "build"))
import _smspp_env  # noqa: E402

VAR_ID_IDX = 0


class SMSppMaxSATEnv:
    def __init__(self, problems_paths, args, problems_list=None,
                 test_mode=False, max_cap_fill_buffer=True, penalty_size=None,
                 max_data_limit_per_set=None, **kwargs):
        if problems_list is not None:
            raise ValueError("SMSppMaxSATEnv reads its problems from files")
        self.args = args
        self.test_mode = test_mode
        self.max_cap_fill_buffer = max_cap_fill_buffer
        self.penalty_size = penalty_size if penalty_size is not None else 0.1
        self.problems_paths = ([realpath(d) for d in problems_paths.split(":")]
                               if problems_paths is not None else [])
        files = [[join(d, f) for f in sorted(listdir(d))
                  if f.endswith(".wcnf") or f.endswith(".cnf")]
                 for d in self.problems_paths]
        if max_data_limit_per_set is not None:
            files = [list(np.random.choice(f, size=min(len(f),
                                                         max_data_limit_per_set),
                                           replace=False)) for f in files]
        self.test_files = [f for fs in files for f in fs]
        self.test_file_num = len(self.test_files)
        self.test_to = 0

        self.metadata = {}
        try:
            for d in self.problems_paths:
                self.metadata[d] = {}
                with open(join(d, "METADATA")) as f:
                    for line in f:
                        k, a, b = line.strip().split(",")
                        self.metadata[d][k] = [int(a), int(b)]
        except OSError:
            print("No metadata available, that is fine for the metadata "
                  "generator.")
            self.metadata = None

        self.features = getattr(args, "bnb_features", 1)
        self.bnb = _smspp_env.BnBEnv(getattr(args, "bnb_solver",
                                             "CaDiCaLSATSolver"),
                                     getattr(args, "bnb_max_iter", 20),
                                     self.features, self.penalty_size)
        self.vertex_in_size = self.bnb.n_col()
        self.edge_in_size = 2
        self.global_in_size = 1
        self.max_decisions_cap = float("inf")
        self.step_ctr = 0
        self.curr_problem = None
        self.curr_state = None
        self.is_solved = None
        self.max_clause_len = 0

    def random_pick_sat_prob(self):
        if self.test_mode:
            filename = self.test_files[self.test_to]
            self.test_to = (self.test_to + 1) % self.test_file_num
            return filename
        return self.test_files[random.randint(0, self.test_file_num - 1)]

    def reset(self, max_decisions_cap=None):
        self.step_ctr = 0
        self.max_decisions_cap = (float("inf") if max_decisions_cap is None
                                  else max_decisions_cap)
        self.curr_problem = self.random_pick_sat_prob()
        self.curr_state, self.is_solved = self.bnb.reset(self.curr_problem)
        return self.curr_state

    def step(self, decision, dummy=False):
        self.step_ctr += 1
        info = {"curr_problem": self.curr_problem, "num_restarts": 0,
                "max_clause_len": 0}
        if dummy or self.step_ctr > self.max_decisions_cap:
            # past the cap the rule of the cores decides; with
            # max_cap_fill_buffer each of its steps is still a transition,
            # otherwise it plays to the end, the reward being the total
            if self.max_cap_fill_buffer or dummy:
                state, r, done = self.bnb.step(-1)
            else:
                r, done = 0.0, False
                while not done:
                    state, rr, done = self.bnb.step(-1)
                    r += rr
        else:
            state, r, done = self.bnb.step(int(decision))
        self.curr_state, self.is_solved = state, done
        return state, r, done, info

    def normalized_score(self, steps, problem):
        pdir, pname = split(problem)
        rule_steps, _ = self.metadata[pdir][pname]
        return rule_steps / max(steps, 1)

    def get_dummy_state(self):
        v = np.zeros((2, self.vertex_in_size), dtype=np.float32)
        v[:, VAR_ID_IDX] = 1
        return (v, np.zeros((0, 2), dtype=np.float32),
                np.zeros((2, 0), dtype=np.int64),
                np.zeros((1, 1), dtype=np.float32))
