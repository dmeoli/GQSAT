# Graph-Q-SAT-UPD. This file is heavily changed and supports variable-sized SAT problems, multiple datasets
# and generates graph-state representations for Graph-Q-SAT.

import random
from os import listdir
from os.path import join, realpath, split

import gym
import numpy as np

from .GymSolver import GymSolver

MINISAT_DECISION_CONSTANT = 32767
VAR_ID_IDX = (
    0  # put 1 at the position of this index to indicate that the node is a variable
)

global_in_size = 1
vertex_in_size = 2
edge_in_size = 2


class gym_sat_Env(gym.Env):

    def __init__(
            self,
            problems_paths,
            args,
            problems_list=None,
            test_mode=False,
            max_cap_fill_buffer=True,
            penalty_size=None,
            with_restarts=None,
            compare_with_restarts=None,
            max_data_limit_per_set=None
    ):

        self.problem_list = problems_list
        self.problems_paths = [realpath(el) for el in problems_paths.split(":")] if problems_paths is not None else []
        self.args = args
        self.test_mode = test_mode

        self.max_data_limit_per_set = max_data_limit_per_set
        pre_test_files = [
            [join(dir, f) for f in listdir(dir) if f.endswith(".cnf")]
            for dir in self.problems_paths
        ]
        if self.max_data_limit_per_set is not None:
            pre_test_files = [
                np.random.choice(el, size=max_data_limit_per_set, replace=False)
                for el in pre_test_files
            ]
        self.test_files = [sl for el in pre_test_files for sl in el]

        self.metadata = {}
        self.max_decisions_cap = float("inf")
        self.max_cap_fill_buffer = max_cap_fill_buffer
        self.penalty_size = penalty_size if penalty_size is not None else 0.0001
        self.with_restarts = True if with_restarts is None else with_restarts
        self.compare_with_restarts = (
            False if compare_with_restarts is None else compare_with_restarts
        )

        try:
            for dir in self.problems_paths:
                self.metadata[dir] = {}
                with open(join(dir, "METADATA")) as f:
                    for l in f:
                        k, rscore, msscore = l.split(",")
                        self.metadata[dir][k] = [int(rscore), int(msscore)]
        except Exception as e:
            print(e)
            print("No metadata available, that is fine for metadata generator.")
            self.metadata = None
        self.test_file_num = len(self.test_files)
        self.test_to = 0

        self.step_ctr = 0
        self.curr_problem = None

        self.global_in_size = global_in_size
        self.vertex_in_size = vertex_in_size
        self.edge_in_size = edge_in_size
        self.max_clause_len = 0
        self.S = None
        self.curr_state = None
        self.is_solved = None

    def parse_state_as_graph(self):

        # if S is already Done, should return a dummy state to store in the buffer.
        if self.S.get_done():
            # to not mess with the C++ code, let's build a dummy graph which will not be used in the q updates anyways
            # since we multiply (1-dones)
            empty_state = self.get_dummy_state()
            self.decision_to_var_mapping = {
                el: el
                for sl in range(empty_state[0].shape[0])
                for el in (2 * sl, 2 * sl + 1)
            }
            return empty_state, True

        # S is not yet Done, parse and return real state

        (
            total_var,
            _,
            current_depth,
            n_init_clauses,
            num_restarts,
            _
        ) = self.S.get_metadata()
        var_assignments = self.S.get_assignments()
        num_var = sum([1 for el in var_assignments if el == 2])

        # only valid decisions
        valid_decisions = [
            el
            for i in range(len(var_assignments))
            for el in (2 * i, 2 * i + 1)
            if var_assignments[i] == 2
        ]
        valid_vars = [
            idx for idx in range(len(var_assignments)) if var_assignments[idx] == 2
        ]
        # we need remapping since we keep only unassigned vars in the observations,
        # however, the environment does know about this, it expects proper indices of the variables
        vars_remapping = {el: i for i, el in enumerate(valid_vars)}
        self.decision_to_var_mapping = {
            i: val_decision for i, val_decision in enumerate(valid_decisions)
        }

        # we should return the vertex/edge numpy objects from the c++ code to make this faster
        clauses = self.S.get_clauses()

        if len(clauses) == 0:
            # this is to avoid feeding empty data structures to our model
            # when the MiniSAT environment returns an empty graph
            # it might return an empty graph since we do not construct it when
            # step > max_cap and max_cap can be zero (all decisions are made by MiniSAT's VSIDS).
            empty_state = self.get_dummy_state()
            self.decision_to_var_mapping = {
                el: el
                for sl in range(empty_state[0].shape[0])
                for el in (2 * sl, 2 * sl + 1)
            }
            return empty_state, False

        clause_counter = 0
        clauses_lens = [len(cl) for cl in clauses]
        self.max_clause_len = max(clauses_lens)
        edge_data = np.zeros((sum(clauses_lens) * 2, 2), dtype=np.float32)
        connectivity = np.zeros((2, edge_data.shape[0]), dtype=np.int)
        ec = 0
        for cl in clauses:
            for l in cl:
                # if positive, create a [0,1] edge from the var to the current clause, else [1,0]
                # data = [1, 0] if l==True else [0, 1]

                # this is not a typo, we want two edge here
                edge_data[ec: ec + 2, int(l > 0)] = 1

                remapped_l = vars_remapping[abs(l) - 1]
                # from var to clause
                connectivity[0, ec] = remapped_l
                connectivity[1, ec] = num_var + clause_counter
                # from clause to var
                connectivity[0, ec + 1] = num_var + clause_counter
                connectivity[1, ec + 1] = remapped_l

                ec += 2
            clause_counter += 1

        vertex_data = np.zeros(
            (num_var + clause_counter, vertex_in_size), dtype=np.float32
        )  # both vars and clauses are vertex in the graph
        vertex_data[:num_var, VAR_ID_IDX] = 1
        vertex_data[num_var:, VAR_ID_IDX + 1] = 1

        return (
            (
                vertex_data,
                edge_data,
                connectivity,
                np.zeros((1, global_in_size), dtype=np.float32)
            ),
            False
        )

    def random_pick_sat_prob(self):
        if self.test_mode:  # in the test mode, just iterate all test files in order
            filename = self.test_files[self.test_to]
            self.test_to += 1
            if self.test_to >= self.test_file_num:
                self.test_to = 0
            return filename
        else:  # not in test mode, return a random file in "uf20-91" folder.
            return self.test_files[random.randint(0, self.test_file_num - 1)]

    def reset(self, max_decisions_cap=None):
        self.step_ctr = 0

        if max_decisions_cap is None:
            # max_decisions_cap = sys.maxsize  # long
            max_decisions_cap = np.iinfo(np.intc).max  # i.e., 2**31 - 1
        self.max_decisions_cap = max_decisions_cap

        if self.problem_list is not None:
            self.curr_problem = "in_memory"
            problem_adj_mat = self.problem_list[random.randrange(len(self.problem_list))]
            self.S = GymSolver(
                sat_prob="",
                adj_mat=problem_adj_mat,
                in_memory=True,
                with_restarts=self.with_restarts,
                max_decision_cap=max_decisions_cap
            )
        else:
            self.curr_problem = self.random_pick_sat_prob()
            self.S = GymSolver(
                sat_prob=self.curr_problem,
                adj_mat=np.array([[0]], dtype=np.intc),
                in_memory=False,
                with_restarts=self.with_restarts,
                max_decision_cap=max_decisions_cap
            )
        self.max_clause_len = 0

        self.curr_state, self.is_solved = self.parse_state_as_graph()
        return self.curr_state

    def step(self, decision, dummy=False):
        # now when we drop variables, we store the mapping
        # convert dropped var decision to the original decision id
        if decision >= 0:
            decision = self.decision_to_var_mapping[decision]
        self.step_ctr += 1

        if dummy:
            self.S.step(MINISAT_DECISION_CONSTANT)
            (
                num_var,
                _,
                current_depth,
                n_init_clauses,
                num_restarts,
                _
            ) = self.S.get_metadata()
            return (
                None,
                None,
                self.S.get_done(),
                {
                    "curr_problem": self.curr_problem,
                    "num_restarts": num_restarts,
                    "max_clause_len": self.max_clause_len,
                }
            )

        if self.step_ctr > self.max_decisions_cap:
            while not self.S.get_done():
                self.S.step(MINISAT_DECISION_CONSTANT)
                if self.max_cap_fill_buffer:
                    # return every next state when param is true
                    break
                self.step_ctr += 1
            else:
                # if we are here, we are not filling the buffer and we need to reduce the counter by one to
                # correct for the increment for the last state
                self.step_ctr -= 1
        else:
            # TODO for debugging purposes, we need to add all the checks
            # I removed this action_set checks for performance optimisation

            # var_values = self.curr_state[0][:, 2]
            # var_values = self.S.get_assignments()
            # action_set = [
            #     a
            #     for v_idx, v in enumerate(var_values)
            #     for a in (v_idx * 2, v_idx * 2 + 1)
            #     if v == 2
            # ]

            if decision < 0:  # this is to say that let minisat pick the decision
                decision = MINISAT_DECISION_CONSTANT
            elif (
                    decision % 2 == 0
            ):  # this is to say that pick decision and assign positive value
                decision = int(decision / 2 + 1)
            else:  # this is to say that pick decision and assign negative value
                decision = 0 - int(decision / 2 + 1)

            # if (decision == MINISAT_DECISION_CONSTANT) or orig_decision in action_set:
            self.S.step(decision)
            # else:
            #    raise ValueError("Illegal action")

        self.curr_state, self.is_solved = self.parse_state_as_graph()
        (
            num_var,
            _,
            current_depth,
            n_init_clauses,
            num_restarts,
            _
        ) = self.S.get_metadata()

        # if we fill the buffer, the rewards are the same as GQSAT was making decisions
        if self.step_ctr > self.max_decisions_cap and not self.max_cap_fill_buffer:
            # if we do not fill the buffer, but play till the end, we still need to penalize
            # since GQSAT hasn't solved the problem
            step_reward = -self.penalty_size
        else:
            step_reward = 0 if self.is_solved else -self.penalty_size

        return (
            self.curr_state,
            step_reward,
            self.is_solved,
            {
                "curr_problem": self.curr_problem,
                "num_restarts": num_restarts,
                "max_clause_len": self.max_clause_len,
            }
        )

    def normalized_score(self, steps, problem):
        pdir, pname = split(problem)
        no_restart_steps, restart_steps = self.metadata[pdir][pname]
        if self.compare_with_restarts:
            return restart_steps / steps
        else:
            return no_restart_steps / steps

    def get_dummy_state(self):
        DUMMY_V = np.zeros((2, vertex_in_size), dtype=np.float32)
        DUMMY_V[:, VAR_ID_IDX] = 1
        DUMMY_STATE = (
            DUMMY_V,
            np.zeros((2, edge_in_size), dtype=np.float32),
            np.eye(2, dtype=np.long),
            np.zeros((1, global_in_size), dtype=np.float32)
        )
        return (
            DUMMY_STATE[0],
            DUMMY_STATE[1],
            DUMMY_STATE[2],
            np.zeros((1, global_in_size), dtype=np.float32)
        )
