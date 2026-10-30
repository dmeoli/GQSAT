# GQSAT: Graph-Q-SAT and GAT-Q-SAT

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dmeoli/GQSAT/blob/main/notebooks/gatqsat.ipynb)

A port of [Graph-Q-SAT](https://github.com/NVIDIA/GraphQSat) (Kurin et al.,
NeurIPS 2020) to the current releases of PyTorch and PyTorch Geometric, and
GAT-Q-SAT, which adds two edge-featured graph attention layers to the core block
of its network. Graph-Q-SAT learns the branching heuristic of MiniSat by Deep
Q-Learning on the bipartite graph of the formula, and its quality is measured by
the Median Relative Iteration Reduction (MRIR), i.e., the median over the
instances of the iterations of MiniSat divided by those of the model (above 1
the model takes fewer decisions than MiniSat). The note in [`report/`](report)
compares the two models at matched training set, together with two controls
that separate the effect of the attention from that of the added parameters.

## Contents

| Path | Description |
|---|---|
| `gqsat/models.py` | the encode-process-decode graph network, with the optional attention layers (`--use_attention`) and the attention aggregation (`--e2v-aggregator attention`) |
| `gqsat/` | learner, agents (also the restricted heuristics), replay buffer, command line (`utils.py`) |
| `minisat/` | the modified MiniSat 2.2 and its environment (SWIG `_GymSolver.so`) |
| `dqn.py`, `train.sh` | the training, and the entry point with the published configuration (`graphqsat`, `gatqsat`, `graphwide`, `attnagg`) |
| `evaluate.py`, `evaluate.sh`, `reeval.sh` | the evaluation, one line per instance with the time, the iterations of the model and those of MiniSat with and without restarts; `reeval.sh` runs the evaluations of the note by phase |
| `make_metadata.py`, `add_metadata.py` | the iterations of MiniSat on each instance of a family (`METADATA`) |
| `paper_numbers.py` | the tables of the note, from the logs |
| `make_plots.py`, `paper_analysis.py`, `transfer_study.py`, `figstyle.py` | the figures of the note, from the same logs; `transfer_study.py` also runs the evaluations on the SATLIB families |
| `data/` | the instances, from [SATLIB](https://www.cs.ubc.ca/~hoos/SATLIB/benchm.html), with the training, validation and test splits (`train_val_test_split.sh`, seed 123456) and the `METADATA` of each family |
| `runs/` | the checkpoints and the logs of the runs of 2021 |
| `report/` | the note (`pdflatex gatqsat.tex`) and its figures |
| `notebooks/gatqsat.ipynb` | every experiment of the note on Colab in one *Run all*, which resumes from what is already on Drive |
| `slides/` | the talk on the note, in English and Italian |

## The port

The original code is written for PyTorch 1.1, with pinned versions of PyTorch
Geometric and of the compiled extensions `torch-scatter` and `torch-sparse`,
which can no longer be installed together. In the port the aggregations come
from `torch_geometric.utils.scatter`, the environment is constructed without
`gym.make` (so that it runs with `gym` and with `gymnasium`), the SWIG wrapper
is rebuilt against `numpy` 2, and the checkpoints of 2021 load with the current
`GATConv` (`SATModel.reconcile_gat_lin_keys`, since the old version stored the
same linear map twice, as `lin_src` and `lin_dst`). The evaluation of the
checkpoints of 2021 on the new stack reproduces the logs of 2021 instance by
instance in 63 of 86 evaluations, and the median ratio changes by at most 0.03
in the others (on `flat30-60` at cap 500 the MRIR is 1.833 for Graph-Q-SAT and
2.000 for GAT-Q-SAT on both stacks). The runs of 2026 add a seed (`--seed`,
or `SEED=<n> bash train.sh ...`).

## Usage

```sh
pip install -r requirements.txt
cd minisat && make python-wrap && cd ..   # the environment of MiniSat

# train with the published configuration (variants: graphqsat, gatqsat, graphwide, attnagg)
bash train.sh graphqsat data/graph-coloring/train/flat50-115 data/graph-coloring/val/flat50-115 runs/graphqsat
SEED=1 bash train.sh gatqsat data/graph-coloring/train/flat50-115 data/graph-coloring/val/flat50-115 runs/gatqsat_s1

# evaluate a checkpoint → runs/*.tsv  (add --no-cuda on CPU)
python evaluate.py --env-name sat-v0 --core-steps -1 --eps-final 0.0 \
    --no_restarts --test_time_max_decisions_allowed 500 \
    --eval-problems-paths data/graph-coloring/flat30-60 \
    --model-dir runs/<run> --model-checkpoint model_50000.chkp

# restricted heuristics (Shirokikh et al. 2023): trade model decisions for time
#   --release_after T    the model takes the first T decisions, VSIDS the others
#   --action_pool_size L one forward pass gives the L best actions, executed in order
#   --warmstart_release  the Q-values of the root initialize the activities of
#                        VSIDS, which then runs unaided
python evaluate.py ... --release_after 5 --action_pool_size 4
python evaluate.py ... --warmstart_release

# re-evaluate the trained checkpoints, logging both MiniSat baselines per problem
bash reeval.sh tables     # cap 500 everywhere; also seeds, aa, colouring, transfer, random
python paper_numbers.py   # the note: with restarts on graph coloring, without on random 3-SAT
python paper_numbers.py --baseline no_restarts     # or with_restarts, best, stronger
python paper_numbers.py --latex                    # the rows of the four tables of the note

# tables + figures from the same logs (OUT_ROOT: where the logs are; PAPER_IMG_DIR
# and --out-dir to write the figures next to the note)
OUT_ROOT=<logs> python make_plots.py --out-dir report/img
OUT_ROOT=<logs> PAPER_IMG_DIR=report/img python paper_analysis.py
OUT_ROOT=<logs> CKPT_ROOT=<checkpoints> PAPER_IMG_DIR=report/img python transfer_study.py
```

### Weighted MaxSAT in the branch and bound of SMS++

`smspp/` is a second environment (`--env-name maxsat-v0`): the branch and bound
of [SMS++](https://gitlab.com/smspp/smspp-project) on weighted MaxSAT instances,
whose relaxation in each node is OLL within a budget of calls of the SAT solver.
A state is the graph of the residual formula of the node to branch on, with the
rows of Graph-Q-SAT or, with `--bnb-features 1`, also the weights, the costs, the
best solution and the cores; an action fixes a variable, and the reward is
`-penalty` per node. A policy learned there, or here on MiniSat, is exported as
a TorchScript module that the `GQSATBranchRule` of SMS++ reads (libtorch only).
SMS++ is installed from any of its channels (Ubuntu PPA, conda-forge, Homebrew,
vcpkg, Conan, Spack, Nix, AUR, MSYS2, xmake), as described in the
[installation guide](https://gitlab.com/smspp/smspp-project/-/wikis/Installing-SMS++).

```sh
# the environment, against an installed SATBlock of SMS++ and pybind11
cmake -S smspp -B smspp/build -DCMAKE_PREFIX_PATH=<SMS++ install> \
      -Dpybind11_DIR=$(python -m pybind11 --cmakedir) -DPYBIND11_FINDPYTHON=ON
cmake --build smspp/build

# the steps of the rule of the cores of SMS++, the reference of the score
python smspp/make_metadata.py <train>:<eval>
bash train_maxsat.sh gatqsat <train> <eval> runs/maxsat_gatqsat

# a checkpoint as a TorchScript module for SMS++
python smspp/export_torchscript.py runs/<run>/model.yaml runs/<run>/model_50000.chkp policy.pt
```

## Cite

```bibtex
@inproceedings{kurin2020graphqsat,
  title     = {Can Q-Learning with Graph Networks Learn a Generalizable Branching Heuristic for a SAT Solver?},
  author    = {Kurin, Vitaly and Godil, Saad and Whiteson, Shimon and Catanzaro, Bryan},
  booktitle = {Advances in Neural Information Processing Systems 33},
  year      = {2020}
}
```

## Acknowledgements & License

Built on [Graph-Q-SAT](https://github.com/NVIDIA/GraphQSat) (Kurin et al.),
[Fei Wang](https://github.com/feiwang3311/minisat)'s env, the
[MiniSat](https://github.com/niklasso/minisat) solver, and
[PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric)'s MetaLayer /
[Graph Nets](https://arxiv.org/abs/1806.01261). See [LICENSE](LICENSE).
