# GQSAT — Graph-Q-SAT · GAT-Q-SAT · GTv2-Q-SAT

> Learning a SAT **branching heuristic** with value-based RL (DQN) over a graph
> neural network on the bipartite CNF graph. Part of the
> [NeuroSAT](https://github.com/dmeoli/NeuroSAT) project.

A modernised PyTorch / PyTorch-Geometric fork of Kurin et al.'s
[Graph-Q-SAT](https://github.com/NVIDIA/GraphQSat), extended with two
attention-based successors and a clean attention-on/off study.

## What's inside

| Path | Description |
|---|---|
| `gqsat/models.py` | the GNN: encode–process–decode `GraphNet` + the three attention variants |
| `gqsat/` | learners, agents, replay buffer, env wrappers, CLI (`utils.py`) |
| `minisat/` | patched MiniSat 2.2 + gym env (SWIG `_GymSolver.so`) — sub-submodule |
| `dqn.py` | DQN training driver |
| `evaluate.py`, `evaluate.sh` | evaluation → per-instance `runs/*.tsv` (MRIR, sec-to-solve) |
| `add_metadata.py` | precompute MiniSat baseline metadata for a dataset |
| `aggregate_results.py` | `runs/*.tsv` → `results/*.md` + `summary.csv` |
| `make_plots.py`, `paper_analysis.py` | regenerate the figures (violet theme) from the logs |
| `runs/`, `results/` | trained checkpoints + evaluation logs |

CNF datasets live **outside** this submodule, in the parent repo's shared
[`../data`](../data) hub (DIMACS `.cnf`).

## Baseline (forked) → our work

**Baseline** — Kurin, Godil, Whiteson & Catanzaro, *Can Q-Learning with Graph
Networks Learn a Generalizable Branching Heuristic for a SAT Solver?* (NeurIPS
2020): a DQN agent whose Q-function is an encode–process–decode graph net over
the variable/clause bipartite graph; metric = **MRIR** (MiniSat iterations /
model iterations; `>1` beats MiniSat).

**Port & fixes (semantics preserved):**
- TensorFlow-era stack → **latest `torch` + `torch-geometric`**, dropping
  `torch-scatter`/`torch-sparse` (now `torch_geometric.utils.scatter`),
  `numpy>=2`, `gymnasium`.
- **Exact reproduction** of the published MRIR from the original checkpoints
  (semantic non-regression oracle).
- Version-agnostic checkpoint loading (`SATModel.reconcile_gat_lin_keys` bridges
  the GATConv `lin_src`/`lin_dst` ↔ `lin` rename across PyG versions).
- Robustness fixes: gym env construction, `int32` decision-cap clamp
  (`OverflowError`), `SummaryWriter` honouring `--logdir` (Drive resume),
  GSL-free / warning-free C++ build.

**Evolutions (our contributions) — a model lineage:**
1. **Graph-Q-SAT** — the reproduced baseline (`--use_attention` off).
2. **GAT-Q-SAT** — inject edge-aware, multi-head **`GATConv`** attention into the
   core block (`--use_attention`). Helps on *structured* problems (graph
   colouring), in-distribution and under transfer, with the advantage **growing
   with problem size**; no help on uniform-random SAT.
3. **GTv2-Q-SAT** — a NeuroBack-inspired successor (`--use_attention
   --attention_type graph_transformer`): a pre-norm **Transformer block on
   `GATv2Conv`** (dynamic edge-featured attention) + parallel FFN + residual,
   replacing the static two-layer GATConv. See
   [`../papers`](../papers) (Wang et al., *NeuroBack*, ICLR 2024).

## Usage

```sh
# build the native env (only if the C++/SWIG changed)
cd minisat && make python-wrap && cd ..

# train (pick the variant via the attention flags)
python dqn.py --train-problems-paths ../data/graph-coloring/train ...           # Graph-Q-SAT
python dqn.py ... --use_attention                                               # GAT-Q-SAT
python dqn.py ... --use_attention --attention_type graph_transformer            # GTv2-Q-SAT

# evaluate a checkpoint → runs/*.tsv  (add --no-cuda on CPU)
python evaluate.py --env-name sat-v0 --core-steps -1 --eps-final 0.0 \
    --no_restarts --test_time_max_decisions_allowed 500 \
    --eval-problems-paths ../data/graph-coloring/flat30-60 \
    --model-dir runs/<run> --model-checkpoint model_50000.chkp

# tables + figures from the logs
python aggregate_results.py && python make_plots.py && python paper_analysis.py
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
