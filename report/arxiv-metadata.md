# arXiv submission of the note

The fields of the submission form, in its order. The abstract is that of `gatqsat.tex`
in plain text (the form does not take LaTeX commands); `make_arxiv.py` builds the
package to upload (`arxiv.tar.gz`).

- **Title**: Graph Attention in a Learned Branching Heuristic for SAT
- **Authors**: Antonio Frangioni, Donato Meoli
- **Primary category**: cs.LG (Machine Learning)
- **Cross-lists**: cs.AI (Artificial Intelligence), cs.LO (Logic in Computer Science)
- **License**: CC BY 4.0
- **Comments**: 20 pages, 8 figures, 4 tables. Code, instances and logs:
  https://github.com/dmeoli/GATQSAT
- **ACM class**: I.2.6; I.2.8
- **MSC class, report number, journal reference, DOI**: none

**Abstract** (1776 characters of at most 1920):

We study the branching heuristic of a Conflict-Driven Clause Learning solver for the Boolean satisfiability problem when it is learned by reinforcement, as in Graph-Q-SAT, where a graph network reads the bipartite graph of the formula and is trained by Deep Q-Learning to reduce the number of decisions of the solver. We add to the core block of that network two edge-featured graph attention layers, which weight the neighbors of each node by learned coefficients before the global attribute is updated, and we compare the resulting model (GAT-Q-SAT) with Graph-Q-SAT at matched training set and configuration, with two controls that separate the effect of the attention from that of the added parameters. On graph coloring GAT-Q-SAT takes fewer decisions than Graph-Q-SAT on all the families we test but the smallest, where the two are level, and the difference tends to grow with the size of the instances, since Graph-Q-SAT approaches the reference solver on the largest ones. A wider Graph-Q-SAT does not close the gap, attention in place of the sum of the messages is worse than Graph-Q-SAT throughout, and attention in the pooling of the global attribute does not improve GAT-Q-SAT. Trained on uniform random 3-SAT and tested on graph coloring, GAT-Q-SAT is again the better of the two, and it appears to be the better one on some structured families not seen in training. Yet, on satisfiable uniform random 3-SAT Graph-Q-SAT is clearly superior, and on instances of this size neither model is competitive with the solver in running time, each decision costing a forward pass of the network. This evidence rests on five training seeds per model on graph coloring (two or three per control), and one run per training family elsewhere. Computational results are reported.
