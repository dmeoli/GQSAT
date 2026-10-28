# arXiv submission of the note

The fields of the submission form, in its order. The abstract is that of `gatqsat.tex`
in plain text (the form does not take LaTeX commands); `make_arxiv.py` builds the
package to upload (`arxiv.tar.gz`) and stops while a placeholder of the campaign is
still in the text. Check the abstract again against the tex once the numbers are in.

- **Title**: Graph Attention in a Learned Branching Heuristic for SAT
- **Authors**: Donato Meoli
- **Primary category**: cs.LG (Machine Learning)
- **Cross-lists**: cs.AI (Artificial Intelligence), cs.LO (Logic in Computer Science)
- **License**: CC BY 4.0
- **Comments**: 18 pages, 8 figures, 4 tables. Code, instances and logs:
  https://github.com/dmeoli/GQSAT
- **ACM class**: I.2.6; I.2.8
- **MSC class, report number, journal reference, DOI**: none

**Abstract** (1616 characters of at most 1920):

We study the branching heuristic of a Conflict-Driven Clause Learning solver for the Boolean satisfiability problem when it is learned by reinforcement, as in Graph-Q-SAT, where a graph network reads the bipartite graph of the formula and is trained by Deep Q-Learning to reduce the number of decisions of the solver. We add to the core block of that network two edge-featured graph attention layers, which weight the neighbors of each node by learned coefficients before the global attribute is updated, and we compare the resulting model (GAT-Q-SAT) with Graph-Q-SAT at matched training set and configuration, together with two controls that separate the effect of the attention from that of the added parameters. On graph coloring GAT-Q-SAT takes fewer decisions than Graph-Q-SAT on all the families we test, and the difference tends to grow with the size of the instances, since Graph-Q-SAT degrades almost to the level of the reference solver on the largest ones. [One sentence on what the controls show.] Trained on uniform random 3-SAT and tested on graph coloring, GAT-Q-SAT is again the better of the two, and it also appears to be the better one on some structured families that neither model has seen in training. Yet, on satisfiable uniform random 3-SAT Graph-Q-SAT is clearly superior, and on instances of this size neither model is competitive with the solver in running time, each decision costing a forward pass of the network. This evidence rests on three training seeds per model on graph coloring (two per control), and on one run per training family elsewhere. Computational results are reported.
