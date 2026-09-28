"""Exports a policy of Graph-Q-SAT, trained on MiniSat (sat-v0) or on the
branch and bound of SMS++ (maxsat-v0), as a TorchScript module that the
GQSATBranchRule of SMS++ (SATBlock/include/GQSATBranchRule.h) reads.

The module is traced (torch.jit.script does not go through the classes of
the network), its forward( x , edge_index , edge_attr , u ) returning the two
Q-values of each vertex of one graph; the trace is checked against the
network on graphs of sizes other than the traced one, so that no size is
frozen in it. The integer attribute "features" of the module says which rows
of the vertices it reads (0 those of Graph-Q-SAT, 1 those of a weighted
MaxSAT node, see SATResidualGraph in SMS++), as given, or as the input
size in model.yaml says.

usage: export_torchscript.py <model.yaml> <checkpoint> <out.pt> [features]
Only torch and torch_geometric are needed, and only here: the module needs
libtorch alone.
"""
import random
import sys
from os.path import abspath, dirname

import torch
import yaml

sys.path.insert(0, dirname(dirname(abspath(__file__))))
from gqsat.models import SATModel  # noqa: E402


class Policy(torch.nn.Module):
    """the Q-values of the vertices (2 per vertex) of one graph"""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x, edge_index, edge_attr, u):
        v = torch.zeros(x.shape[0], dtype=torch.long)
        e = torch.zeros(edge_attr.shape[0], dtype=torch.long)
        vout, _, _ = self.net(x, edge_index, edge_attr, u, v, e)
        return vout


def graph(nv, nc, k, ncol, seed):
    """a random residual formula as its graph: variables then clauses, two
    edges per literal, polarity [0,1] if positive and [1,0] if not, the
    other columns of the rows at random"""
    rnd = random.Random(seed)
    src, dst, ea = [], [], []
    for c in range(nc):
        for v in rnd.sample(range(nv), min(k, nv)):
            pos = rnd.random() < 0.5
            for a, b in ((v, nv + c), (nv + c, v)):
                src.append(a)
                dst.append(b)
                ea.append([0.0, 1.0] if pos else [1.0, 0.0])
    x = torch.zeros(nv + nc, ncol)
    x[:nv, 0] = 1
    x[nv:, 1] = 1
    if ncol > 2:
        x[:, 2:] = torch.rand(nv + nc, ncol - 2,
                              generator=torch.Generator().manual_seed(seed))
    return (x, torch.tensor([src, dst], dtype=torch.long),
            torch.tensor(ea), torch.zeros(1, 1))


def main():
    with open(sys.argv[1]) as f:
        ncol = int(yaml.load(f, Loader=yaml.Loader)["call_args"]["in_dims"][0])
    features = int(sys.argv[4]) if len(sys.argv) > 4 else (1 if ncol == 7
                                                          else 0)
    if ncol != (7 if features == 1 else 2):
        sys.exit(f"features {features} do not go with {ncol} input columns")
    net = SATModel.load_from_yaml(sys.argv[1])
    sd = torch.load(sys.argv[2], map_location="cpu")
    net.load_state_dict(SATModel.reconcile_gat_lin_keys(net, sd))
    net.eval()
    pol = Policy(net).eval()
    with torch.no_grad():
        traced = torch.jit.trace(pol, graph(20, 60, 3, ncol, 0),
                                 check_trace=False)
        worst = 0.0
        for nv, nc, k, s in [(5, 9, 3, 1), (50, 210, 3, 2), (120, 500, 4, 3),
                             (1, 1, 1, 4), (300, 40, 7, 5)]:
            g = graph(nv, nc, k, ncol, s)
            a, b = pol(*g), traced(*g)
            assert a.shape == b.shape, (a.shape, b.shape)
            worst = max(worst, (a - b).abs().max().item())
        assert worst == 0.0, f"the trace differs by {worst}"
        traced._c._register_attribute("features", torch._C.IntType.get(),
                                      features)
        traced.save(sys.argv[3])
        back = torch.jit.load(sys.argv[3])
        g = graph(77, 300, 3, ncol, 9)
        assert (back(*g) - pol(*g)).abs().max().item() == 0.0
        assert back.features == features
    print(f"{sys.argv[3]}: features {features}, {ncol} columns, the trace "
          "equal to the network on 6 graphs")


if __name__ == "__main__":
    main()
