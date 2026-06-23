import inspect
import sys

import torch
import yaml
from gqsat.meta import ModifiedMetaLayer
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, LayerNorm
from torch_geometric.nn import Sequential, GATConv as EdgeGATConv
from torch_geometric.nn.conv.gatv2_conv import GATv2Conv
from torch_geometric.utils import scatter


class SATModel(torch.nn.Module):

    def __init__(self, save_name=None):
        super().__init__()
        if save_name is not None:
            self.save_to_yaml(save_name)

    @classmethod
    def save_to_yaml(cls, model_name):
        # -2 is here because I want to know how many layers below lies the final child and get its init params.
        # I do not need nn.Module and 'object'
        # this WILL NOT work with multiple inheritance of the leaf children
        frame, filename, line_number, function_name, lines, index = inspect.stack()[
            len(cls.mro()) - 2
            ]
        args, _, _, values = inspect.getargvalues(frame)

        save_dict = {
            "class_name": values["self"].__class__.__name__,
            "call_args": {
                k: values[k] for k in args if k != "self" and k != "save_name"
            },
        }
        with open(model_name, "w") as f:
            yaml.dump(save_dict, f, default_flow_style=False)

    @staticmethod
    def load_from_yaml(fname):
        with open(fname, "r") as f:
            res = yaml.load(f, Loader=yaml.Loader)
        return getattr(sys.modules[__name__], res["class_name"])(**res["call_args"])

    @staticmethod
    def reconcile_gat_lin_keys(model, state_dict):
        """Bridge the GATConv linear-layer naming across PyG versions.

        Older PyG stored the homogeneous GATConv linear as a shared
        ``lin_src``/``lin_dst`` pair (identical weights); newer PyG uses a single
        ``lin``. Remap the checkpoint keys in whichever direction the live model
        expects, so checkpoints load with no missing/unexpected keys regardless of
        the installed torch-geometric version. Lossless because lin_src == lin_dst.
        """
        model_keys = set(model.state_dict().keys())
        out = {}
        for k, v in state_dict.items():
            if k.endswith("lin_src.weight") and k not in model_keys:
                single = k.replace("lin_src.weight", "lin.weight")
                if single in model_keys:           # new PyG: single lin
                    out[single] = v
                    continue
            if k.endswith("lin_dst.weight") and k not in model_keys:
                if k.replace("lin_dst.weight", "lin.weight") in model_keys:
                    continue                        # redundant (== lin_src), drop
            if k.endswith("lin.weight") and k not in model_keys:
                src = k.replace("lin.weight", "lin_src.weight")
                dst = k.replace("lin.weight", "lin_dst.weight")
                if src in model_keys and dst in model_keys:   # old PyG: split lin
                    out[src] = v
                    out[dst] = v
                    continue
            out[k] = v
        return out


def get_mlp(
        in_size,
        out_size,
        n_hidden,
        hidden_size,
        activation=ReLU,
        activate_last=True,
        layer_norm=True
):
    arch = []
    l_in = in_size
    for l_idx in range(n_hidden):
        arch.append(Lin(l_in, hidden_size))
        arch.append(activation())
        l_in = hidden_size

    arch.append(Lin(l_in, out_size))

    if activate_last:
        arch.append(activation())

        if layer_norm:
            arch.append(LayerNorm(out_size))

    return Seq(*arch)


class GraphTransformerAttn(torch.nn.Module):
    """NeuroBack-inspired graph-transformer attention block (the GAT-Q-SAT successor).

    Replaces GAT-Q-SAT's plain two-layer edge-aware ``GATConv`` with a single
    pre-norm Transformer-style block built on ``GATv2Conv`` --- i.e. *dynamic*
    edge-featured attention (Brody et al., 2022) instead of the static GATConv
    scoring --- followed by a parallel feed-forward branch and a residual
    connection, mirroring the GSA / GTBlock design of NeuroBack (Wang et al.,
    ICLR 2024). Operates on the bipartite variable/clause graph (no self-loops).
    """

    def __init__(self, dim, edge_dim, heads=3, dropout=0.0, activation=ReLU,
                 layer_norm=True):
        super().__init__()
        per_head = max(dim // heads, 1)
        self.norm = LayerNorm(dim)
        self.gat = GATv2Conv(
            dim, per_head, heads=heads, concat=True, edge_dim=edge_dim,
            # no self-connections since the SAT representation is a bipartite graph
            add_self_loops=False, dropout=dropout, share_weights=False
        )
        gat_out = per_head * heads
        self.proj = Lin(gat_out, dim) if gat_out != dim else None
        self.ffn = Seq(Lin(dim, dim), activation(), Lin(dim, dim))
        self.out_norm = LayerNorm(dim) if layer_norm else None

    def forward(self, x, edge_index, edge_attr):
        h = self.norm(x)                                  # pre-norm
        a = self.gat(h, edge_index, edge_attr)            # dynamic edge-aware attention
        if self.proj is not None:
            a = self.proj(a)
        x = x + a + self.ffn(h)                           # residual + parallel FFN (GTBlock)
        return self.out_norm(x) if self.out_norm is not None else x


class IndependentGraphNet(SATModel):
    """
    Independent Graph Network block.
    A graph block that applies models to the graph elements *independently*.
    The inputs and outputs are graphs. The corresponding models are applied to
    each element of the graph (edges, nodes and globals) in parallel and
    independently of the other elements.
    It can be used to encode or decode the elements of a graph.
    """

    def __init__(
            self,
            in_dims,
            out_dims,
            save_name=None,
            n_hidden=1,
            hidden_size=64,
            activation=ReLU,
            layer_norm=True
    ):
        super().__init__(save_name)

        v_in = in_dims[0]  # in_node_features
        e_in = in_dims[1]  # in_edge_features
        u_in = in_dims[2]  # in_global_features

        v_out = out_dims[0]  # out_node_features
        e_out = out_dims[1]  # out_edge_features
        u_out = out_dims[2]  # out_global_features

        class EdgeModel(torch.nn.Module):

            def __init__(self):
                super(EdgeModel, self).__init__()
                self.edge_mlp = get_mlp(
                    e_in,  # in_edge_features
                    e_out,  # out_edge_features
                    n_hidden,
                    hidden_size,
                    activation=activation,
                    layer_norm=layer_norm
                )

            def forward(self, src, target, edge_attr, u=None, e_indices=None):
                # src, target: [E, F_x], where E is the number of edges.
                # edge_attr: [E, F_e], where E is the number of edges and F_e is the number of edge features.
                # u: [B, F_u], where B is the number of graphs and F_u is the number of global features.
                # e_indices: [E] with max entry B - 1.
                return self.edge_mlp(edge_attr)

        class NodeModel(torch.nn.Module):

            def __init__(self):
                super(NodeModel, self).__init__()
                self.node_mlp = get_mlp(
                    v_in,  # in_node_features
                    v_out,  # out_node_features
                    n_hidden,
                    hidden_size,
                    activation=activation,
                    layer_norm=layer_norm
                )

            def forward(self, x, edge_index, edge_attr, u=None, v_indices=None):
                # x: node feature matrix of shape [N, F_x], where N is the number of nodes
                #    and F_x is the number of node features.
                # edge_index: graph connectivity matrix of shape [2, E] with max entry N - 1.
                # edge_attr: [E, F_e], where E is the number of edges and F_e is the number of edge features.
                # u: [B, F_u], where B is the number of graphs and F_u is the number of global features.
                # v_indices: [N] with max entry B - 1.
                return self.node_mlp(x)

        class GlobalModel(torch.nn.Module):

            def __init__(self):
                super(GlobalModel, self).__init__()
                self.global_mlp = get_mlp(
                    u_in,  # in_global_features
                    u_out,  # out_global_features
                    n_hidden,
                    hidden_size,
                    activation=activation,
                    layer_norm=layer_norm
                )

            def forward(self, x, edge_attr, u, v_indices, e_indices):
                # x: node feature matrix of shape [N, F_x], where N is the number of nodes
                #    and F_x is the number of node features.
                # edge_attr: [E, F_e], where E is the number of edges and F_e is the number of edge features.
                # u: [B, F_u], where B is the number of graphs and F_u is the number of global features.
                # v_indices: [N] with max entry B - 1.
                # e_indices: [E] with max entry B - 1.
                return self.global_mlp(u)

        self.op = ModifiedMetaLayer(EdgeModel(), NodeModel(), GlobalModel())

    def forward(self, x, edge_index, edge_attr=None, u=None, v_indices=None, e_indices=None):
        return self.op(x, edge_index, edge_attr, u, v_indices, e_indices)


class GraphNet(SATModel):
    """
    Full Graph Network block.
    First apply edge block, then node block, and finally, the global block.
    """

    def __init__(
            self,
            in_dims,
            out_dims,
            save_name=None,
            e2v_agg='sum',
            n_hidden=1,
            hidden_size=64,
            activation=ReLU,
            layer_norm=True,
            use_attention=False,
            heads=3,
            attention_type='gat'
    ):
        super().__init__(save_name)
        if e2v_agg not in ['sum', 'mean']:
            raise ValueError('unknown aggregation function {}'.format(e2v_agg))

        v_in = in_dims[0]  # in_node_features
        e_in = in_dims[1]  # in_edge_features
        u_in = in_dims[2]  # in_global_features

        v_out = out_dims[0]  # out_node_features
        e_out = out_dims[1]  # out_edge_features
        u_out = out_dims[2]  # out_global_features

        class EdgeModel(torch.nn.Module):

            def __init__(self):
                super(EdgeModel, self).__init__()
                self.edge_mlp = get_mlp(
                    e_in + 2 * v_in + u_in,  # in_edge_features + 2 * in_node_features + in_global_features
                    e_out,
                    n_hidden,
                    hidden_size,
                    activation=activation,
                    layer_norm=layer_norm
                )

            def forward(self, src, target, edge_attr, u=None, e_indices=None):
                # src, target: [E, F_x], where E is the number of edges.
                # edge_attr: [E, F_e], where E is the number of edges and F_e is the number of edge features.
                # u: [B, F_u], where B is the number of graphs and F_u is the number of global features.
                # e_indices: [E] with max entry B - 1.
                out = torch.cat([src, target, edge_attr, u[e_indices]], dim=1)
                return self.edge_mlp(out)

        class NodeModel(torch.nn.Module):

            def __init__(self):
                super(NodeModel, self).__init__()
                self.node_mlp = get_mlp(
                    v_in + e_out + u_in,  # in_node_features + out_edge_features + in_global_features
                    v_out,  # out_node_features
                    n_hidden,
                    hidden_size,
                    activation=activation,
                    layer_norm=layer_norm
                )

            def forward(self, x, edge_index, edge_attr, u=None, v_indices=None):
                # x: node feature matrix of shape [N, F_x], where N is the number of nodes
                #    and F_x is the number of node features.
                # edge_index: graph connectivity matrix of shape [2, E] with max entry N - 1.
                # edge_attr: [E, F_e], where E is the number of edges and F_e is the number of edge features.
                # u: [B, F_u], where B is the number of graphs and F_u is the number of global features.
                # v_indices: [N] with max entry B - 1.
                row, col = edge_index  # source and target nodes in edge index
                if e2v_agg == 'sum':
                    # global_add_pool(edge_attr, row, size=x.size(0))
                    out = scatter(edge_attr, row, dim=0, dim_size=x.size(0), reduce='sum')
                elif e2v_agg == 'mean':
                    # global_mean_pool(edge_attr, row, size=x.size(0))
                    out = scatter(edge_attr, row, dim=0, dim_size=x.size(0), reduce='mean')
                out = torch.cat([x, out, u[v_indices]], dim=1)
                return self.node_mlp(out)

        class GlobalModel(torch.nn.Module):

            def __init__(self):
                super(GlobalModel, self).__init__()
                self.global_mlp = get_mlp(
                    u_in + v_out + e_out,  # in_global_features + out_node_features + out_edge_features
                    u_out,  # out_global_features
                    n_hidden,
                    hidden_size,
                    activation=activation,
                    layer_norm=layer_norm
                )

            def forward(self, x, edge_attr, u, v_indices, e_indices):
                # x: node feature matrix of shape [N, F_x], where N is the number of nodes
                #    and F_x is the number of node features.
                # edge_attr: [E, F_e], where E is the number of edges and F_e is the number of edge features.
                # u: [B, F_u], where B is the number of graphs and F_u is the number of global features.
                # v_indices: [N] with max entry B - 1.
                # e_indices: [E] with max entry B - 1.
                out = torch.cat([u,
                                 scatter(x, v_indices, dim=0, dim_size=u.size(0), reduce='mean'),
                                 scatter(edge_attr, e_indices, dim=0, dim_size=u.size(0), reduce='mean')], dim=1)
                return self.global_mlp(out)

        if use_attention and attention_type == 'graph_transformer':

            # GAT-Q-SAT successor: GATv2 + Transformer block (NeuroBack-inspired)
            self.attention_model = GraphTransformerAttn(
                v_out, e_out, heads=heads, activation=activation, layer_norm=layer_norm
            )

            self.op = ModifiedMetaLayer(EdgeModel(), NodeModel(), GlobalModel(),
                                        attention_model=self.attention_model)

        elif use_attention:

            self.attention_model = Sequential('mlp_x, edge_index, mlp_edge_attr', [
                (EdgeGATConv(
                    in_channels=v_out,  # out_node_features
                    out_channels=hidden_size, heads=heads, concat=True,  # heads * hidden_size
                    edge_dim=e_out,  # out_edge_features
                    # no self-connections since the SAT representation is a bipartite graph
                    add_self_loops=False,
                    aggr='add' if e2v_agg == 'sum' else e2v_agg),
                 'mlp_x, edge_index, mlp_edge_attr -> gat_x'),
                activation(inplace=True),
                (EdgeGATConv(
                    in_channels=heads * hidden_size,  # heads * hidden_size
                    out_channels=v_out,  # out_node_features
                    edge_dim=e_out,  # out_edge_features
                    # no self-connections since the SAT representation is a bipartite graph
                    add_self_loops=False,
                    aggr='add' if e2v_agg == 'sum' else e2v_agg),
                 'gat_x, edge_index, mlp_edge_attr -> gat_x'),
                activation(inplace=True),
                (LayerNorm(v_out) if layer_norm else lambda gat_x: gat_x)
            ])

            self.op = ModifiedMetaLayer(EdgeModel(), NodeModel(), GlobalModel(),
                                        attention_model=self.attention_model)

        else:

            self.op = ModifiedMetaLayer(EdgeModel(), NodeModel(), GlobalModel())

    def forward(self, x, edge_index, edge_attr=None, u=None, v_indices=None, e_indices=None):
        return self.op(x, edge_index, edge_attr, u, v_indices, e_indices)


class EncodeProcessDecode(SATModel):
    """
    Full Encode-Process-Decode model.
    - An "Encoder" graph net, which independently encodes the edge, node, and
      global attributes (does not compute relations, etc.).
    - A "Core" graph net, which performs N rounds of processing (message-passing)
      steps. The input to the Core is the concatenation of the Encoder's output
      and the previous output of the Core (labeled "Hidden(t)" below, where "t" is
      the processing step).
    - A "Decoder" graph net, which independently decodes the edge, node, and
      global attributes (does not compute relations, etc.).

                        Hidden(t)   Hidden(t+1)
                           |            ^
              *---------*  |  *------*  |  *---------*
              |         |  |  |      |  |  |         |
    Input --->| Encoder |  *->| Core |--*->| Decoder |---> Output(t)
              |         |---->|      |     |         |
              *---------*     *------*     *---------*

    The model is trained by supervised learning. Input graphs are procedurally
    generated, and output graphs have the same structure with the nodes and edges
    of the shortest path labeled (using 2-element 1-hot vectors).
    """

    def __init__(
            self,
            in_dims,
            core_out_dims,
            out_dims,
            core_steps=1,
            encoder_out_dims=None,
            dec_out_dims=None,
            save_name=None,
            e2v_agg='sum',
            n_hidden=1,
            hidden_size=64,
            activation=ReLU,
            independent_block_layers=1,
            layer_norm=True,
            use_attention=False,
            heads=3,
            attention_type='gat'
    ):
        super().__init__(save_name)
        # all dims are tuples with (v,e) feature sizes
        self.steps = core_steps
        # if dec_out_dims is None, there will not be a decoder
        self.in_dims = in_dims
        self.core_out_dims = core_out_dims
        self.dec_out_dims = dec_out_dims

        self.encoder = None
        if encoder_out_dims is not None:
            self.encoder = IndependentGraphNet(
                in_dims,
                encoder_out_dims,
                n_hidden=independent_block_layers,
                hidden_size=hidden_size,
                activation=activation,
                layer_norm=layer_norm
            )

        core_in_dims = in_dims if self.encoder is None else encoder_out_dims

        self.core = GraphNet(
            (
                core_in_dims[0] + core_out_dims[0],
                core_in_dims[1] + core_out_dims[1],
                core_in_dims[2] + core_out_dims[2]
            ),
            core_out_dims,
            e2v_agg=e2v_agg,
            n_hidden=n_hidden,
            hidden_size=hidden_size,
            activation=activation,
            layer_norm=layer_norm,
            use_attention=use_attention,
            heads=heads,
            attention_type=attention_type
        )

        self.decoder = None
        if dec_out_dims is not None:
            self.decoder = IndependentGraphNet(
                core_out_dims,
                dec_out_dims,
                n_hidden=independent_block_layers,
                hidden_size=hidden_size,
                activation=activation,
                layer_norm=layer_norm
            )

        pre_out_dims = core_out_dims if self.decoder is None else dec_out_dims

        self.vertex_out_transform = (
            Lin(pre_out_dims[0], out_dims[0]) if out_dims[0] is not None else None
        )
        self.edge_out_transform = (
            Lin(pre_out_dims[1], out_dims[1]) if out_dims[1] is not None else None
        )
        self.global_out_transform = (
            Lin(pre_out_dims[2], out_dims[2]) if out_dims[2] is not None else None
        )

    def get_init_state(self, n_v, n_e, n_u, device):
        return (
            torch.zeros((n_v, self.core_out_dims[0]), device=device),
            torch.zeros((n_e, self.core_out_dims[1]), device=device),
            torch.zeros((n_u, self.core_out_dims[2]), device=device)
        )

    def forward(self, x, edge_index, edge_attr, u, v_indices=None, e_indices=None):
        # if v_indices and e_indices are both None, then we have only one graph without a batch
        if v_indices is None and e_indices is None:
            v_indices = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
            e_indices = torch.zeros(
                edge_attr.shape[0], dtype=torch.long, device=edge_attr.device
            )

        if self.encoder is not None:
            x, edge_attr, u = self.encoder(
                x, edge_index, edge_attr, u, v_indices, e_indices
            )

        latent0 = (x, edge_attr, u)
        latent = self.get_init_state(
            x.shape[0], edge_attr.shape[0], u.shape[0], x.device
        )
        for st in range(self.steps):  # message passing steps
            latent = self.core(
                torch.cat([latent0[0], latent[0]], dim=1),
                edge_index,
                torch.cat([latent0[1], latent[1]], dim=1),
                torch.cat([latent0[2], latent[2]], dim=1),
                v_indices,
                e_indices
            )

        if self.decoder is not None:
            latent = self.decoder(
                latent[0], edge_index, latent[1], latent[2], v_indices, e_indices
            )

        v_out = (
            latent[0]
            if self.vertex_out_transform is None
            else self.vertex_out_transform(latent[0])
        )
        e_out = (
            latent[1]
            if self.edge_out_transform is None
            else self.edge_out_transform(latent[1])
        )
        u_out = (
            latent[2]
            if self.global_out_transform is None
            else self.global_out_transform(latent[2])
        )
        return v_out, e_out, u_out
