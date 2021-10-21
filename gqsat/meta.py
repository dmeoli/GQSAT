### The code in this file was originally copied from the Pytorch Geometric library and modified later:
### https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/meta.html#MetaLayer
### Pytorch geometric license is below

# Copyright (c) 2019 Matthias Fey <matthias.fey@tu-dortmund.de>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from torch_geometric.nn import MetaLayer


class ModifiedMetaLayer(MetaLayer):

    def __init__(self, edge_model=None, node_model=None, global_model=None, attention_model=None):
        super().__init__(edge_model, node_model, global_model)
        self.attention_model = attention_model

    def forward(self, x, edge_index, edge_attr=None, u=None, v_indices=None, e_indices=None):
        row, col = edge_index

        if self.edge_model is not None:
            edge_attr = self.edge_model(x[row], x[col], edge_attr, u, e_indices)

        if self.node_model is not None:
            x = self.node_model(x, edge_index, edge_attr, u, v_indices)

        # apply `attention_model` using the updated (x, edge_attr) before passing through
        # `global_model` in order to extract some extra node/edge high level features
        if self.attention_model:
            x = self.attention_model(x, edge_index, edge_attr)

        if self.global_model is not None:
            u = self.global_model(x, edge_attr, u, v_indices, e_indices)

        return x, edge_attr, u
