"""Hyperbolic layers."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch_geometric.utils import add_remaining_self_loops, remove_self_loops, softmax, add_self_loops
from torch_scatter import scatter, scatter_add
from torch_geometric.nn.conv import MessagePassing, GATConv
from torch.nn.parameter import Parameter
from torch_geometric.nn.inits import glorot, zeros
from manifolds.poincare import PoincareBall
import itertools

class HGATConv(nn.Module):
    """
    Hyperbolic graph convolution layer.。
    """
    def __init__(self, manifold, in_features, out_features, c_in, c_out, act=F.leaky_relu,
                 dropout=0.6, att_dropout=0.6, use_bias=True, heads=2, concat=False):
        super(HGATConv, self).__init__()
        out_features = out_features * heads
        self.linear = HypLinear(manifold, in_features, out_features, c_in, dropout=dropout, use_bias=use_bias)
        self.agg = HypAttAgg(manifold, c_in, out_features, att_dropout, heads=heads, concat=concat)
        self.hyp_act = HypAct(manifold, c_in, c_out, act)
        self.manifold = manifold
        self.c_in = c_in

    def forward(self, x, edge_index):
        h = self.linear.forward(x)
        h = self.agg.forward(h, edge_index)
        h = self.hyp_act.forward(h)
        return h


class HGCNConv1(nn.Module):
    """
    Hyperbolic graph convolution layer, from hgcn。
    """
    def __init__(self, manifold, in_features, out_features, dropout=0.6, act=F.leaky_relu,
                 use_bias=True):
        super(HGCNConv1, self).__init__()
        self.linear = HypLinear(manifold, in_features, out_features, dropout=dropout)
        self.agg = HypAgg(manifold, out_features, bias=use_bias,dropout=dropout)
        self.hyp_act = HypAct(manifold, act)
        self.manifold = manifold
        # self.c_in = c_in

    def forward(self, x, adj,c):
        h = self.linear.forward(x,c)
        h = self.agg.forward(h, adj,c)
        h = self.hyp_act.forward(h,c)
        return h


class HypLinear(nn.Module):
    """
    Hyperbolic linear layer.
    """
    def __init__(self, manifold, in_features, out_features, dropout=0.3, use_bias=True):
        super(HypLinear, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        # self.c = c
        self.dropout = dropout
        self.use_bias = use_bias
        self.bias = nn.Parameter(torch.Tensor(out_features), requires_grad=True)
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

    def forward(self, x,c):
        #drop_weight = F.dropout(self.weight, p=self.dropout, training=self.training)
        drop_weight = self.manifold.proj_tan0(F.dropout(self.weight, p=self.dropout, training=self.training), c)
        drop_weight = self.manifold.expmap0(drop_weight, c)
        hyp_weight = self.manifold.proj(drop_weight, c)
        mv = self.manifold.mobius_matvec(hyp_weight, x, c)
        res = self.manifold.proj(mv, c)
        if self.use_bias:
            bias = self.manifold.proj_tan0(self.bias.view(1, -1), c)
            hyp_bias = self.manifold.expmap0(bias, c)
            hyp_bias = self.manifold.proj(hyp_bias, c)
            res = self.manifold.mobius_add(res, hyp_bias, c)
            res = self.manifold.proj(res, c)
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, c={}'.format(
            self.in_features, self.out_features, self.c)

class HypAgg(MessagePassing):
    """
    Hyperbolic aggregation layer using degree.
    """
    def __init__(self, manifold,  out_features, bias=True,dropout=0.3):
        super(HypAgg, self).__init__()
        self.manifold = manifold
        self.manifold = PoincareBall()
        self.dropout = dropout
        self.use_bias = bias
        self.weight = Parameter(torch.FloatTensor(out_features, out_features), requires_grad=True)
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        zeros(self.bias)
        self.mlp = nn.Sequential(nn.Linear(out_features * 2, 1))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, adj,c):
        # self.weight = F.dropout(self.weight, p=self.dropout, training=self.training)
        #x_tangent = self.manifold.logmap0(x, c)
        drop_weight = self.manifold.proj_tan0(
            F.dropout(self.weight, p=self.dropout, training=self.training), c)
        drop_weight = self.manifold.expmap0(drop_weight, c)
        hypagg_weight = self.manifold.proj(drop_weight, c)
        support = self.manifold.mobius_matvec(hypagg_weight, x, c)
        output = self.manifold.mobius_matvec(support,adj,c)
        if self.bias is not None:
            output= self.manifold.mobius_add(output,self.bias,c)
            output = self.manifold.proj(output, c)
        else:
            output = self.manifold.proj(support, c)
        return output

    def extra_repr(self):
        return 'c={}'.format(self.c)

class HypAct(Module):
    """
    Hyperbolic activation layer.
    """
    def __init__(self, manifold, act):
        super(HypAct, self).__init__()
        self.manifold = manifold
        # self.c_in = c_in
        # self.c_out = c_out
        self.act = act

    def forward(self, x,c):
        xt = self.act(self.manifold.logmap0(x, c))
        xt = self.manifold.proj_tan0(xt, c)
        return self.manifold.proj(self.manifold.expmap0(xt, c), 1-c)

    def extra_repr(self):
        return 'c_in={}, c_out={}'.format(
            self.c_in, self.c_out)


class HypAttAgg(MessagePassing):
    def __init__(self, manifold, c, out_features, att_dropout=0.6, heads=1, concat=False):
        super(HypAttAgg, self).__init__()
        self.manifold = manifold
        self.dropout = att_dropout
        self.out_channels = out_features // heads
        self.negative_slope = 0.2
        self.heads = heads
        self.c = c
        self.concat = concat
        self.att_i = Parameter(torch.Tensor(1, heads, self.out_channels), requires_grad=True)
        self.att_j = Parameter(torch.Tensor(1, heads, self.out_channels), requires_grad=True)
        glorot(self.att_i)
        glorot(self.att_j)

    def forward(self, x, edge_index):
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index,num_nodes=x.size(self.node_dim))
        edge_index_i = edge_index[0]
        edge_index_j = edge_index[1]
        x_tangent0 = self.manifold.logmap0(x, c=self.c)  # project to origin
        x_i = torch.nn.functional.embedding(edge_index_i, x_tangent0)
        x_j = torch.nn.functional.embedding(edge_index_j, x_tangent0)
        x_i = x_i.view(-1, self.heads, self.out_channels)
        x_j = x_j.view(-1, self.heads, self.out_channels)
        alpha = (x_i * self.att_i).sum(-1) + (x_j * self.att_j).sum(-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, num_nodes=x_i.size(0))
        alpha = F.dropout(alpha, self.dropout, training=self.training)
        support_t = scatter(x_j * alpha.view(-1, self.heads, 1), edge_index_i, dim=0)
        if self.concat:
            support_t = support_t.view(-1, self.heads * self.out_channels)
        else:
            support_t = support_t.mean(dim=1)
        support_t = self.manifold.proj(self.manifold.expmap0(support_t, c=self.c), c=self.c)
        return support_t

