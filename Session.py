import torch
from torch import nn

import dgl
import dgl.ops as F


class update_gate(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.f_in = nn.Linear(in_size, 2 * out_size, bias=True)
        self.f_out = nn.Linear(out_size, 2 * out_size, bias=False)

    def forward(self, h_i, hidden_size):
        f_i, f_n = self.f_in(h_i).chunk(2, 1)
        b_z, b_h = self.f_out(hidden_size).chunk(2, 1)
        input_gate = torch.sigmoid(f_i + b_z)
        new_gate = torch.tanh(f_n + b_h)
        return new_gate + input_gate * (hidden_size - new_gate)


class GRU(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_steps=1,
        batch_norm=True,
        feat_drop=0.0,
        activation=None,
    ):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(input_dim) if batch_norm else None
        self.feat_drop = nn.Dropout(feat_drop) if feat_drop > 0 else None
        self.fc_i2h = nn.Linear(input_dim, hidden_dim, bias=False) if input_dim != hidden_dim else None
        self.fc_in = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.fc_out = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.gate_updated = update_gate(2 * hidden_dim, hidden_dim)
        self.fc_h2o = nn.Linear(
            hidden_dim, output_dim, bias=False
        ) if hidden_dim != output_dim else None
        self.hidden_dim = hidden_dim
        self.num_steps = num_steps
        self.activation = activation

    def propagate(self, g, reverseg, feat):
        if g.number_of_edges() > 0:
            feat_in = self.fc_in(feat)
            feat_out = self.fc_out(feat)
            a_in = F.u_mul_e_sum(g, feat_in, g.edata['iw'])
            a_out = F.u_mul_e_sum(reverseg, feat_out, reverseg.edata['ow'])
            a = torch.cat((a_in, a_out), dim=1)
        else:
            num_nodes = g.number_of_nodes()
            a = feat.new_zeros((num_nodes, 2 * self.hidden_dim))
        hn = self.gate_updated(a, feat)
        return hn

    def forward(self, g, reverseg, feat):
        if self.feat_drop is not None:
            feat = self.feat_drop(feat)
        if self.fc_i2h is not None:
            feat = self.fc_i2h(feat)
        for _ in range(self.num_steps):
            feat = self.propagate(g, reverseg, feat)
        if self.fc_h2o is not None:
            feat = self.fc_h2o(feat)
        if self.activation is not None:
            feat = self.activation(feat)
        return feat


class PGAT(nn.Module):
    def __init__(self, embedding_dim, batch_norm=False, feat_drop=0.0, activation=None):
        super().__init__()
        if batch_norm:
            self.batch_norm = nn.ModuleDict({
                'user': nn.BatchNorm1d(embedding_dim),
                'item': nn.BatchNorm1d(embedding_dim)
            })
        else:
            self.batch_norm = None
        self.feat_drop = nn.Dropout(feat_drop) if feat_drop > 0 else None
        self.fc_user = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.fc_key = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.fc_last = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.fc_e = nn.Linear(embedding_dim, 1, bias=False)
        self.activation = activation

    def forward(self, g, feat_i, U_feat, last_nodes):
        if self.feat_drop is not None:
            feat_i = self.feat_drop(feat_i)
            U_feat = self.feat_drop(U_feat)
        feat_val = feat_i
        feat_key = self.fc_key(feat_i)
        U_feat = self.fc_user(U_feat)
        last_feat = self.fc_last(feat_i[last_nodes])
        feat_qry = dgl.broadcast_nodes(g, U_feat + last_feat)
        e = self.fc_e(torch.sigmoid(feat_qry + feat_key))
        e = e + g.ndata['cnt'].log().view_as(e)
        alpha = F.segment.segment_softmax(g.batch_num_nodes(), e)
        rst = F.segment.segment_reduce(g.batch_num_nodes(), alpha * feat_val, 'sum')
        if self.activation is not None:
            rst = self.activation(rst)
        return rst


class session_model(nn.Module):
    def __init__(self, embedding_dim, num_steps=1, batch_norm=True, feat_drop=0.0, relu=False):
        super().__init__()
        self.fc_i = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.fc_u = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.gru = GRU(embedding_dim, embedding_dim, embedding_dim, num_steps=num_steps, batch_norm=batch_norm, feat_drop=feat_drop, activation=nn.ReLU() if relu else nn.PReLU(embedding_dim))
        self.Attention = PGAT(embedding_dim, batch_norm=batch_norm, feat_drop=feat_drop, activation=nn.ReLU() if relu else nn.PReLU(embedding_dim))

    def forward(self, g, h_i, HKGU):
        reverseg = dgl.reverse(g, False, False)
        if g.number_of_edges() > 0:
            edge_weight = g.edata['w']
            in_deg = F.copy_e_sum(g, edge_weight)
            g.edata['iw'] = F.e_div_v(g, edge_weight, in_deg)
            out_deg = F.copy_e_sum(reverseg, edge_weight)
            reverseg.edata['ow'] = F.e_div_v(reverseg, edge_weight, out_deg)
        h_i = self.gru(g, reverseg, h_i)
        last_nodes = g.filter_nodes(lambda nodes: nodes.data['last'] == 1)
        e_last = h_i[last_nodes] #last term
        Per_session = self.Attention(g, h_i, HKGU, last_nodes)
        score = torch.cat((e_last, Per_session), dim=1)
        return score
