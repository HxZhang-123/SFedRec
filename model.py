import torch
from torch import nn
import dgl

from longterm import HKG
from Session import session_model


class SFedRec(HKG):
    def __init__(self, item_embedding, embedding_dim, knowledge_graph, num_items, num_layers, relu=False, batch_norm=True, feat_drop=0.0):
        super().__init__(item_embedding, embedding_dim, knowledge_graph, num_layers, batch_norm=batch_norm, feat_drop=feat_drop)
        self.num_items = num_items
        self.item_embedding = item_embedding
        self.fc_i = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.fc_u = nn.Linear(embedding_dim, embedding_dim, bias=False)
        in_size = 3 * embedding_dim
        self.fc_score = nn.Linear(in_size, embedding_dim, bias=False)
        self.Session_layer = session_model(embedding_dim, num_steps=1, batch_norm=batch_norm, feat_drop=feat_drop, relu=relu)

    def forward(self, g, new_seq_id, extra_inputs=None, feats=None):
        item_indices = nn.Parameter(torch.arange(self.num_items, dtype=torch.long), requires_grad=False)
        all_ieb = self.item_embedding(item_indices)
        HKGE = super().forward(extra_inputs, feats)
        new_seq_id_tensor = torch.tensor(new_seq_id).long()
        item_feat = self.item_embedding(new_seq_id_tensor)
        U_feat = HKGE['user'][0]
        h_i = self.fc_i(item_feat) + dgl.broadcast_nodes(g, self.fc_u(U_feat))
        HKGU = U_feat.unsqueeze(0)
        feat_ii = self.Session_layer(g, h_i, HKGU)
        score = torch.cat([feat_ii, HKGU], dim=1)
        s_i = self.fc_score(score)
        logits = s_i @ all_ieb.t()
        return logits

