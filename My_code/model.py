import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from dgl.nn.pytorch import SAGEConv
from dgl.nn.pytorch import GraphConv
import numpy as np
from layers import MultiHeadGATLayer, HAN_metapath_specific


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)  # 权重矩阵，获得元路径的重要性 [2, 1] .mean(0):每个meta_path上的均值（/|V|）;
        beta = torch.sigmoid(w)
        beta = beta.expand((z.shape[0],) + beta.shape)  # [x,2,1] 扩展到N个节点上的metapath的值
        return (beta * z).sum(1)


class MAHN(nn.Module):
    def __init__(self, G_D0, G_D1, G_ME0, G_ME1, hidden_dim, G, meta_paths_list, feature_attn_size, num_heads, num_diseases, num_metabolite, num_microbe,
                 d_sim_dim, me_sim_dim, mi_sim_dim, out_dim, dropout, slope, device):
        super(MAHN, self).__init__()
        self.device = device
        self.G_D0 = G_D0
        self.G_D1 = G_D1
        self.G_ME0 = G_ME0
        self.G_ME1 = G_ME1
        self.G = G
        self.meta_paths = meta_paths_list
        self.num_heads = num_heads
        self.num_diseases = num_diseases
        self.num_metabolite = num_metabolite
        self.num_microbe = num_microbe
        self.conv1 = GraphConv(d_sim_dim, hidden_dim)
        self.sageconv = SAGEConv(me_sim_dim, hidden_dim, 'mean')
        self.gat = MultiHeadGATLayer(G, feature_attn_size, num_heads, dropout, slope, device, merge='cat')
        self.heads = nn.ModuleList()
        self.metapath_layers = nn.ModuleList()
        for i in range(self.num_heads):
            self.metapath_layers.append(HAN_metapath_specific(G, feature_attn_size, out_dim, dropout, slope,device))
        self.dropout = nn.Dropout(dropout)
        self.me_fc = nn.Linear(feature_attn_size * num_heads + me_sim_dim, out_dim*2)
        self.d_fc = nn.Linear(feature_attn_size * num_heads + d_sim_dim, out_dim*2)
        self.semantic_attention = SemanticAttention(in_size=out_dim * num_heads)
        self.h_fc = nn.Linear(out_dim*4, out_dim)
        self.predict = nn.Linear(out_dim * 2, 1)
        self.BilinearDecoder = BilinearDecoder(feature_size=64)

    def forward(self, G_D0, G_D1, G_ME0, G_ME1, G, G0, diseases, metabolite):
        G_D0 = G_D0.to(self.device)
        G_D0.ndata['d_sim'] = G_D0.ndata['d_sim'].to(self.device)
        h_D0 = self.conv1(G_D0, G_D0.ndata['d_sim']).to(self.device)
        h_D0 = self.dropout(F.elu(h_D0)).to(self.device)

        G_D1 = G_D1.to(self.device)
        G_D1.ndata['d_sim'] = G_D1.ndata['d_sim'].to(self.device)
        h_D1 = self.conv1(G_D1, G_D1.ndata['d_sim']).to(self.device)
        h_D1 = self.dropout(F.elu(h_D1)).to(self.device)

        G_ME0 = G_ME0.to(self.device)
        G_ME0.ndata['me_sim'] = G_ME0.ndata['me_sim'].to(self.device)
        h_ME0 = self.sageconv(G_ME0, G_ME0.ndata['me_sim']).to(self.device)
        h_ME0 = self.dropout(F.elu(h_ME0)).to(self.device)

        G_ME1 = G_ME1.to(self.device)
        G_ME1.ndata['me_sim'] = G_ME1.ndata['me_sim'].to(self.device)
        h_ME1 = self.sageconv(G_ME1, G_ME1.ndata['me_sim']).to(self.device)
        h_ME1 = self.dropout(F.elu(h_ME1)).to(self.device)

        h_D = torch.cat((h_D0, h_D1), dim=1).to(self.device)
        h_ME = torch.cat((h_ME0, h_ME1), dim=1).to(self.device)

        index1 = 0
        for meta_path in self.meta_paths:
            if meta_path == 'dme' or meta_path == 'med':
                if index1 == 0:
                    h_agg0 = self.gat(G).to(self.device)
                    index1 = 1
            elif meta_path == 'dmi':
                dmi_edges = G0.filter_edges(lambda edges: edges.data['dmi']).to(self.device)
                g_dmi = G0.edge_subgraph(dmi_edges, preserve_nodes=True)
                g_dmi = g_dmi.to(self.device)
                head_outs0 = [attn_head(g_dmi, meta_path) for attn_head in self.metapath_layers]
                h_agg1 = torch.cat(head_outs0, dim=1).to(self.device)
            elif meta_path == 'mime':
                mime_edges = G0.filter_edges(lambda edges: edges.data['mime']).to(self.device)
                g_mime = G0.edge_subgraph(mime_edges, preserve_nodes=True)
                g_mime=g_mime.to(self.device)
                head_outs1 = [attn_head(g_mime, meta_path) for attn_head in self.metapath_layers]
                h_agg2 = torch.cat(head_outs1, dim=1).to(self.device)

# 不同元路径疾病特征和不同元路径节点特征
        disease0 = h_agg0[:self.num_diseases].to(self.device)
        metabolite0 = h_agg0[self.num_diseases:self.num_diseases + self.num_metabolite].to(self.device)
        disease1 = h_agg1[:self.num_diseases].to(self.device)
        metabolite1 = h_agg2[self.num_diseases:self.num_diseases + self.num_metabolite].to(self.device)

        semantic_embeddings1 = torch.stack((disease0, disease1), dim=1).to(self.device)
        h1 = self.semantic_attention(semantic_embeddings1).to(self.device)
        semantic_embeddings2 = torch.stack((metabolite0, metabolite1), dim=1).to(self.device)
        h2 = self.semantic_attention(semantic_embeddings2).to(self.device)

        h_d = torch.cat((h1, self.G.ndata['d_sim'][:self.num_diseases]), dim=1).to(self.device)
        h_me = torch.cat((h2, self.G.ndata['me_sim'][self.num_diseases: self.num_diseases+self.num_metabolite]), dim=1).to(self.device)

        h_me = self.dropout(F.elu(self.me_fc(h_me))).to(self.device)
        h_d = self.dropout(F.elu(self.d_fc(h_d))).to(self.device)

        h_me_final = torch.cat((h_ME, h_me), dim=1).to(self.device)
        h_d_final = torch.cat((h_D, h_d), dim=1).to(self.device)

        h = torch.cat((h_d_final, h_me_final), dim=0).to(self.device)
        h = self.dropout(F.elu(self.h_fc(h))).to(self.device)

# 获取训练边或测试边的点的特征
        h_diseases = h[diseases].to(self.device)
        h_metabolite = h[metabolite].to(self.device)
# 解码
        predict_score = self.BilinearDecoder(h_diseases, h_metabolite).to(self.device)
        return predict_score

# 双线性解码器
class BilinearDecoder(nn.Module):
    def __init__(self, feature_size):
        super(BilinearDecoder, self).__init__()
        self.W = Parameter(torch.randn(feature_size, feature_size))

    def forward(self, h_diseases, h_metabolite):
        h_diseases0 = torch.mm(h_diseases, self.W)
        h_metabolite0 = torch.mul(h_diseases0, h_metabolite)
        h0 = h_metabolite0.sum(1)
        h = torch.sigmoid(h0)
        return h.unsqueeze(1)




