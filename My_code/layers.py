import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    def __init__(self, G, feature_attn_size, dropout, slope, device):
        super(GATLayer, self).__init__()
        self.device = device
        self.disease_nodes = G.filter_nodes(lambda nodes: nodes.data['type'] == 1)
        self.metabolite_nodes = G.filter_nodes(lambda nodes: nodes.data['type'] == 0)
        self.G = G
        self.slope = slope
        self.me_fc = nn.Linear(G.ndata['me_sim'].shape[1], feature_attn_size, bias=False)
        self.d_fc = nn.Linear(G.ndata['d_sim'].shape[1], feature_attn_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.me_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.d_fc.weight, gain=gain)

# 边的注意力系数
    def edge_attention(self, edges):
        '''通过逐元素相乘的方式计算边注意力系数，a是每条边的注意力系数'''
        a = torch.sum(edges.src['z'].mul(edges.dst['z']), dim=1).unsqueeze(1)
        return {'e': F.leaky_relu(a, negative_slope=self.slope)}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': F.elu(h)}

    def forward(self, G):
        self.G.apply_nodes(lambda nodes: {'z': self.dropout(self.d_fc(nodes.data['d_sim']))}, self.disease_nodes)
        self.G.apply_nodes(lambda nodes: {'z': self.dropout(self.me_fc(nodes.data['me_sim']))}, self.metabolite_nodes)
        self.G.apply_edges(self.edge_attention)
        self.G.update_all(self.message_func, self.reduce_func)
        return self.G.ndata.pop('h')


class MultiHeadGATLayer(nn.Module):  # 多头注意力
    def __init__(self, G, feature_attn_size, num_heads, dropout, slope,device, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.device = device
        self.G = G
        self.dropout = dropout
        self.slope = slope
        self.merge = merge
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(G, feature_attn_size, dropout, slope, device))

    def forward(self, G):
        head_outs = [attn_head(G) for attn_head in self.heads]
        if self.merge == 'cat':
            return torch.cat(head_outs, dim=1).to(self.device)
        else:
            return torch.mean(torch.stack(head_outs), dim=0).to(self.device)


class HAN_metapath_specific(nn.Module):
    def __init__(self, G, feature_attn_size, out_dim, dropout, slope, device):
        super(HAN_metapath_specific, self).__init__()
        self.device = device
        self.metabolite_nodes = G.filter_nodes(lambda nodes: nodes.data['type'] == 0)
        self.microbe_nodes = G.filter_nodes(lambda nodes: nodes.data['type'] == 2)
        self.disease_nodes = G.filter_nodes(lambda nodes: nodes.data['type'] == 1)
        self.G = G
        self.slope = slope
        self.me_fc = nn.Linear(G.ndata['me_sim'].shape[1], feature_attn_size, bias=False)  # 统一节点特征维数
        self.mi_fc = nn.Linear(G.ndata['mi_sim'].shape[1], feature_attn_size, bias=False)
        self.d_fc = nn.Linear(G.ndata['d_sim'].shape[1], feature_attn_size, bias=False)
        self.me_fc1 = nn.Linear(feature_attn_size + 495, out_dim)   # 设置全连接层
        self.d_fc1 = nn.Linear(feature_attn_size + 383, out_dim)
        self.attn_fc = nn.Linear(feature_attn_size * 2, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.me_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.mi_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.d_fc.weight, gain=gain)

    def edge_attention(self, edges):
        a = torch.sum(edges.src['z'].mul(edges.dst['z']), dim=1).unsqueeze(1)
        '''z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)'''
        return {'e': F.leaky_relu(a, negative_slope=self.slope)}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': F.elu(h)}

    def forward(self, new_g, meta_path):
        if meta_path == 'dmi':
            new_g.apply_nodes(lambda nodes: {'z': self.dropout(self.d_fc(nodes.data['d_sim']))}, self.disease_nodes)
            new_g.apply_nodes(lambda nodes: {'z': self.dropout(self.mi_fc(nodes.data['mi_sim']))}, self.microbe_nodes)
            new_g.apply_edges(self.edge_attention)
            new_g.update_all(self.message_func, self.reduce_func)
            h_dmi = new_g.ndata.pop('h').to(self.device)
            return h_dmi

        elif meta_path == 'mime':
            new_g.apply_nodes(lambda nodes: {'z': self.dropout(self.mi_fc(nodes.data['mi_sim']))}, self.microbe_nodes)
            new_g.apply_nodes(lambda nodes: {'z': self.dropout(self.me_fc(nodes.data['me_sim']))}, self.metabolite_nodes)
            new_g.apply_edges(self.edge_attention)
            new_g.update_all(self.message_func, self.reduce_func)
            h_mime = new_g.ndata.pop('h').to(self.device)
            return h_mime

