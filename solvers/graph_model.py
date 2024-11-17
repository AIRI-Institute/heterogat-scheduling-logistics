import dgl.function as fn
from dgl.sampling import sample_neighbors
import dgl
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Adam
from pytorch_lightning import LightningModule
from utils import total_cost_from_graph, os_type, ss_type, so_type


def cat_s_ss(edges):
    return {'s_ss': torch.cat([edges.src['s_feat'], edges.data['feat']], dim=1)}


def cat_o_os(edges):
    return {'o_os': torch.cat([edges.src['o_feat'], edges.data['feat']], dim=1)}


def cat_h_os_s(edges):
    return {'h_os_s': torch.cat([edges.data['h_os'], edges.dst['h_s']], dim=1)}


def cat_h_ss_s(edges):
    return {'h_ss_s': torch.cat([edges.data['h_ss'], edges.dst['h_s']], dim=1)}


def edge_den_os(nodes):
    return {'den_os': torch.sum(torch.exp(nodes.mailbox['e_os']), dim=1)}


def edge_den_ss(nodes):
    return {'den_ss': torch.sum(torch.exp(nodes.mailbox['e_ss']), dim=1)}


def sample_so(graph, logits):
    with graph.local_scope():
        graph.edata['prob'] = {'so': torch.sigmoid(logits)}
        subg = sample_neighbors(
            graph, 
            nodes={'o': graph.nodes('o')}, 
            fanout={'backward': 0, 'forward': 0, 'os': 0, 'so': 1, 'ss': 0},
            prob='prob'
        )
    return subg.edges(etype='so')


class AttnConvLayer(nn.Module):
    def __init__(self, ins_dim, ino_dim, out_dim):
        super().__init__()
        self.W_s = nn.Linear(ins_dim, out_dim)
        self.W_os = nn.Linear(ino_dim + 2, out_dim)
        self.W_ss = nn.Linear(ins_dim + 1, out_dim)
        self.attn = nn.Linear(out_dim * 2, 1)
        self.W_in = nn.Linear(ino_dim, out_dim)
        self.W_self = nn.Linear(ino_dim, out_dim)
        self.W_out = nn.Linear(ino_dim, out_dim)
        self.W_o = nn.Linear(out_dim*3, out_dim)
    
    def forward(self, graph, s_feat, o_feat):
        with graph.local_scope():
            z = self._conv_z(graph, s_feat, o_feat)
            x = self._conv_x(graph, o_feat)
        return z, x

    def _conv_z(self, graph, s_feat, o_feat):
        graph.ndata['s_feat'] = {'s': s_feat}
        graph.ndata['o_feat'] = {'o': o_feat}

        graph.apply_edges(cat_s_ss, etype='ss')
        graph.apply_edges(cat_o_os, etype='os')

        graph.edata['h_ss'] = {'ss': self.W_ss(graph.edata['s_ss'][ss_type])}
        graph.edata['h_os'] = {'os': self.W_os(graph.edata['o_os'][os_type])}
        graph.ndata['h_s'] = {'s': self.W_s(s_feat)}

        graph.apply_edges(cat_h_ss_s, etype='ss')
        graph.apply_edges(cat_h_os_s, etype='os')

        graph.edata['e_ss'] = {'ss': F.leaky_relu(self.attn(graph.edata['h_ss_s'][ss_type]))}
        graph.edata['e_os'] = {'os': F.leaky_relu(self.attn(graph.edata['h_os_s'][os_type]))}

        graph.multi_update_all({
            'os': (fn.copy_e('e_os', 'e_os'), edge_den_os),
            'ss': (fn.copy_e('e_ss', 'e_ss'), edge_den_ss),
        }, 'sum')

        graph.edata['nom_os'] = {'os': torch.exp(graph.edata['e_os'][os_type])}
        graph.edata['nom_ss'] = {'ss': torch.exp(graph.edata['e_ss'][ss_type])}

        graph.apply_edges(fn.e_div_v('nom_os', 'den_os', 'alpha_os'), etype='os')
        graph.apply_edges(fn.e_div_v('nom_ss', 'den_ss', 'alpha_ss'), etype='ss')

        graph.edata['alpha_h_ss'] = {'ss': graph.edata['alpha_ss'][ss_type] * graph.edata['h_ss'][ss_type]}
        graph.edata['alpha_h_os'] = {'os': graph.edata['alpha_os'][os_type] * graph.edata['h_os'][os_type]}

        graph.multi_update_all({
            'ss': (fn.copy_e('alpha_h_ss', 'alpha_h_ss'), fn.sum('alpha_h_ss', 'z_ss')),
            'os': (fn.copy_e('alpha_h_os', 'alpha_h_os'), fn.sum('alpha_h_os', 'z_os')),
        }, 'sum')

        z = graph.ndata['z_ss']['s'] + graph.ndata['z_os']['s']
        return z
    
    def _conv_x(self, graph, o_feat):
        graph.ndata['h_in'] = {'o': self.W_in(o_feat)}
        graph.ndata['h_self'] = {'o': self.W_self(o_feat)}
        graph.ndata['h_out'] = {'o': self.W_out(o_feat)}

        graph.multi_update_all({
            'forward': (fn.copy_u('h_in', 'h_in'), fn.sum('h_in', 'h_in')),
            'backward': (fn.copy_u('h_out', 'h_out'), fn.sum('h_out', 'h_out')),
        }, 'sum')
        x = torch.cat([
            graph.ndata['h_in']['o'],
            graph.ndata['h_self']['o'],
            graph.ndata['h_out']['o'],
        ], dim=1)
        x = self.W_o(torch.relu(x))
        return x


class DotProductDecoder(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, graph, z, x):
        with graph.local_scope():
            graph.ndata['z'] = {'s': z}
            graph.ndata['x'] = {'o': x}
            graph.apply_edges(fn.u_dot_v('z', 'x', 'dot'), etype='so')
            logits = graph.edata['dot'][so_type]
            return logits


class BaseGNN(LightningModule):
    def forward(self, graph):
        s_feat = graph.ndata['feat']['s']
        o_feat = graph.ndata['feat']['o']
        s_hid, o_hid = self.enc[0](graph, s_feat, o_feat)
        for conv in self.enc[1:]:
            s_hid, o_hid = self.dropout(s_hid), self.dropout(o_hid)
            s_hid, o_hid = conv(graph, torch.relu(s_hid), torch.relu(o_hid))
        logits = self.dec(graph, s_hid, o_hid)
        return logits
    
    def predict(self, graph):
        logits = self.forward(graph)
        s, o = sample_so(graph, logits)
        edge_ids = graph.edge_ids(o, s, etype=os_type)
        pred = torch.zeros(graph.number_of_edges(etype=os_type), 1)
        pred[edge_ids] = 1
        return pred
    
    def training_step(self, batch, batch_idx):
        graph, _ = batch
        target = graph.edata['target'][os_type]
        logits = self.forward(graph)
        loss = F.binary_cross_entropy_with_logits(logits, target)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        graph, _ = batch
        target = graph.edata['target'][os_type]
        logits = self.forward(graph)
        loss = F.binary_cross_entropy_with_logits(logits, target)
        self.log("val_loss", loss)

        pred = self.predict(graph)
        total_cost = 0
        i = 0
        for subgraph in dgl.unbatch(graph):
            num_edges = subgraph.num_edges(os_type)
            total_cost += total_cost_from_graph(subgraph, pred[i:i+num_edges])
            i = num_edges

        self.log("val_total_cost", total_cost/graph.batch_size)
        return loss, total_cost
    
    def configure_optimizers(self):
        opt = Adam(self.parameters(), lr=self.lr)
        return opt
    

class AttentionGNN(BaseGNN):
    def __init__(self, ins_dim, ino_dim, out_dim, n_layers, lr, dropout_rate=0.):
        super().__init__()
        convs = [AttnConvLayer(ins_dim, ino_dim, out_dim)]
        for _ in range(n_layers-1):
            convs.append(AttnConvLayer(out_dim, out_dim, out_dim))
        self.enc = nn.ModuleList(convs)
        self.dropout = nn.Dropout(dropout_rate)
        self.dec = DotProductDecoder()
        self.lr = lr


class ConvLayer(nn.Module):
    def __init__(self, ins_dim, ino_dim, out_dim):
        super().__init__()
        self.W_os = nn.Linear(ino_dim + 2, out_dim)
        self.W_ss = nn.Linear(ins_dim + 1, out_dim)
        
        self.W_in = nn.Linear(ino_dim, out_dim)
        self.W_self = nn.Linear(ino_dim, out_dim)
        self.W_out = nn.Linear(ino_dim, out_dim)
        self.W_o = nn.Linear(out_dim*3, out_dim)
    
    def forward(self, graph, s_feat, o_feat):
        with graph.local_scope():
            z = self._conv_z(graph, s_feat, o_feat)
            x = self._conv_x(graph, o_feat)
        return z, x

    def _conv_z(self, graph, s_feat, o_feat):
        graph.ndata['s_feat'] = {'s': s_feat}
        graph.ndata['o_feat'] = {'o': o_feat}

        graph.apply_edges(cat_s_ss, etype='ss')
        graph.apply_edges(cat_o_os, etype='os')
        
        graph.edata['h_ss'] = {'ss': self.W_ss(graph.edata['s_ss'][ss_type])}
        graph.edata['h_os'] = {'os': self.W_os(graph.edata['o_os'][os_type])}

        graph.multi_update_all({
            'ss': (fn.copy_e('h_ss', 'h_ss'), fn.mean('h_ss', 'z_ss')),
            'os': (fn.copy_e('h_os', 'h_os'), fn.mean('h_os', 'z_os')),
        }, 'sum')
        
        z = graph.ndata['z_ss']['s'] + graph.ndata['z_os']['s']
        return z
    
    def _conv_x(self, graph, o_feat):
        graph.ndata['h_in'] = {'o': self.W_in(o_feat)}
        graph.ndata['h_self'] = {'o': self.W_self(o_feat)}
        graph.ndata['h_out'] = {'o': self.W_out(o_feat)}

        graph.multi_update_all({
            'forward': (fn.copy_u('h_in', 'h_in'), fn.sum('h_in', 'h_in')),
            'backward': (fn.copy_u('h_out', 'h_out'), fn.sum('h_out', 'h_out')),
        }, 'sum')
        x = torch.cat([
            graph.ndata['h_in']['o'],
            graph.ndata['h_self']['o'],
            graph.ndata['h_out']['o'],
        ], dim=1)
        x = self.W_o(torch.relu(x))
        return x


class ConvGNN(BaseGNN):
    def __init__(self, ins_dim, ino_dim, out_dim, n_layers, lr, dropout_rate=0.):
        super().__init__()
        convs = [ConvLayer(ins_dim, ino_dim, out_dim)]
        for _ in range(n_layers-1):
            convs.append(ConvLayer(out_dim, out_dim, out_dim))
        self.enc = nn.ModuleList(convs)
        self.dropout = nn.Dropout(dropout_rate)
        self.dec = DotProductDecoder()
        self.lr = lr


class LinearLayer(nn.Module):
    def __init__(self, ins_dim, ino_dim, out_dim):
        super().__init__()
        self.W_s = nn.Linear(ins_dim, out_dim)
        self.W_self = nn.Linear(ino_dim, out_dim)
    
    def forward(self, graph, s_feat, o_feat):
        with graph.local_scope():
            z = self._conv_z(graph, s_feat, o_feat)
            x = self._conv_x(graph, o_feat)
        return z, x

    def _conv_z(self, graph, s_feat, o_feat):
        return self.W_s(s_feat)
    
    def _conv_x(self, graph, o_feat):
        return self.W_self(o_feat)


class MLP(BaseGNN):
    def __init__(self, ins_dim, ino_dim, out_dim, n_layers, lr, dropout_rate=0.):
        super().__init__()
        convs = [LinearLayer(ins_dim, ino_dim, out_dim)]
        for _ in range(n_layers-1):
            convs.append(LinearLayer(out_dim, out_dim, out_dim))
        self.enc = nn.ModuleList(convs)
        self.dropout = nn.Dropout(dropout_rate)
        self.dec = DotProductDecoder()
        self.lr = lr
