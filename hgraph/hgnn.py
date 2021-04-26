import torch
import torch.nn as nn
import rdkit.Chem as Chem
import torch.nn.functional as F
from hgraph.mol_graph import MolGraph
from hgraph.encoder import HierMPNEncoder
from hgraph.decoder import HierMPNDecoder
from hgraph.nnutils import *


def make_cuda(tensors):
    tree_tensors, graph_tensors = tensors
    make_tensor = lambda x: x if type(x) is torch.Tensor else torch.tensor(x)
    tree_tensors = [make_tensor(x).cuda().long() for x in tree_tensors[:-1]] + [tree_tensors[-1]]
    graph_tensors = [make_tensor(x).cuda().long() for x in graph_tensors[:-1]] + [graph_tensors[-1]]
    return tree_tensors, graph_tensors


class HierVAE(nn.Module):

    def __init__(self, args):
        super(HierVAE, self).__init__()
        self.encoder = HierMPNEncoder(args.vocab, args.atom_vocab, args.rnn_type, args.embed_size, args.hidden_size, args.depthT, args.depthG, args.dropout)
        self.decoder = HierMPNDecoder(args.vocab, args.atom_vocab, args.rnn_type, args.embed_size, args.hidden_size, args.latent_size, args.diterT, args.diterG, args.dropout)
        self.encoder.tie_embedding(self.decoder.hmpn)
        self.latent_size = args.latent_size

        self.R_mean = nn.Linear(args.hidden_size, args.latent_size)
        self.R_var = nn.Linear(args.hidden_size, args.latent_size)

    def rsample(self, z_vecs, W_mean, W_var, perturb=True):
        batch_size = z_vecs.size(0)
        z_mean = W_mean(z_vecs)
        z_log_var = -torch.abs( W_var(z_vecs) )
        kl_loss = -0.5 * torch.sum(1.0 + z_log_var - z_mean * z_mean - torch.exp(z_log_var)) / batch_size
        epsilon = torch.randn_like(z_mean).cuda()
        z_vecs = z_mean + torch.exp(z_log_var / 2) * epsilon if perturb else z_mean
        return z_vecs, kl_loss

    def sample(self, batch_size, greedy):
        root_vecs = torch.randn(batch_size, self.latent_size).cuda()
        return self.decoder.decode((root_vecs, root_vecs, root_vecs), greedy=greedy, max_decode_step=150)

    def reconstruct(self, batch):
        graphs, tensors, _ = batch
        tree_tensors, graph_tensors = tensors = make_cuda(tensors)
        root_vecs, tree_vecs, _, graph_vecs = self.encoder(tree_tensors, graph_tensors)

        root_vecs, root_kl = self.rsample(root_vecs, self.R_mean, self.R_var, perturb=False)
        return self.decoder.decode((root_vecs, root_vecs, root_vecs), greedy=True, max_decode_step=150)
       
    def forward(self, graphs, tensors, orders, beta, perturb_z=True):
        tree_tensors, graph_tensors = tensors = make_cuda(tensors)

        root_vecs, tree_vecs, _, graph_vecs = self.encoder(tree_tensors, graph_tensors)
        root_vecs, root_kl = self.rsample(root_vecs, self.R_mean, self.R_var, perturb_z)
        kl_div = root_kl

        loss, wacc, iacc, tacc, sacc = self.decoder((root_vecs, root_vecs, root_vecs), graphs, tensors, orders)
        return loss + beta * kl_div, kl_div.item(), wacc, iacc, tacc, sacc


class HierVGNN(nn.Module):

    def __init__(self, args):
        super(HierVGNN, self).__init__()
        self.latent_size = args.latent_size
        self.encoder = HierMPNEncoder(args.vocab, args.atom_vocab, args.rnn_type, args.embed_size, args.hidden_size, args.depthT, args.depthG, args.dropout)
        self.decoder = HierMPNDecoder(args.vocab, args.atom_vocab, args.rnn_type, args.embed_size, args.hidden_size, args.hidden_size, args.diterT, args.diterG, args.dropout, attention=True)
        self.encoder.tie_embedding(self.decoder.hmpn)

        self.T_mean = nn.Linear(args.hidden_size, args.latent_size)
        self.T_var = nn.Linear(args.hidden_size, args.latent_size)
        self.G_mean = nn.Linear(args.hidden_size, args.latent_size)
        self.G_var = nn.Linear(args.hidden_size, args.latent_size)

        self.W_tree = nn.Sequential( nn.Linear(args.hidden_size + args.latent_size, args.hidden_size), nn.ReLU() )
        self.W_graph = nn.Sequential( nn.Linear(args.hidden_size + args.latent_size, args.hidden_size), nn.ReLU() )

    def encode(self, tensors):
        tree_tensors, graph_tensors = tensors
        root_vecs, tree_vecs, _, graph_vecs = self.encoder(tree_tensors, graph_tensors)
        tree_vecs = stack_pad_tensor( [tree_vecs[st : st + le] for st,le in tree_tensors[-1]] )
        graph_vecs = stack_pad_tensor( [graph_vecs[st : st + le] for st,le in graph_tensors[-1]] )
        return root_vecs, tree_vecs, graph_vecs

    def translate(self, tensors, num_decode, enum_root, greedy=True):
        tensors = make_cuda(tensors)
        root_vecs, tree_vecs, graph_vecs = self.encode(tensors)
        all_smiles = []
        if enum_root:
            repeat = num_decode // len(root_vecs)
            modulo = num_decode % len(root_vecs)
            root_vecs = torch.cat([root_vecs] * repeat + [root_vecs[:modulo]], dim=0)
            tree_vecs = torch.cat([tree_vecs] * repeat + [tree_vecs[:modulo]], dim=0)
            graph_vecs = torch.cat([graph_vecs] * repeat + [graph_vecs[:modulo]], dim=0)
        
        batch_size = len(root_vecs)
        z_tree = torch.randn(batch_size, 1, self.latent_size).expand(-1, tree_vecs.size(1), -1).cuda()
        z_graph = torch.randn(batch_size, 1, self.latent_size).expand(-1, graph_vecs.size(1), -1).cuda()
        z_tree_vecs = self.W_tree( torch.cat([tree_vecs, z_tree], dim=-1) )
        z_graph_vecs = self.W_graph( torch.cat([graph_vecs, z_graph], dim=-1) )
        return self.decoder.decode( (root_vecs, z_tree_vecs, z_graph_vecs), greedy=greedy)

    def rsample(self, z_vecs, W_mean, W_var):
        batch_size = z_vecs.size(0)
        z_mean = W_mean(z_vecs)
        z_log_var = -torch.abs( W_var(z_vecs) )
        kl_loss = -0.5 * torch.sum(1.0 + z_log_var - z_mean * z_mean - torch.exp(z_log_var)) / batch_size
        epsilon = torch.randn_like(z_mean).cuda()
        z_vecs = z_mean + torch.exp(z_log_var / 2) * epsilon
        return z_vecs, kl_loss

    def forward(self, x_graphs, x_tensors, y_graphs, y_tensors, y_orders, beta):
        x_tensors = make_cuda(x_tensors)
        y_tensors = make_cuda(y_tensors)
        x_root_vecs, x_tree_vecs, x_graph_vecs = self.encode(x_tensors)
        _, y_tree_vecs, y_graph_vecs = self.encode(y_tensors)

        diff_tree_vecs = y_tree_vecs.sum(dim=1) - x_tree_vecs.sum(dim=1)
        diff_graph_vecs = y_graph_vecs.sum(dim=1) - x_graph_vecs.sum(dim=1)
        diff_tree_vecs, tree_kl = self.rsample(diff_tree_vecs, self.T_mean, self.T_var)
        diff_graph_vecs, graph_kl = self.rsample(diff_graph_vecs, self.G_mean, self.G_var)
        kl_div = tree_kl + graph_kl

        diff_tree_vecs = diff_tree_vecs.unsqueeze(1).expand(-1, x_tree_vecs.size(1), -1)
        diff_graph_vecs = diff_graph_vecs.unsqueeze(1).expand(-1, x_graph_vecs.size(1), -1)
        x_tree_vecs = self.W_tree( torch.cat([x_tree_vecs, diff_tree_vecs], dim=-1) )
        x_graph_vecs = self.W_graph( torch.cat([x_graph_vecs, diff_graph_vecs], dim=-1) )

        loss, wacc, iacc, tacc, sacc = self.decoder((x_root_vecs, x_tree_vecs, x_graph_vecs), y_graphs, y_tensors, y_orders)
        return loss + beta * kl_div, kl_div.item(), wacc, iacc, tacc, sacc

class HierCondVGNN(HierVGNN):

    def __init__(self, args):
        super(HierCondVGNN, self).__init__(args)
        self.W_tree = nn.Sequential( nn.Linear(args.hidden_size + args.latent_size + args.cond_size, args.hidden_size), nn.ReLU() )
        self.W_graph = nn.Sequential( nn.Linear(args.hidden_size + args.latent_size + args.cond_size, args.hidden_size), nn.ReLU() )

        self.U_tree = nn.Sequential( nn.Linear(args.hidden_size + args.cond_size, args.hidden_size), nn.ReLU() )
        self.U_graph = nn.Sequential( nn.Linear(args.hidden_size + args.cond_size, args.hidden_size), nn.ReLU() )

    def translate(self, tensors, cond, num_decode, enum_root):
        assert enum_root 
        tensors = make_cuda(tensors)
        root_vecs, tree_vecs, graph_vecs = self.encode(tensors)

        cond = cond.view(1,1,-1)
        tree_cond = cond.expand(num_decode, tree_vecs.size(1), -1)
        graph_cond = cond.expand(num_decode, graph_vecs.size(1), -1)

        if enum_root:
            repeat = num_decode // len(root_vecs)
            modulo = num_decode % len(root_vecs)
            root_vecs = torch.cat([root_vecs] * repeat + [root_vecs[:modulo]], dim=0)
            tree_vecs = torch.cat([tree_vecs] * repeat + [tree_vecs[:modulo]], dim=0)
            graph_vecs = torch.cat([graph_vecs] * repeat + [graph_vecs[:modulo]], dim=0)

        z_tree = torch.randn(num_decode, 1, self.latent_size).expand(-1, tree_vecs.size(1), -1).cuda()
        z_graph = torch.randn(num_decode, 1, self.latent_size).expand(-1, graph_vecs.size(1), -1).cuda()
        z_tree_vecs = self.W_tree( torch.cat([tree_vecs, z_tree, tree_cond], dim=-1) )
        z_graph_vecs = self.W_graph( torch.cat([graph_vecs, z_graph, graph_cond], dim=-1) )
        return self.decoder.decode( (root_vecs, z_tree_vecs, z_graph_vecs) )

    def forward(self, x_graphs, x_tensors, y_graphs, y_tensors, y_orders, cond, beta):
        x_tensors = make_cuda(x_tensors)
        y_tensors = make_cuda(y_tensors)
        cond = torch.tensor(cond).float().cuda()

        x_root_vecs, x_tree_vecs, x_graph_vecs = self.encode(x_tensors)
        _, y_tree_vecs, y_graph_vecs = self.encode(y_tensors)

        diff_tree_vecs = y_tree_vecs.sum(dim=1) - x_tree_vecs.sum(dim=1)
        diff_graph_vecs = y_graph_vecs.sum(dim=1) - x_graph_vecs.sum(dim=1)
        diff_tree_vecs = self.U_tree( torch.cat([diff_tree_vecs, cond], dim=-1) ) #combine condition for posterior
        diff_graph_vecs = self.U_graph( torch.cat([diff_graph_vecs, cond], dim=-1) ) #combine condition for posterior

        diff_tree_vecs, tree_kl = self.rsample(diff_tree_vecs, self.T_mean, self.T_var)
        diff_graph_vecs, graph_kl = self.rsample(diff_graph_vecs, self.G_mean, self.G_var)
        kl_div = tree_kl + graph_kl

        diff_tree_vecs = torch.cat([diff_tree_vecs, cond], dim=-1) #combine condition for posterior
        diff_graph_vecs = torch.cat([diff_graph_vecs, cond], dim=-1) #combine condition for posterior

        diff_tree_vecs = diff_tree_vecs.unsqueeze(1).expand(-1, x_tree_vecs.size(1), -1)
        diff_graph_vecs = diff_graph_vecs.unsqueeze(1).expand(-1, x_graph_vecs.size(1), -1)
        x_tree_vecs = self.W_tree( torch.cat([x_tree_vecs, diff_tree_vecs], dim=-1) )
        x_graph_vecs = self.W_graph( torch.cat([x_graph_vecs, diff_graph_vecs], dim=-1) )

        loss, wacc, iacc, tacc, sacc = self.decoder((x_root_vecs, x_tree_vecs, x_graph_vecs), y_graphs, y_tensors, y_orders)
        return loss + beta * kl_div, kl_div.item(), wacc, iacc, tacc, sacc

