import torch
import torch.nn as nn
import rdkit.Chem as Chem
import torch.nn.functional as F
from agraph.mol_graph import MolGraph
from agraph.encoder import GraphEncoder
from agraph.decoder import GraphDecoder
from agraph.nnutils import *

def make_cuda(graph_tensors):
    graph_tensors = [x.cuda().long() for x in graph_tensors[:-1]] + [graph_tensors[-1]]
    return graph_tensors


class AtomVGNN(nn.Module):

    def __init__(self, args):
        super(AtomVGNN, self).__init__()
        self.latent_size = args.latent_size
        self.encoder = GraphEncoder(args.atom_vocab, args.rnn_type, args.embed_size, args.hidden_size, args.depth)
        self.decoder = GraphDecoder(args.atom_vocab, args.rnn_type, args.embed_size, args.hidden_size, args.hidden_size, args.diter)

        self.G_mean = nn.Linear(args.hidden_size, args.latent_size)
        self.G_var = nn.Linear(args.hidden_size, args.latent_size)
        self.W_graph = nn.Sequential( nn.Linear(args.hidden_size + args.latent_size, args.hidden_size), nn.ReLU() )

    def encode(self, graph_tensors):
        graph_vecs = self.encoder(graph_tensors)
        graph_vecs = stack_pad_tensor( [graph_vecs[st : st + le] for st,le in graph_tensors[-1]] )
        return graph_vecs

    def translate(self, x_tensors, num_decode, enum_root):
        x_tensors = make_cuda(x_tensors)
        graph_vecs = self.encode(x_tensors)
        if not enum_root:
            graph_vecs = graph_vecs.expand(num_decode, -1, -1)
        z_graph = torch.randn(num_decode, 1, self.latent_size).expand(-1, graph_vecs.size(1), -1).cuda()
        z_graph_vecs = self.W_graph( torch.cat([graph_vecs, z_graph], dim=-1) )
        return self.decoder.decode( z_graph_vecs )

    def reconstruct(self, x_graphs, x_tensors, y_graphs, y_tensors, y_orders):
        x_tensors = make_cuda(x_tensors)
        y_tensors = make_cuda(y_tensors)
        x_graph_vecs = self.encode(x_tensors)
        y_graph_vecs = self.encode(y_tensors)

        diff_graph_vecs = y_graph_vecs.sum(dim=1) - x_graph_vecs.sum(dim=1)
        diff_graph_vecs, kl_div = self.rsample(diff_graph_vecs, self.G_mean, self.G_var, mean_only=True)

        diff_graph_vecs = diff_graph_vecs.unsqueeze(1).expand(-1, x_graph_vecs.size(1), -1)
        x_graph_vecs = self.W_graph( torch.cat([x_graph_vecs, diff_graph_vecs], dim=-1) )
        return self.decoder.decode( x_graph_vecs )

    def rsample(self, z_vecs, W_mean, W_var, mean_only=False):
        batch_size = z_vecs.size(0)
        z_mean = W_mean(z_vecs)
        if mean_only: return z_mean, z_mean.new_tensor([0.])

        z_log_var = -torch.abs( W_var(z_vecs) )
        kl_loss = -0.5 * torch.sum(1.0 + z_log_var - z_mean * z_mean - torch.exp(z_log_var)) / batch_size
        epsilon = torch.randn_like(z_mean).cuda()
        z_vecs = z_mean + torch.exp(z_log_var / 2) * epsilon
        return z_vecs, kl_loss

    def forward(self, x_graphs, x_tensors, y_graphs, y_tensors, y_orders, beta):
        x_tensors = make_cuda(x_tensors)
        y_tensors = make_cuda(y_tensors)
        x_graph_vecs = self.encode(x_tensors)
        y_graph_vecs = self.encode(y_tensors)

        diff_graph_vecs = y_graph_vecs.sum(dim=1) - x_graph_vecs.sum(dim=1)
        diff_graph_vecs, kl_div = self.rsample(diff_graph_vecs, self.G_mean, self.G_var)

        diff_graph_vecs = diff_graph_vecs.unsqueeze(1).expand(-1, x_graph_vecs.size(1), -1)
        x_graph_vecs = self.W_graph( torch.cat([x_graph_vecs, diff_graph_vecs], dim=-1) )

        loss, wacc, tacc, sacc = self.decoder(x_graph_vecs, y_graphs, y_tensors, y_orders)
        return loss + beta * kl_div, kl_div.item(), wacc, tacc, sacc


class AtomCondVGNN(nn.Module):

    def __init__(self, args):
        super(AtomCondVGNN, self).__init__()
        self.latent_size = args.latent_size
        self.encoder = GraphEncoder(args.atom_vocab, args.rnn_type, args.embed_size, args.hidden_size, args.depth)
        self.decoder = GraphDecoder(args.atom_vocab, args.rnn_type, args.embed_size, args.hidden_size, args.hidden_size, args.diter)

        self.G_mean = nn.Linear(args.hidden_size, args.latent_size)
        self.G_var = nn.Linear(args.hidden_size, args.latent_size)
        self.U_graph = nn.Sequential( nn.Linear(args.hidden_size + args.cond_size, args.hidden_size), nn.ReLU() )
        self.W_graph = nn.Sequential( nn.Linear(args.hidden_size + args.latent_size + args.cond_size, args.hidden_size), nn.ReLU() )

    def encode(self, graph_tensors):
        graph_vecs = self.encoder(graph_tensors)
        graph_vecs = stack_pad_tensor( [graph_vecs[st : st + le] for st,le in graph_tensors[-1]] )
        return graph_vecs

    def translate(self, x_tensors, cond, num_decode, enum_root):
        x_tensors = make_cuda(x_tensors)
        graph_vecs = self.encode(x_tensors)
        cond = cond.view(1,1,-1)
        graph_cond = cond.expand(graph_vecs.size(0), graph_vecs.size(1), -1)
        if not enum_root:
            graph_vecs = graph_vecs.expand(num_decode, -1, -1)

        z_graph = torch.randn(num_decode, 1, self.latent_size).expand(-1, graph_vecs.size(1), -1).cuda()
        z_graph_vecs = self.W_graph( torch.cat([graph_vecs, z_graph, graph_cond], dim=-1) )
        return self.decoder.decode( z_graph_vecs )

    def reconstruct(self, x_graphs, x_tensors, y_graphs, y_tensors, y_orders):
        x_tensors = make_cuda(x_tensors)
        y_tensors = make_cuda(y_tensors)
        x_graph_vecs = self.encode(x_tensors)
        y_graph_vecs = self.encode(y_tensors)

        diff_graph_vecs = y_graph_vecs.sum(dim=1) - x_graph_vecs.sum(dim=1)
        diff_graph_vecs, kl_div = self.rsample(diff_graph_vecs, self.G_mean, self.G_var, mean_only=True)

        diff_graph_vecs = diff_graph_vecs.unsqueeze(1).expand(-1, x_graph_vecs.size(1), -1)
        x_graph_vecs = self.W_graph( torch.cat([x_graph_vecs, diff_graph_vecs], dim=-1) )
        return self.decoder.decode( x_graph_vecs )

    def rsample(self, z_vecs, W_mean, W_var, mean_only=False):
        batch_size = z_vecs.size(0)
        z_mean = W_mean(z_vecs)
        if mean_only: return z_mean, z_mean.new_tensor([0.])

        z_log_var = -torch.abs( W_var(z_vecs) )
        kl_loss = -0.5 * torch.sum(1.0 + z_log_var - z_mean * z_mean - torch.exp(z_log_var)) / batch_size
        epsilon = torch.randn_like(z_mean).cuda()
        z_vecs = z_mean + torch.exp(z_log_var / 2) * epsilon
        return z_vecs, kl_loss

    def forward(self, x_graphs, x_tensors, y_graphs, y_tensors, y_orders, cond, beta):
        x_tensors = make_cuda(x_tensors)
        y_tensors = make_cuda(y_tensors)
        cond = cond.cuda()

        x_graph_vecs = self.encode(x_tensors)
        y_graph_vecs = self.encode(y_tensors)

        diff_graph_vecs = y_graph_vecs.sum(dim=1) - x_graph_vecs.sum(dim=1)
        diff_graph_vecs = self.U_graph( torch.cat([diff_graph_vecs, cond], dim=-1) ) #combine condition for posterior
        diff_graph_vecs, kl_div = self.rsample(diff_graph_vecs, self.G_mean, self.G_var)

        diff_graph_vecs = torch.cat([diff_graph_vecs, cond], dim=-1) #combine condition for posterior
        diff_graph_vecs = diff_graph_vecs.unsqueeze(1).expand(-1, x_graph_vecs.size(1), -1)
        x_graph_vecs = self.W_graph( torch.cat([x_graph_vecs, diff_graph_vecs], dim=-1) )

        loss, wacc, tacc, sacc = self.decoder(x_graph_vecs, y_graphs, y_tensors, y_orders)
        return loss + beta * kl_div, kl_div.item(), wacc, tacc, sacc

