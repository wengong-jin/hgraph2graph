import torch
import torch.nn as nn
import rdkit.Chem as Chem
import torch.nn.functional as F
from agraph.nnutils import *
from agraph.encoder import GraphEncoder
from agraph.mol_graph import MolGraph
from agraph.inc_graph import IncGraph
from collections import deque

class HTuple():
    def __init__(self, node=None, mess=None, vmask=None, emask=None):
        self.node, self.mess = node, mess
        self.vmask, self.emask = vmask, emask

class GraphDecoder(nn.Module):

    def __init__(self, avocab, rnn_type, embed_size, hidden_size, latent_size, depth):
        super(GraphDecoder, self).__init__()
        self.avocab = avocab
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.latent_size = latent_size
        self.itensor = torch.LongTensor([]).cuda()

        self.mpn = GraphEncoder(avocab, rnn_type, embed_size, hidden_size, depth)
        self.rnn_cell = self.mpn.encoder.rnn

        self.topoNN = nn.Sequential(
                nn.Linear(hidden_size * 2 + latent_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1)
        )
        self.atomNN = nn.Sequential(
                nn.Linear(hidden_size * 2 + latent_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, avocab.size())
        )
        self.bondNN = nn.Sequential(
                nn.Linear(hidden_size * 3 + latent_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, len(MolGraph.BOND_LIST))
        )

        self.A_topo = nn.Linear(hidden_size * 2, latent_size)
        self.A_atom = nn.Linear(hidden_size * 2, latent_size)
        self.A_bond = nn.Linear(hidden_size * 3, latent_size)
        self.R_bond = nn.Sequential(
                nn.Linear(hidden_size + avocab.size(), hidden_size),
                nn.ReLU()
        )
        self.W_bond = nn.Sequential(
                nn.Linear(hidden_size + len(MolGraph.BOND_LIST), hidden_size),
                nn.ReLU()
        )
        self.E_a = torch.eye( avocab.size() ).cuda() 
        self.E_b = torch.eye( len(MolGraph.BOND_LIST) ).cuda()

        self.topo_loss = nn.BCEWithLogitsLoss(size_average=False)
        self.atom_loss = nn.CrossEntropyLoss(size_average=False)
        self.bond_loss = nn.CrossEntropyLoss(size_average=False)
        
    def apply_graph_mask(self, tensors, hgraph):
        fnode, fmess, agraph, bgraph, scope = tensors
        agraph = agraph * index_select_ND(hgraph.emask, 0, agraph)
        bgraph = bgraph * index_select_ND(hgraph.emask, 0, bgraph)
        return fnode, fmess, agraph, bgraph, scope

    def update_graph_mask(self, graph_batch, new_atoms, hgraph, visited):
        new_atom_index = hgraph.vmask.new_tensor(new_atoms)
        hgraph.vmask.scatter_(0, new_atom_index, 1)

        visited.update(new_atoms)
        new_bonds = [] 
        for yid in new_atoms:
            for zid in graph_batch[yid]:
                if zid in visited:
                    new_bonds.append( graph_batch[yid][zid]['mess_idx'] )
                    new_bonds.append( graph_batch[zid][yid]['mess_idx'] )

        new_bond_index = hgraph.emask.new_tensor(new_bonds)
        if len(new_bonds) > 0:
            hgraph.emask.scatter_(0, new_bond_index, 1)

    def attention(self, src_vecs, batch_idx, queries, W_att):
        src_vecs = src_vecs.index_select(0, batch_idx)
        att_score = torch.bmm( src_vecs, W_att(queries).unsqueeze(-1) )
        att_vecs = F.softmax(att_score, dim=1) * src_vecs
        att_vecs = att_vecs.sum(dim=1)
        return att_vecs 

    def get_topo_score(self, src_graph_vecs, batch_idx, topo_vecs):
        topo_cxt = self.attention(src_graph_vecs, batch_idx, topo_vecs, self.A_topo)
        return self.topoNN( torch.cat([topo_vecs, topo_cxt], dim=-1) ).squeeze(-1)

    def get_atom_score(self, src_graph_vecs, batch_idx, atom_vecs):
        atom_cxt = self.attention(src_graph_vecs, batch_idx, atom_vecs, self.A_atom)
        atom_vecs = torch.cat([atom_vecs, atom_cxt], dim=-1)
        return self.atomNN(atom_vecs)

    def get_bond_score(self, src_graph_vecs, batch_idx, bond_vecs):
        bond_cxt = self.attention(src_graph_vecs, batch_idx, bond_vecs, self.A_bond)
        bond_vecs = torch.cat([bond_vecs, bond_cxt], dim=-1)
        return self.bondNN(bond_vecs)

    def forward(self, src_graph_vecs, graph_batch, graph_tensors, orders):
        batch_size = len(orders)

        hgraph = HTuple(
            node = src_graph_vecs.new_zeros(graph_tensors[0].size(0), self.hidden_size),
            mess = self.rnn_cell.get_init_state(graph_tensors[1]),
            vmask = self.itensor.new_zeros(graph_tensors[0].size(0)),
            emask = self.itensor.new_zeros(graph_tensors[1].size(0))
        )
        
        all_topo_preds, all_atom_preds, all_bond_preds = [], [], []
        graph_tensors = self.mpn.embed_graph(graph_tensors) + (graph_tensors[-1],) #preprocess graph tensors

        visited = set()
        new_atoms = []
        zero = src_graph_vecs.new_zeros(self.hidden_size * 2)
        for i in range(batch_size):
            root = orders[i][0][0]
            root_label = graph_batch.nodes[root]['label']
            atom_type = self.avocab[ root_label ]
            all_atom_preds.append( (zero, i, atom_type) )
            new_atoms.append(root)

        self.update_graph_mask(graph_batch, new_atoms, hgraph, visited)

        maxt = max([len(x) for x in orders])
        for t in range(maxt):
            batch_list = [i for i in range(batch_size) if t < len(orders[i])]
            assert hgraph.vmask[0].item() == 0 and hgraph.emask[0].item() == 0

            cur_graph_tensors = self.apply_graph_mask(graph_tensors, hgraph)
            vmask = hgraph.vmask.unsqueeze(-1).float()
            hgraph.node, _ = self.mpn.encoder(*cur_graph_tensors[:-1], mask=vmask)

            new_atoms = []
            for i in batch_list:
                xid, yid, front, fbond = orders[i][t]
                st,le = graph_tensors[-1][i]
                stop = 1 if yid is None else 0

                gvec = hgraph.node[st : st + le].sum(dim=0)
                cxt_vec = torch.cat( (gvec, hgraph.node[xid]), dim=-1)
                all_topo_preds.append( (cxt_vec, i, stop) ) 

                if stop == 0:
                    new_atoms.append(yid)
                    ylabel = graph_batch.nodes[yid]['label']
                    atom_type = self.avocab[ ylabel ]
                    all_atom_preds.append( (cxt_vec, i, atom_type) )

                    hist = hgraph.node[xid]
                    atom_vec = self.E_a[atom_type]
                    for zid,bt in zip(front, fbond):
                        cur_hnode = torch.cat([hist, atom_vec], dim=-1)
                        cur_hnode = self.R_bond(cur_hnode)
                        pairs = torch.cat([gvec, cur_hnode, hgraph.node[zid]], dim=-1)
                        all_bond_preds.append( (pairs, i, bt) )
                        if bt > 0: 
                            bond_vec = self.E_b[bt]
                            z_hnode = torch.cat([hgraph.node[zid], bond_vec], dim=-1)
                            hist += self.W_bond(z_hnode)

            self.update_graph_mask(graph_batch, new_atoms, hgraph, visited)

        topo_vecs, batch_idx, topo_labels = zip_tensors(all_topo_preds)
        topo_scores = self.get_topo_score(src_graph_vecs, batch_idx, topo_vecs)
        topo_loss = self.topo_loss(topo_scores, topo_labels.float())
        topo_acc = get_accuracy_bin(topo_scores, topo_labels)

        atom_vecs, batch_idx, atom_labels = zip_tensors(all_atom_preds)
        atom_scores = self.get_atom_score(src_graph_vecs, batch_idx, atom_vecs)
        atom_loss = self.atom_loss(atom_scores, atom_labels)
        atom_acc = get_accuracy(atom_scores, atom_labels)

        bond_vecs, batch_idx, bond_labels = zip_tensors(all_bond_preds)
        bond_scores = self.get_bond_score(src_graph_vecs, batch_idx, bond_vecs)
        bond_loss = self.bond_loss(bond_scores, bond_labels)
        bond_acc = get_accuracy(bond_scores, bond_labels)

        loss = (topo_loss + atom_loss + bond_loss) / batch_size
        return loss, atom_acc, topo_acc, bond_acc

    def decode(self, src_graph_vecs, max_decode_step=150):
        batch_size = len(src_graph_vecs)
        graph_batch = IncGraph(self.avocab, batch_size, node_fdim=self.mpn.atom_size, edge_fdim=self.mpn.atom_size + self.mpn.bond_size)
        queue = [deque() for i in range(batch_size)]

        init_vecs = src_graph_vecs.new_zeros(batch_size, self.hidden_size * 2)
        batch_idx = self.itensor.new_tensor(range(batch_size))

        atom_scores = self.get_atom_score(src_graph_vecs, batch_idx, init_vecs)
        root_atoms = atom_scores.max(dim=-1)[1]
        root_atoms = root_atoms.tolist()

        for bid in range(batch_size):
            atom_type = self.avocab.get_smiles( root_atoms[bid] )
            root_idx = graph_batch.add_atom( bid, atom_type )
            queue[bid].append(root_idx)
        
        graph_tensors = graph_batch.get_tensors()
        hgraph = HTuple() 
        
        for t in range(max_decode_step):
            batch_list = [ bid for bid in range(batch_size) if len(queue[bid]) > 0 ]
            if len(batch_list) == 0: break

            batch_idx = batch_idx.new_tensor(batch_list)
            cur_nodes = [queue[bid][0] for bid in batch_list]
            cur_nodes = self.itensor.new_tensor(cur_nodes)

            hgraph.node, _ = self.mpn.encoder(*graph_tensors, mask=None)

            for i,bid in enumerate(batch_list):
                xid = queue[bid][0]
                bid_tensor = self.itensor.new_tensor([bid]) 
                bid_nodes = self.itensor.new_tensor(graph_batch.batch[bid])
                gvec = hgraph.node.index_select(0, bid_nodes).sum(dim=0)
                cxt_vec = torch.cat( (gvec, hgraph.node[xid]), dim=-1).unsqueeze(0)
                stop_score = self.get_topo_score(src_graph_vecs, bid_tensor, cxt_vec)
                stop_score = stop_score.item()

                if stop_score > 0 or graph_batch.can_expand(xid) is False:
                    queue[bid].popleft()
                    continue

                atom_score = self.get_atom_score(src_graph_vecs, bid_tensor, cxt_vec)
                atom_type_id = atom_score.max(dim=-1)[1].item()
                atom_type = self.avocab.get_smiles( atom_type_id )
                yid = graph_batch.add_atom( bid, atom_type )

                cands = list(queue[bid])
                hist = hgraph.node[xid]
                atom_vec = self.E_a[atom_type_id]
                for zid in cands:
                    cur_hnode = torch.cat([hist, atom_vec], dim=-1)
                    cur_hnode = self.R_bond(cur_hnode)
                    pairs = torch.cat([gvec, cur_hnode, hgraph.node[zid]], dim=-1)
                    pairs = pairs.unsqueeze(0)

                    bid_tensor = self.itensor.new_tensor( [bid] )
                    bond_scores = self.get_bond_score( src_graph_vecs, bid_tensor, pairs )
                    bt = bond_scores.max(dim=-1)[1].item()
                    if bt > 0: 
                        graph_batch.add_bond(yid, zid, bt)
                        bond_vec = self.E_b[bt]
                        z_hnode = torch.cat([hgraph.node[zid], bond_vec], dim=-1)
                        hist += self.W_bond(z_hnode)

                queue[bid].append(yid)
               
        return graph_batch.get_mol()

