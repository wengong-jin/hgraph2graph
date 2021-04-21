import torch
import torch.nn as nn
import rdkit.Chem as Chem
import torch.nn.functional as F
from poly_hgraph.nnutils import *
from poly_hgraph.encoder import IncHierMPNEncoder
from poly_hgraph.mol_graph import MolGraph
from poly_hgraph.inc_graph import IncTree, IncGraph

class HTuple():
    def __init__(self, node=None, mess=None, vmask=None, emask=None):
        self.node, self.mess = node, mess
        self.vmask, self.emask = vmask, emask

class HierMPNDecoder(nn.Module):

    def __init__(self, vocab, avocab, rnn_type, embed_size, hidden_size, latent_size, depthT, depthG, dropout, attention=False):
        super(HierMPNDecoder, self).__init__()
        self.vocab = vocab
        self.avocab = avocab
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.latent_size = latent_size
        self.use_attention = attention
        self.itensor = torch.LongTensor([]).cuda()

        self.hmpn = IncHierMPNEncoder(vocab, avocab, rnn_type, embed_size, hidden_size, depthT, depthG, dropout)
        self.rnn_cell = self.hmpn.tree_encoder.rnn
        self.E_assm = self.hmpn.E_i 
        self.E_order = torch.eye(MolGraph.MAX_POS).cuda()

        self.topoNN = nn.Sequential(
                nn.Linear(hidden_size + latent_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, 1)
        )
        self.clsNN = nn.Sequential(
                nn.Linear(hidden_size + latent_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, vocab.size()[0])
        )
        self.iclsNN = nn.Sequential(
                nn.Linear(hidden_size + latent_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, vocab.size()[1])
        )
        self.matchNN = nn.Sequential(
                nn.Linear(hidden_size + embed_size + MolGraph.MAX_POS, hidden_size),
                nn.ReLU(),
        )
        self.W_assm = nn.Linear(hidden_size, latent_size)

        if latent_size != hidden_size:
            self.W_root = nn.Linear(latent_size, hidden_size)

        if self.use_attention:
            self.A_topo = nn.Linear(hidden_size, latent_size)
            self.A_cls = nn.Linear(hidden_size, latent_size)
            self.A_assm = nn.Linear(hidden_size, latent_size)

        self.topo_loss = nn.BCEWithLogitsLoss(size_average=False)
        self.cls_loss = nn.CrossEntropyLoss(size_average=False)
        self.icls_loss = nn.CrossEntropyLoss(size_average=False)
        self.assm_loss = nn.CrossEntropyLoss(size_average=False)
        
    def apply_tree_mask(self, tensors, cur, prev):
        fnode, fmess, agraph, bgraph, cgraph, scope = tensors
        agraph = agraph * index_select_ND(cur.emask, 0, agraph)
        bgraph = bgraph * index_select_ND(cur.emask, 0, bgraph)
        cgraph = cgraph * index_select_ND(prev.vmask, 0, cgraph)
        return fnode, fmess, agraph, bgraph, cgraph, scope

    def apply_graph_mask(self, tensors, hgraph):
        fnode, fmess, agraph, bgraph, scope = tensors
        agraph = agraph * index_select_ND(hgraph.emask, 0, agraph)
        bgraph = bgraph * index_select_ND(hgraph.emask, 0, bgraph)
        return fnode, fmess, agraph, bgraph, scope

    def update_graph_mask(self, graph_batch, new_atoms, hgraph):
        new_atom_index = hgraph.vmask.new_tensor(new_atoms)
        hgraph.vmask.scatter_(0, new_atom_index, 1)

        new_atom_set = set(new_atoms)
        new_bonds = [] #new bonds are the subgraph induced by new_atoms
        for zid in new_atoms:
            for nid in graph_batch[zid]:
                if nid not in new_atom_set: continue
                new_bonds.append( graph_batch[zid][nid]['mess_idx'] )

        new_bond_index = hgraph.emask.new_tensor(new_bonds)
        if len(new_bonds) > 0:
            hgraph.emask.scatter_(0, new_bond_index, 1)
        return new_atom_index, new_bond_index

    def init_decoder_state(self, tree_batch, tree_tensors, src_root_vecs):
        batch_size = len(src_root_vecs)
        num_mess = len(tree_tensors[1])
        agraph = tree_tensors[2].clone()
        bgraph = tree_tensors[3].clone()

        for i,tup in enumerate(tree_tensors[-1]):
            root = tup[0]
            assert agraph[root,-1].item() == 0
            agraph[root,-1] = num_mess + i
            for v in tree_batch.successors(root):
                mess_idx = tree_batch[root][v]['mess_idx'] 
                assert bgraph[mess_idx,-1].item() == 0
                bgraph[mess_idx,-1] = num_mess + i

        new_tree_tensors = tree_tensors[:2] + [agraph, bgraph] + tree_tensors[4:]
        htree = HTuple()
        htree.mess = self.rnn_cell.get_init_state(tree_tensors[1], src_root_vecs)
        htree.emask = torch.cat( [bgraph.new_zeros(num_mess), bgraph.new_ones(batch_size)], dim=0 )

        return htree, new_tree_tensors

    def attention(self, src_vecs, batch_idx, queries, W_att):
        size = batch_idx.size()
        if batch_idx.dim() > 1:
            batch_idx = batch_idx.view(-1)
            queries = queries.view(-1, queries.size(-1))

        src_vecs = src_vecs.index_select(0, batch_idx)
        att_score = torch.bmm( src_vecs, W_att(queries).unsqueeze(-1) )
        att_vecs = F.softmax(att_score, dim=1) * src_vecs
        att_vecs = att_vecs.sum(dim=1)
        return att_vecs if len(size) == 1 else att_vecs.view(size[0], size[1], -1)

    def get_topo_score(self, src_tree_vecs, batch_idx, topo_vecs):
        if self.use_attention:
            topo_cxt = self.attention(src_tree_vecs, batch_idx, topo_vecs, self.A_topo)
        else:
            topo_cxt = src_tree_vecs.index_select(index=batch_idx, dim=0)
        return self.topoNN( torch.cat([topo_vecs, topo_cxt], dim=-1) ).squeeze(-1)

    def get_cls_score(self, src_tree_vecs, batch_idx, cls_vecs, cls_labs):
        if self.use_attention:
            cls_cxt = self.attention(src_tree_vecs, batch_idx, cls_vecs, self.A_cls)
        else:
            cls_cxt = src_tree_vecs.index_select(index=batch_idx, dim=0)

        cls_vecs = torch.cat([cls_vecs, cls_cxt], dim=-1)
        cls_scores = self.clsNN(cls_vecs)

        if cls_labs is None: #inference mode
            icls_scores = self.iclsNN(cls_vecs) #no masking
        else:
            vocab_masks = self.vocab.get_mask(cls_labs)
            icls_scores = self.iclsNN(cls_vecs) + vocab_masks #apply mask by log(x + mask): mask=0 or -INF
        return cls_scores, icls_scores

    def get_assm_score(self, src_graph_vecs, batch_idx, assm_vecs):
        if self.use_attention:
            assm_cxt = self.attention(src_graph_vecs, batch_idx, assm_vecs, self.A_assm)
        else:
            assm_cxt = index_select_ND(src_graph_vecs, 0, batch_idx)
        return (self.W_assm(assm_vecs) * assm_cxt).sum(dim=-1)

    def forward(self, src_mol_vecs, graphs, tensors, orders):
        batch_size = len(orders)
        tree_batch, graph_batch = graphs
        tree_tensors, graph_tensors = tensors
        inter_tensors = tree_tensors

        src_root_vecs, src_tree_vecs, src_graph_vecs = src_mol_vecs
        init_vecs = src_root_vecs if self.latent_size == self.hidden_size else self.W_root(src_root_vecs)

        htree, tree_tensors = self.init_decoder_state(tree_batch, tree_tensors, init_vecs)
        hinter = HTuple(
            mess = self.rnn_cell.get_init_state(inter_tensors[1]),
            emask = self.itensor.new_zeros(inter_tensors[1].size(0))
        )
        hgraph = HTuple(
            mess = self.rnn_cell.get_init_state(graph_tensors[1]),
            vmask = self.itensor.new_zeros(graph_tensors[0].size(0)),
            emask = self.itensor.new_zeros(graph_tensors[1].size(0))
        )
        
        all_topo_preds, all_cls_preds, all_assm_preds = [], [], []
        new_atoms = []
        tree_scope = tree_tensors[-1]
        for i in range(batch_size):
            root = tree_batch.nodes[ tree_scope[i][0] ]
            clab, ilab = self.vocab[ root['label'] ]
            all_cls_preds.append( (init_vecs[i], i, clab, ilab) ) #cluster prediction
            new_atoms.extend(root['cluster'])

        subgraph = self.update_graph_mask(graph_batch, new_atoms, hgraph)
        graph_tensors = self.hmpn.embed_graph(graph_tensors) + (graph_tensors[-1],) #preprocess graph tensors

        maxt = max([len(x) for x in orders])
        max_cls_size = max( [len(attr) * 2 for node,attr in tree_batch.nodes(data='cluster')] )

        for t in range(maxt):
            batch_list = [i for i in range(batch_size) if t < len(orders[i])]
            assert htree.emask[0].item() == 0 and hinter.emask[0].item() == 0 and hgraph.vmask[0].item() == 0 and hgraph.emask[0].item() == 0

            subtree = [], []
            for i in batch_list:
                xid, yid, tlab = orders[i][t]
                subtree[0].append(xid)
                if yid is not None:
                    mess_idx = tree_batch[xid][yid]['mess_idx']
                    subtree[1].append(mess_idx)

            subtree = htree.emask.new_tensor(subtree[0]), htree.emask.new_tensor(subtree[1]) 
            htree.emask.scatter_(0, subtree[1], 1)
            hinter.emask.scatter_(0, subtree[1], 1)

            cur_tree_tensors = self.apply_tree_mask(tree_tensors, htree, hgraph)
            cur_inter_tensors = self.apply_tree_mask(inter_tensors, hinter, hgraph)
            cur_graph_tensors = self.apply_graph_mask(graph_tensors, hgraph)
            htree, hinter, hgraph = self.hmpn(cur_tree_tensors, cur_inter_tensors, cur_graph_tensors, htree, hinter, hgraph, subtree, subgraph)

            new_atoms = []
            for i in batch_list:
                xid, yid, tlab = orders[i][t]
                all_topo_preds.append( (htree.node[xid], i, tlab) ) #topology prediction
                if yid is not None:
                    mess_idx = tree_batch[xid][yid]['mess_idx']
                    new_atoms.extend( tree_batch.nodes[yid]['cluster'] ) #NOTE: regardless of tlab = 0 or 1

                if tlab == 0: continue

                cls = tree_batch.nodes[yid]['smiles']
                clab, ilab = self.vocab[ tree_batch.nodes[yid]['label'] ]
                mess_idx = tree_batch[xid][yid]['mess_idx']
                hmess = self.rnn_cell.get_hidden_state(htree.mess)
                all_cls_preds.append( (hmess[mess_idx], i, clab, ilab) ) #cluster prediction using message
                
                inter_label = tree_batch.nodes[yid]['inter_label']
                inter_label = [ (pos, self.vocab[(cls, icls)][1]) for pos,icls in inter_label ]
                inter_size = self.vocab.get_inter_size(ilab)

                if len(tree_batch.nodes[xid]['cluster']) > 2: #uncertainty occurs only when previous cluster is a ring
                    nth_child = tree_batch[yid][xid]['label'] #must be yid -> xid (graph order labeling is different from tree)
                    cands = tree_batch.nodes[yid]['assm_cands']
                    icls = list(zip(*inter_label))[1]
                    cand_vecs = self.enum_attach(hgraph, cands, icls, nth_child)

                    if len(cand_vecs) < max_cls_size:
                        pad_len = max_cls_size - len(cand_vecs)
                        cand_vecs = F.pad(cand_vecs, (0,0,0,pad_len))

                    batch_idx = hgraph.emask.new_tensor( [i] * max_cls_size )
                    all_assm_preds.append( (cand_vecs, batch_idx, 0) ) #the label is always the first of assm_cands

            subgraph = self.update_graph_mask(graph_batch, new_atoms, hgraph)

        topo_vecs, batch_idx, topo_labels = zip_tensors(all_topo_preds)
        topo_scores = self.get_topo_score(src_tree_vecs, batch_idx, topo_vecs)
        topo_loss = self.topo_loss(topo_scores, topo_labels.float())
        topo_acc = get_accuracy_bin(topo_scores, topo_labels)

        cls_vecs, batch_idx, cls_labs, icls_labs = zip_tensors(all_cls_preds)
        cls_scores, icls_scores = self.get_cls_score(src_tree_vecs, batch_idx, cls_vecs, cls_labs)
        cls_loss = self.cls_loss(cls_scores, cls_labs) + self.icls_loss(icls_scores, icls_labs)
        cls_acc = get_accuracy(cls_scores, cls_labs)
        icls_acc = get_accuracy(icls_scores, icls_labs)

        if len(all_assm_preds) > 0:
            assm_vecs, batch_idx, assm_labels = zip_tensors(all_assm_preds)
            assm_scores = self.get_assm_score(src_graph_vecs, batch_idx, assm_vecs)
            assm_loss = self.assm_loss(assm_scores, assm_labels)
            assm_acc = get_accuracy_sym(assm_scores, assm_labels)
        else:
            assm_loss, assm_acc = 0, 1
        
        loss = (topo_loss + cls_loss + assm_loss) / batch_size
        return loss, cls_acc, icls_acc, topo_acc, assm_acc

    def enum_attach(self, hgraph, cands, icls, nth_child):
        cands = self.itensor.new_tensor(cands)
        icls_vecs = self.itensor.new_tensor(icls * len(cands))
        icls_vecs = self.E_assm( icls_vecs )

        nth_child = self.itensor.new_tensor([nth_child] * len(cands.view(-1)))
        order_vecs = self.E_order.index_select(0, nth_child)

        cand_vecs = hgraph.node.index_select(0, cands.view(-1))
        cand_vecs = torch.cat( [cand_vecs, icls_vecs, order_vecs], dim=-1 )
        cand_vecs = self.matchNN(cand_vecs)

        if len(icls) == 2:
            cand_vecs = cand_vecs.view(-1, 2, self.hidden_size).sum(dim=1)
        return cand_vecs

    def decode(self, src_mol_vecs, greedy=True, max_decode_step=100, beam=5):
        src_root_vecs, src_tree_vecs, src_graph_vecs = src_mol_vecs
        batch_size = len(src_root_vecs)

        tree_batch = IncTree(batch_size, node_fdim=2, edge_fdim=3)
        graph_batch = IncGraph(self.avocab, batch_size, node_fdim=self.hmpn.atom_size, edge_fdim=self.hmpn.atom_size + self.hmpn.bond_size)
        stack = [[] for i in range(batch_size)]

        init_vecs = src_root_vecs if self.latent_size == self.hidden_size else self.W_root(src_root_vecs)
        batch_idx = self.itensor.new_tensor(range(batch_size))
        cls_scores, icls_scores = self.get_cls_score(src_tree_vecs, batch_idx, init_vecs, None)
        root_cls = cls_scores.max(dim=-1)[1]
        icls_scores = icls_scores + self.vocab.get_mask(root_cls)
        root_cls, root_icls = root_cls.tolist(), icls_scores.max(dim=-1)[1].tolist()

        super_root = tree_batch.add_node() 
        for bid in range(batch_size):
            clab, ilab = root_cls[bid], root_icls[bid]
            root_idx = tree_batch.add_node( batch_idx.new_tensor([clab, ilab]) )
            tree_batch.add_edge(super_root, root_idx) 
            stack[bid].append(root_idx)

            root_smiles = self.vocab.get_ismiles(ilab)
            new_atoms, new_bonds, attached = graph_batch.add_mol(bid, root_smiles, [], 0)
            tree_batch.register_cgraph(root_idx, new_atoms, new_bonds, attached)
        
        #invariance: tree_tensors is equal to inter_tensors (but inter_tensor's init_vec is 0)
        tree_tensors = tree_batch.get_tensors()
        graph_tensors = graph_batch.get_tensors()

        htree = HTuple( mess = self.rnn_cell.get_init_state(tree_tensors[1]) )
        hinter = HTuple( mess = self.rnn_cell.get_init_state(tree_tensors[1]) )
        hgraph = HTuple( mess = self.rnn_cell.get_init_state(graph_tensors[1]) )
        h = self.rnn_cell.get_hidden_state(htree.mess)
        h[1 : batch_size + 1] = init_vecs #wiring root (only for tree, not inter)
        
        for t in range(max_decode_step):
            batch_list = [ bid for bid in range(batch_size) if len(stack[bid]) > 0 ]
            if len(batch_list) == 0: break

            batch_idx = batch_idx.new_tensor(batch_list)
            cur_tree_nodes = [stack[bid][-1] for bid in batch_list]
            subtree = batch_idx.new_tensor(cur_tree_nodes), batch_idx.new_tensor([])
            subgraph = batch_idx.new_tensor( tree_batch.get_cluster_nodes(cur_tree_nodes) ), batch_idx.new_tensor( tree_batch.get_cluster_edges(cur_tree_nodes) )

            htree, hinter, hgraph = self.hmpn(tree_tensors, tree_tensors, graph_tensors, htree, hinter, hgraph, subtree, subgraph)
            topo_scores = self.get_topo_score(src_tree_vecs, batch_idx, htree.node.index_select(0, subtree[0]))
            topo_scores = torch.sigmoid(topo_scores)
            if greedy:
                topo_preds = topo_scores.tolist()
            else:
                topo_preds = torch.bernoulli(topo_scores).tolist()

            new_mess = []
            expand_list = []
            for i,bid in enumerate(batch_list):
                if topo_preds[i] > 0.5 and tree_batch.can_expand(stack[bid][-1]):
                    expand_list.append( (len(new_mess), bid) )
                    new_node = tree_batch.add_node() #new node label is yet to be predicted
                    edge_feature = batch_idx.new_tensor( [stack[bid][-1], new_node, 0] ) #parent to child is 0
                    new_edge = tree_batch.add_edge(stack[bid][-1], new_node, edge_feature) 
                    stack[bid].append(new_node)
                    new_mess.append(new_edge)
                else:
                    child = stack[bid].pop()
                    if len(stack[bid]) > 0:
                        nth_child = tree_batch.graph.in_degree(stack[bid][-1]) #edge child -> father has not established
                        edge_feature = batch_idx.new_tensor( [child, stack[bid][-1], nth_child] )
                        new_edge = tree_batch.add_edge(child, stack[bid][-1], edge_feature)
                        new_mess.append(new_edge)

            subtree = subtree[0], batch_idx.new_tensor(new_mess)
            subgraph = [], []
            htree, hinter, hgraph = self.hmpn(tree_tensors, tree_tensors, graph_tensors, htree, hinter, hgraph, subtree, subgraph)
            cur_mess = self.rnn_cell.get_hidden_state(htree.mess).index_select(0, subtree[1])

            if len(expand_list) > 0:
                idx_in_mess, expand_list = zip(*expand_list)
                idx_in_mess = batch_idx.new_tensor( idx_in_mess )
                expand_idx = batch_idx.new_tensor( expand_list )
                forward_mess = cur_mess.index_select(0, idx_in_mess)
                cls_scores, icls_scores = self.get_cls_score(src_tree_vecs, expand_idx, forward_mess, None)
                scores, cls_topk, icls_topk = hier_topk(cls_scores, icls_scores, self.vocab, beam)
                if not greedy:
                    scores = torch.exp(scores) #score is output of log_softmax
                    shuf_idx = torch.multinomial(scores, beam, replacement=False).tolist()

            for i,bid in enumerate(expand_list):
                new_node, fa_node = stack[bid][-1], stack[bid][-2]
                success = False
                cls_beam = range(beam) if greedy else shuf_idx[i]
                for kk in cls_beam: #try until one is chemically valid
                    if success: break
                    clab, ilab = cls_topk[i][kk], icls_topk[i][kk]
                    node_feature = batch_idx.new_tensor( [clab, ilab] )
                    tree_batch.set_node_feature(new_node, node_feature)
                    smiles, ismiles = self.vocab.get_smiles(clab), self.vocab.get_ismiles(ilab)
                    fa_cluster, _, fa_used = tree_batch.get_cluster(fa_node)
                    inter_cands, anchor_smiles, attach_points = graph_batch.get_assm_cands(fa_cluster, fa_used, ismiles)

                    if len(inter_cands) == 0:
                        continue
                    elif len(inter_cands) == 1:
                        sorted_cands = [(inter_cands[0], 0)]
                        nth_child = 0
                    else:
                        nth_child = tree_batch.graph.in_degree(fa_node)
                        icls = [self.vocab[ (smiles,x) ][1] for x in anchor_smiles]
                        cands = inter_cands if len(attach_points) <= 2 else [ (x[0],x[-1]) for x in inter_cands ]
                        cand_vecs = self.enum_attach(hgraph, cands, icls, nth_child)

                        batch_idx = batch_idx.new_tensor( [bid] * len(inter_cands) )
                        assm_scores = self.get_assm_score(src_graph_vecs, batch_idx, cand_vecs).tolist()
                        sorted_cands = sorted( list(zip(inter_cands, assm_scores)), key = lambda x:x[1], reverse=True )

                    for inter_label,_ in sorted_cands:
                        inter_label = list(zip(inter_label, attach_points))
                        if graph_batch.try_add_mol(bid, ismiles, inter_label):
                            new_atoms, new_bonds, attached = graph_batch.add_mol(bid, ismiles, inter_label, nth_child)
                            tree_batch.register_cgraph(new_node, new_atoms, new_bonds, attached)
                            tree_batch.update_attached(fa_node, inter_label)
                            success = True
                            break

                if not success: #force backtrack
                    child = stack[bid].pop() #pop the dummy new_node which can't be added
                    nth_child = tree_batch.graph.in_degree(stack[bid][-1]) 
                    edge_feature = batch_idx.new_tensor( [child, stack[bid][-1], nth_child] )
                    new_edge = tree_batch.add_edge(child, stack[bid][-1], edge_feature)

                    child = stack[bid].pop() 
                    if len(stack[bid]) > 0:
                        nth_child = tree_batch.graph.in_degree(stack[bid][-1]) 
                        edge_feature = batch_idx.new_tensor( [child, stack[bid][-1], nth_child] )
                        new_edge = tree_batch.add_edge(child, stack[bid][-1], edge_feature)

        return graph_batch.get_mol()

