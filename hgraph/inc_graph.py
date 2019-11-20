import torch
import rdkit.Chem as Chem
import networkx as nx
from hgraph.mol_graph import MolGraph
from hgraph.chemutils import *
from collections import defaultdict

class IncBase(object):

    def __init__(self, batch_size, node_fdim, edge_fdim, max_nodes=100, max_edges=200, max_nb=12):
        self.max_nb = max_nb
        self.graph = nx.DiGraph()
        self.graph.add_node(0) #make sure node is 1 index
        self.edge_dict = {None : 0} #make sure edge is 1 index

        self.fnode = torch.zeros(max_nodes * batch_size, node_fdim).long().cuda()
        self.fmess = self.fnode.new_zeros(max_edges * batch_size, edge_fdim)
        self.agraph = self.fnode.new_zeros(max_edges * batch_size, max_nb)
        self.bgraph = self.fnode.new_zeros(max_edges * batch_size, max_nb)

    def add_node(self, feature=None):
        idx = len(self.graph)
        self.graph.add_node(idx)
        if feature is not None:
            self.fnode[idx, :len(feature)] = feature
        return idx

    def set_node_feature(self, idx, feature):
        self.fnode[idx, :len(feature)] = feature

    def can_expand(self, idx):
        return self.graph.in_degree(idx) < self.max_nb

    def add_edge(self, i, j, feature=None):
        if (i,j) in self.edge_dict: 
            return self.edge_dict[(i,j)]

        self.graph.add_edge(i, j)
        self.edge_dict[(i,j)] = idx = len(self.edge_dict)

        self.agraph[j, self.graph.in_degree(j) - 1] = idx
        if feature is not None:
            self.fmess[idx, :len(feature)] = feature

        in_edges = [self.edge_dict[(k,i)] for k in self.graph.predecessors(i) if k != j]
        self.bgraph[idx, :len(in_edges)] = self.fnode.new_tensor(in_edges)

        for k in self.graph.successors(j):
            if k == i: continue
            nei_idx = self.edge_dict[(j,k)]
            self.bgraph[nei_idx, self.graph.in_degree(j) - 2] = idx

        return idx


class IncTree(IncBase):

    def __init__(self, batch_size, node_fdim, edge_fdim, max_nodes=100, max_edges=200, max_nb=12, max_sub_nodes=20):
        super(IncTree, self).__init__(batch_size, node_fdim, edge_fdim, max_nodes, max_edges, max_nb)
        self.cgraph = self.fnode.new_zeros(max_nodes * batch_size, max_sub_nodes)

    def get_tensors(self):
        return self.fnode, self.fmess, self.agraph, self.bgraph, self.cgraph, None 

    def register_cgraph(self, i, nodes, edges, attached):
        self.cgraph[i, :len(nodes)] = self.fnode.new_tensor(nodes)
        self.graph.nodes[i]['cluster'] = nodes
        self.graph.nodes[i]['cluster_edges'] = edges
        self.graph.nodes[i]['attached'] = attached

    def update_attached(self, i, attached):
        if len(self.graph.nodes[i]['cluster']) > 1: 
            used = list(zip(*attached))[0]
            self.graph.nodes[i]['attached'].extend(used)

    def get_cluster(self, node_idx):
        cluster = self.graph.nodes[node_idx]['cluster']
        edges = self.graph.nodes[node_idx]['cluster_edges']
        used = self.graph.nodes[node_idx]['attached']
        return cluster, edges, used

    def get_cluster_nodes(self, node_list):
        return [ c for node_idx in node_list for c in self.graph.nodes[node_idx]['cluster'] ]

    def get_cluster_edges(self, node_list):
        return [ e for node_idx in node_list for e in self.graph.nodes[node_idx]['cluster_edges'] ]


class IncGraph(IncBase):

    def __init__(self, avocab, batch_size, node_fdim, edge_fdim, max_nodes=100, max_edges=300, max_nb=10):
        super(IncGraph, self).__init__(batch_size, node_fdim, edge_fdim, max_nodes, max_edges, max_nb)
        self.avocab = avocab
        self.mol = Chem.RWMol()
        self.mol.AddAtom( Chem.Atom('C') ) #make sure node is 1 index, consistent to self.graph
        self.fnode = self.fnode.float()
        self.fmess = self.fmess.float()
        self.batch = defaultdict(list)

    def get_mol(self):
        mol_list = [None] * len(self.batch)
        for batch_idx, batch_atoms in self.batch.items():
            mol = get_sub_mol(self.mol, batch_atoms)
            mol = sanitize(mol, kekulize=False)
            if mol is None: 
                mol_list[batch_idx] = None
            else:
                for atom in mol.GetAtoms():
                    atom.SetAtomMapNum(0)
                mol_list[batch_idx] = Chem.MolToSmiles(mol)
        return mol_list

    def get_tensors(self):
        return self.fnode, self.fmess, self.agraph, self.bgraph, None 

    def add_mol(self, batch_idx, smiles, inter_label, nth_child):
        emol = get_mol(smiles)
        atom_map = {y : x for x,y in inter_label}
        new_atoms, new_bonds, attached = [], [], []

        for atom in emol.GetAtoms(): #atoms must be inserted in order given by emol.GetAtoms() (for rings assembly)
            if atom.GetIdx() in atom_map: 
                idx = atom_map[atom.GetIdx()]
                new_atoms.append(idx)
                attached.append(idx)
            else:
                new_atom = copy_atom(atom)
                new_atom.SetAtomMapNum( batch_idx ) 
                idx = self.mol.AddAtom( new_atom )
                assert idx == self.add_node( self.get_atom_feature(new_atom) ) #mol and nx graph must have the same indexing
                atom_map[atom.GetIdx()] = idx
                new_atoms.append(idx)
                self.batch[batch_idx].append(idx)
                if atom.GetAtomMapNum() > 0: attached.append(idx)

        for bond in emol.GetBonds():
            a1 = atom_map[bond.GetBeginAtom().GetIdx()]
            a2 = atom_map[bond.GetEndAtom().GetIdx()]
            if a1 == a2: continue
            bond_type = bond.GetBondType()
            existing_bond = self.mol.GetBondBetweenAtoms(a1, a2)
            if existing_bond is None:
                self.mol.AddBond(a1, a2, bond_type)
                self.add_edge(a1, a2, self.get_mess_feature(bond.GetBeginAtom(), bond_type, nth_child if a2 in attached else 0) ) #only child to father node (in intersection) have non-zero nth_child
                self.add_edge(a2, a1, self.get_mess_feature(bond.GetEndAtom(), bond_type, nth_child if a1 in attached else 0) ) 
            else:
                attached.extend( [(a1,a2),(a2,a1)] )
            new_bonds.extend( [ self.edge_dict[(a1,a2)], self.edge_dict[(a2,a1)] ] )

        if emol.GetNumAtoms() == 1: #singletons always attached = []
            attached = []
        return new_atoms, new_bonds, attached

    #validity check function
    def try_add_mol(self, batch_idx, smiles, inter_label):
        emol = get_mol(smiles)
        for x,y in inter_label:
            if not atom_equal(self.mol.GetAtomWithIdx(x), emol.GetAtomWithIdx(y)):
                return False

        atom_map = {y : x for x,y in inter_label}
        new_atoms, new_bonds = [], []

        for atom in emol.GetAtoms(): #atoms must be inserted in order given by emol.GetAtoms() (for rings assembly)
            if atom.GetIdx() not in atom_map: 
                new_atom = copy_atom(atom)
                new_atom.SetAtomMapNum( batch_idx ) 
                idx = self.mol.AddAtom( new_atom )
                atom_map[atom.GetIdx()] = idx
                new_atoms.append(idx)

        valid = True
        for bond in emol.GetBonds():
            a1 = atom_map[bond.GetBeginAtom().GetIdx()]
            a2 = atom_map[bond.GetEndAtom().GetIdx()]
            if a1 == a2: #self loop must be an error
                valid = False
                break
            bond_type = bond.GetBondType()
            if self.mol.GetBondBetweenAtoms(a1, a2) is None: #later maybe check bond type match
                self.mol.AddBond(a1, a2, bond_type)
                new_bonds.append( (a1,a2) )

        if valid: 
            tmp_mol = get_sub_mol(self.mol, self.batch[batch_idx] + new_atoms)
            tmp_mol = sanitize(tmp_mol, kekulize=False)
        
        #revert trial
        for a1,a2 in new_bonds:
            self.mol.RemoveBond(a1, a2)
        for atom in sorted(new_atoms, reverse=True): 
            self.mol.RemoveAtom(atom)

        return valid and (tmp_mol is not None)

    def get_atom_feature(self, atom):
        f = torch.zeros(self.avocab.size())
        symbol, charge = atom.GetSymbol(), atom.GetFormalCharge()
        f[ self.avocab[(symbol,charge)] ] = 1
        return f.cuda()

    def get_mess_feature(self, atom, bond_type, nth_child):
        f1 = torch.zeros(self.avocab.size())
        f2 = torch.zeros(len(MolGraph.BOND_LIST))
        f3 = torch.zeros(MolGraph.MAX_POS)
        symbol, charge = atom.GetSymbol(), atom.GetFormalCharge()
        f1[ self.avocab[(symbol,charge)] ] = 1
        f2[ MolGraph.BOND_LIST.index(bond_type) ] = 1
        f3[ nth_child ] = 1
        return torch.cat( [f1,f2,f3], dim=-1 ).cuda()

    def get_assm_cands(self, cluster, used, smiles):
        emol = get_mol(smiles)
        if emol.GetNumAtoms() == 1:
            attach_points = [0]
        else:
            attach_points = [atom.GetIdx() for atom in emol.GetAtoms() if atom.GetAtomMapNum() > 0]

        inter_size = len(attach_points)
        idxfunc = lambda x:x.GetIdx()
        anchors = attach_points

        if inter_size == 1:
            anchor_smiles = [smiles]
        elif inter_size == 2:
            anchor_smiles = [get_anchor_smiles(emol, a, idxfunc) for a in anchors]
        else:
            anchors = [a for a in attach_points if is_anchor(emol.GetAtomWithIdx(a), [0])] #all attach points are labeled with 1
            attach_points = [a for a in attach_points if a not in anchors]
            attach_points = [anchors[0]] + attach_points + [anchors[1]] #force the attach_points to be a chain like anchor ... anchor
            anchor_smiles = [get_anchor_smiles(emol, a, idxfunc) for a in anchors]

        assert len(anchors) <= 2

        if inter_size == 1:
            cands = [ [x] for x in cluster if x not in used ]

        elif anchor_smiles[0] == anchor_smiles[1]:
            cluster2 = cluster + cluster
            cands = [cluster2[i : i + inter_size] for i in range(len(cluster))] #not pairs if inter_size >= 3
            cands = [c for c in cands if (c[0], c[-1]) not in used
                    and bond_match(self.mol, c[0], c[-1], emol, attach_points[0], attach_points[-1]) ] #weak matching
        else: 
            cluster2 = cluster + cluster
            cands = [cluster2[i : i + inter_size] for i in range(len(cluster))]
            cluster2 = cluster2[::-1]
            cands += [cluster2[i : i + inter_size] for i in range(len(cluster))]
            cands = [c for c in cands if (c[0], c[-1]) not in used
                    and bond_match(self.mol, c[0], c[-1], emol, attach_points[0], attach_points[-1]) ] #weak matching

        return cands, anchor_smiles, attach_points


