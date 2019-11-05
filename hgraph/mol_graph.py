import torch
import rdkit
import rdkit.Chem as Chem
import networkx as nx
from hgraph.chemutils import *
from hgraph.nnutils import *

add = lambda x,y : x + y if type(x) is int else (x[0] + y, x[1] + y)

class MolGraph(object):

    BOND_LIST = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC] 
    MAX_POS = 20

    def __init__(self, smiles):
        self.smiles = smiles
        self.mol = get_mol(smiles)

        self.mol_graph = self.build_mol_graph()
        self.clusters, self.atom_cls = self.find_clusters()
        self.mol_tree = self.tree_decomp()
        self.order = self.label_tree()

    def find_clusters(self):
        mol = self.mol
        n_atoms = mol.GetNumAtoms()
        if n_atoms == 1: #special case
            return [(0,)], [[0]]

        clusters = []
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            if not bond.IsInRing():
                clusters.append( (a1,a2) )

        ssr = [tuple(x) for x in Chem.GetSymmSSSR(mol)]
        clusters.extend(ssr)

        if 0 not in clusters[0]: #root is not node[0]
            for i,cls in enumerate(clusters):
                if 0 in cls:
                    clusters = [clusters[i]] + clusters[:i] + clusters[i+1:]
                    #clusters[i], clusters[0] = clusters[0], clusters[i]
                    break

        atom_cls = [[] for i in range(n_atoms)]
        for i in range(len(clusters)):
            for atom in clusters[i]:
                atom_cls[atom].append(i)

        return clusters, atom_cls

    def tree_decomp(self):
        clusters = self.clusters
        graph = nx.empty_graph( len(clusters) )
        for atom, nei_cls in enumerate(self.atom_cls):
            if len(nei_cls) <= 1: continue
            bonds = [c for c in nei_cls if len(clusters[c]) == 2]
            rings = [c for c in nei_cls if len(clusters[c]) > 4] #need to change to 2

            if len(nei_cls) > 2 and len(bonds) >= 2:
                clusters.append([atom])
                c2 = len(clusters) - 1
                graph.add_node(c2)
                for c1 in nei_cls:
                    graph.add_edge(c1, c2, weight = 100)

            elif len(rings) > 2: #Bee Hives, len(nei_cls) > 2 
                clusters.append([atom]) #temporary value, need to change
                c2 = len(clusters) - 1
                graph.add_node(c2)
                for c1 in nei_cls:
                    graph.add_edge(c1, c2, weight = 100)
            else:
                for i,c1 in enumerate(nei_cls):
                    for c2 in nei_cls[i + 1:]:
                        inter = set(clusters[c1]) & set(clusters[c2])
                        graph.add_edge(c1, c2, weight = len(inter))

        n, m = len(graph.nodes), len(graph.edges)
        assert n - m <= 1 #must be connected
        return graph if n - m == 1 else nx.maximum_spanning_tree(graph)

    def label_tree(self):
        def dfs(order, pa, prev_sib, x, fa):
            pa[x] = fa 
            sorted_child = sorted([ y for y in self.mol_tree[x] if y != fa ]) #better performance with fixed order
            for idx,y in enumerate(sorted_child):
                self.mol_tree[x][y]['label'] = 0 
                self.mol_tree[y][x]['label'] = idx + 1 #position encoding
                prev_sib[y] = sorted_child[:idx] 
                prev_sib[y] += [x, fa] if fa >= 0 else [x]
                order.append( (x,y,1) )
                dfs(order, pa, prev_sib, y, x)
                order.append( (y,x,0) )

        order, pa = [], {}
        self.mol_tree = nx.DiGraph(self.mol_tree)
        prev_sib = [[] for i in range(len(self.clusters))]
        dfs(order, pa, prev_sib, 0, -1)

        order.append( (0, None, 0) ) #last backtrack at root
        
        mol = get_mol(self.smiles)
        for a in mol.GetAtoms():
            a.SetAtomMapNum( a.GetIdx() + 1 )

        tree = self.mol_tree
        for i,cls in enumerate(self.clusters):
            inter_atoms = set(cls) & set(self.clusters[pa[i]]) if pa[i] >= 0 else set([0])
            cmol, inter_label = get_inter_label(mol, cls, inter_atoms)
            tree.nodes[i]['ismiles'] = ismiles = get_smiles(cmol)
            tree.nodes[i]['inter_label'] = inter_label
            tree.nodes[i]['smiles'] = smiles = get_smiles(set_atommap(cmol))
            tree.nodes[i]['label'] = (smiles, ismiles) if len(cls) > 1 else (smiles, smiles)
            tree.nodes[i]['cluster'] = cls 
            tree.nodes[i]['assm_cands'] = []

            if pa[i] >= 0 and len(self.clusters[ pa[i] ]) > 2: #uncertainty occurs in assembly
                hist = [a for c in prev_sib[i] for a in self.clusters[c]] 
                pa_cls = self.clusters[ pa[i] ]
                tree.nodes[i]['assm_cands'] = get_assm_cands(mol, hist, inter_label, pa_cls, len(inter_atoms)) 

                child_order = tree[i][pa[i]]['label']
                diff = set(cls) - set(pa_cls)
                for fa_atom in inter_atoms:
                    for ch_atom in self.mol_graph[fa_atom]:
                        if ch_atom in diff:
                            label = self.mol_graph[ch_atom][fa_atom]['label']
                            if type(label) is int: #in case one bond is assigned multiple times
                                self.mol_graph[ch_atom][fa_atom]['label'] = (label, child_order)
        return order
       
    def build_mol_graph(self):
        mol = self.mol
        graph = nx.DiGraph(Chem.rdmolops.GetAdjacencyMatrix(mol))
        for atom in mol.GetAtoms():
            graph.nodes[atom.GetIdx()]['label'] = (atom.GetSymbol(), atom.GetFormalCharge())

        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            btype = MolGraph.BOND_LIST.index( bond.GetBondType() )
            graph[a1][a2]['label'] = btype
            graph[a2][a1]['label'] = btype

        return graph
    
    @staticmethod
    def tensorize(mol_batch, vocab, avocab):
        mol_batch = [MolGraph(x) for x in mol_batch]
        tree_tensors, tree_batchG = MolGraph.tensorize_graph([x.mol_tree for x in mol_batch], vocab)
        graph_tensors, graph_batchG = MolGraph.tensorize_graph([x.mol_graph for x in mol_batch], avocab)
        tree_scope = tree_tensors[-1]
        graph_scope = graph_tensors[-1]

        max_cls_size = max( [len(c) for x in mol_batch for c in x.clusters] )
        cgraph = torch.zeros(len(tree_batchG) + 1, max_cls_size).int()
        for v,attr in tree_batchG.nodes(data=True):
            bid = attr['batch_id']
            offset = graph_scope[bid][0]
            tree_batchG.nodes[v]['inter_label'] = inter_label = [(x + offset, y) for x,y in attr['inter_label']]
            tree_batchG.nodes[v]['cluster'] = cls = [x + offset for x in attr['cluster']]
            tree_batchG.nodes[v]['assm_cands'] = [add(x, offset) for x in attr['assm_cands']]
            cgraph[v, :len(cls)] = torch.IntTensor(cls)

        all_orders = []
        for i,hmol in enumerate(mol_batch):
            offset = tree_scope[i][0]
            order = [(x + offset, y + offset, z) for x,y,z in hmol.order[:-1]] + [(hmol.order[-1][0] + offset, None, 0)]
            all_orders.append(order)

        tree_tensors = tree_tensors[:4] + (cgraph, tree_scope)
        return (tree_batchG, graph_batchG), (tree_tensors, graph_tensors), all_orders

    @staticmethod
    def tensorize_graph(graph_batch, vocab):
        fnode,fmess = [None],[(0,0,0,0)] 
        agraph,bgraph = [[]], [[]] 
        scope = []
        edge_dict = {}
        all_G = []

        for bid,G in enumerate(graph_batch):
            offset = len(fnode)
            scope.append( (offset, len(G)) )
            G = nx.convert_node_labels_to_integers(G, first_label=offset)
            all_G.append(G)
            fnode.extend( [None for v in G.nodes] )

            for v, attr in G.nodes(data='label'):
                G.nodes[v]['batch_id'] = bid
                fnode[v] = vocab[attr]
                agraph.append([])

            for u, v, attr in G.edges(data='label'):
                if type(attr) is tuple:
                    fmess.append( (u, v, attr[0], attr[1]) )
                else:
                    fmess.append( (u, v, attr, 0) )
                edge_dict[(u, v)] = eid = len(edge_dict) + 1
                G[u][v]['mess_idx'] = eid
                agraph[v].append(eid)
                bgraph.append([])

            for u, v in G.edges:
                eid = edge_dict[(u, v)]
                for w in G.predecessors(u):
                    if w == v: continue
                    bgraph[eid].append( edge_dict[(w, u)] )

        fnode[0] = fnode[1]
        fnode = torch.IntTensor(fnode)
        fmess = torch.IntTensor(fmess)
        agraph = create_pad_tensor(agraph)
        bgraph = create_pad_tensor(bgraph)
        return (fnode, fmess, agraph, bgraph, scope), nx.union_all(all_G)

if __name__ == "__main__":
    import sys
    
    test_smiles = ['CCC(NC(=O)c1scnc1C1CC1)C(=O)N1CCOCC1','O=C1OCCC1Sc1nnc(-c2c[nH]c3ccccc23)n1C1CC1', 'CCN(C)S(=O)(=O)N1CCC(Nc2cccc(OC)c2)CC1', 'CC(=O)Nc1cccc(NC(C)c2ccccn2)c1', 'Cc1cc(-c2nc3sc(C4CC4)nn3c2C#N)ccc1Cl', 'CCOCCCNC(=O)c1cc(OC)ccc1Br', 'Cc1nc(-c2ccncc2)[nH]c(=O)c1CC(=O)NC1CCCC1', 'C#CCN(CC#C)C(=O)c1cc2ccccc2cc1OC(F)F', 'CCOc1ccc(CN2c3ccccc3NCC2C)cc1N', 'NC(=O)C1CCC(CNc2cc(-c3ccccc3)nc3ccnn23)CC1', 'CC1CCc2noc(NC(=O)c3cc(=O)c4ccccc4o3)c2C1', 'c1cc(-n2cnnc2)cc(-n2cnc3ccccc32)c1', 'Cc1ccc(-n2nc(C)cc2NC(=O)C2CC3C=CC2C3)nn1', 'O=c1ccc(c[nH]1)C1NCCc2ccc3OCCOc3c12']

    for s in sys.stdin:#test_smiles:
        print(s.strip("\r\n "))
        #mol = Chem.MolFromSmiles(s)
        #for a in mol.GetAtoms():
        #    a.SetAtomMapNum( a.GetIdx() )
        #print(Chem.MolToSmiles(mol))

        hmol = MolGraph(s)
        print(hmol.clusters)
        #print(list(hmol.mol_tree.edges))
        print(nx.get_node_attributes(hmol.mol_tree, 'label'))
        #print(nx.get_node_attributes(hmol.mol_tree, 'inter_label'))
        #print(nx.get_node_attributes(hmol.mol_tree, 'assm_cands'))
        #print(hmol.order)
