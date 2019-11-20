import torch
import rdkit
import rdkit.Chem as Chem
import networkx as nx
from agraph.chemutils import *
from agraph.nnutils import *
from collections import deque

add = lambda x,y : x + y if type(x) is int else (x[0] + y, x[1] + y)

class MolGraph(object):

    BOND_LIST = [None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC] 
    MAX_POS = 40

    def __init__(self, smiles):
        self.smiles = smiles
        self.mol = get_mol(smiles)
        self.mol_graph = self.build_mol_graph()
        self.order = self.get_bfs_order()

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

    def get_bfs_order(self):
        order = []
        visited = set([0])
        self.mol_graph.nodes[0]['pos'] = 0
        root = self.mol.GetAtomWithIdx(0)
        queue = deque([root])

        while len(queue) > 0:
            x = queue.popleft()
            x_idx = x.GetIdx()
            for y in x.GetNeighbors():
                y_idx = y.GetIdx()
                if y_idx in visited: continue

                frontier = [x_idx] + [a.GetIdx() for a in list(queue)]
                bonds = [0] * len(frontier)
                y_neis = set([z.GetIdx() for z in y.GetNeighbors()])

                for i,z_idx in enumerate(frontier):
                    if z_idx in y_neis: 
                        bonds[i] = self.mol_graph[y_idx][z_idx]['label']

                order.append( (x_idx, y_idx, frontier, bonds) )
                self.mol_graph.nodes[y_idx]['pos'] = min( MolGraph.MAX_POS - 1, len(visited) )
                visited.add( y_idx )
                queue.append(y)

            order.append( (x_idx, None, None, None) )

        return order
    
    @staticmethod
    def tensorize(mol_batch, avocab):
        mol_batch = [MolGraph(x) for x in mol_batch]
        graph_tensors, graph_batchG = MolGraph.tensorize_graph([x.mol_graph for x in mol_batch], avocab)
        graph_scope = graph_tensors[-1]

        add = lambda a,b : None if a is None else a + b
        add_list = lambda alist,b : None if alist is None else [a + b for a in alist]

        all_orders = []
        for i,hmol in enumerate(mol_batch):
            offset = graph_scope[i][0]
            order = [(x + offset, add(y, offset), add_list(z, offset), t) for x,y,z,t in hmol.order]
            all_orders.append(order)

        return graph_batchG, graph_tensors, all_orders

    @staticmethod
    def tensorize_graph(graph_batch, vocab):
        fnode,fmess = [None],[(0,0,0)] 
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
                fnode[v] = (vocab[attr], G.nodes[v]['pos'])
                agraph.append([])

            for u, v, attr in G.edges(data='label'):
                fmess.append( (u, v, attr) )
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
        fnode = torch.LongTensor(fnode)
        fmess = torch.LongTensor(fmess)
        agraph = create_pad_tensor(agraph)
        bgraph = create_pad_tensor(bgraph)
        return (fnode, fmess, agraph, bgraph, scope), nx.union_all(all_G)

if __name__ == "__main__":
    import sys
    from vocab import common_atom_vocab
    
    test_smiles = ['CCC(NC(=O)c1scnc1C1CC1)C(=O)N1CCOCC1','O=C1OCCC1Sc1nnc(-c2c[nH]c3ccccc23)n1C1CC1', 'CCN(C)S(=O)(=O)N1CCC(Nc2cccc(OC)c2)CC1', 'CC(=O)Nc1cccc(NC(C)c2ccccn2)c1', 'Cc1cc(-c2nc3sc(C4CC4)nn3c2C#N)ccc1Cl', 'CCOCCCNC(=O)c1cc(OC)ccc1Br', 'Cc1nc(-c2ccncc2)[nH]c(=O)c1CC(=O)NC1CCCC1', 'C#CCN(CC#C)C(=O)c1cc2ccccc2cc1OC(F)F', 'CCOc1ccc(CN2c3ccccc3NCC2C)cc1N', 'NC(=O)C1CCC(CNc2cc(-c3ccccc3)nc3ccnn23)CC1', 'CC1CCc2noc(NC(=O)c3cc(=O)c4ccccc4o3)c2C1', 'c1cc(-n2cnnc2)cc(-n2cnc3ccccc32)c1', 'Cc1ccc(-n2nc(C)cc2NC(=O)C2CC3C=CC2C3)nn1', 'O=c1ccc(c[nH]1)C1NCCc2ccc3OCCOc3c12']

    for s in test_smiles[:2]:
        print(s.strip("\r\n "))
        mol = Chem.MolFromSmiles(s)
        for a in mol.GetAtoms():
            a.SetAtomMapNum( a.GetIdx() )
        print(Chem.MolToSmiles(mol))

        hmol = MolGraph(s)
        print(hmol.order)

    print(MolGraph.tensorize(test_smiles[:2], common_atom_vocab))
