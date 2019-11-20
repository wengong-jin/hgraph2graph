import torch
import os, random, gc

from rdkit import Chem
from torch.utils.data import Dataset
from agraph.chemutils import get_leaves
from agraph.mol_graph import MolGraph

class MoleculeDataset(Dataset):

    def __init__(self, data, avocab, batch_size):
        self.batches = [data[i : i + batch_size] for i in range(0, len(data), batch_size)]
        self.avocab = avocab

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        return MolGraph.tensorize(self.batches[idx], self.avocab)

class MolEnumRootDataset(Dataset):

    def __init__(self, data, avocab, num_decode):
        self.batches = data
        self.avocab = avocab
        self.num_decode = num_decode

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        mol = Chem.MolFromSmiles(self.batches[idx])
        leaves = get_leaves(mol)
        smiles_list = [Chem.MolToSmiles(mol, rootedAtAtom=i, isomericSmiles=False) for i in leaves]
        smiles_list = list( set(smiles_list) )
        while len(smiles_list) < self.num_decode:
            smiles_list = smiles_list + smiles_list
        smiles_list = smiles_list[:self.num_decode]
        return MolGraph.tensorize(smiles_list, self.avocab)

class MolPairDataset(Dataset):

    def __init__(self, data, avocab, batch_size):
        self.batches = [data[i : i + batch_size] for i in range(0, len(data), batch_size)]
        self.avocab = avocab

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        x, y = list(zip(*self.batches[idx]))
        x = MolGraph.tensorize(x, self.avocab)[:-1] #no need of order for x
        y = MolGraph.tensorize(y, self.avocab)
        return x + y

class CondPairDataset(Dataset):

    def __init__(self, data, avocab, batch_size):
        self.batches = [data[i : i + batch_size] for i in range(0, len(data), batch_size)]
        self.avocab = avocab

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        x, y, cond = list(zip(*self.batches[idx]))
        cond = [map(int, c.split(',')) for c in cond]
        cond = torch.tensor(cond).float()
        x = MolGraph.tensorize(x, self.avocab)[:-1] #no need of order for x
        y = MolGraph.tensorize(y, self.avocab)
        return x + y + (cond,)


