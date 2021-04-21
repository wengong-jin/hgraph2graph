import sys
from hgraph import *
from rdkit import Chem
from multiprocessing import Pool
from collections import Counter

def process(data):
    vocab = set()
    for line in data:
        s = line.strip("\r\n ")
        hmol = MolGraph(s)
        for node,attr in hmol.mol_tree.nodes(data=True):
            smiles = attr['smiles']
            vocab[attr['label']] += 1
            for i,s in attr['inter_label']:
                vocab[(smiles, s)] += 1
    return vocab

if __name__ == "__main__":

    data = [mol for line in sys.stdin for mol in line.split()[:2]]
    data = list(set(data))

    ncpu = 15
    batch_size = len(data) // ncpu + 1
    batches = [data[i : i + batch_size] for i in range(0, len(data), batch_size)]

    pool = Pool(ncpu)
    vocab_list = pool.map(process, batches)

    vocab = Counter()
    for c in vocab_list:
        vocab |= c

    for (x,y),c in vocab:
        print(x, y, c)
