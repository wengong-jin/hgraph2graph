import sys
import argparse
from hgraph import *
from rdkit import Chem
from multiprocessing import Pool


def process(data):
    vocab = set()
    for line in data:
        s = line.strip("\r\n ")
        hmol = MolGraph(s)
        for node, attr in hmol.mol_tree.nodes(data=True):
            smiles = attr["smiles"]
            vocab.add(attr["label"])
            for i, s in attr["inter_label"]:
                vocab.add((smiles, s))
    return vocab


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--ncpu", type=int, default=1)
    args = parser.parse_args()

    data = [mol for line in sys.stdin for mol in line.split()[:2]]
    data = list(set(data))

    batch_size = len(data) // args.ncpu + 1
    batches = [data[i : i + batch_size] for i in range(0, len(data), batch_size)]

    pool = Pool(args.ncpu)
    vocab_list = pool.map(process, batches)
    vocab = [(x, y) for vocab in vocab_list for x, y in vocab]
    vocab = list(set(vocab))

    for x, y in sorted(vocab):
        print(x, y)
