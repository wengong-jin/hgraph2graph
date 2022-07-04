from multiprocessing import Pool
import math, random, sys
import pickle
import argparse
from functools import partial
import torch
import numpy

from poly_hgraph import MolGraph, common_atom_vocab, PairVocab
import rdkit


def to_numpy(tensors):
    convert = lambda x: x.numpy() if type(x) is torch.Tensor else x
    a, b, c = tensors
    b = [convert(x) for x in b[0]], [convert(x) for x in b[1]]
    return a, b, c


def tensorize(mol_batch, vocab):
    x = MolGraph.tensorize(mol_batch, vocab, common_atom_vocab)
    return to_numpy(x)


if __name__ == "__main__":
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--ncpu", type=int, default=4)
    args = parser.parse_args()

    with open(args.vocab) as f:
        vocab = [x.strip("\r\n ").split() for x in f]
    MolGraph.load_fragments([x[0] for x in vocab if eval(x[-1])])
    args.vocab = PairVocab([(x, y) for x, y, _ in vocab], cuda=False)

    pool = Pool(args.ncpu)
    random.seed(1)

    with open(args.train) as f:
        data = [line.strip("\r\n ").split()[0] for line in f]

    random.shuffle(data)

    batches = [
        data[i : i + args.batch_size] for i in range(0, len(data), args.batch_size)
    ]
    func = partial(tensorize, vocab=args.vocab)
    all_data = pool.map(func, batches)
    num_splits = len(all_data) // 1000

    le = (len(all_data) + num_splits - 1) // num_splits

    for split_id in range(num_splits):
        st = split_id * le
        sub_data = all_data[st : st + le]

        with open("tensors-%d.pkl" % split_id, "wb") as f:
            pickle.dump(sub_data, f, pickle.HIGHEST_PROTOCOL)
