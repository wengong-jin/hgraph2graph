import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

import math, random, sys
import numpy as np
import argparse

from agraph import *
import rdkit

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument('--test', required=True)
parser.add_argument('--atom_vocab', default=common_atom_vocab)
parser.add_argument('--model', required=True)

parser.add_argument('--num_decode', type=int, default=20)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--enum_root', action='store_true')

parser.add_argument('--rnn_type', type=str, default='LSTM')
parser.add_argument('--hidden_size', type=int, default=400)
parser.add_argument('--embed_size', type=int, default=400)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--latent_size', type=int, default=4)
parser.add_argument('--depth', type=int, default=20)
parser.add_argument('--diter', type=int, default=3)

args = parser.parse_args()
args.enum_root = True

args.test = [line.strip("\r\n ") for line in open(args.test)]

model = AtomVGNN(args).cuda()
model.load_state_dict(torch.load(args.model))
model.eval()

if args.enum_root:
    dataset = MolEnumRootDataset(args.test, args.atom_vocab, num_decode=args.num_decode)
else:
    dataset = MoleculeDataset(args.test, args.atom_vocab, batch_size=1)

loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x:x[0])
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

with torch.no_grad():
    for i,batch in enumerate(loader):
        smiles = args.test[i]
        new_mols = model.translate(batch[1], args.num_decode, args.enum_root)
        for k in range(args.num_decode):
            print(smiles, new_mols[k])

