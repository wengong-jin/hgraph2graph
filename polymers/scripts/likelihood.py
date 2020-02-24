import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

import math, random, sys
import numpy as np
import argparse
from tqdm import tqdm

from poly_hgraph import *
import rdkit

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument('--test', required=True)
parser.add_argument('--vocab', required=True)
parser.add_argument('--model', required=True)
parser.add_argument('--atom_vocab', default=common_atom_vocab)

parser.add_argument('--rnn_type', type=str, default='LSTM')
parser.add_argument('--hidden_size', type=int, default=240)
parser.add_argument('--embed_size', type=int, default=240)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--latent_size', type=int, default=8)
parser.add_argument('--depth', type=str, default="(20,20,20)")
parser.add_argument('--diter', type=str, default="(1,3,3)")
parser.add_argument('--dropout', type=float, default=0.0)

args = parser.parse_args()
args.depth = eval(args.depth)
args.diter = eval(args.diter)

vocab = [x.strip("\r\n ").split() for x in open(args.vocab)] 
args.vocab = PairVocab(vocab)

model = HierVAE(args).cuda()
model.load_state_dict(torch.load(args.model))
model.eval()

with open(args.test) as f:
    testdata = [line.strip("\r\n ") for line in f]

dataset = MoleculeDataset(testdata, args.vocab, args.atom_vocab, args.batch_size)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x:x[0], drop_last=False)

total_nll, total_kl = 0, 0
total_mol = 0
with torch.no_grad():
    for batch in tqdm(dataloader):
        total_mol += len(batch[-1])
        loss, kl_div, wacc, iacc, tacc, sacc = model(*batch, beta=0, perturb_z=False)
        total_nll += loss.item() * len(batch[-1])
        total_kl += kl_div * len(batch[-1])
        if args.batch_size == 1:
            print(testdata[total_mol - 1], loss.item() + kl_div)

total_nll /= total_mol
total_kl /= total_mol
print('NLL:', total_nll, 'KL:', total_kl, 'ELBO:', total_nll + total_kl)
