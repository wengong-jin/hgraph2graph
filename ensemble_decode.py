import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

import rdkit
import math, random, sys, os
import numpy as np
import argparse

from hgraph import *
from collections import defaultdict

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument('--test', required=True)
parser.add_argument('--vocab', required=True)
parser.add_argument('--atom_vocab', default=common_atom_vocab)
parser.add_argument('--model_dir', required=True)

parser.add_argument('--num_decode', type=int, default=100)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--min_similarity', type=float, default=0.4)
parser.add_argument('--max_epoch', type=int, default=10)

parser.add_argument('--rnn_type', type=str, default='LSTM')
parser.add_argument('--hidden_size', type=int, default=270)
parser.add_argument('--embed_size', type=int, default=270)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--latent_size', type=int, default=4)
parser.add_argument('--depthT', type=int, default=20)
parser.add_argument('--depthG', type=int, default=20)
parser.add_argument('--diterT', type=int, default=1)
parser.add_argument('--diterG', type=int, default=3)
parser.add_argument('--dropout', type=float, default=0.0)

args = parser.parse_args()
args.enum_root = True

args.test = [line.strip("\r\n ") for line in open(args.test)]
vocab = [x.strip("\r\n ").split() for x in open(args.vocab)] 
args.vocab = PairVocab(vocab) 

model = HierVGNN(args).cuda()

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

all_outcomes = defaultdict(list)
for fn in os.listdir(args.model_dir):
    if not fn.startswith('model'): continue
    epoch = int(fn.split('.')[1])
    if epoch > args.max_epoch or epoch == 0: continue

    torch.manual_seed(args.seed)
    fn = os.path.join(args.model_dir, fn)
    model.load_state_dict(torch.load(fn))
    model.eval()

    dataset = MolEnumRootDataset(args.test, args.vocab, args.atom_vocab)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x:x[0])

    with torch.no_grad():
        for i,batch in enumerate(loader):
            x = args.test[i]
            ylist = model.translate(batch[1], args.num_decode, args.enum_root)
            outcomes = [restore_stereo(x,y) for y in set(ylist)]
            outcomes = [(x,y,sim) for x,y,sim in outcomes if sim >= args.min_similarity]
            all_outcomes[x].extend(outcomes)

print('lead compound smiles,new compound smiles,similarity')
for x, outcomes in all_outcomes.items():
    outcomes = list(set(outcomes))
    for x,y,sim in outcomes:
        print('%s,%s,%.4f' % (x, y, sim))

