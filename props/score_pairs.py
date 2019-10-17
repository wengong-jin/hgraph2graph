import sys
import argparse
from props.properties import *

parser = argparse.ArgumentParser()
parser.add_argument('--property', type=str, required=True)
args = parser.parse_args()

if args.property == 'qed':
    f = qed
elif args.cond == 'logp':
    f = penalized_logp
elif args.cond == 'drd2':
    f = drd2
else:
    raise ValueError('Property not supported.')


for line in sys.stdin:
    x,y = line.split()[:2]
    if y == "None": y = None
    sim2D = similarity(x, y)
    try:
        print(x, y, sim2D, f(y))
    except Exception as e:
        print(x, y, sim2D, 0.0)
