import sys
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num_decode", type=int, default=20)
parser.add_argument("--delta", type=float, default=0.6)
args = parser.parse_args()

data = [line.split() for line in sys.stdin]
data = [(a, b, float(c), float(d)) for a, b, c, d in data]
n_mols = len(data) / args.num_decode
assert len(data) % args.num_decode == 0

all_logp = []

for i in range(0, len(data), args.num_decode):
    set_x = set([x[0] for x in data[i : i + args.num_decode]])
    assert len(set_x) == 1

    good = [
        (sim, logp)
        for _, y, sim, logp in data[i : i + args.num_decode]
        if 1 > sim >= args.delta and "." not in y
    ]
    if len(good) > 0:
        sim, logp = max(good, key=lambda x: x[1])
        all_logp.append(max(0, logp))
    else:
        all_logp.append(0.0)  # No improvement

assert len(all_logp) == n_mols
all_logp = np.array(all_logp)

print("Evaluated on %d samples" % (n_mols,))
print("average improvement", np.mean(all_logp), "stdev", np.std(all_logp))
