import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num_decode", type=int, default=20)
parser.add_argument("--sim_delta", type=float, default=0.4)
parser.add_argument("--prop_delta", type=float, default=0.5)
args = parser.parse_args()

data = [line.split() for line in sys.stdin]
data = [(a, b, float(c), float(d)) for a, b, c, d in data]
n_mols = len(data) / args.num_decode
assert len(data) % args.num_decode == 0

n_succ = 0.0
for i in range(0, len(data), args.num_decode):
    set_x = set([x[0] for x in data[i : i + args.num_decode]])
    assert len(set_x) == 1

    good = [
        (sim, prop)
        for _, _, sim, prop in data[i : i + args.num_decode]
        if 1 > sim >= args.sim_delta and prop >= args.prop_delta
    ]
    if len(good) > 0:
        n_succ += 1

print("Evaluated on %d samples" % (n_mols,))
print("success rate", n_succ / n_mols)
