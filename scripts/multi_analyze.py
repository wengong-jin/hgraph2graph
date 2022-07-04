import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num_decode", type=int, default=20)
parser.add_argument("--sim_delta", type=float, default=0.4)
parser.add_argument("--qed_delta", type=float, default=0.9)
parser.add_argument("--drd2_delta", type=float, default=0.5)
parser.add_argument("--cond", type=str, default="1,0,1,0")
args = parser.parse_args()

data = [line.split() for line in sys.stdin]
data = [(a, b, float(c), float(d), float(e)) for a, b, c, d, e in data]
n_mols = len(data) / args.num_decode
assert len(data) % args.num_decode == 0

if args.cond == "1,0,1,0":
    f_success = lambda x, y: x >= args.qed_delta and y >= args.drd2_delta
elif args.cond == "1,0,0,1":
    f_success = lambda x, y: x >= args.qed_delta and y < args.drd2_delta
elif args.cond == "0,1,1,0":
    f_success = lambda x, y: x < args.qed_delta and y >= args.drd2_delta
else:
    raise ValueError("condition not supported")

n_succ = 0.0
for i in range(0, len(data), args.num_decode):
    set_x = set([x[0] for x in data[i : i + args.num_decode]])
    assert len(set_x) == 1

    good = [
        (sim, qed, drd2)
        for _, _, sim, qed, drd2 in data[i : i + args.num_decode]
        if 1 > sim >= args.sim_delta and f_success(qed, drd2)
    ]
    if len(good) > 0:
        n_succ += 1

print("Evaluated on %d samples with condition %s" % (n_mols, args.cond))
print("success rate", n_succ / n_mols)
