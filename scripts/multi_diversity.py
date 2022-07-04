import sys
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--num_decode", type=int, default=20)
parser.add_argument("--sim_delta", type=float, default=0.4)
parser.add_argument("--qed_delta", type=float, default=0.9)
parser.add_argument("--drd2_delta", type=float, default=0.5)
parser.add_argument("--cond", type=str, default="1,0,1,0")

args = parser.parse_args()

if args.cond == "1,0,1,0":
    f_success = lambda x, y: x >= args.qed_delta and y >= args.drd2_delta
elif args.cond == "1,0,0,1":
    f_success = lambda x, y: x >= args.qed_delta and y < args.drd2_delta
elif args.cond == "0,1,1,0":
    f_success = lambda x, y: x < args.qed_delta and y >= args.drd2_delta
else:
    raise ValueError("condition not supported")

data = [line.split() for line in sys.stdin]
data = [(a, b, float(c), float(d), float(e)) for a, b, c, d, e in data]


def convert(x):
    return None if x == "None" else x


all_div = []
n_succ = 0
for i in range(0, len(data), args.num_decode):
    set_x = set([x[0] for x in data[i : i + args.num_decode]])
    assert len(set_x) == 1

    good = [
        convert(y)
        for x, y, sim, qed, drd2 in data[i : i + args.num_decode]
        if 1 > sim >= args.sim_delta and f_success(qed, drd2)
    ]
    if len(good) == 0:
        continue

    good = list(set(good))
    if len(good) == 1:
        all_div.append(0.0)
        continue
    n_succ += 1

    div = 0.0
    tot = 0
    for i in range(len(good)):
        for j in range(i + 1, len(good)):
            div += 1 - similarity(good[i], good[j])
            tot += 1
    div /= tot
    all_div.append(div)

all_div = np.array(all_div)
print(np.mean(all_div), np.std(all_div))
print(n_succ)
