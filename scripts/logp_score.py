import sys
from props import *

for line in sys.stdin:
    x, y = line.split()[:2]
    if y == "None":
        y = None
    sim = similarity(x, y)
    try:
        prop = penalized_logp(y) - penalized_logp(x)
        print(x, y, sim, prop)
    except Exception as e:
        print(x, y, sim, 0.0)
