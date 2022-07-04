import sys
from props import *

for line in sys.stdin:
    x, y = line.split()[:2]
    if y == "None":
        y = None
    sim2D = similarity(x, y)
    try:
        qq = qed(y)
        print(x, y, sim2D, qq)
    except Exception as e:
        print(x, y, sim2D, 0.0)
