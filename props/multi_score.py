import sys
from props.properties import *

for line in sys.stdin:
    x,y = line.split()
    if y == "None": y = None
    sim2D = similarity(x, y)
    try:
        qed_y = qed(y)
        drd2_y = drd2(y)
        print(x, y, sim2D, qed_y, drd2_y)
    except Exception as e:
        print(x, y, sim2D, 0, 0)
