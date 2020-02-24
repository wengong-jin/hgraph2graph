import sys
from collections import defaultdict
from rdkit import Chem

d = defaultdict(list)
for line in sys.stdin:
    x,y = line.split()
    n = Chem.MolFromSmiles(x).GetNumAtoms()
    if n > 110 and x == y:
        print(x,y)
    
