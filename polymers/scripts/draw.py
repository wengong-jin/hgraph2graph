import rdkit
import random
import rdkit.Chem as Chem
from rdkit.Chem import Draw
import sys

smiles = [line.strip("\r\n ") for line in sys.stdin]
smiles = sorted(smiles, key=lambda x:len(x), reverse=True)
mols = [Chem.MolFromSmiles(s) for s in smiles[300:400]]
random.shuffle(mols)
img = Draw.MolsToGridImage(mols[:36], molsPerRow=4, subImgSize=(500,300), useSVG=True)
print(img)

