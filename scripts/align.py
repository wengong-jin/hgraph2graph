from rdkit import Chem
import sys
import Levenshtein
from multiprocessing import Pool


def get_leaves(mol):
    leaf_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetDegree() == 1]

    clusters = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        if not bond.IsInRing():
            clusters.append(set([a1, a2]))

    rings = [set(x) for x in Chem.GetSymmSSSR(mol)]
    clusters.extend(rings)

    leaf_rings = []
    for r in rings:
        inters = [c for c in clusters if r != c and len(r & c) > 0]
        if len(inters) == 1:
            leaf_rings.extend([i for i in r if mol.GetAtomWithIdx(i).GetDegree() == 2])

    return leaf_atoms + leaf_rings


def align(xy_tuple):
    x, y = xy_tuple
    xmol, ymol = Chem.MolFromSmiles(x), Chem.MolFromSmiles(y)
    x = Chem.MolToSmiles(xmol, isomericSmiles=False)
    xmol = Chem.MolFromSmiles(x)

    xleaf = get_leaves(xmol)
    yleaf = get_leaves(ymol)

    best_i, best_j = 0, 0
    best = 1000000
    for i in xleaf:
        for j in yleaf:
            new_x = Chem.MolToSmiles(xmol, rootedAtAtom=i, isomericSmiles=False)
            new_y = Chem.MolToSmiles(ymol, rootedAtAtom=j, isomericSmiles=False)
            le = min(len(new_x), len(new_y)) // 2
            dist = Levenshtein.distance(new_x[:le], new_y[:le])
            if dist < best:
                best_i, best_j = i, j
                best = dist

    return (
        Chem.MolToSmiles(xmol, rootedAtAtom=best_i, isomericSmiles=False),
        Chem.MolToSmiles(ymol, rootedAtAtom=best_j, isomericSmiles=False),
    )


if __name__ == "__main__":
    data = [line.split()[:2] for line in sys.stdin]
    pool = Pool(30)
    aligned_data = pool.map(align, data)
    for x, y in aligned_data:
        print(x, y)
