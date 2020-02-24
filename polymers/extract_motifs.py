import sys
from rdkit import Chem
from collections import Counter

def copy_atom(atom):
    new_atom = Chem.Atom(atom.GetSymbol())
    new_atom.SetFormalCharge(atom.GetFormalCharge())
    return new_atom

def find_clusters(mol):
    n_atoms = mol.GetNumAtoms()
    clusters = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        if not bond.IsInRing():
            clusters.append( (a1,a2) )

    ssr = [tuple(x) for x in Chem.GetSymmSSSR(mol)]
    clusters.extend(ssr)

    atom_cls = [[] for i in range(n_atoms)]
    for i in range(len(clusters)):
        for atom in clusters[i]:
            atom_cls[atom].append(i)

    return clusters, atom_cls

def find_fragments(mol):
    Chem.Kekulize(mol)
    new_mol = Chem.RWMol(mol)
    for atom in new_mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())

    for bond in mol.GetBonds():
        if bond.IsInRing(): continue
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()

        if a1.IsInRing() and a2.IsInRing():
            new_mol.RemoveBond(a1.GetIdx(), a2.GetIdx())

        elif a1.IsInRing() and a2.GetDegree() > 1:
            new_idx = new_mol.AddAtom(copy_atom(a1))
            new_mol.GetAtomWithIdx(new_idx).SetAtomMapNum(a1.GetIdx())
            new_mol.AddBond(new_idx, a2.GetIdx(), bond.GetBondType())
            new_mol.RemoveBond(a1.GetIdx(), a2.GetIdx())

        elif a2.IsInRing() and a1.GetDegree() > 1:
            new_idx = new_mol.AddAtom(copy_atom(a2))
            new_mol.GetAtomWithIdx(new_idx).SetAtomMapNum(a2.GetIdx())
            new_mol.AddBond(new_idx, a1.GetIdx(), bond.GetBondType())
            new_mol.RemoveBond(a1.GetIdx(), a2.GetIdx())
    
    new_mol = new_mol.GetMol()
    new_smiles = Chem.MolToSmiles(new_mol)

    hopts = []
    for fragment in new_smiles.split('.'):
        fmol = Chem.MolFromSmiles(fragment)
        indices = set([atom.GetAtomMapNum() for atom in fmol.GetAtoms()])
        for atom in fmol.GetAtoms():
            atom.SetAtomMapNum(0)
        fsmiles = Chem.MolToSmiles(fmol)
        hopts.append((fsmiles, indices))
    
    return hopts

counter = Counter()

for line in sys.stdin:
    smiles = line.strip("\r\n ")
    mol = Chem.MolFromSmiles(smiles)
    fragments = find_fragments(mol)
    
    for fsmiles, _ in fragments:
        counter[fsmiles] += 1

for fragment, cnt in counter.most_common():
    print(fragment, cnt)

