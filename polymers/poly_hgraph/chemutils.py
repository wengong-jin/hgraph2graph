import rdkit
import rdkit.Chem as Chem
from collections import defaultdict

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

idxfunc = lambda a : a.GetAtomMapNum() - 1

def set_atommap(mol, num=0):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(num)
    return mol

def get_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None: Chem.Kekulize(mol)
    return mol

def get_smiles(mol):
    return Chem.MolToSmiles(mol, kekuleSmiles=True)

def sanitize(mol, kekulize=True):
    try:
        smiles = get_smiles(mol) if kekulize else Chem.MolToSmiles(mol)
        mol = get_mol(smiles) if kekulize else Chem.MolFromSmiles(smiles)
    except:
        mol = None
    return mol

def is_aromatic_ring(mol):
    if mol.GetNumAtoms() == mol.GetNumBonds(): 
        aroma_bonds = [b for b in mol.GetBonds() if b.GetBondType() == Chem.rdchem.BondType.AROMATIC]
        return len(aroma_bonds) == mol.GetNumBonds()
    else:
        return False

def find_fragments(mol):
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
        fmol = get_clique_mol(mol, indices)
        fmol = sanitize(fmol, kekulize=False)
        fsmiles = Chem.MolToSmiles(fmol)
        hopts.append((fsmiles, indices))
    
    return hopts

def get_leaves(mol):
    leaf_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetDegree() == 1]

    clusters = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        if not bond.IsInRing():
            clusters.append( set([a1,a2]) )

    rings = [set(x) for x in Chem.GetSymmSSSR(mol)]
    clusters.extend(rings)

    leaf_rings = []
    for r in rings:
        inters = [c for c in clusters if r != c and len(r & c) > 0]
        if len(inters) > 1: continue
        nodes = [i for i in r if mol.GetAtomWithIdx(i).GetDegree() == 2]
        leaf_rings.append( max(nodes) )

    return leaf_atoms + leaf_rings

def atom_equal(a1, a2):
    return a1.GetSymbol() == a2.GetSymbol() and a1.GetFormalCharge() == a2.GetFormalCharge()

def bond_match(mol1, a1, b1, mol2, a2, b2):
    a1,b1 = mol1.GetAtomWithIdx(a1), mol1.GetAtomWithIdx(b1)
    a2,b2 = mol2.GetAtomWithIdx(a2), mol2.GetAtomWithIdx(b2)
    return atom_equal(a1,a2) and atom_equal(b1,b2)

def copy_atom(atom, atommap=True):
    new_atom = Chem.Atom(atom.GetSymbol())
    new_atom.SetFormalCharge(atom.GetFormalCharge())
    if atommap: 
        new_atom.SetAtomMapNum(atom.GetAtomMapNum())
    return new_atom

#mol must be RWMol object
def get_sub_mol(mol, sub_atoms):
    new_mol = Chem.RWMol()
    atom_map = {}
    for idx in sub_atoms:
        atom = mol.GetAtomWithIdx(idx)
        atom_map[idx] = new_mol.AddAtom(atom)

    sub_atoms = set(sub_atoms)
    for idx in sub_atoms:
        a = mol.GetAtomWithIdx(idx)
        for b in a.GetNeighbors():
            if b.GetIdx() not in sub_atoms: continue
            bond = mol.GetBondBetweenAtoms(a.GetIdx(), b.GetIdx())
            bt = bond.GetBondType()
            if a.GetIdx() < b.GetIdx(): #each bond is enumerated twice
                new_mol.AddBond(atom_map[a.GetIdx()], atom_map[b.GetIdx()], bt)

    return new_mol.GetMol()

def copy_edit_mol(mol):
    new_mol = Chem.RWMol(Chem.MolFromSmiles(''))
    for atom in mol.GetAtoms():
        new_atom = copy_atom(atom)
        new_mol.AddAtom(new_atom)

    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        bt = bond.GetBondType()
        new_mol.AddBond(a1, a2, bt)
        #if bt == Chem.rdchem.BondType.AROMATIC and not aromatic:
        #    bt = Chem.rdchem.BondType.SINGLE
    return new_mol

def get_clique_mol(mol, atoms):
    smiles = Chem.MolFragmentToSmiles(mol, atoms, kekuleSmiles=True)
    new_mol = Chem.MolFromSmiles(smiles, sanitize=False)
    new_mol = copy_edit_mol(new_mol).GetMol()
    new_mol = sanitize(new_mol) 
    #if tmp_mol is not None: new_mol = tmp_mol
    return new_mol

def get_assm_cands(mol, atoms, inter_label, cluster, inter_size):
    atoms = list(set(atoms))
    mol = get_clique_mol(mol, atoms)
    atom_map = [idxfunc(atom) for atom in mol.GetAtoms()]
    mol = set_atommap(mol)
    rank = Chem.CanonicalRankAtoms(mol, breakTies=False)
    rank = { x:y for x,y in zip(atom_map, rank) }

    pos, icls = zip(*inter_label)
    if inter_size == 1:
        cands = [pos[0]] + [ x for x in cluster if rank[x] != rank[pos[0]] ] 
    
    elif icls[0] == icls[1]: #symmetric case
        shift = cluster[inter_size - 1:] + cluster[:inter_size - 1]
        cands = zip(cluster, shift)
        cands = [pos] + [ (x,y) for x,y in cands if (rank[min(x,y)],rank[max(x,y)]) != (rank[min(pos)], rank[max(pos)]) ]
    else: 
        shift = cluster[inter_size - 1:] + cluster[:inter_size - 1]
        cands = zip(cluster + shift, shift + cluster)
        cands = [pos] + [ (x,y) for x,y in cands if (rank[x],rank[y]) != (rank[pos[0]], rank[pos[1]]) ]

    return cands

def get_inter_label(mol, atoms, inter_atoms, atom_cls):
    new_mol = get_clique_mol(mol, atoms)
    if new_mol.GetNumBonds() == 0: 
        inter_atom = list(inter_atoms)[0]
        for a in new_mol.GetAtoms():
            a.SetAtomMapNum(0)
        return new_mol, [ (inter_atom, Chem.MolToSmiles(new_mol)) ]

    inter_label = []
    for a in new_mol.GetAtoms():
        idx = idxfunc(a)
        if idx in inter_atoms and is_anchor(a, inter_atoms):
            inter_label.append( (idx, get_anchor_smiles(new_mol, idx)) )

    for a in new_mol.GetAtoms():
        idx = idxfunc(a)
        if idx in inter_atoms:
            a.SetAtomMapNum(1) 
        elif len(atom_cls[idx]) > 1:
            a.SetAtomMapNum(2)
        else:
            a.SetAtomMapNum(0)

    return new_mol, inter_label

def is_anchor(atom, inter_atoms):
    for a in atom.GetNeighbors():
        if idxfunc(a) not in inter_atoms:
            return True
    return False
            
def get_anchor_smiles(mol, anchor, idxfunc=idxfunc):
    copy_mol = Chem.Mol(mol)
    for a in copy_mol.GetAtoms():
        idx = idxfunc(a)
        if idx == anchor: a.SetAtomMapNum(1)
        else: a.SetAtomMapNum(0)

    return get_smiles(copy_mol)
