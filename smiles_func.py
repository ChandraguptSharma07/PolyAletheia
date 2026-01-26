# smiles functions for rdkit
from rdkit import Chem


def is_valid_smiles(smiles):
    # check if smiles parses correctly
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None


def canonicalize(smiles):
    # convert to canonical form
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def get_atom_count(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    return mol.GetNumAtoms()


if __name__ == "__main__":
    # quick test
    test = "*CC(*)c1ccccc1"
    print(f"Valid: {is_valid_smiles(test)}")
    print(f"Canonical: {canonicalize(test)}")
    print(f"Atoms: {get_atom_count(test)}")
