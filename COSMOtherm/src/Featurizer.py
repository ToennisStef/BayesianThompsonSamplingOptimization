import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
from rdkit.DataStructs import ConvertToNumpyArray


def compute_morgan_fingerprint(mol, radius=2, n_bits=2048):
    """
    Compute the Morgan fingerprint for a molecule.
    
    Parameters:
        mol (rdkit.Chem.Mol): RDKit Mol object.
        radius (int): Radius for the fingerprint.
        n_bits (int): Length of the bit vector.
        
    Returns:
        np.ndarray: Binary vector of the Morgan fingerprint.
    """

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)

    arr = np.zeros((n_bits,), dtype=int)
    ConvertToNumpyArray(fp, arr)
    return arr


def featurize_dataset(smiles_names, include_fingerprints=False):
    """
    Generate physicochemical descriptors, functional group presence, and optionally Morgan fingerprints.
    
    Parameters:
        smiles_names (list): List of SMILES strings representing molecules.
        include_fingerprints (bool): Whether to include Morgan fingerprints.
        
    Returns:
        pd.DataFrame: DataFrame with molecular features.
    """

    mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_names]

    functional_groups = {
        'Alcohol': '[CX4][OH]',
        'Carboxylic Acid': 'C(=O)[OH]',
        'Amine (Primary)': '[NX3;H2][CX4]',
        'Amine (Secondary)': '[NX3;H1][CX4][CX4]',
        'Amine (Tertiary)': '[NX3]([CX4])([CX4])[CX4]',
        'Ester': 'C(=O)O[CX4]',
        'Ether': '[OD2]([CX4])[CX4]',
        'Ketone': 'C(=O)[CX4]',
        'Aromatic Ring': 'a1aaaaa1',
        'Phenol': 'c1ccc(cc1)[OH]',
    }

    data = []
    fingerprint_data = []

    for i, mol in enumerate(mols):
        if mol is None:
            print(f"Warning: Invalid SMILES '{smiles_names[i]}'")
            continue
        
        row = {'Molecule': smiles_names[i]}
        
        for name, smarts in functional_groups.items():
            patt = Chem.MolFromSmarts(smarts)
            row[name] = mol.HasSubstructMatch(patt) if patt else False
        
        row['MolWt'] = Descriptors.MolWt(mol)
        row['TPSA'] = rdMolDescriptors.CalcTPSA(mol)
        row['LogP'] = Descriptors.MolLogP(mol)
        row['NumHDonors'] = rdMolDescriptors.CalcNumHBD(mol)
        row['NumHAcceptors'] = rdMolDescriptors.CalcNumHBA(mol)
        row['NumRotatableBonds'] = Descriptors.NumRotatableBonds(mol)
        row['RingCount'] = rdMolDescriptors.CalcNumRings(mol)
        
        data.append(row)

        if include_fingerprints:
            fingerprint_data.append([compute_morgan_fingerprint(mol)])

    df = pd.DataFrame(data)
    
    if include_fingerprints:
        fingerprint_data = np.array(fingerprint_data)
        df['MorganFingerprint'] = list(fingerprint_data)
    print(df)
    return df


if __name__ == "__main__":
    # Example usage
    smiles_names = [
        'CC(=O)OC1=CC=CC=C1C(=O)O',  # Acetic acid
        'CC(C(=O)O)C(=O)O',         # Lactic acid
        'CC(C(=O)O)C(=O)OC1=CC=CC=C1C(=O)O'  # Lactic acid with aromatic ring
    ]
    
    df = featurize_molecules(smiles_names, include_fingerprints=True)
    print(df.columns)