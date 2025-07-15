import numpy as np
from rdkit import Chem
from mordred import Calculator, descriptors
from mordred import (HydrogenBond, TopoPSA, SLogP, RotatableBond,
                     Weight, VdwVolumeABC, Constitutional, Polarizability, CPSA)
from transformers import AutoModel, AutoTokenizer
import torch

import pandas as pd
from gauche.dataloader import MolPropLoader

import os
import hashlib
import joblib

# from chemeleon_fingerprint import CheMeleonFingerprint

def _get_cache_path(function_name, smiles_list):
    os.makedirs('featurizer_cache', exist_ok=True)
    smiles_str = '\n'.join(smiles_list)
    hash_digest = hashlib.sha256(smiles_str.encode('utf-8')).hexdigest()
    return os.path.join('featurizer_cache', f"{function_name}_{hash_digest}.joblib")


def expert_featurization(smiles_list):
    """
    Featurize SMILES strings using expert-selected Mordred descriptors.
    """
    cache_path = _get_cache_path('expert', smiles_list)
    if os.path.exists(cache_path):
        features = joblib.load(cache_path)
        print(f"Loaded expert featurization from cache: {features.shape}")
        return features
    calc = Calculator([
        HydrogenBond.HBondAcceptor,
        HydrogenBond.HBondDonor,
        TopoPSA.TopoPSA,
        SLogP.SLogP,
        RotatableBond.RotatableBondsCount,
        Weight,
        VdwVolumeABC,
        Constitutional,
        Polarizability,
        CPSA,
    ])
    mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    valid_idx = [i for i, mol in enumerate(mols) if mol is not None]
    valid_mols = [mols[i] for i in valid_idx]
    X_calc = calc.pandas(valid_mols, nproc=1)
    X_calc = X_calc.fillna(0)
    X_calc = X_calc.select_dtypes(include=[np.number])
    print(f"Expert featurization dims: {X_calc.shape}")
    features = X_calc.to_numpy()
    joblib.dump(features, cache_path)
    return features


def molformer_featurization(smiles_list):
    """
    Featurize SMILES strings using the MoLFormer transformer model.
    """
    cache_path = _get_cache_path('molformer', smiles_list)
    if os.path.exists(cache_path):
        features = joblib.load(cache_path)
        print(f"Loaded molformer featurization from cache: {features.shape}")
        return features
    model = AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct", deterministic_eval=True, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)
    inputs = tokenizer(smiles_list, padding=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    features = outputs.pooler_output.cpu().numpy()
    print(f"MolFormer featurization dims: {features.shape}")    
    joblib.dump(features, cache_path)
    return features


def chemeleon_featurization(smiles_list):
    """
    Featurize SMILES strings using CheMeleon fingerprint.
    """
    cache_path = _get_cache_path('chemeleon', smiles_list)
    if os.path.exists(cache_path):
        features = joblib.load(cache_path)
        print(f"Loaded chemeleon featurization from cache: {features.shape}")
        return features
    from chemeleon_fingerprint import CheMeleonFingerprint
    chemeleon = CheMeleonFingerprint()
    features = chemeleon(smiles_list)
    features = np.array(features)
    print(f"CheMeleon featurization dims: {features.shape}")
    joblib.dump(features, cache_path)
    return features 


class MolPropLoaderEXTENDED(MolPropLoader):
    """
    Extended MolPropLoader with additional features for featurization and validation.
    Stores original SMILES and labels so featurize can be called multiple times without reloading.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.smiles = None
        self.labels = None

    def read_csv(self, path, smiles_column="SMILES", label_column="Selectivity", clip_index=None, random=True, **kwargs):

        if clip_index is not None:
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
                if random:
                    df = pd.read_csv(path)
                    df = df.sample(n=clip_index, random_state=42)
                else:
                    df = pd.read_csv(path, nrows=clip_index)
                df.to_csv(temp_file.name, index=False)
                path = temp_file.name

        super().read_csv(path=path, smiles_column=smiles_column, label_column=label_column, **kwargs)
        self.smiles = self.features 

    def featurize(self, representation, **kwargs):
        valid_representations = [
            "ecfp_fingerprints",
            "fragments",
            "ecfp_fragprints",
            "molecular_graphs",
            "bag_of_smiles",
            "bag_of_selfies",
            "mqn",
        ]
        self.features = self.smiles 
        if self.smiles is None:
            raise ValueError("No SMILES loaded. Please call read_csv first.")
        if representation in valid_representations:
            return super().featurize(representation, **kwargs)
        if representation == "expert":
            self.features = expert_featurization(self.smiles)
        elif representation == "molformer":
            self.features = molformer_featurization(self.smiles)
        elif representation == "chemeleon":
            self.features = chemeleon_featurization(self.smiles)
        else:
            raise ValueError(
                f"Invalid representation '{representation}'. "
                f"Valid options are: {', '.join(valid_representations + ['expert', 'molformer', 'chemeleon'])}."
            )
