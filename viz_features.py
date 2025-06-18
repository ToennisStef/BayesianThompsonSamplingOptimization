import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rdkit import Chem
from mordred import Calculator, descriptors
from mordred import (SLogP, Weight, CPSA) # Simplified from original for brevity

from gauche.dataloader import MolPropLoader
import umap # Make sure umap is imported

def expert_featurization(smiles_list):

    calc = Calculator([
        # HydrogenBond.HBondAcceptor,
        # HydrogenBond.HBondDonor,
        # TopoPSA.TopoPSA,
        SLogP.SLogP,
        # RotatableBond.RotatableBondsCount,
        Weight,
        # VdwVolumeABC,
        # Constitutional,
        # Polarizability,
        CPSA,
    ])

    # calc = Calculator(descriptors.all, ignore_3D=False)

    # Convert SMILES to RDKit Mol objects, skipping invalids
    mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    valid_idx = [i for i, mol in enumerate(mols) if mol is not None]
    valid_mols = [mols[i] for i in valid_idx]
    
    # Compute descriptors
    X_calc = calc.pandas(valid_mols, nproc=1)

    # Fill missing values with 0 or another strategy
    X_calc = X_calc.fillna(0)
    X_calc = X_calc.select_dtypes(include=[np.number])  # Ensure only numeric columns are kept
    print(f"Expert featurization dims: {X_calc.shape}")
    return X_calc.to_numpy()


if __name__ == "__main__":
    # Load the dataset
    try:
        df = pd.read_csv("./output_data_full.csv")
    except FileNotFoundError:
        print("Error: 'output_data_full.csv' not found. Make sure the file is in the correct directory.")
        exit()

    loader = MolPropLoader()
    loader.read_csv(path="./output_data_full.csv", smiles_column="SMILES", label_column="KV")
    y_target = loader.labels # Load the target variable y
    # Ensure y_target is a 1D array for coloring
    if y_target.ndim > 1 and y_target.shape[1] == 1:
        y_target = y_target.ravel()


    unique_smiles = df["SMILES"].unique()
    solvents_id_solvent_dict = {smiles: i for i, smiles in enumerate(unique_smiles)}
    df["solvent_id"] = df["SMILES"].map(solvents_id_solvent_dict)

    # --- Feature Generation ---
    print("Generating ECFP Fragprints...")
    loader.featurize('ecfp_fragprints')
    X_fp = loader.features
    X_temp = df["Temperature (°C)"].values.reshape(-1, 1)
    X_fp_temp = np.concatenate((X_fp, X_temp), axis=1)
    print(f"ECFP Fragprints + Temperature shape: {X_fp_temp.shape}")

    print("\nGenerating Alphabetical Solvent ID features...")
    X_alphab = df["solvent_id"].values.reshape(-1, 1)
    X_alphab_temp = np.concatenate((X_alphab, X_temp), axis=1)
    print(f"Alphabetical Solvent ID + Temperature shape: {X_alphab_temp.shape}")

    print("\nGenerating Expert (Mordred) features...")
    smiles_for_expert = df["SMILES"].tolist()
    X_expert = expert_featurization(smiles_for_expert)
    X_expert_temp = np.concatenate((X_expert, X_temp), axis=1)
    print(f"Expert (Mordred) + Temperature shape: {X_expert_temp.shape}")

    # --- UMAP Plotting ---
    features_dict = {
        "ECFP Fragprints + Temp": X_fp_temp,
        "Alphabetical Solvent ID + Temp": X_alphab_temp,
        "Expert (Mordred) Descriptors + Temp": X_expert_temp
    }

    n_neighbors_list = [5, 10, 15, 20]

    for feature_name, X_data in features_dict.items():
        print(f"\nRunning UMAP for {feature_name} with varying n_neighbors...")
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f"UMAP Projection of {feature_name}\nColored by Target Variable (KV)", fontsize=16, y=1.02)
        axes = axes.ravel() # Flatten the 2x2 array of axes for easy iteration

        # Check for NaNs or Infs in X_data before passing to UMAP
        if np.isnan(X_data).any() or np.isinf(X_data).any():
            print(f"Warning: NaNs or Infs found in {feature_name}. Replacing with 0. This might affect UMAP results.")
            X_data = np.nan_to_num(X_data, nan=0.0, posinf=0.0, neginf=0.0) # Replace NaNs/Infs

        for i, n_neighbors_val in enumerate(n_neighbors_list):
            print(f"  n_neighbors = {n_neighbors_val}")
            reducer = umap.UMAP(n_neighbors=n_neighbors_val, random_state=42, n_components=2, min_dist=0.1)
            
            # Ensure X_data is C-contiguous, float32 or float64, and has at least 2 samples.
            if not X_data.flags['C_CONTIGUOUS']:
                X_data_contiguous = np.ascontiguousarray(X_data, dtype=np.float64)
            else:
                X_data_contiguous = X_data.astype(np.float64)

            if X_data_contiguous.shape[0] < 2 :
                print(f"Skipping UMAP for {feature_name} with n_neighbors={n_neighbors_val} due to insufficient samples ({X_data_contiguous.shape[0]})")
                axes[i].text(0.5, 0.5, 'Too few samples', horizontalalignment='center', verticalalignment='center', transform=axes[i].transAxes)
                axes[i].set_title(f"n_neighbors = {n_neighbors_val}\n(Too few samples)")
                axes[i].set_xticks([])
                axes[i].set_yticks([])
                continue

            try:
                embedding = reducer.fit_transform(X_data_contiguous)
                
                scatter = axes[i].scatter(embedding[:, 0], embedding[:, 1], c=y_target, cmap='viridis', s=10)
                axes[i].set_title(f"n_neighbors = {n_neighbors_val}")
                axes[i].set_xlabel("UMAP Component 1")
                axes[i].set_ylabel("UMAP Component 2")
                # Add a colorbar to the first plot for reference, or one per plot if preferred
                if i == 0: # Or for each plot: fig.colorbar(scatter, ax=axes[i], label="Target Value (KV)")
                     fig.colorbar(scatter, ax=axes[:], label="Target Value (KV)", aspect=40, pad=0.08)

            except Exception as e:
                print(f"Error during UMAP for {feature_name} with n_neighbors={n_neighbors_val}: {e}")
                axes[i].text(0.5, 0.5, 'UMAP Error', horizontalalignment='center', verticalalignment='center', transform=axes[i].transAxes)
                axes[i].set_title(f"n_neighbors = {n_neighbors_val}\n(Error)")
                axes[i].set_xticks([])
                axes[i].set_yticks([])


        plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust layout to make space for suptitle
        plt.savefig(f"UMAP_{feature_name.replace(' ', '_').replace('+', 'and')}_neighbors_grid.pdf")
        plt.show()

    print("\nUMAP plotting complete.")