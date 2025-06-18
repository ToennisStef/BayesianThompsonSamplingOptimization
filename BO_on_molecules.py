import warnings
warnings.filterwarnings("ignore")

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from sklearn.model_selection import train_test_split
from concurrent.futures import ProcessPoolExecutor

from rdkit import Chem
from mordred import Calculator, descriptors
from mordred import (HydrogenBond, TopoPSA, SLogP, RotatableBond,
                     Weight, VdwVolumeABC, Constitutional, Polarizability, CPSA)

from botorch.exceptions.warnings import BadInitialCandidatesWarning
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import LogExpectedImprovement as ExpectedImprovement
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize

from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.kernels import RBFKernel, ScaleKernel, ProductKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.distributions import MultivariateNormal

from gauche.dataloader import MolPropLoader
from gauche.kernels.fingerprint_kernels.tanimoto_kernel import TanimotoKernel

from transformers import AutoModel, AutoTokenizer

# Experiment parameters
N_TRIALS = 10
holdout_set_size = 0.95
N_ITERS = 10
verbose = True


class TanimotoGP(SingleTaskGP):
    def __init__(self, train_X, train_Y, d_fingerprint=2048):
        super().__init__(train_X, train_Y, likelihood=GaussianLikelihood(), 
                         input_transform=Normalize(d=train_X.shape[1]),
                         outcome_transform=Standardize(m=train_Y.shape[1]))
        self.mean_module = ConstantMean()
        tanimoto_kernel = TanimotoKernel(active_dims=list(range(d_fingerprint)))
        rbf_kernel = RBFKernel(active_dims=[d_fingerprint])
        self.covar_module = ScaleKernel(ProductKernel(tanimoto_kernel, rbf_kernel))
        self.to(train_X)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

def initialize_model_tanimoto_gp(train_x, train_y, state_dict=None):
    model = TanimotoGP(train_x, train_y).to(train_x)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    if state_dict:
        model.load_state_dict(state_dict)
    return mll, model

def initialize_model_se_gp(train_x, train_y, state_dict=None):
    model = SingleTaskGP(
        train_X=train_x, 
        train_Y=train_y,
        input_transform=Normalize(d=train_x.shape[1]),
        outcome_transform=Standardize(m=train_y.shape[1])
    ).to(train_x)
    
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    if state_dict:
        model.load_state_dict(state_dict)
    return mll, model

def optimize_acqf_and_get_observation(acq_func, heldout_x, heldout_y):
    acq_vals = torch.tensor([acq_func(x.unsqueeze(0)) for x in heldout_x])
    best_idx = torch.argmax(acq_vals)
    new_x = heldout_x[best_idx].unsqueeze(0)
    new_y = heldout_y[best_idx].unsqueeze(0)
    heldout_x = torch.cat([heldout_x[:best_idx], heldout_x[best_idx+1:]])
    heldout_y = torch.cat([heldout_y[:best_idx], heldout_y[best_idx+1:]])
    return new_x, new_y, heldout_x, heldout_y

def update_random_observations(best_random, heldout_x, heldout_y):
    idx = torch.randperm(len(heldout_y))[0]
    next_y = heldout_y[idx]
    best_random.append(max(best_random[-1], next_y))
    heldout_x = torch.cat([heldout_x[:idx], heldout_x[idx+1:]])
    heldout_y = torch.cat([heldout_y[:idx], heldout_y[idx+1:]])
    return best_random, heldout_x, heldout_y

def expert_featurization(smiles_list):
    calc = Calculator([
        HydrogenBond.HBondAcceptor,
        # HydrogenBond.HBondDonor,
        TopoPSA.TopoPSA,
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

def molformer_featurization(smiles_list):
    

    model = AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct", deterministic_eval=True, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)

    inputs = tokenizer(smiles_list, padding=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    # outputs.pooler_output.shape  # (batch_size, hidden_size)
    features = outputs.pooler_output.cpu().numpy()
    
    print(f"MolFormer featurization dims: {features.shape}")    
    
    return features



# Convert to lists and save as json
def save_to_json(np_array_list, filename):
    import json
    
    if type(np_array_list) is not list:
        np_array_list = np_array_list.tolist()
    else:
        np_array_list = [arr.tolist() for arr in np_array_list]
    with open(f"{filename}.json", 'w') as f:
        json.dump(np_array_list, f)
        
def load_from_json(filename):
    import json
    with open(f"{filename}.json", 'r') as f:
        data = json.load(f)
    return [np.array(arr) for arr in data]





def run_trial(args):
    """
    Run a single trial of Bayesian optimization.
    
    Args:
        trial (int): Trial number (used as random seed).
        
    Returns:
        tuple: (best_observed_ei, best_random) as torch tensors.
    """
    
    trial, X_fp, X_alphab, X_expert, y = args
    
    print(f"\nTrial {trial:>2} of {N_TRIALS} ", end="")
    best_observed_ei, best_observed_ei_alphab, best_observed_ei_expert, best_random = [],[],[],[]

    #create index
    data_index = [i for i in range(len(y))]
    np.random.seed(trial)
    torch.manual_seed(trial)
    
    train_idx, heldout_idx = train_test_split(
        data_index, test_size=holdout_set_size, random_state=trial
    )
    train_x_ei = X_fp[train_idx]
    heldout_x_ei = X_fp[heldout_idx]
    train_y_ei = y[train_idx]
    heldout_y_ei = y[heldout_idx]
    
    best_observed_value_ei = torch.tensor(np.max(train_y_ei))

    # Convert numpy arrays to PyTorch tensors and flatten the label vectors
    train_x_ei = torch.tensor(train_x_ei.astype(np.float64))
    heldout_x_ei = torch.tensor(heldout_x_ei.astype(np.float64))
    train_y_ei = torch.tensor(train_y_ei)
    heldout_y_ei = torch.tensor(heldout_y_ei)
    
    train_x_alphab = torch.tensor(X_alphab[train_idx].astype(np.float64))
    heldout_x_alphab = torch.tensor(X_alphab[heldout_idx].astype(np.float64))
    train_y_alphab = train_y_ei.clone()
    heldout_y_alphab = heldout_y_ei.clone()
    
    train_x_expert = torch.tensor(X_expert[train_idx].astype(np.float64))
    heldout_x_expert = torch.tensor(X_expert[heldout_idx].astype(np.float64))
    train_y_expert = train_y_ei.clone()
    heldout_y_expert = heldout_y_ei.clone()

    # The initial heldout set is the same for random search
    heldout_x_random = heldout_x_ei
    heldout_y_random = heldout_y_ei

    mll_ei, model_ei = initialize_model_tanimoto_gp(train_x_ei, train_y_ei)
    # Initialize the model for the alphab optimization
    mll_alphab, model_alphab = initialize_model_se_gp(train_x_alphab, train_y_alphab)
    # Initialize the model for the expert optimization
    mll_expert, model_expert = initialize_model_se_gp(train_x_expert, train_y_expert)

    best_observed_ei.append(best_observed_value_ei)
    best_observed_ei_alphab.append(best_observed_value_ei)
    best_observed_ei_expert.append(best_observed_value_ei)
    best_random.append(best_observed_value_ei)

    # Run N_ITERS rounds of BayesOpt
    for iteration in range(1, N_ITERS + 1):
        t0 = time.time()

        # Fit the model
        fit_gpytorch_mll(mll_ei)
        fit_gpytorch_mll(mll_alphab)
        fit_gpytorch_mll(mll_expert)

        # Use analytic acquisition function for batch size of 1
        EI = ExpectedImprovement(model=model_ei, best_f=(train_y_ei.to(train_y_ei)).max())
        EI_alphab = ExpectedImprovement(
            model=model_alphab, best_f=(train_y_alphab.to(train_y_alphab)).max()
        )
        EI_expert = ExpectedImprovement(
            model=model_expert, best_f=(train_y_expert.to(train_y_expert)).max()
        )

        new_x_ei, new_obj_ei, heldout_x_ei, heldout_y_ei = optimize_acqf_and_get_observation(
            EI, heldout_x_ei, heldout_y_ei
        )
        # Update training points
        train_x_ei = torch.cat([train_x_ei, new_x_ei])
        train_y_ei = torch.cat([train_y_ei, new_obj_ei])
        
        new_x_ei_alphab, new_obj_ei_alphab, heldout_x_alphab, heldout_y_alphab = optimize_acqf_and_get_observation(
            EI_alphab, heldout_x_alphab, heldout_y_alphab
        )
        # Update training points for alphab optimization
        train_x_alphab = torch.cat([train_x_alphab, new_x_ei_alphab])
        train_y_alphab = torch.cat([train_y_alphab, new_obj_ei_alphab])

        new_x_ei_expert, new_obj_ei_expert, heldout_x_expert, heldout_y_expert = optimize_acqf_and_get_observation(
            EI_expert, heldout_x_expert, heldout_y_expert
        )
        # Update training points for expert optimization
        train_x_expert = torch.cat([train_x_expert, new_x_ei_expert])
        train_y_expert = torch.cat([train_y_expert, new_obj_ei_expert])

        # Update random search progress
        best_random, heldout_x_random, heldout_y_random = update_random_observations(
            best_random, heldout_x=heldout_x_random, heldout_y=heldout_y_random
        )
        best_value_ei = torch.max(new_obj_ei, best_observed_ei[-1])
        best_observed_ei.append(best_value_ei.squeeze())
        
        best_value_ei_alphab = torch.max(new_obj_ei_alphab, best_observed_ei_alphab[-1])
        best_observed_ei_alphab.append(best_value_ei_alphab.squeeze())
        
        best_value_ei_expert = torch.max(new_obj_ei_expert, best_observed_ei_expert[-1])
        best_observed_ei_expert.append(best_value_ei_expert.squeeze())
        
        # Reinitialize the model for the next iteration
        mll_ei, model_ei = initialize_model_tanimoto_gp(
            train_x_ei, train_y_ei, model_ei.state_dict()
        )
        
        mll_alphab, model_alphab = initialize_model_se_gp(
            train_x_alphab, train_y_alphab, model_alphab.state_dict()
        )
        
        mll_expert, model_expert = initialize_model_se_gp(
            train_x_expert, train_y_expert, model_expert.state_dict()
        )

        t1 = time.time()

        if verbose:
            print(
                f"\nBatch {iteration:>2}: best_value (random, qEI, alphab, expert) = "
                f"({max(best_random).item():>4.2f}, {best_value_ei.item():>4.2f}, {best_value_ei_alphab.item():>4.2f}, {best_value_ei_expert.item():>4.2f} "
                f"time = {t1 - t0:>4.2f}s.", end=""
            )
        else:
            print(".", end="")

    return torch.hstack(best_observed_ei), torch.hstack(best_observed_ei_alphab), torch.hstack(best_observed_ei_expert), torch.hstack(best_random)

if __name__ == "__main__":
    # Load the Lactate Solubility dataset
    loader = MolPropLoader()
    loader.read_csv(path="./output_data_full.csv", smiles_column="SMILES", label_column="KV")
    df = pd.read_csv("./output_data_full.csv")

    #change
    solvents_id_solvent_dict = {v:k for k, v in enumerate(df["SMILES"].unique())}

    df["solvent_id"] = df["SMILES"].map(solvents_id_solvent_dict)

    # Featurize using fragprints representations
    
    loader.featurize('ecfp_fragprints')
    y = loader.labels
    X_temp = df["Temperature (°C)"]

    X_fp = loader.features
    X_fp = np.concatenate((X_fp, X_temp.values.reshape(-1, 1)), axis=1)  # Add temperature as a feature
    # X_alphab = np.concatenate((df["solvent_id"].values.reshape(-1, 1), X_temp.values.reshape(-1, 1)), axis=1)
    X_expert = expert_featurization(df["SMILES"])
    X_expert = np.concatenate((X_expert, X_temp.values.reshape(-1, 1)), axis=1)  # Add temperature as a feature
    X_llm = molformer_featurization(df["SMILES"].to_list())
    X_alphab = np.concatenate((X_llm, X_temp.values.reshape(-1, 1)), axis=1)  # Add temperature as a feature
    # run_trial(1)
    
    fname = f"BOvsRS_{N_ITERS}iters_{N_TRIALS}trials_{holdout_set_size}holdout"
    
    # warnings.filterwarnings('ignore', category=BadInitialCandidatesWarning)
    # warnings.filterwarnings('ignore', category=RuntimeWarning)
    
    iterable = [
        (i, X_fp, X_alphab, X_expert, y)
        for i in range(1, N_TRIALS + 1)
    ]

    t_start = time.time()
    best_observed_all_ei, best_observed_all_ei_alphab, best_observed_all_ei_expert, best_random_all = [], [],[], []
    with ProcessPoolExecutor(max_workers=os.cpu_count()-2) as executor:
        results = list(executor.map(run_trial, iterable))

    t_end = time.time()
    print(f"time = {t_end - t_start:>4.2f}.")

    # Unpack results
    for best_observed_ei, best_observed_ei_alphab, best_observed_ei_expert, best_random in results:
        best_observed_all_ei.append(best_observed_ei)
        best_observed_all_ei_alphab.append(best_observed_ei_alphab)
        best_observed_all_ei_expert.append(best_observed_ei_expert)
        best_random_all.append(best_random)

    # Prepare data for plotting
    iters = np.arange(N_ITERS + 1)
    y_ei = np.asarray(torch.stack(best_observed_all_ei))
    y_ei_alphab = np.asarray(torch.stack(best_observed_all_ei_alphab))
    y_ei_expert = np.asarray(torch.stack(best_observed_all_ei_expert))
    y_rnd = np.asarray(torch.stack(best_random_all))
            
    save_to_json([y_ei, y_ei_alphab, y_rnd], fname)
    # Load data from json
    # y_ei, y_ei_alphab, y_rnd = load_from_json(fname)

    # Compute quantiles (2.5th and 97.5th percentiles for 95% CI)
    y_rnd_lower = np.percentile(y_rnd, 2.5, axis=0)
    y_rnd_upper = np.percentile(y_rnd, 97.5, axis=0)
    y_ei_lower = np.percentile(y_ei, 2.5, axis=0)
    y_ei_upper = np.percentile(y_ei, 97.5, axis=0)
    y_ei_alphab_lower = np.percentile(y_ei_alphab, 2.5, axis=0)
    y_ei_alphab_upper = np.percentile(y_ei_alphab, 97.5, axis=0)
    y_ei_expert_lower = np.percentile(y_ei_expert, 2.5, axis=0)
    y_ei_expert_upper = np.percentile(y_ei_expert, 97.5, axis=0)
    
    y_rnd_mean = np.mean(y_rnd, axis=0)
    y_ei_mean = np.mean(y_ei, axis=0)
    y_ei_alphab_mean = np.mean(y_ei_alphab, axis=0)
    y_ei_expert_mean = np.mean(y_ei_expert, axis=0)

    # Plotting
    plt.figure(figsize=(7, 6))
    plt.plot(iters, y_ei_mean, label="BO Tanimoto & Fragprints", color="orange", marker='o')
    plt.fill_between(iters, y_ei_lower, y_ei_upper, color="orange", alpha=0.2)
    plt.plot(iters, y_ei_alphab_mean, label="BO SE & Alphab.", color="green", marker='o')
    plt.fill_between(iters, y_ei_alphab_lower, y_ei_alphab_upper, color="green", alpha=0.2)
    plt.plot(iters, y_ei_expert_mean, label="BO SE & Expert", color="red", marker='o')
    plt.fill_between(iters, y_ei_expert_lower, y_ei_expert_upper, color="red", alpha=0.2)
    plt.plot(iters, y_rnd_mean, label="Random Search", color="blue", marker='o')
    plt.fill_between(iters, y_rnd_lower, y_rnd_upper, color="blue", alpha=0.2)
    plt.title("BO w/ Tanimoto Kernel & Fragprints vs.\n SE Kernel & alphabetical. vs. SE Kernel & Mordred")
    plt.xlabel("Iteration")
    plt.ylabel("Best Observed Value")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{fname}.pdf")