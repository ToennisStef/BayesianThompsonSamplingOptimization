import warnings
warnings.filterwarnings("ignore")

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import Tensor
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

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
from gpytorch.kernels import RBFKernel, ScaleKernel, ProductKernel, RFFKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.distributions import MultivariateNormal

from gauche.dataloader import MolPropLoader
from gauche.kernels.fingerprint_kernels.tanimoto_kernel import TanimotoKernel

from transformers import AutoModel, AutoTokenizer

from chemeleon_fingerprint import CheMeleonFingerprint

import os
from botorch import fit_fully_bayesian_model_nuts
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.transforms import Standardize as BoTorchStandardize

# Experiment parameters
N_TRIALS = 10
holdout_set_size = 0.95
N_ITERS = 50
verbose = True


class TanimotoGP(SingleTaskGP):
    def __init__(self, train_X, train_Y, d_fingerprint=2048):
        super().__init__(train_X, train_Y, likelihood=GaussianLikelihood(), 
                         input_transform=Normalize(d=train_X.shape[1]),
                         outcome_transform=Standardize(m=train_Y.shape[1]))
        self.mean_module = ConstantMean()
        # tanimoto_kernel = TanimotoKernel(active_dims=list(range(d_fingerprint)))
        tanimoto_kernel = TanimotoKernel()
        # rbf_kernel = RBFKernel(active_dims=[d_fingerprint])
        # self.covar_module = ScaleKernel(ProductKernel(tanimoto_kernel, rbf_kernel))
        self.covar_module = tanimoto_kernel
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

def initialize_model_RFF_se_gp(train_x, train_y, state_dict=None):

    base_kernel = RFFKernel(ard_num_dims=train_x.shape[-1], num_samples=1024)
    covar_module = ScaleKernel(base_kernel)

    model = SingleTaskGP(
        train_X=train_x, 
        train_Y=train_y,
        covar_module=covar_module,
        input_transform=Normalize(d=train_x.shape[1]),
        outcome_transform=Standardize(m=train_y.shape[1])
    ).to(train_x)
    
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    if state_dict:
        model.load_state_dict(state_dict)
    return mll, model

def initialize_model_saas_gp(train_x, train_y, state_dict=None):
    model = SaasFullyBayesianSingleTaskGP(
        train_X=train_x,
        train_Y=train_y,
        outcome_transform=Standardize(m=1)
    )
    if state_dict:
        model.load_state_dict(state_dict)
    # mll is not used for SAAS GP, but return a dummy for compatibility
    return None, model


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
def chemeleon_featurization(smiles_list):
    """
    Featurize SMILES strings using CheMeleon fingerprint.
    """
    chemeleon = CheMeleonFingerprint()
    features = chemeleon(smiles_list)
    features = np.array(features)
    
    print(f"CheMeleon featurization dims: {features.shape}")
    
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






# Helper for random search
class RandomSearchHelper:
    def __init__(self, y_train, heldout_x, heldout_y):
        self.best_random = [torch.tensor(np.max(y_train))]
        self.heldout_x = heldout_x
        self.heldout_y = heldout_y
    def step(self):
        idx = torch.randperm(len(self.heldout_y))[0]
        next_y = self.heldout_y[idx]
        best_val = torch.max(torch.stack([self.best_random[-1], next_y]))
        self.best_random.append(best_val)
        self.heldout_x = torch.cat([self.heldout_x[:idx], self.heldout_x[idx+1:]])
        self.heldout_y = torch.cat([self.heldout_y[:idx], self.heldout_y[idx+1:]])
    def run(self, n_iters):
        for _ in range(n_iters):
            self.step()
        return torch.hstack(self.best_random)

def run_trial_general(args):
    """
    General-purpose run_trial function for Bayesian optimization with arbitrary surrogate models and feature sets.
    Each model/feature set is run independently, including random search as a pseudo-model.
    Args:
        trial (int): Trial number (used as random seed).
        feature_sets (dict): Mapping from feature set name to numpy array of features.
        y (np.ndarray): Target values.
        model_initializers (dict): Mapping from feature set name to model initialization function.
    Returns:
        dict: Mapping from feature set name to best observed values (torch tensor), including 'Random_Search'.
    """
    trial, feature_sets, y, model_initializers = args
    print(f"\nTrial {trial:>2} of {N_TRIALS} (general)", end="")

    # Create index
    data_index = [i for i in range(len(y))]
    np.random.seed(trial)
    torch.manual_seed(trial)
    train_idx, heldout_idx = train_test_split(
        data_index, test_size=holdout_set_size, random_state=trial
    )

    results = {}
    # Run each model/feature set independently
    for name, X in feature_sets.items():
        train_x = torch.tensor(X[train_idx].astype(np.float64))
        heldout_x = torch.tensor(X[heldout_idx].astype(np.float64))
        train_y = torch.tensor(y[train_idx])
        heldout_y = torch.tensor(y[heldout_idx])
        best_observed = [torch.tensor(np.max(y[train_idx]))]
        mll, model = model_initializers[name](train_x, train_y)
        for iteration in range(1, N_ITERS + 1):
            t0 = time.time()
            # Special handling for SAAS GP
            if name.lower().startswith("saas") or isinstance(model, SaasFullyBayesianSingleTaskGP):
                fit_fully_bayesian_model_nuts(
                    model,
                    warmup_steps=256,
                    num_samples=128,
                    thinning=16,
                    disable_progbar=True,
                )
            else:
                fit_gpytorch_mll(mll)
            EI = ExpectedImprovement(model=model, best_f=(train_y.to(train_y)).max())
            new_x, new_obj, heldout_x, heldout_y = optimize_acqf_and_get_observation(
                EI, heldout_x, heldout_y
            )
            train_x = torch.cat([train_x, new_x])
            train_y = torch.cat([train_y, new_obj])
            best_value = torch.max(new_obj, best_observed[-1])
            best_observed.append(best_value.squeeze())
            mll, model = model_initializers[name](train_x, train_y, getattr(model, 'state_dict', lambda: None)())
            t1 = time.time()
            if verbose:
                print(f"\n{name} Batch {iteration:>2}: best_value = {best_value.item():>4.2f} time = {t1 - t0:>4.2f}s.", end="")
            else:
                print(".", end="")
        results[name] = torch.hstack(best_observed)
    # Random search as a pseudo-model
    # Use the first feature set's heldout split for random search
    first_name = next(iter(feature_sets))
    train_y = torch.tensor(y[train_idx])
    heldout_x = torch.tensor(feature_sets[first_name][heldout_idx].astype(np.float64))
    heldout_y = torch.tensor(y[heldout_idx])
    random_search = RandomSearchHelper(train_y, heldout_x, heldout_y)
    results['Random_Search'] = random_search.run(N_ITERS)
    return results

def cross_validation(model_initializer, X, y, n_splits=5, random_state=None):
    """
    Perform k-fold cross-validation for a given model and feature set.
    Args:
        model_initializer: function (train_x, train_y) -> (mll, model)
        X: numpy array of features
        y: numpy array of targets
        n_splits: number of folds
        random_state: random seed
    Returns:
        dict: {'mse': avg_mse, 'r2': avg_r2, 'mse_list': [...], 'r2_list': [...],
               'y_true_folds': [...], 'y_pred_folds': [...]}
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    mses = []
    r2s = []
    y_true_folds = []
    y_pred_folds = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        train_x = torch.tensor(X_train.astype(np.float64))
        train_y = torch.tensor(y_train)
        test_x = torch.tensor(X_test.astype(np.float64))
        mll, model = model_initializer(train_x, train_y)
        # Special handling for SAAS GP
        if isinstance(model, SaasFullyBayesianSingleTaskGP):
            fit_fully_bayesian_model_nuts(
                model,
                warmup_steps=256,
                num_samples=128,
                thinning=16,
                disable_progbar=True,
            )
        else:
            fit_gpytorch_mll(mll)
        with torch.no_grad():
            pred = model(test_x)
            # Handle different prediction types
            if isinstance(pred, MultivariateNormal):
                y_pred = pred.mean.cpu().numpy().squeeze()
            elif isinstance(pred, Tensor):
                y_pred = pred.cpu().numpy().squeeze()
            else:
                # Only check for .mean if not a LinearOperator
                try:
                    from gpytorch.lazy import LinearOperator
                except ImportError:
                    LinearOperator = None
                if LinearOperator is not None and isinstance(pred, LinearOperator):
                    raise RuntimeError(f"Unexpected LinearOperator prediction type: {type(pred)}")
                elif hasattr(pred, 'mean') and isinstance(pred.mean, Tensor):
                    y_pred = pred.mean.cpu().numpy().squeeze()
                else:
                    raise RuntimeError(f"Unknown prediction type: {type(pred)}")
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"Fold MSE: {mse:.4f}, R2: {r2:.4f}")
        mses.append(mse)
        r2s.append(r2)
        y_true_folds.append(y_test)
        y_pred_folds.append(y_pred)
    return {'mse': float(np.mean(mses)), 'r2': float(np.mean(r2s)), 'mse_list': mses, 'r2_list': r2s,
            'y_true_folds': y_true_folds, 'y_pred_folds': y_pred_folds}

def parity_plot(y_true, y_pred, title="", save_path=None):
    """
    Create a parity plot (y_true vs y_pred) with a y=x reference line.
    Args:
        y_true: array-like of true values
        y_pred: array-like of predicted values
        title: optional plot title
        save_path: if provided, save the plot to this path
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolor='k')
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    plt.title(f"Parity Plot\nMSE: {mse:.4f}, R2: {r2:.4f}\n{title}")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    # plt.show()

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


if __name__ == "__main__":
    # Load the Lactate Solubility dataset
    loader = MolPropLoaderEXTENDED()

    path = r"U:\Github\BayesianThompsonSamplingOptimization\Datafiles\LIQEX_COSMObase.csv"

    loader.read_csv(path=path, smiles_column="SMILES", label_column="Selectivity", clip_index=100)
    # df = pd.read_csv(path)

    # Featurize using fragprints representations
    loader.featurize('ecfp_fragprints')
    y = loader.labels
    X_fp = loader.features
    print(f"Fragprints featurization dims: {X_fp.shape}, y dims: {y.shape}")
    loader.featurize('expert')
    X_expert = loader.features
    loader.featurize('molformer')
    X_molformer = loader.features
    # loader.featurize('chemeleon')
    # X_chemeleon = loader.features

    # Prepare feature sets and model initializers
    feature_sets = {
        "SE_Expert": X_expert,
        # "Tanimoto_Fragprints": X_fp,
        # "MolFormer": X_molformer,
        # "CheMeleon": X_chemeleon,
        # "Saas_CheMeleon": X_chemeleon,
        "Saas_Expert": X_expert,
        # "Saas_Fragprints": X_fp,
        "Saas_MolFormer": X_molformer,

    }
    model_initializers = {
        "SE_Expert": initialize_model_RFF_se_gp,
        "Tanimoto_Fragprints": initialize_model_tanimoto_gp,
        "MolFormer": initialize_model_se_gp,  # Use SE GP for MolFormer as well
        "CheMeleon": initialize_model_se_gp,  # Use SE GP for CheMeleon as well
        "Saas_CheMeleon": initialize_model_saas_gp, # Add Saas GP for CheMeleon
        "Saas_Expert": initialize_model_saas_gp,
        "Saas_Fragprints": initialize_model_saas_gp,
        "Saas_MolFormer": initialize_model_saas_gp,
    }

    fname = f"BO_COSMObase_{N_ITERS}iters_{N_TRIALS}trials_{holdout_set_size}holdout"

    # iterable = [
    #     (i, feature_sets, y, model_initializers)
    #     for i in range(1, N_TRIALS + 1)
    # ]

    # t_start = time.time()
    # # Each result is a dict: {feature_set_name: tensor, ..., 'random': tensor}
    # all_results = []
    # with ThreadPoolExecutor(max_workers=1) as executor:
    #     all_results = list(executor.map(run_trial_general, iterable))
    # t_end = time.time()
    # print(f"time = {t_end - t_start:>4.2f}.")

    # # Aggregate results for each feature set
    # feature_names = list(feature_sets.keys())
    # n_iters = N_ITERS + 1
    # results_by_feature = {name: [] for name in feature_names}
    # results_by_feature['random'] = []
    # for res in all_results:
    #     for name in feature_names:
    #         results_by_feature[name].append(res[name])
    #     results_by_feature['random'].append(res['Random_Search'])

    # # Convert to numpy arrays for plotting
    # results_np = {name: np.asarray(torch.stack(results_by_feature[name])) for name in results_by_feature}

    # # Save results
    # save_to_json([results_np[name] for name in feature_names] + [results_np['Random_Search']], fname)

    # # Compute quantiles and means for plotting
    # stats = {}
    # for name in results_np:
    #     stats[name] = {
    #         'mean': np.mean(results_np[name], axis=0),
    #         'lower': np.percentile(results_np[name], 2.5, axis=0),
    #         'upper': np.percentile(results_np[name], 97.5, axis=0)
    #     }

    # # Plotting
    # plt.figure(figsize=(7, 6))
    # colors = ["orange", "red", "blue", "green", "purple", "brown"]
    # markers = ['o', 's', '^', 'D', 'v', 'x']
    # for idx, name in enumerate(feature_names):
    #     plt.plot(range(n_iters), stats[name]['mean'], label=name.replace('_', ' '), color=colors[idx % len(colors)], marker=markers[idx % len(markers)])
    #     plt.fill_between(range(n_iters), stats[name]['lower'], stats[name]['upper'], color=colors[idx % len(colors)], alpha=0.2)
    # # Plot random search
    # plt.plot(range(n_iters), stats['Random_Search']['mean'], label="Random Search", color="blue", marker='o')
    # plt.fill_between(range(n_iters), stats['Random_Search']['lower'], stats['Random_Search']['upper'], color="blue", alpha=0.2)
    # plt.title("BO with Multiple Surrogate Models and Feature Sets")
    # plt.xlabel("Iteration")
    # plt.ylabel("Best Observed Value")
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(f"{fname}.pdf")

    # --- Cross-validation demonstration ---
    print("\nCross-validation results:")
    for name in feature_sets:
        print(f"\n{name}:")
        cv = cross_validation(model_initializers[name], feature_sets[name], y, n_splits=3, random_state=42)
        print(f"  MSE: {cv['mse']:.4f}")
        print(f"  R^2: {cv['r2']:.4f}")
        # Concatenate all folds for parity plot
        y_true_all = np.concatenate(cv['y_true_folds'])
        y_pred_all = np.concatenate(cv['y_pred_folds'])
        parity_plot(y_true_all, y_pred_all, title=f"Parity Plot: {name}", save_path=f"parity_plot_{name}.png")

