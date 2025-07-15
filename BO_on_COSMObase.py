import warnings
warnings.filterwarnings("ignore")

import time
import numpy as np
import pandas as pd

import torch
from sklearn.model_selection import train_test_split

from gpytorch import ExactMarginalLogLikelihood


from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import LogExpectedImprovement as ExpectedImprovement

from BayesianOptimization.featurizers import MolPropLoaderEXTENDED
from BayesianOptimization.model import (
    initialize_model_tanimoto_gp,
    initialize_model_se_gp,
    initialize_model_RFF_se_gp,
    initialize_model_dkl_gp,
    initialize_model_deep_gp,
    optimize_acqf_and_get_observation,
    RandomSearchHelper,
)
from BayesianOptimization.plotting import parity_plot 
from BayesianOptimization.eval import cross_validation

import os

# Experiment parameters
N_TRIALS = 10
holdout_set_size = 0.95
N_ITERS = 50
verbose = True



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
            if isinstance(mll, ExactMarginalLogLikelihood):
                # Fit the model using ExactMarginalLogLikelihood
                fit_gpytorch_mll(mll)
            else:  # assume is function to train model
                mll(model)
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


if __name__ == "__main__":
    # Load the Lactate Solubility dataset
    loader = MolPropLoaderEXTENDED()

    # path = r"U:\Github\BayesianThompsonSamplingOptimization\Datafiles\LIQEX_COSMObase.csv"
    path = r"C:\Users\kabe02-lokal\Documents\Github\BayesianThompsonSamplingOptimization\Datafiles\LIQEX_COSMObase.csv"

    loader.read_csv(path=path, smiles_column="SMILES", label_column="Selectivity", clip_index=1000)
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
        # "SE_Expert": X_expert,
        # "Tanimoto_Fragprints": X_fp,
        # "MolFormer": X_molformer,
        "DKL_GP": X_expert,  # Use expert features for DKL as example
        # "DKL_GP": X_molformer,  # Use MolFormer features for DKL as example
        # "D_GP": X_expert,  # Use MolFormer features for DGP as example
    }
    model_initializers = {
        "SE_Expert": initialize_model_RFF_se_gp,
        "Tanimoto_Fragprints": initialize_model_tanimoto_gp,
        "MolFormer": initialize_model_se_gp,  # Use SE GP for MolFormer as well
        "DKL_GP": initialize_model_dkl_gp,    # Add DKL model
        "D_GP": initialize_model_deep_gp    # Add DGP model
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

