from src.thompsonsampling.MultiFidelity_TestFunctions import MF1
from src.thompsonsampling.util import get_initial_points
from src.thompsonsampling.visualize import visualize, Frame
from src.thompsonsampling.Arms import Arm
import torch

import numpy as np
from botorch.sampling import SobolEngine
from gpytorch import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
import logging

"""
Thompson Sampling Implementation
This script implements Thompson Sampling for a multi-fidelity test functions using Gaussian Processes.
It utilizes the BoTorch and GPyTorch libraries for Gaussian Process modeling and optimization.
Modules:
    src.thompsonsampling.MultiFidelity_TestFunctions: Contains the multi-fidelity test functions.
    torch: PyTorch library for tensor computations.
    botorch: Bayesian optimization library built on top of PyTorch.
    gpytorch: Gaussian Process library built on top of PyTorch.
    numpy: Library for numerical computations.
Functions:
    main(): Main function to execute the Thompson Sampling algorithm.
Usage:
    Run this script directly to execute the Thompson Sampling algorithm.
"""     
        
def main():
    """
    Main function to execute the Thompson Sampling algorithm.
    This function initializes the multi-fidelity test functions, performs space-filling sampling,
    fits Gaussian Process models, optimizes acquisition functions, and updates the training data
    iteratively. It also visualizes the results at each iteration.
    Steps:
        1. Initialization
            1.1 Test function,
            1.1 Initialize the multi-fidelity test functions and dictionaries and lists for the training data and visualization.
        2. Perform space-filling sampling for the initial points.
        3. Perform the Bayesian optimization loop:
        3.1. Define and train the Gaussian Process models using the marginal log likelihood.
        3.3. Define and optimize the acquisition functions to select new points.
        3.4. Perform the Thompson Sampling by sampling from the posterior at the new points.
        3.5. Update the training data with new observations.
        3.6. Get data for plotting (GP predictions, acquisition functions, etc.).
        4. Visualize the results at each iteration.
    Returns:
        None
    Notes: 
        The covariance kernel is defined as a scaled RBF kernel with Automatic Relevance Determination (ARD).
        The output scale is fixed to 100, to enable arms to still be potentially selected even if the datasample are not in favour of the respective arm.
    """
    # ------------------------------------------------------------------
    # --- 1 Initialization ---
    # ------------------------------------------------------------------
    
    # Define Global Variables
    dtype = torch.float64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    n_pts = 2           # Number of initial points per fidelity
    n_samples = 10000   # Number of samples for calculating arm probabilities
    n_iterations = 1   # Number of iterations for the Bayesian optimization loop

    testfunction = MF1()
    bounds = testfunction.bounds
    dim = testfunction.dim
    funcs = testfunction.funcs_neg
    q_batch = 5
    seed = 0
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
                
    
    # ------------------------------------------------------------------
    # --- 1.1 Arms ---
    # ------------------------------------------------------------------
    
    arms = [Arm(y_func=funcs[f], 
                dim=dim, 
                n_pts=n_pts, 
                n_samples=n_samples,
                bounds=bounds,
                q_batch=2, 
                seed=seed+f,
                dtype=dtype,
                device=device) 
            for f in testfunction.fidelities]

    for arm in arms:
        print(f"Arm {arm}:")
        
    # ------------------------------------------------------------------
    # --- 3 Bayesian Loop ---
    # ------------------------------------------------------------------
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    
    
    for Iter in range(n_iterations):
        for arm in arms:
            arm.CalcGP()
            arm.Predict()
            arm.OptimizeAcqf()
            arm.SampleAtCandidate()
                    
        # ------------------------------------------------------------------
        # --- 3.5 Arm selection ---
        # ------------------------------------------------------------------
        for q in range(q_batch):
            
            # Identify the arm with the highest sample value
            highest_value = float('-inf')
            best_arm = None
            for arm in arms:
                max_sample_value = arm.Samples[0].max().item()                
                if max_sample_value > highest_value:
                    highest_value = max_sample_value
                    best_arm = arm
                    
            # Update data for the best arm
            if best_arm is not None:
                logging.info(f"Best arm selected: {best_arm.seed} with value: {highest_value}")
                best_arm.UpdateData()
                best_arm.SampleAtCandidate() # resample at the next candidate point
            else:
                logging.warning("No best arm found.")
        
        
        # ------------------------------------------------------------------
        # --- 4 Visualisation ---
        # ------------------------------------------------------------------
        
        for i, arm in enumerate(arms):
            
            # Visualize the GP predictions
            plt.plot(arm.pred_x.detach().numpy(), 
                     arm.pred_mean.detach().numpy(),
                     label=f'Fidelity {i}',
                     color=f"C{i}")
            plt.plot(arm.train_X.detach().numpy(),
                     arm.train_Y.detach().numpy(),
                     'x',
                     color=f"C{i}")
            plt.fill_between(arm.pred_x.detach().numpy().reshape(-1),
                             arm.pred_lower.detach().numpy(),
                             arm.pred_upper.detach().numpy(),
                             alpha=0.5,
                             color=f"C{i}")

            # Visualize the test functions
            plt.plot(arm.pred_x.detach().numpy(), 
                     arm.y_func(arm.pred_x).detach().numpy()
                     , label=f'Fidelity {i}', color=f"C{i}", linestyle='dashed')
        plt.show()

        
        
        
            
    
if __name__ == "__main__":
    main()