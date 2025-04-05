from src.thompsonsampling.MultiFidelity_TestFunctions import MF1
from src.thompsonsampling.util import get_initial_points
from src.thompsonsampling.visualize import visualize, Frame
import torch
import botorch
import gpytorch
import numpy as np
from botorch.sampling import SobolEngine
from gpytorch import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll

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

# Define Global Variables
dtype = torch.float64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_pts = 2           # Number of initial points per fidelity
n_samples = 10000   # Number of samples for calculating arm probabilities
N_Iterations = 20   # Number of iterations for the Bayesian optimization loop


class Arm():
    def __init__(self):
        self.train_X = None
        self.train_Y = None
        self.bounds = None
        self.model = None
        self.mll = None
        self.acqf = None
        
        self.new_x = None
        self.acqf_value = None
        # self.acqf_values_plot = None
        
    def OptimizeAcqf(self):
        self.acqf = botorch.acquisition.LogExpectedImprovement(self.model, best_f=self.train_Y.max())
        self.candidates, self.acqf_value = botorch.optim.optimize_acqf(self.acqf, bounds=self.bounds, q=1, num_restarts=10, raw_samples=512)
        
    
        
def main():
    """
    Main function to execute the Thompson Sampling algorithm.
    This function initializes the multi-fidelity test functions, performs space-filling sampling,
    fits Gaussian Process models, optimizes acquisition functions, and updates the training data
    iteratively. It also visualizes the results at each iteration.
    Steps:
        1. Initialize the multi-fidelity test functions and dictionaries and lists for the training data and visualization.
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
    mf1 = MF1()
    frames = []
    train_X = {}
    train_Y = {}
    seed = 3
    
    # ------------------------------------------------------------------
    # --- 2 Initial Points ---
    # ------------------------------------------------------------------
    for i in range(mf1.n_fidelities):
        train_X[i] = get_initial_points(dim=1, n_pts=n_pts, bounds=mf1.bounds, seed=seed)
        train_Y[i] = mf1.neg(train_X[i], mf1.fidelities[i])
        seed += 1

    resolution = 1000
    Plot_X = torch.linspace(mf1.lower_bound.item(), mf1.upper_bound.item(), resolution, dtype=dtype)
    Plot_Y = {}
    for i in range(mf1.n_fidelities):
        Plot_Y[i] = mf1.neg(Plot_X, mf1.fidelities[i])
    
    
    # ------------------------------------------------------------------
    # --- 3 Bayesian Loop ---
    # ------------------------------------------------------------------
    for _ in range(N_Iterations):
        # Define the model list
        models = {}
        mlls = {}
        new_X = {}
        GRV = {}
        Samples = {}
        kernel = {}
        Preds = {}
        Means = {}
        Upper = {}
        Lower = {}
        Acqf = {}
        
        # Define and optimize the acquisition function
        # Get the overal best obseravtion
        for j in range(mf1.n_fidelities):
            if i == 0:
                best_observation = train_Y[j].max()
            else:
                if train_Y[j].max() > best_observation:
                    best_observation = train_Y[j].max()
            
        
        for i in range(mf1.n_fidelities):
            # ------------------------------------------------------------------
            # --- 3.1 Define and train GP with MLL/MLE ---
            # ------------------------------------------------------------------
            kernel[i] = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(
                ard_num_dims=1, 
                lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(0.1, 1.0, sigma=0.1)))
            models[i] = botorch.models.SingleTaskGP(
                train_X=train_X[i], 
                train_Y=train_Y[i], 
                covar_module=kernel[i])
            mlls[i] = ExactMarginalLogLikelihood(models[i].likelihood, models[i])
            models[i].covar_module.outputscale = 100
            models[i].covar_module.raw_outputscale.requires_grad_(False)
            fit_gpytorch_mll(mlls[i])
            
            
            # ------------------------------------------------------------------
            # --- 3.2 Optimize Acquisition Function ---
            # ------------------------------------------------------------------
            acqf = botorch.acquisition.LogExpectedImprovement(models[i], train_Y[i].max())
            acqf = botorch.acquisition.LogExpectedImprovement(models[i], best_f=best_observation)
        
            new_X[i], _ = botorch.optim.optimize_acqf(acqf, bounds=mf1.bounds, q=1, num_restarts=10, raw_samples=512)
            
            # ------------------------------------------------------------------
            # --- 3.3 Thompson Sampling (Sample from GP) ---
            # ------------------------------------------------------------------
            
            GRV[i] = models[i].posterior(new_X[i])
            temp = []
            for _ in range(n_samples):
                temp.append(GRV[i].sample())
            Samples[i] = torch.tensor(temp)
            
            # ------------------------------------------------------------------
            # --- 3.4 Data for Visualization ---
            # ------------------------------------------------------------------
            
            # Get GP prediction data for plotting
            Preds[i] = models[i].posterior(Plot_X.view(-1,1))
            Means[i] = Preds[i].mean
            Lower[i], Upper[i] = Preds[i].confidence_region()

            # Get acquisition function values for plotting
            Acqf[i] = torch.exp(acqf(Plot_X.view(-1,1,1)))

        # ------------------------------------------------------------------
        # --- 3.4 Update Training Data ---
        # ------------------------------------------------------------------
        temp = []
        for j in range(mf1.n_fidelities):
            temp.append(Samples[j][i])
        temp = torch.tensor(temp)
        max_index = torch.argmax(temp).item()
        print(max_index)
        new_Y = mf1.neg(new_X[max_index], mf1.fidelities[max_index])
        
        train_X[max_index] = torch.cat((train_X[max_index].clone(), new_X[max_index].clone()))
        train_Y[max_index] = torch.cat((train_Y[max_index].clone(), new_Y.clone()))
        
        # ------------------------------------------------------------------
        # --- 4 Visualization ----------------------------------------------
        # ------------------------------------------------------------------
        
        # Calculate the arm selection probability from drawn samples:
        for i in range(n_samples):
            temp = []
            for j in range(mf1.n_fidelities):
                temp.append(Samples[j][i])
            temp = torch.tensor(temp)
            max_index = torch.argmax(temp)
            if i == 0:
                P = torch.zeros(mf1.n_fidelities, dtype=dtype)
            P[max_index] += 1
        P = P / n_samples
        
        frames.append(Frame(
            N_fidelities=mf1.n_fidelities,
            train_X={k: v.clone() for k, v in train_X.items()},
            train_Y={k: v.clone() for k, v in train_Y.items()},
            Plot_X=Plot_X,
            Plot_Y=Plot_Y,
            Preds=Preds,
            Means=Means,
            Upper=Upper,
            Lower=Lower,
            Acqf=Acqf,
            new_X=new_X,
            Sample=Samples,
            Sample_index=max_index,
            Probabilities=P
        ))
    
    visualize(frames)
    
    # visualize(frame.mf1, frame.models, frame.train_X, frame.train_Y, frame.new_X, frame.Sample, frame.P)

    # Calculate the probability of the individual samples beeing the highest:
    # P_matrix = torch.ones((mf1.n_fidelities, mf1.n_fidelities))
    # for i in range(mf1.n_fidelities):
    #     for j in range(mf1.n_fidelities):
    #         if i != j:
    #             P_matrix[i,j] = norm.cdf((mean[i].numpy() - mean[j].numpy()) / np.sqrt(var[i].numpy() + var[j].numpy())).item()
    
    # P = P_matrix.prod(dim=1)
    # P_ = P_matrix.prod(dim=0)
    # print(P)
    # print(P_)
    # print(P_matrix)
    # print(torch.sum(P))
    # print(torch.sum(P_))
    
if __name__ == "__main__":
    main()