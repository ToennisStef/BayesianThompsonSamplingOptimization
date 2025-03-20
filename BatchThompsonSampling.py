from src.thompsonsampling.MultiFidelity_TestFunctions import MF1
from src.thompsonsampling.Arms import Arm
from src.thompsonsampling.batchVisualize import *

import torch
import logging
import copy
from matplotlib.widgets import Slider

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
            1.1 global variable initialization.
            1.2 Arm initialization.
        2. Perform the Bayesian optimization loop:
            2.1. Arm GP calculations (GP fitting, prediction, and acquisition function optimization).
            2.2. q-best Arm selection ()
        3. Visualize the results at each iteration.
    Returns:
        None
    Notes: 
        The covariance kernel is defined as a scaled RBF kernel with Automatic Relevance Determination (ARD).
        The output scale is fixed to 100, to enable arms to still be potentially selected even if the data samples are not in favor of the respective arm.
    """
    # ------------------------------------------------------------------
    # --- 1 Initialization ---
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # --- 1.1 Initialization ---
    # ------------------------------------------------------------------
    
    # Define Global Variables
    dtype = torch.float64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    n_pts = 2           # Number of initial points per fidelity
    n_samples = 10000   # Number of samples for calculating arm probabilities
    n_iterations = 5   # Number of iterations for the Bayesian optimization loop

    testfunction = MF1()
    bounds = testfunction.bounds
    dim = testfunction.dim
    funcs = testfunction.funcs_neg
    q_batch = 5
    arm_q_batch = 2
    seed = 0
    
    frames = []
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
                
    
    # ------------------------------------------------------------------
    # --- 1.2 Arm initialization ---
    # ------------------------------------------------------------------
    
    arms = [Arm(y_func=funcs[f], 
                dim=dim, 
                n_pts=n_pts, 
                n_samples=n_samples,
                bounds=bounds,
                q_batch=arm_q_batch, 
                seed=seed+f,
                dtype=dtype,
                device=device) 
            for f in testfunction.fidelities]

    total_arm_q_batch = 0
    for arm in arms:
        total_arm_q_batch += arm.q_batch
        
    assert total_arm_q_batch >= q_batch, "Total arm_q_batch must be greater than q_batch"
    
    
    # ------------------------------------------------------------------
    # --- 3 Bayesian Loop ---
    # ------------------------------------------------------------------    
    for Iter in range(n_iterations):
        # Copy each arm to a list for plotting
        arms_init = []
        # ------------------------------------------------------------------
        # --- 3.1 Arm GP calculations ---
        # ------------------------------------------------------------------
        for arm in arms:
            arm.CalcGP()
            arm.Predict()
            arm.OptimizeAcqf()
            arm.SampleAtCandidate()
            
            arms_init.append(PArm(pred_x=arm.pred_x,
                                  pred_mean=arm.pred_mean,
                                  pred_lower=arm.pred_lower,
                                  pred_upper=arm.pred_upper,
                                  train_X=arm.train_X,
                                  train_Y=arm.train_Y,
                                  Samples=arm.Samples,
                                  candidates=arm.candidates,
                                  instance_id=arm.instance_id,
                                  GRV=arm.GRV,
                                  y_func=arm.y_func))
                
        # ------------------------------------------------------------------
        # --- 3.5 q-best Arm selection ---
        # ------------------------------------------------------------------
        arms_q = [] # List of best arms at each q-step
        q_step = [] # List of all arms at each q-step (List of List of arms)
        armSelectProb = [] # Probability of selecting each arm at each q-step
        for q in range(q_batch):
            
            # Copy each arm to a list for plotting
            arms_q_step = [] # List of all arms at current q-step
            for arm in arms:
                arms_q_step.append(PArm(pred_x=arm.pred_x,
                                  pred_mean=arm.pred_mean,
                                  pred_lower=arm.pred_lower,
                                  pred_upper=arm.pred_upper,
                                  train_X=arm.train_X,
                                  train_Y=arm.train_Y,
                                  Samples=arm.Samples,
                                  candidates=arm.candidates,
                                  instance_id=arm.instance_id,
                                  GRV=arm.GRV,
                                  y_func=arm.y_func))
            q_step.append((arms_q_step))    
            
            # Identify the arm with the highest sample value
            highest_value = float('-inf')
            best_arm = None
            for arm in arms:
                max_sample_value = arm.Samples[0].max().item()                
                if max_sample_value > highest_value:
                    highest_value = max_sample_value
                    best_arm = arm
            
            
            # Calculate the probability of selecting each arm
            samples = torch.stack([arm.Samples.reshape(-1,1) for arm in arms], dim=1)
            max_indices = samples.argmax(dim=1).flatten()
            counts = torch.bincount(max_indices, minlength=len(arms))
            probabilities = counts / n_samples
            armSelectProb.append({arm.instance_id: probabilities[i].item() for i, arm in enumerate(arms)})
            
            # Update data for the best arm
            if best_arm is not None:
                
                # Copy the best arm to a list for plotting
                arms_q.append(PArm(pred_x=best_arm.pred_x,
                                  pred_mean=best_arm.pred_mean,
                                  pred_lower=best_arm.pred_lower,
                                  pred_upper=best_arm.pred_upper,
                                  train_X=best_arm.train_X,
                                  train_Y=best_arm.train_Y,
                                  Samples=best_arm.Samples,
                                  candidates=best_arm.candidates,
                                  instance_id=best_arm.instance_id,
                                  GRV=best_arm.GRV,
                                  y_func=best_arm.y_func))
                
                logging.info(f"Best arm selected: {best_arm.seed} with value: {highest_value}")
                best_arm.UpdateData()
                best_arm.SampleAtCandidate() # resample at the next candidate point
            else:
                logging.warning("No best arm found.")
        
        # ------------------------------------------------------------------
        # --- 4 Visualisation ---
        # ------------------------------------------------------------------
        
        # fig1, axs1 = create_figure_layout(rows=1, cols=1, width=10, height=12)
        # colors = get_arm_colors(arms)
        # ids = get_arm_ids(arms)
        # plot_arms(arms_init, axs1)
        # plot_q_arms(arms_q, axs1, colors=colors)
        # fig2, axs2 = create_figure_layout(rows=1, cols=1, width=10, height=12)
        # plot_armSelectProb(armSelectProb, axs2, colors=colors)
        # fig3, axs3 = create_figure_layout(rows=1, cols=1, width=10, height=12)
        # plot_bestSample(q_step, axs3, colors=colors)
        
        # plt.show()

        frames.append(Frame(arms_init=arms_init, 
                       arms_q=arms_q, 
                       armSelectProb=armSelectProb, 
                       q_step=q_step))
        
    colors = get_arm_colors(arms)
    ids = get_arm_ids(arms)
    import matplotlib.pyplot as plt

    # Create figures and axes for the interactive plots
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(left=0.1, bottom=0.25)

    # Initial plot for the first iteration
    plot_arms(frames[0].arms_init, ax1, ids=ids, colors=colors)
    plot_q_arms(frames[0].arms_q, ax1, colors=colors)
    plot_armSelectProb(frames[0].armSelectProb, ax2, ids=ids, colors=colors)
    plot_bestSample(frames[0].q_step, ax3, ids=ids, colors=colors)

    # Slider for selecting the iteration
    ax_slider1 = plt.axes([0.1, 0.1, 0.8, 0.05], facecolor='lightgoldenrodyellow')
    slider1 = Slider(ax_slider1, 'Iteration', 0, len(frames) - 1, valinit=0, valstep=1)

    # Update function for the slider
    def update(val):
        iteration = int(slider1.val)
        ax1.clear()
        ax2.clear()
        ax3.clear()
        plot_arms(frames[iteration].arms_init, ax1, ids=ids, colors=colors)
        plot_q_arms(frames[iteration].arms_q, ax1, colors=colors)
        plot_armSelectProb(frames[iteration].armSelectProb, ax2, ids=ids, colors=colors)
        plot_bestSample(frames[iteration].q_step, ax3, ids=ids, colors=colors)
        fig1.canvas.draw_idle()
        fig2.canvas.draw_idle()
        fig3.canvas.draw_idle()

    # Connect the update function to the slider
    slider1.on_changed(update)

    plt.show()
                    
    
if __name__ == "__main__":
    main()