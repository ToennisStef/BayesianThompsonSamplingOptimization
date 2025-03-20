from dataclasses import dataclass
import matplotlib.pyplot as plt
import torch

@dataclass
class Frame:
    """
    Dataclass for Frame
    Stores data for visualization of Batch Thompson Sampling algorithm
    """
    arms_init: list
    arms_q: list
    armSelectProb: list

@dataclass
class PArm:
    """
    Dataclass for PArm
    Stores data for visualization of Batch Thompson Sampling algorithm
    """
    pred_x: torch.Tensor
    pred_mean: torch.Tensor
    pred_lower: torch.Tensor
    pred_upper: torch.Tensor
    train_X: torch.Tensor
    train_Y: torch.Tensor
    Samples: torch.Tensor
    candidates: torch.Tensor
    instance_id: str
    y_func: callable

def create_figure_layout(rows=1, cols=1, width=10, height=6):
    """
    Creates and returns a figure/axes layout.
    """
    fig, axs = plt.subplots(rows, cols, figsize=(width, height))
    return fig, axs


def plot_arms(
    arms: list, #List of PArm instances
    ax,
    colors: dict = None
    ) -> None:
        
    # Assign unique colors to each arm
    if colors is None:
        colors = {arm.instance_id: f"C{i}" for i, arm in enumerate(arms)}


    for i, arm in enumerate(arms):            
        # Visualize the GP predictions
        ax.plot(arm.pred_x.detach().numpy(), 
                    arm.pred_mean.detach().numpy(),
                    label=f'Fidelity {i}',
                    color=colors[arm.instance_id])
        ax.plot(arm.train_X.detach().numpy(),
                    arm.train_Y.detach().numpy(),
                    'x',
                    color=colors[arm.instance_id])
        ax.fill_between(arm.pred_x.detach().numpy().reshape(-1),
                            arm.pred_lower.detach().numpy(),
                            arm.pred_upper.detach().numpy(),
                            alpha=0.5,
                            color=colors[arm.instance_id])
        # Visualize the test functions
        ax.plot(arm.pred_x.detach().numpy(), 
                    arm.y_func(arm.pred_x).detach().numpy()
                    , label=f'Fidelity {i}', color=colors[arm.instance_id], linestyle='dashed')
        
        # Visualize the samples
        if arm.candidates.shape[0] > 0:        
            ax.scatter(torch.full_like(arm.Samples.flatten() ,arm.candidates[0,...].detach().item()), 
                        arm.Samples.detach().flatten(), 
                        color=colors[arm.instance_id], 
                        marker='o', 
                        label=f'f{i+1} sample', 
                        s=3)
    
    # ax.legend()

def get_arm_colors(arms):
    """
    Returns a dictionary mapping arm instance ids to unique colors.
    """
    colors = {arm.instance_id: f"C{i}" for i, arm in enumerate(arms)}
    return colors

def get_arm_ids(arms):
    """
    Returns a dictionary mapping arm instance ids to unique colors.
    """
    ids = {arm.instance_id: i+1 for i, arm in enumerate(arms)}
    return ids
        
def plot_q_arms(arms, ax, colors):
        
    # Assign unique colors to each arm
    # colors = {arm.instance_id: f"C{i}" for i, arm in enumerate(arms)}

    for i, arm in enumerate(arms):            
        # Visualize the test functions
        if arm.candidates.shape[0] > 0:
            ax.vlines(arm.candidates[0,...].detach().item(), 
                        ymin=-10, 
                        ymax=10, 
                        color=colors[arm.instance_id], 
                        )        
            ax.scatter(torch.full_like(arm.Samples ,arm.candidates[0,...].detach().item()), 
                        arm.Samples.detach(), 
                        color=colors[arm.instance_id], 
                        marker='o', 
                        label=f'f{i+1} sample', 
                        s=3)
    # ax.legend()
    
def plot_armSelectProb(armSelectProb, ax, colors):
    
    for q, q_step in enumerate(armSelectProb):
        bottom = 0
        for instance_id, prob in q_step.items():
            ax.bar(q + 1, prob, bottom=bottom, color=colors[instance_id])
            bottom += prob

    # ax.legend()
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Probability')
    
def plot_bestSample(arms, ax, colors, ids):
    """
    Visualizes the best samples from each arm.
    """
    for i, arm in enumerate(arms):
        ax.scatter(torch.full_like(arm.Samples[0,...].flatten(), ids[arm.instance_id]), 
                arm.Samples[0,...].flatten(), 
                color=colors[arm.instance_id], 
                marker='o', 
                label=f'f{i+1} sample', 
                s=10)
        
    # ax.legend()
    ax.set_xlabel('Arm')
    ax.set_ylabel('Sample')
    ax.set_title('Best Samples from Each Arm')
    