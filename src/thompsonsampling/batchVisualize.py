from dataclasses import dataclass
import matplotlib.pyplot as plt
import torch
from scipy.stats import norm

@dataclass
class Frame:
    """
    Dataclass for Frame
    Stores data for visualization of Batch Thompson Sampling algorithm
    """
    arms_init: list
    arms_q: list
    armSelectProb: list
    q_step: list

@dataclass
class PArm:
    """
    Dataclass for PArm
    Stores data for visualization of Batch Thompson Sampling algorithm
    """
    model: torch.nn.Module
    pred_x: torch.Tensor
    pred_mean: torch.Tensor
    pred_lower: torch.Tensor
    pred_upper: torch.Tensor
    train_X: torch.Tensor
    train_Y: torch.Tensor
    Samples: torch.Tensor
    candidates: torch.Tensor
    instance_id: str
    GRV: torch.distributions.MultivariateNormal
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
    ids: dict = None,
    colors: dict = None,
    ) -> None:
        
    # Assign unique colors to each arm
    if colors is None:
        colors = {arm.instance_id: f"C{i}" for i, arm in enumerate(arms)}

    if ids is None:
        ids = {arm.instance_id: i+1 for i, arm in enumerate(arms)}

    for i, arm in enumerate(arms):            
        # Visualize the GP predictions
        ax.plot(arm.pred_x.detach().numpy(), 
                    arm.pred_mean.detach().numpy(),
                    label=f"Arm {ids[arm.instance_id]} mean pred", 
                    color=colors[arm.instance_id]
                    )
        
        ax.plot(arm.train_X.detach().numpy(),
                    arm.train_Y.detach().numpy(),
                    'x',
                    label=f"Arm {ids[arm.instance_id]} data points", 
                    color=colors[arm.instance_id]
                    )
        
        ax.fill_between(arm.pred_x.detach().numpy().reshape(-1),
                            arm.pred_lower.detach().numpy(),
                            arm.pred_upper.detach().numpy(),
                            alpha=0.5,
                            color=colors[arm.instance_id],
                            label=f"Arm {ids[arm.instance_id]} CI",
                            )
        # Visualize the test functions
        ax.plot(arm.pred_x.detach().numpy(),     
                arm.y_func(arm.pred_x).detach().numpy(), 
                color=colors[arm.instance_id], 
                linestyle='dashed',
                label=f"Arm {ids[arm.instance_id]} true func",
                )

        # Visualize the best candidates:
        for candidate in arm.candidates:
            
            m = arm.model.posterior(candidate).mean.detach()
            std = arm.model.posterior(candidate).variance.sqrt().detach()
            ax.vlines(candidate.detach().item(), 
                        ymin=m - 3*std, 
                        ymax=m + 3*std, 
                        color=colors[arm.instance_id], 
                        # linestyle='dashed',
                        label=f"Arm {ids[arm.instance_id]} candidate",
                        )
                   
        # Visualize the samples
        # if arm.candidates.shape[0] > 0:        
        #     ax.scatter(torch.full_like(arm.Samples.flatten() ,arm.candidates[0,...].detach().item()), 
        #                 arm.Samples.detach().flatten(), 
        #                 color=colors[arm.instance_id], 
        #                 marker='o', 
        #                 label=f"Arm {ids[arm.instance_id]} candidate samples",    
        #                 s=3)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), ncol=len(arms))
    ax.set_xlabel('x')
    ax.set_ylabel('y')

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
                        # label=arm.instance_id, 
                        s=3)
    
def plot_armSelectProb(armSelectProb, ax, ids, colors):
    
    for q, q_step in enumerate(armSelectProb):
        bottom = 0
        for instance_id, prob in q_step.items():
            ax.bar(q + 1, 
                   prob, 
                   bottom=bottom, 
                   color=colors[instance_id],
                   label=f"Arm {ids[instance_id]}"
                   )
            bottom += prob

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    
    # ax.legend()
    ax.set_xlabel('q-step')
    ax.set_ylabel('Arm Selection Probability')
    
def plot_bestSample(q_step, ax, ids, colors):
    """
    Visualizes the best samples from each arm.
    """
    
    for q, q_step in enumerate(q_step):
        for arm in q_step:
            # Slightly adjust the color based on the index i
            color = colors[arm.instance_id]
            
            if arm.GRV is None:
                continue
            
            mean = arm.GRV.mean.detach()
            stdd = arm.GRV.stddev.detach()
            lower = arm.GRV.mean - 4*arm.GRV.stddev
            upper = arm.GRV.mean + 4*arm.GRV.stddev
            x = torch.linspace(lower.item(), upper.item(), 200)
            y = norm.pdf(x ,loc=mean, scale=stdd) # arm.GRV.log_prob(x).exp()
            y = y / y.max() * 0.95
            ax.fill_betweenx(x.flatten(), 
                    q + 1, 
                    q + 1 + y.flatten(), 
                    color=color, 
                    alpha=0.5)
            
            ax.scatter(q+1, 
                arm.Samples[0,...].flatten(), 
                color=color, 
                marker='o',
                label=f"Arm {ids[arm.instance_id]} best sample",
                s=10)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    ax.set_xlabel('q-step')
    ax.set_ylabel('arm sample')
    ax.set_title('Best Samples from Each Arm')
    