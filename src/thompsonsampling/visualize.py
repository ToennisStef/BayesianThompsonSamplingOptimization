from dataclasses import dataclass
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
import torch

@dataclass
class Frame:
    """
    Dataclass for Frame
    Stores data for visualization of Thompson Sampling algorithm
    """
    N_fidelities: int   # Number of fidelities
    train_X: dict       # Training data
    train_Y: dict       # Training labels
    Plot_X: dict        # Plotting data
    Plot_Y: dict        # Plotting labels
    Preds: dict         # Predictions
    Means: dict         # Mean of GP predictions
    Upper: dict         # Upper confidence bound of GP predictions
    Lower: dict         # Lower confidence bound of GP predictions
    Acqf: dict          # Acquisition function
    new_X: dict         # New data point
    Sample: dict
    Sample_index: int
    Probabilities: torch.tensor



def visualize(Frames):   
    
    fig, ax = plt.subplots(3, 1, figsize=(8, 12))
    plt.subplots_adjust(bottom=0.25)

    slider_ax = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    frame_slider = Slider(slider_ax, 'Frame', 0, len(Frames) - 1, valinit=0, valstep=1)

    def update(frame_idx):
        ax[0].clear()
        ax[1].clear()
        ax[2].clear()

        frame = Frames[frame_idx]

        N_fidelities = frame.N_fidelities
        train_X = frame.train_X
        train_Y = frame.train_Y
        Plot_X = frame.Plot_X
        Plot_Y = frame.Plot_Y
        Means = frame.Means
        Upper = frame.Upper
        Lower = frame.Lower
        new_X = frame.new_X
        Sample = frame.Sample
        Probs = frame.P

        for i in range(N_fidelities):
            ax[0].plot(Plot_X.numpy(), Plot_Y[i].numpy(), label=f'f{i+1}', color=f'C{i}')
            ax[0].scatter(train_X[i].numpy(), train_Y[i].numpy(), color=f'C{i}', marker='x', label=f'f{i+1} training data')
            ax[0].plot(Plot_X.numpy(), Means[i].detach().numpy(), color=f'C{i}', linestyle='dashed', label=f'f{i+1} mean prediction')
            ax[0].fill_between(Plot_X.numpy(), Lower[i].detach().numpy(), Upper[i].detach().numpy(), color=f'C{i}', alpha=0.25, label=f'f{i+1} confidence region')
            ax[0].scatter(torch.full_like(Sample[i][1], new_X[i].item()), Sample[i][1].numpy(), color=f'C{i}', marker='o', label=f'f{i+1} sample')

            ax[1].plot(Plot_X.numpy(), frame.Acqf[i].detach().numpy(), label=f'f{i+1} acquisition function', color=f'C{i}')
            
        # ax[0].legend()
        ax[0].set_xlabel('x')
        ax[0].set_ylabel('y')
        ax[0].set_title('Function Optimization Results')

        ax[1].set_xlabel('x')
        ax[1].set_ylabel(r'$\alpha_{LogEI}$')

        P_numpy = Probs.numpy()
        bottom = 0
        labels = [f'X{i+1}' for i in range(len(P_numpy))]

        for frame_idx in range(frame_idx + 1):
            frame = Frames[frame_idx]
            P_numpy = frame.P.numpy()
            bottom = 0
            for i, prob in enumerate(P_numpy):
                ax[2].bar(frame_idx + 1, prob, bottom=bottom, color=f"C{i}", label=labels[i] if frame_idx == 0 else "")
                bottom += prob

        if frame_idx == 0:
            ax[2].legend()
        ax[2].set_title('Probability of Each Fidelity')
        ax[2].set_xlabel('Frame')
        ax[2].set_ylabel('Probability')

    frame_slider.on_changed(update)
    update(0)  # Initialize the plot with the first frame
    plt.show()
