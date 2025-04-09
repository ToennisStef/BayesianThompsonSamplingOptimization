from src.thompsonsampling.MultiFidelity_TestFunctions import MF1
import torch
import botorch
import gpytorch
import numpy as np
from botorch.sampling import SobolEngine
import matplotlib.pyplot as plt
from gpytorch import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from scipy.stats import norm
from dataclasses import dataclass
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider

dtype = torch.float64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_pts = 2
n_samples = 10000


@dataclass
class Frame:
    N_fidelities: int
    models: botorch.models.gpytorch.GPyTorchModel
    train_X: dict
    train_Y: dict
    Plot_X: torch.tensor
    Plot_Y: dict
    Preds: dict
    Means: dict
    Upper: dict
    Lower: dict
    Acqf: dict
    new_X: dict
    Sample: dict
    Sample_index: int
    P: torch.tensor


def get_initial_points(dim: int, n_pts: int, bounds: torch.tensor = torch.tensor([-1, 1]), seed=0):
    sobol = SobolEngine(dimension=dim, scramble=True, seed=seed)
    X_init = sobol.draw(n=n_pts).to(dtype=dtype, device=device)
    lower_bound, upper_bound = bounds
    X_init = lower_bound + (upper_bound - lower_bound) * X_init  # scale points to the given bounds
    return X_init

#def visualize(mf1, models, train_X, train_Y, new_X, Sample, P):   
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
            
        ax[0].legend()
        ax[0].set_xlabel('x')
        ax[0].set_ylabel('y')
        ax[0].set_title('Function Optimization Results')

        P_numpy = Probs.numpy()
        bottom = 0
        labels = [f'X{i+1}' for i in range(len(P_numpy))]

        for i, prob in enumerate(P_numpy):
            ax[2].bar(1, prob, bottom=bottom, color=f"C{i}", label=labels[i])
            bottom += prob

        ax[2].legend()
        ax[2].set_title('Probability of Each Fidelity')

    frame_slider.on_changed(update)
    update(0)  # Initialize the plot with the first frame
    plt.show()


def main():
    # Define space filling sampling for the functions
    mf1 = MF1()
    N_Iterations = 20
    frames = []
    train_X = {}
    train_Y = {}
    seed = 3
    for i in range(mf1.n_fidelities):
        train_X[i] = get_initial_points(dim=1, n_pts=n_pts, bounds=mf1.bounds, seed=seed)
        train_Y[i] = mf1.neg(train_X[i], mf1.fidelities[i])
        seed += 1

    resolution = 1000
    Plot_X = torch.linspace(mf1.lower_bound.item(), mf1.upper_bound.item(), resolution, dtype=dtype)
    Plot_Y = {}
    for i in range(mf1.n_fidelities):
        Plot_Y[i] = mf1.neg(Plot_X, mf1.fidelities[i])
    
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
        Plot_Acqf = {}
        for i in range(mf1.n_fidelities):
            # Define the model and the marginal log likelihood and fit the model
            kernel[i] = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=1, lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(0.1, 1.0, sigma=0.1)))
            models[i] = botorch.models.SingleTaskGP(train_X[i], train_Y[i], covar_module=kernel[i])
            mlls[i] = ExactMarginalLogLikelihood(models[i].likelihood, models[i])
            models[i].covar_module.outputscale = 100
            models[i].covar_module.raw_outputscale.requires_grad_(False)
            fit_gpytorch_mll(mlls[i])
            
            # Define and optimize the acquisition function
            # Get the overal best obseravtion
            for j in range(mf1.n_fidelities):
                if i == 0:
                    best_observation = train_Y[j].max()
                else:
                    if train_Y[j].max() > best_observation:
                        best_observation = train_Y[j].max()
            
            # acqf = botorch.acquisition.LogExpectedImprovement(models[i], train_Y[i].max())
            acqf = botorch.acquisition.LogExpectedImprovement(models[i], best_f=best_observation)
            new_X[i], Acqf[i] = botorch.optim.optimize_acqf(acqf, bounds=mf1.bounds, q=1, num_restarts=10, raw_samples=512)
            
            # Sample from the posterior at the new point
            GRV[i] = models[i].posterior(new_X[i])
            temp = []
            for _ in range(n_samples):
                temp.append(GRV[i].sample())
            Samples[i] = torch.tensor(temp)
            
            # Get the prediction data for plotting
            Preds[i] = models[i].posterior(Plot_X.view(-1,1))
            Means[i] = Preds[i].mean
            Lower[i], Upper[i] = Preds[i].confidence_region()

            # Get acquisition function values
            Plot_Acqf[i] = torch.exp(acqf(Plot_X.view(-1,1,1)))

        # Get new Y value
        temp = []
        temp_acqf = []
        for j in range(mf1.n_fidelities):
            temp.append(Samples[j][i])
            temp_acqf.append(Acqf[j])
        temp = torch.tensor(temp)
        temp_acqf = torch.tensor(temp_acqf)
        max_index = torch.argmax(temp).item()
        max_index_acqf = torch.argmax(temp_acqf).item()
        print(max_index)
        print('max_index_acqf: ', max_index_acqf)
        
        #new_Y = mf1.neg(new_X[max_index], mf1.fidelities[max_index])
        new_Y = mf1.neg(new_X[max_index_acqf], mf1.fidelities[max_index_acqf])
        
                
        # Update the training data
        train_X[max_index_acqf] = torch.cat((train_X[max_index_acqf].clone(), new_X[max_index_acqf].clone()))
        train_Y[max_index_acqf] = torch.cat((train_Y[max_index_acqf].clone(), new_Y.clone()))
        
        # For Visualization
        for i in range(n_samples):
            temp = []
            for j in range(mf1.n_fidelities):
                temp.append(Samples[j][i])
            temp = torch.tensor(temp)
            max_index_ = torch.argmax(temp)
            if i == 0:
                P = torch.zeros(mf1.n_fidelities, dtype=dtype)
            P[max_index_] += 1
        P = P / n_samples
        
        frames.append(Frame(
            N_fidelities=mf1.n_fidelities,
            models=models,
            train_X={k: v.clone() for k, v in train_X.items()},
            train_Y={k: v.clone() for k, v in train_Y.items()},
            Plot_X=Plot_X,
            Plot_Y=Plot_Y,
            Preds=Preds,
            Means=Means,
            Upper=Upper,
            Lower=Lower,
            Acqf=Plot_Acqf,
            new_X=new_X,
            Sample=Samples,
            Sample_index=max_index,
            P=P
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