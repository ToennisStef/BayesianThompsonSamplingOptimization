import torch
from torch import Tensor
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.kernels import RBFKernel, ScaleKernel, ProductKernel, RFFKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.distributions import MultivariateNormal
from botorch import fit_fully_bayesian_model_nuts
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.transforms import Standardize as BoTorchStandardize
from gauche.kernels.fingerprint_kernels.tanimoto_kernel import TanimotoKernel

import gpytorch


import numpy as np

class TanimotoGP(SingleTaskGP):
    """
    Gaussian Process model with Tanimoto kernel for molecular fingerprints.
    """
    def __init__(self, train_X, train_Y, d_fingerprint=2048):
        super().__init__(train_X, train_Y, likelihood=GaussianLikelihood(), 
                         input_transform=Normalize(d=train_X.shape[1]),
                         outcome_transform=Standardize(m=train_Y.shape[1]))
        self.mean_module = ConstantMean()
        tanimoto_kernel = TanimotoKernel()
        self.covar_module = ScaleKernel(tanimoto_kernel)
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
        train_Yvar=torch.ones_like(train_y)* 1e-4,  # Small noise variance
        input_transform=Normalize(d=train_x.shape[1]),
        outcome_transform=Standardize(m=train_y.shape[1])
    )
    if state_dict:
        model.load_state_dict(state_dict)
    # mll is not used for SAAS GP, but return a dummy for compatibility
    return None, model 


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


# dkl model
def initialize_model_dkl_gp(train_x, train_y, state_dict=None, training_iterations=100, lr=0.01):
    """
    Initializes a Deep Kernel Learning (DKL) GP model and returns a train function and the model.
    Args:
        train_x: torch.Tensor, shape (n_samples, n_features)
        train_y: torch.Tensor, shape (n_samples,)
        state_dict: optional, model state dict for warm starting
        training_iterations: int, number of training iterations
        lr: float, learning rate
    Returns:
        train_fn: function(model, train_x, train_y) -> None, trains the model in-place
        model: the DKL GP model
    """
    import gpytorch
    import torch
    from torch import nn
    from tqdm import tqdm

    data_dim = train_x.shape[1]

    class LargeFeatureExtractor(nn.Sequential):
        def __init__(self):
            super().__init__()
            self.add_module('linear1', nn.Linear(data_dim, 256))
            self.add_module('relu1', nn.ReLU())
            self.add_module('linear2', nn.Linear(256, 128))
            self.add_module('relu2', nn.ReLU())
            self.add_module('linear3', nn.Linear(128, 64))
            self.add_module('relu3', nn.ReLU())
            self.add_module('linear4', nn.Linear(64, 32))
            self.add_module('relu4', nn.ReLU())
            self.add_module('linear5', nn.Linear(32, 2))

    feature_extractor = LargeFeatureExtractor()

    class GPRegressionModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super().__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.GridInterpolationKernel(
                gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=2)),
                num_dims=2, grid_size=100
            )
            self.feature_extractor = feature_extractor
            self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)

        def forward(self, x):
            projected_x = self.feature_extractor(x)
            projected_x = self.scale_to_bounds(projected_x)
            mean_x = self.mean_module(projected_x)
            covar_x = self.covar_module(projected_x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GPRegressionModel(train_x, train_y, likelihood)

    if torch.cuda.is_available():
        model = model.cuda()
        likelihood = likelihood.cuda()


    if state_dict:
        model.load_state_dict(state_dict)

    def train_fn(model_to_train, train_x, train_y):
        model_to_train.train()
        model_to_train.likelihood.train()
        optimizer = torch.optim.Adam([
            {'params': model_to_train.feature_extractor.parameters()},
            {'params': model_to_train.covar_module.parameters()},
            {'params': model_to_train.mean_module.parameters()},
            {'params': model_to_train.likelihood.parameters()},
        ], lr=lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(model_to_train.likelihood, model_to_train)
        iterator = tqdm(range(training_iterations), desc="DKL Training", leave=False)
        for i in iterator:
            optimizer.zero_grad()
            # output is a gpytorch.distributions.MultivariateNormal
            output: gpytorch.distributions.MultivariateNormal = model_to_train(train_x)
            # mll(output, train_y) returns a scalar tensor (the marginal log likelihood)
            loss = -mll(output, train_y)
            if loss.dim() != 0:
                loss = loss.sum()  # Ensure scalar
            loss.backward()
            iterator.set_postfix(loss=float(loss.item()))
            optimizer.step()

    return train_fn, model



