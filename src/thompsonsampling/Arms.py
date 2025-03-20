from gpytorch import ExactMarginalLogLikelihood
from .util import get_initial_points
from botorch.fit import fit_gpytorch_mll
import logging
import torch
import botorch 
import gpytorch
import uuid



class Arm():
    def __init__(self, 
                 y_func: callable, 
                 dim: int, 
                 n_pts: int,
                 n_samples: int, 
                 bounds: torch.Tensor, 
                 train_X: torch.Tensor = None, 
                 train_Y: torch.Tensor = None, 
                 q_batch: int = 1,
                 seed: int = 0,
                 dtype: torch.dtype = torch.float64,
                 device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self._dtype = dtype
        self._device = device
        self._y_func = y_func
        self._dim = dim
        self._n_pts = n_pts
        self._n_samples = n_samples
        self._bounds = bounds
        self._q_batch = q_batch
        self._seed = seed
        self._train_X = train_X if train_X is not None else get_initial_points(dim=dim, n_pts=n_pts, bounds=bounds, seed=seed)
        self._train_Y = train_Y if train_Y is not None else y_func(self._train_X)
        self._model = None
        self._mll = None
        self._acqf = None
        self._candidates = None
        self._acqf_value = None
        self._GRV = None
        self._Samples = None
        self._pred_x = None
        self._pred_mean = None
        self._pred_lower = None
        self._pred_upper = None
        self._instance_id = uuid.uuid4()
        self._logstr = f"{self.__class__.__name__} {self._instance_id}"
        
    @property
    def instance_id(self):
        return self._instance_id
    
    @property
    def pred_x(self):
        return self._pred_x
    
    @property
    def pred_mean(self):
        return self._pred_mean
    
    @property
    def pred_lower(self):
        return self._pred_lower
    
    @property
    def pred_upper(self):
        return self._pred_upper

    @property
    def y_func(self):
        return self._y_func

    @property
    def dtype(self):
        return self._dtype
    
    @property
    def device(self):
        return self._device

    @property
    def dim(self):
        return self._dim

    @property
    def n_pts(self):
        return self._n_pts

    @property
    def n_samples(self):
        return self._n_samples
    
    @property
    def bounds(self):
        return self._bounds

    @property
    def train_X(self):
        return self._train_X

    @property
    def train_Y(self):
        return self._train_Y

    @property
    def model(self):
        return self._model

    @property
    def mll(self):
        return self._mll

    @property
    def acqf(self):
        return self._acqf

    @acqf.setter
    def acqf(self, value):
        if not callable(value):
            raise ValueError("acqf must be callable")
        self._acqf = value

    @property
    def candidates(self):
        return self._candidates

    @property
    def acqf_value(self):
        return self._acqf_value
    
    @property
    def seed(self):
        return self._seed
    
    @property
    def q_batch(self):
        return self._q_batch

    @property
    def GRV(self):
        return self._GRV
    
    @property
    def Samples(self):
        return self._Samples

    def OptimizeAcqf(self):
        if self.q_batch == 1:
            self._acqf = botorch.acquisition.LogExpectedImprovement(self._model, 
                                                                    best_f=self._train_Y.max()
                                                                    )
        else:
            self._acqf = botorch.acquisition.qLogExpectedImprovement(self._model, 
                                                                     best_f=self._train_Y.max()
                                                                     )
        self._candidates, self._acqf_value = botorch.optim.optimize_acqf(self._acqf, 
                                                                         bounds=self._bounds, 
                                                                         q=self.q_batch, 
                                                                         num_restarts=10, 
                                                                         raw_samples=512
                                                                         )

    def CalcGP(self, verbose=False):
        if verbose:
            logging.info(f"{self._logstr} CalcGP: Initializing kernel and model.")
        kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(
            ard_num_dims=self._dim, 
            lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(0.1, 1.0, sigma=0.1)))
        self._model = botorch.models.SingleTaskGP(
            train_X=self._train_X, 
            train_Y=self._train_Y, 
            covar_module=kernel)
        self._model.covar_module.outputscale = 100
        self._model.covar_module.raw_outputscale.requires_grad_(False)
        self._mll = ExactMarginalLogLikelihood(self._model.likelihood, self._model)
        if verbose:
            logging.info(f"{self._logstr} CalcGP: Fitting model.")
        fit_gpytorch_mll(self._mll)
        if verbose:
            logging.info(f"{self._logstr} CalcGP: Model fitting completed")
            logging.info(f"fitted hyperparameter:")
            for name, param in self._model.named_parameters():
                logging.info(f"{name}: {param}")
        
    def UpdateData(self, verbose=False):
        if verbose:
            logging.info(f"{self._logstr} Updating data with candidate: {self._candidates[0,...].view(1, -1)}")
        self._train_X = torch.cat((self._train_X, self._candidates[0,...].view(1, -1)))
        self._train_Y = torch.cat((self._train_Y, self._y_func(self._candidates[0,...].view(1, -1))))
        self._candidates = self._candidates[1:,...]
        if verbose:
            logging.info(f"{self._logstr}Updated train_X: {self._train_X}")
            logging.info(f"{self._logstr}Updated train_Y: {self._train_Y}")
            logging.info(f"{self._logstr}Updated candidates: {self._candidates}")

    def SampleAtCandidate(self, verbose=False):
        if self._candidates is None or len(self._candidates) == 0 or self._candidates.shape[0] == 0:
            logging.warning(f"{self._logstr} SampleAtCandidate: No candidates to sample from, setting Samples to -inf")
            self._Samples = torch.full((self.n_samples,1), float('-inf'), device=self._device)
            self._GRV = None
            return
        self._GRV = self._model.posterior(self._candidates[0,...].unsqueeze(0))
        self._Samples = self._GRV.sample(torch.Size([self.n_samples]))
        if verbose:
            logging.info(f"{self._logstr} SampleAtCandidate: Sampled {self.n_samples} times at candidate {self._candidates[0,...].unsqueeze(0)}")
    
    def Predict(self, verbose=False):
        if verbose:
            logging.info(f"{self._logstr} Predict: Generating prediction points.")
        self._pred_x = torch.linspace(self.bounds[0].item(), self.bounds[1].item(), 100, dtype=self._dtype, device=self._device).view(-1, 1)
        if verbose:
            logging.info(f"{self._logstr} Predict: Predicting mean and confidence region.")
        self._pred_mean = self._model.posterior(self.pred_x).mean
        self._pred_lower, self._pred_upper = self._model.posterior(self.pred_x).confidence_region()
        if verbose:
            logging.info(f"{self._logstr} Predict: Prediction completed.")

    def __repr__(self):
        return (
            f"(\n"
            f"  id={self._instance_id},\n"
            f"  dim={self._dim},\n"
            f"  n_pts={self._n_pts},\n"
            f"  n_samples={self._n_samples},\n"
            f"  bounds={self._bounds},\n"
            f"  train_X={self._train_X},\n"
            f"  train_Y={self._train_Y},\n"
            f"  q_batch={self._q_batch},\n"
            f"  seed={self._seed},\n"
            f"  dtype={self._dtype},\n"
            f"  device={self._device}\n"
            f"  candidates={self._candidates},\n"
            f")\n"
            "----------------------------------"
        )