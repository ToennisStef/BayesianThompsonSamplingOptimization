from botorch.sampling import SobolEngine
import torch

def get_initial_points(
        dim: int, 
        n_pts: int, 
        bounds: torch.tensor = torch.tensor([-1, 1]), 
        seed=0, 
        dtype=torch.float64,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ) -> torch.tensor:
    sobol = SobolEngine(dimension=dim, scramble=True, seed=seed)
    X_init = sobol.draw(n=n_pts).to(dtype=dtype, device=device)
    # lower_bound, upper_bound = bounds
    
    # X_init = lower_bound + (upper_bound - lower_bound) * X_init  # scale points to the given bounds

    for i in range(dim):
        lower_bound, upper_bound = bounds[...,i]
        X_init[:, i] = lower_bound + (upper_bound - lower_bound) * X_init[:, i]  # scale points to the given bounds
    return X_init
