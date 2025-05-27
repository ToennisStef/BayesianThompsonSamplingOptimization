import pandas as pd
import torch
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
import botorch
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
import re
import logging 

logger = logging.getLogger(__name__)

def get_next_candidate(
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
    bounds: torch.Tensor,
    solvents: pd.DataFrame,
    )-> tuple:
    """
    Generates the next candidate for the optimization process using a Gaussian Process model.
    The function uses the training data (train_X and train_Y) to fit a Gaussian Process model,
    and then uses an acquisition function to find the next candidate point.
    The candidate is rounded to the nearest integer for the solvent ID.
    
    Parameters:
        train_X (torch.Tensor): The training input data.
        train_Y (torch.Tensor): The training output data.
        bounds (torch.Tensor): The bounds for the optimization process.
        solvents (pd.DataFrame): DataFrame containing solvent information.
    Returns:
        tuple: The next candidate values for temperature (tC), x1_lacticacid, and solvent in that order.
    """
    
    # Generate Gaussian Process model
    logger.debug(f"Creating Gaussian Process model...")
    model = botorch.models.SingleTaskGP(
        train_X=train_X, 
        train_Y=train_Y,
        input_transform=Normalize(d=train_X.shape[1]),
        outcome_transform=Standardize(m=train_Y.shape[1]),
        )
    
    # Fit the model
    logger.debug(f"Fitting Gaussian Process model...")
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll=mll)
    
    # Define the acquisition function
    logger.debug(f"Defining acquisition function...")
    acqf = LogExpectedImprovement(
        model=model, 
        best_f=train_Y.max()
        )
    
    # Optimize the acquisition function
    logger.debug(f"Optimizing acquisition function...")
    candidate, acq_value = optimize_acqf(
        acq_function=acqf,
        bounds=bounds,
        q=1,  # Number of candidates to sample
        num_restarts=50,
        raw_samples=512
    )

    # Round the candidate to the nearest integer for the solvent ID
    candidate_rounded_down = candidate.clone()
    candidate_rounded_up = candidate.clone()

    # ATTENTION: This is a hardcoded assumption, that the last column of the data is the DISCRETE solvent id
    candidate_rounded_down[0, -1] = torch.floor(candidate_rounded_down[0, -1])
    candidate_rounded_up[0, -1] = torch.ceil(candidate_rounded_up[0, -1])

    # Evaluate the acquisition function value for each rounded candidate
    acq_value_down = acqf(candidate_rounded_down.unsqueeze(0))
    acq_value_up = acqf(candidate_rounded_up.unsqueeze(0))

    best_candidate = None
    # Compare the acquisition function values and choose the better one
    if acq_value_down > acq_value_up:
        best_candidate = candidate_rounded_down
    else:
        best_candidate = candidate_rounded_up

    
    next_tC = round(best_candidate[:,0].item(), 6) # Round to 6 decimal places and convert to Celsius
    next_x1_lacticacid = round(best_candidate[:,1].item(), 8) # Round to 8 decimal places
    next_solvent_id = best_candidate[:,2]
    next_solvent = solvents.loc[next_solvent_id.item(), "COSMO_name"] # Get the solvent name from the DataFrame
    
    return next_tC, next_x1_lacticacid, next_solvent



def get_training_data(
    files: list,
    solvents: pd.DataFrame,
    )-> list:
    
    train_X = []
    train_Y = []
    for file in files:
        # Load the data from the file
        with open(file, 'r') as f:
            for i, line in enumerate(f):
                if i == 2:  # Line 3 (0-based index)
                    settings_line = line.strip()
                    break

        # Extract Temperature (T) and x(3) value
        temperature_match = re.search(r'T= (\d+\.\d+) K', settings_line)
        x3_match = re.search(r'x\(3\)= ([\d\.E\-]+)', settings_line)
        
        temperature = float(temperature_match.group(1)) if temperature_match else None
        x3_value = float(x3_match.group(1)) if x3_match else None

        # print(f"Extracted Temperature: {temperature} K, x(3): {x3_value}")
        
        
        # Determine if the descriptor is 4 or 5 lines long
        with open(file, 'r') as f:
            lines = [next(f) for _ in range(6)]
        # Check if the 5th line (index 4) contains column headers (e.g., 'Compound')
        header_line = lines[4].strip()
        if re.match(r'Compound', header_line):
            skiprows = 4
        else:
            skiprows = 5

        data = pd.read_csv(file, sep=r'\s+', skiprows=skiprows)
        
        solvent_name = data['Compound'][1]
        
        solvent_index = solvents[solvents['COSMO_name'] == solvent_name].index[0]
        x2_lacticacid_p2 = data['phase_2_x'][2] # [mol/mol] Mole fraction of lactic acid in solvent phase
        x2_lacticacid_p1 = data['phase_1_x'][2] # [mol/mol] Mole fraction of lactic acid in water phase
        KV = x2_lacticacid_p2 / x2_lacticacid_p1 # [mol/mol] Partition coefficient of lactic acid in solvent phase
        
        tC = temperature - 273.15 # Convert to Celsius
        
        train_X.append([tC, x3_value, solvent_index])
        train_Y.append([KV])
        
    train_X = torch.tensor(train_X, dtype=torch.float64)
    train_Y = torch.tensor(train_Y, dtype=torch.float64)
    
    return train_X, train_Y