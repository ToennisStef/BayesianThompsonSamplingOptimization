import os
import pandas as pd
import pyDOE3
import subprocess
import glob
import torch
import re
import gpytorch
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
import botorch
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize



def gen_LIQEX_inp_file(
    tC:float,
    x1_lacticacid:float, 
    solvent:str, 
    ctd_file:str, 
    cdir:str, 
    ldir:str, 
    odir:str, 
    fdir:str,
    file_name:str = None,  
    inputfiles_folder:str =".")->None:
    """
    Generates an input file for LIQEX calculations based on the provided parameters.
    Parameters:
        tC (float): Temperature in degrees Celsius.
        x1_lacticacid (float): Mole fraction of lactic acid.
        solvent (str): Name of the solvent.
        ctd_file (str): Path to the CT-Data file.
        cdir (str): Directory for the CT-data/parameterization file.  
        ldir (str): Directory for the license file.
        odir (str): Directory for the output files.
        fdir (str): Directory for the compound data files.
        file_name (str, optional): Name of the inputfile. If None, a default name will be generated.
        inputfiles_folder (str, optional): Directory where the inputfile will be saved. Default is the current directory.
    Returns:
        None
    """
    
    # Ensure the output directory exists
    os.makedirs(inputfiles_folder, exist_ok=True)
    
    if file_name is None:
        file_name = f"LIQEX_{solvent}_tc{tC}_x{x1_lacticacid}.inp"
    file_fullpath = os.path.join(inputfiles_folder, file_name)
    
    
    # Calculate the mole fraction of water
    x1_h2o = 1-x1_lacticacid
    
    # Define content
    content = f"""ctd={ctd_file} CDIR="{cdir}" LDIR="{ldir}" odir="{odir}" # Global command line
FDIR="{fdir}" vpfile CTAB WCONF AUTOC                                 # Global command line
!! Multi-Component-2-Phase-Equilibrium calculation                        # Comment line
f = h2o                      # Compound input (water)
f = {solvent}                # Compound input (solvent)
f = lacticacid               # Compound input (lactic acid)
tc={tC} LIQ_EX x1={{{x1_h2o} 0 {x1_lacticacid}}} x2={{0 1 0}}
# LIQ_EX: Liquid phase equilibrium calculation
# x1: Mole fraction of the first component in the input stream
# x2: Mole fraction of the second component in the input stream
# tc: Temperature in degrees Celsius
# ctd: CT-Data file name / parameterization file
# CDIR: Directory for the CT-data/parameterization file
# LDIR: Directory for the licence file
# odir: Directory for the output files
# FDIR: Directory for the compound data files
# vpfile: ???
# CTAB: some additional output configuration (see documentation)
# WCONF: some additional output configuration (see documentation)
# AUTOC: Specifies automatic conformere search & consideration for all compound
"""
    # Check if the output file already exists
    if os.path.exists(file_fullpath):
        # overwrite = input(f"The file '{file_fullpath}' already exists. Do you want to overwrite it? [y/n]: ").strip().lower()
        overwrite = 'n'
        if overwrite != 'y':
            print("Operation cancelled. The file was not overwritten. returning the existing file path and name.")
        else:    
        # Write content to file
            with open(file_fullpath, "w") as file:
                file.write(content)
            print(f"Input file '{file_fullpath}' has been created successfully. returning the file path and name")
    else:
        with open(file_fullpath, "w") as file:
            file.write(content)
            print(f"Input file '{file_fullpath}' has been created successfully. returning the file path and name")
    return file_fullpath, file_name

def sort_solvents_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adjusts the DataFrame by removing rows with NaN values in 'SMILES' and sorting it based on a custom sortingkey.
    The sorting key is a tuple derived from the 'SMILES' string of each row. The tuple consists of:
        1. The count of 'C' atoms in the SMILES string.
        2. The count of 'O' atoms in the SMILES string.
        3. The position of the first occurrence of 'O' in the SMILES string (or infinity if 'O' is not present).
        4. The count of 'C' atoms within parentheses, but only if the SMILES string contains '=O'.
    
    Parameters:
        df (pd.DataFrame): The DataFrame to be adjusted and sorted.
    Returns:
        pd.DataFrame: The adjusted and sorted DataFrame.
    """
    
    import math
    import re

    def parse_smiles(smiles):
        c_count = smiles.count('C')
        o_count = smiles.count('O')
        o_position = smiles.find('O') if 'O' in smiles else math.inf

        # Check if '=O' is present
        has_eq_o = '=O' in smiles

        # Count how many 'C' are within parentheses only if '=O' is present
        c_paren_count = 0
        if has_eq_o:
            paren_matches = re.findall(r'\([^()]*\)', smiles)
            for match in paren_matches:
                c_paren_count += match.count('C')

        return (c_count, o_count, o_position, c_paren_count)

    df['smiles_sort_key'] = df['SMILES'].apply(parse_smiles)
    df = df.sort_values(by='smiles_sort_key').reset_index(drop=True).drop(columns='smiles_sort_key')
    return df

def load_and_sort_solvents(solvents_fullpath):
    solvents = pd.read_csv(solvents_fullpath)
    solvents = sort_solvents_df(solvents)
    return solvents

def build_design_matrix(
    tC_range:list, 
    x1_lacticacid_range:list, 
    solvents:pd.DataFrame,
    reduction:int = 5
    )-> pd.DataFrame:
    """
    Generate a reduced full factorial design matrix based on the provided ranges and solvents.
    The design matrix is generated using the pyDOE3 library.
    The function returns a DataFrame with the generated design matrix.
    Parameters:
        tC_range (list): List of temperature values.
        x1_lacticacid_range (list): List of lactic acid mole fraction values.
        solvents_ids (list): List of solvent IDs.
        solvents (pd.DataFrame): DataFrame containing solvent information.
    Returns:
        pd.DataFrame: DataFrame containing the generated design matrix.
    """
    solvents_ids = solvents.index.tolist()
    levels = [len(tC_range), len(x1_lacticacid_range), len(solvents_ids)]
    reduced_design = pyDOE3.gsd(levels=levels, reduction=reduction)
    reduced_fullfact = pd.DataFrame(reduced_design, columns=['temperature', 'x1_lacticacid', 'solvent'])
    reduced_fullfact['temperature'] = reduced_fullfact['temperature'].map({
        i: tC_range[i] for i in range(len(tC_range))
    })
    reduced_fullfact['x1_lacticacid'] = reduced_fullfact['x1_lacticacid'].map({
        i: x1_lacticacid_range[i] for i in range(len(x1_lacticacid_range))
    })
    reduced_fullfact['solvent'] = reduced_fullfact['solvent'].map({
        i: solvents_ids[i] for i in range(len(solvents_ids))
    })
    reduced_fullfact = reduced_fullfact.join(solvents['COSMO_name'], on='solvent')
    return reduced_fullfact


def build_design_matrix_fullfactorial(
    tC_range:list, 
    x1_lacticacid_range:list, 
    solvents:pd.DataFrame,
    )-> pd.DataFrame:
    """
    Generate a reduced full factorial design matrix based on the provided ranges and solvents.
    The design matrix is generated using the pyDOE3 library.
    The function returns a DataFrame with the generated design matrix.
    Parameters:
        tC_range (list): List of temperature values.
        x1_lacticacid_range (list): List of lactic acid mole fraction values.
        solvents_ids (list): List of solvent IDs.
        solvents (pd.DataFrame): DataFrame containing solvent information.
    Returns:
        pd.DataFrame: DataFrame containing the generated design matrix.
    """
    solvents_ids = solvents.index.tolist()
    levels = [len(tC_range), len(x1_lacticacid_range), len(solvents_ids)]
    fullfact_design = pyDOE3.fullfact(levels=levels)
    # reduced_design = pyDOE3.gsd(levels=levels, reduction=reduction)
    fullfact = pd.DataFrame(fullfact_design, columns=['temperature', 'x1_lacticacid', 'solvent'])
    fullfact['temperature'] = fullfact['temperature'].map({
        i: tC_range[i] for i in range(len(tC_range))
    })
    fullfact['x1_lacticacid'] = fullfact['x1_lacticacid'].map({
        i: x1_lacticacid_range[i] for i in range(len(x1_lacticacid_range))
    })
    fullfact['solvent'] = fullfact['solvent'].map({
        i: solvents_ids[i] for i in range(len(solvents_ids))
    })
    fullfact = fullfact.join(solvents['COSMO_name'], on='solvent')
    return fullfact


def run_COSMOtherm_calculations(
    COSMOtherm_exe_fullpath: str, 
    files: list,
    )-> list:
    """
    Run COSMOtherm calculations for the provided files using the specified executable.
    Parameters:
        COSMO_exe_fullpath (str): Full path to the COSMO executable.
        files (list): List of file paths to be processed. 
        outputfiles_dir (str): Directory where the output files will be saved.
    Returns:
        results (list): List of results from the calculations.
    """
    results = []
    for file in files:
        # Run the calculations for each file
        
        subprocess_str = '"'+ COSMOtherm_exe_fullpath + '"' + ' ' + '"' + file + '"'
        result = subprocess.run(subprocess_str, shell=True)
        results.append(result)
    return results

def ensure_full_path(input_path: str) -> str:
    """
    Ensures the input path is a full path by appending the current working directory (cwd)
    if any part of the cwd is not already in the input path.

    Parameters:
        input_path (str): The input path to be checked and adjusted.

    Returns:
        str: The full path.
    """
    cwd = os.getcwd()
    full_path = os.path.abspath(input_path)

    if not full_path.startswith(cwd):
        full_path = os.path.join(cwd, input_path)

    return os.path.normpath(full_path)

def list_files_with_extension(folder_path, file_extension):
        search_pattern = os.path.join(folder_path, f"*.{file_extension}")
        files = glob.glob(search_pattern)
        return files


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
        data = pd.read_csv(file, sep=r'\s+', skiprows=4)
        solvent_name = data['Compound'][1]
        
        solvent_index = solvents[solvents['COSMO_name'] == solvent_name].index[0]
        x2_lacticacid = data['phase_2_x'][2] # [mol/mol] Mole fraction of lactic acid in solvent phase
        
        tC = temperature - 273.15 # Convert to Celsius
        
        train_X.append([tC, x3_value, solvent_index])
        train_Y.append([x2_lacticacid])
        
    train_X = torch.tensor(train_X, dtype=torch.float64)
    train_Y = torch.tensor(train_Y, dtype=torch.float64)
    
    return train_X, train_Y

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
        tuple: The next candidate values for temperature (tC), x1_lacticacid, and solvent.
    """
    
    model = botorch.models.SingleTaskGP(
        train_X=train_X, 
        train_Y=train_Y,
        input_transform=Normalize(d=train_X.shape[1]),
        outcome_transform=Standardize(m=train_Y.shape[1]),
        )
    
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll=mll)
    acqf = LogExpectedImprovement(
        model=model, 
        best_f=train_Y.max()
        )
    
    candidate, acq_value = optimize_acqf(
    acq_function=acqf,
    bounds=bounds,
    q=1,  # Number of candidates to sample
    num_restarts=50,
    raw_samples=512
    )

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


