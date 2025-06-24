# --- Imports ---
import os
from COSMOtherm.Configfiles import Config, COSMOthermConfig
from COSMOtherm.src.Chemfuncs import calc_la_molefrac
from COSMOtherm.src.Design_Matrix import build_design_matrix
from COSMOtherm.src.COSMOtherm_functions.Inputfile_Generation import gen_LIQEX_inp_file
from COSMOtherm.src.COSMOtherm_functions.Run_COSMOtherm_Calculations import run_COSMOtherm_calculations
from COSMOtherm.src.funcs import load_and_sort_solvents, run_initialSample_calculations, get_filelist_from_design_matrix
from COSMOtherm.src.BayesianOptimization import get_training_data, get_next_candidate
import logging
import torch


# --- Constants & Global Variables ---
N_steps = 50  # Number of iterations for Bayesian optimization
Run_initialSampling = False  # Set to False if you want to skip the initial sampling step

# --- Logger Setup ---
log_file = "BayesianOptimization.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)


# --- Main Execution ---
if __name__ == "__main__":
    
    solvents = load_and_sort_solvents(
        solvents_fullpath=Config.solvents_fullpath
        )
    
    # This is still ugly and should be improved
    solvents_ids = solvents.index.tolist()
    solvents_ids_range = [solvents_ids[0], solvents_ids[-1]]
    tC_range = Config.tC_range
    massconcentration_lacticacid_range = Config.massconcentration_lacticacid_range
    
    x1_lacticacid_range = calc_la_molefrac(
        massconcentration_lacticacid=massconcentration_lacticacid_range
        )
    
    # --- Initial Sampling ---    
    # Build the design matrix using the reduced full factorial design for initial sampling
    design_matrix = build_design_matrix(
        tC_range=tC_range, 
        x1_lacticacid_range=x1_lacticacid_range, 
        solvents=solvents,
        reduction=4
        )
    if Run_initialSampling:    
        # Run the calculations for the design matrix 
        run_initialSample_calculations(design_matrix)
    
    # --- Bayesian Optimization ---
    # Define the bounds for Bayesian optimization
    bounds = torch.tensor([tC_range, x1_lacticacid_range, solvents_ids_range]).T
    logging.info(f"Bounds for Bayesian optimization: {bounds}")
    
    # Load the initial samples from the design matrix
    initialSample_files = get_filelist_from_design_matrix(
        design_matrix=design_matrix,
        output_dir= Config.initialSamples_dir
    )

    # Load the results from the initial calculations
    train_X, train_Y = get_training_data(initialSample_files, solvents)
    logging.info(f"Number of initial training data samples: {train_Y.shape[0]}")
    logging.info(f"Initial training data; train_X: {train_X}")
    logging.info(f"Initial training data; train_Y: {train_Y}")
    
    # bayesaian optimization loop:
    logging.info(f"Starting Bayesian optimization with {N_steps} iterations...")
    for Iter in range(N_steps):

        next_tc, next_x1_lacticacid, next_solvent = get_next_candidate(
            train_X=train_X, 
            train_Y=train_Y, 
            bounds=bounds,
            solvents=solvents
        )
        logging.info(f"Iteration {Iter}: New candidate - T: {next_tc}, x1: {next_x1_lacticacid}, solvent: {next_solvent}")
        
        # Run the calculations for the new candidate
        inputfile_fullpath, file_name = gen_LIQEX_inp_file(
            tC=next_tc,
            x1_lacticacid=next_x1_lacticacid,
            solvent=next_solvent,
            ctd_file=COSMOthermConfig.ctd_file,
            cdir=COSMOthermConfig.cdir,
            ldir=COSMOthermConfig.ldir,
            odir=COSMOthermConfig.odir,
            fdir=COSMOthermConfig.fdir,
            inputfiles_folder=Config.inputfile_dir
        )
        
        # Run the calculations for the new candidate
        results = run_COSMOtherm_calculations(
            COSMOtherm_exe_fullpath=Config.COSMOtherm_exe_fullpath,
            files=[inputfile_fullpath]
        )
        
        # Load the results from the new calculations
        outputfile_fullpath = os.path.join(COSMOthermConfig.odir, file_name[:-3] + "tab")
        
        # Get the training data from the new calculations
        train_X_, train_Y_ = get_training_data(
            files=[outputfile_fullpath],
            solvents=solvents,
        )
        
        logging.info(f"New training data train_X: {train_X_}")
        logging.info(f"New training data train_Y: {train_Y_}")
        
        # Append the new data to the training data
        train_X = torch.cat([train_X, train_X_])
        train_Y = torch.cat([train_Y, train_Y_])
        