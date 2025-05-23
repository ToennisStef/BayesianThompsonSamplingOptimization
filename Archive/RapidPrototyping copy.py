# --- Imports ---
import pandas as pd
import subprocess
import os
from COSMOtherm.Configfiles import Config, COSMOthermConfig
from COSMOtherm.src.funcs import gen_LIQEX_inp_file, load_and_sort_solvents, run_COSMOtherm_calculations, list_files_with_extension
from COSMOtherm.src.funcs import get_training_data, get_next_candidate, build_design_matrix
import torch


# --- Constants & Global Variables ---
N_steps = 10  # Number of iterations for Bayesian optimization

Run_initialSampling = False  # Set to False if you want to skip the initial sampling step

tC_range = [20, 40]  # [°C]
massc1_lacticacid_range = [5, 20]  # [g/L]
M_h2o = 18.01528  # [g/mol]
M_lacticacid = 90.078  # [g/mol]
rho_lacticacid = 1.209  # [g/mL]
rho_h2o = 1.0  # [g/mL]
c_p_h2o = 55.5  # [mol/L]
c_p_lacticacid = rho_lacticacid / M_lacticacid * (10**3)  # [mol/L]

# --- Helper Functions ---
def calculate_molar_fractions():

    c1_lacticacid_range = [
        massc1_lacticacid_range[0]/M_lacticacid,
        massc1_lacticacid_range[1]/M_lacticacid
    ]
    c1_h2o_range = [
        c_p_h2o - (c_p_h2o/c_p_lacticacid)*c1_lacticacid_range[0],
        c_p_h2o - (c_p_h2o/c_p_lacticacid)*c1_lacticacid_range[1]
    ]
    x1_lacticacid_range = [
        c1_lacticacid_range[0]/(c1_lacticacid_range[0]+c1_h2o_range[0]),
        c1_lacticacid_range[1]/(c1_lacticacid_range[1]+c1_h2o_range[1])
    ]
    return x1_lacticacid_range


def run_initialSample_calculations(reduced_fullfact):
    for index, row in reduced_fullfact.iterrows():
        
        temperature = row['temperature']
        x1_lacticacid = row['x1_lacticacid']
        solvent = row['COSMO_name']
        
        inputfile_fullpath, inputfile_filename = gen_LIQEX_inp_file(
            temperature=temperature,
            x1_lacticacid=x1_lacticacid,
            solvent=solvent,
            ctd_file=COSMOthermConfig.ctd_file,
            cdir=COSMOthermConfig.cdir,
            ldir=COSMOthermConfig.ldir,
            odir=COSMOthermConfig.odir,
            fdir=COSMOthermConfig.fdir,
            inputfiles_folder=Config.inputfile_path
        )
        outputfile_fullpath = COSMOthermConfig.odir + "\\" + inputfile_filename[:-3] + "tab"
        subprocess_str = Config.cosmotherm_filepath + '"' + inputfile_fullpath + '"'

        if os.path.exists(outputfile_fullpath):
            overwrite = input(f"The file '{outputfile_fullpath}' already exists. Overwrite? [y/n]: ").strip().lower()
            if overwrite != 'y':
                print("Operation cancelled. Returning existing file path and name.")
            else:
                print("Running COSMO-RS calculation and overwriting the existing file.")
                result = subprocess.run(subprocess_str, shell=True)
                print("Return code:", result.returncode)
        else:
            print("Running COSMO-RS calculation.")
            result = subprocess.run(subprocess_str, shell=True)
            print("Return code:", result.returncode)

    


# --- Main Execution ---
if __name__ == "__main__":
    
    x1_lacticacid_range = calculate_molar_fractions()
    solvents = load_and_sort_solvents(Config.solvents_fullpath)
    
    # This is still ugly and should be improved
    solvents_ids = solvents.index.tolist()
    solvents_ids_range = [solvents_ids[0], solvents_ids[-1]]
    
    # --- Initial Sampling ---    
    if Run_initialSampling:
        # Build the design matrix using the reduced full factorial design for initial sampling
        reduced_fullfact = build_design_matrix(
            tC_range=tC_range, 
            x1_lacticacid_range=x1_lacticacid_range, 
            solvents=solvents,
            reduction=5
            )
        
        # Run the calculations for the design matrix 
        run_initialSample_calculations(reduced_fullfact)
    
    # --- Bayesian Optimization ---
    # Define the bounds for Bayesian optimization
    bounds = torch.tensor([tC_range, x1_lacticacid_range, solvents_ids_range]).T

    # Load the results from the initial calculations
    initialSample_files = list_files_with_extension(Config.initialSamples_dir, "tab")
    train_X, train_Y = get_training_data(initialSample_files, solvents)
    
    print(f"Initial training data: {train_X.shape[0]} samples")
    print(f"Initial training data: {train_Y.shape[0]} samples")
    print(f"Initial training data train_Y: {train_X}")
    print(f"Initial training data train_Y: {train_Y}")
    
    print(f"Running Bayesian optimization with {N_steps} iterations...")
    # bayesaian optimization loop:
    for Iter in range(N_steps):

        next_tc, next_x1_lacticacid, next_solvent = get_next_candidate(
            train_X=train_X, 
            train_Y=train_Y, 
            bounds=bounds,
            solvents=solvents
        )
        
        print(f"Iteration {Iter}: New candidate - T: {next_tc}, x1: {next_x1_lacticacid}, solvent: {next_solvent}")

        print("Running calculations for the new candidate...")
        # Run the calculations for the new candidate
        inputfile_fullpath, file_name = gen_LIQEX_inp_file(
            temperature=next_tc,
            x1_lacticacid=next_x1_lacticacid,
            solvent=next_solvent,
            ctd_file=COSMOthermConfig.ctd_file,
            cdir=COSMOthermConfig.cdir,
            ldir=COSMOthermConfig.ldir,
            odir=COSMOthermConfig.odir,
            fdir=COSMOthermConfig.fdir,
            inputfiles_folder=Config.inputfile_dir
        )
        print(f"Input file created: {inputfile_fullpath}")
        
        print("Running COSMO-RS calculation...")
        # Run the calculations for the new candidate
        run_COSMOtherm_calculations(
            COSMOtherm_exe_fullpath=Config.COSMOtherm_exe_fullpath,
            files=[inputfile_fullpath]
            )
        
        
        # Load the results from the new calculations
        outputfile_fullpath = os.path.join(COSMOthermConfig.odir, file_name[:-3] + "tab")
        # outputfile_fullpath = Config.outputfile_dir
        # outputfile_fullpath = COSMOthermConfig.odir + "\\" + file_name[:-3] + "tab"
        
        # Get the training data from the new calculations
        train_X_, train_Y_ = get_training_data(
            files=[outputfile_fullpath],
            solvents=solvents,
        )
        
        print(f"New training data: {train_X_.shape[0]} samples")
        print(f"New training data: {train_Y_.shape[0]} samples")
        print(f"New training data train_X: {train_X_}")
        print(f"New training data train_Y: {train_Y_}")
        
        # Append the new data to the training data
        train_X = torch.cat([train_X, train_X_])
        train_Y = torch.cat([train_Y, train_Y_])
        