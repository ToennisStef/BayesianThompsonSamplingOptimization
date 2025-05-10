# --- Imports ---
import pandas as pd
import subprocess
import os
from COSMOtherm.Configfiles import Config, COSMOthermConfig
from COSMOtherm.funcs import gen_LIQEX_inp_file, load_and_sort_solvents, run_COSMOtherm_calculations, list_files_with_extension
from COSMOtherm.funcs import get_training_data, get_next_candidate, build_design_matrix_fullfactorial
import torch


# --- Constants & Global Variables ---
N_steps = 10  # Number of iterations for Bayesian optimization

Run_initialSampling = True  # Set to False if you want to skip the initial sampling step

tC_range = [20, 30, 40]  # [°C]
massc1_lacticacid_range = [5, 10, 20, 50, 100, 150, 200, 250]  # [g/L] mass concentration of lactic acid in the solution [250]
# Konzentrationen 
# 5g/L 10g/L 20 g/L (Für kontinuierliche Abtrennung aus dem Fermenter relevant)
# 50g/L 100 g/L 150g/L 200g/L 250 g/L (Das ist laut Literatur die maximale Konzentration die bisher bei Fermentation erreicht wird)

M_h2o = 18.01528  # [g/mol] molar mass of water
M_lacticacid = 90.078  # [g/mol] molar mass of lactic acid
rho_lacticacid = 1.209  # [g/mL] density of lactic acid
rho_h2o = 1.0  # [g/mL] density of water
c_p_h2o = 55.5  # [mol/L] concentration of pure water at 25 °C
c_p_lacticacid = rho_lacticacid / M_lacticacid * (10**3)  # [mol/L] concentration of pure lactic acid at 25 °C

# --- Helper Functions ---
def calculate_molar_fractions():
    c1_lacticacid_range = [mass / M_lacticacid for mass in massc1_lacticacid_range]
    c1_h2o_range = [
        c_p_h2o - (c_p_h2o / c_p_lacticacid) * c1_lacticacid
        for c1_lacticacid in c1_lacticacid_range
    ]
    x1_lacticacid_range = [
        c1_lacticacid / (c1_lacticacid + c1_h2o)
        for c1_lacticacid, c1_h2o in zip(c1_lacticacid_range, c1_h2o_range)
    ]
    return x1_lacticacid_range


def run_initialSample_calculations(reduced_fullfact):
    for index, row in reduced_fullfact.iterrows():
        
        temperature = row['temperature']
        x1_lacticacid = row['x1_lacticacid']
        solvent = row['COSMO_name']
        
        inputfile_fullpath, inputfile_filename = gen_LIQEX_inp_file(
            tC=temperature,
            x1_lacticacid=x1_lacticacid,
            solvent=solvent,
            ctd_file=COSMOthermConfig.ctd_file,
            cdir=COSMOthermConfig.cdir,
            ldir=COSMOthermConfig.ldir,
            odir=COSMOthermConfig.odir,
            fdir=COSMOthermConfig.fdir,
            inputfiles_folder=Config.inputfile_dir
        )
        outputfile_fullpath = COSMOthermConfig.odir + "\\" + inputfile_filename[:-3] + "tab"
        subprocess_str = '"' + Config.COSMOtherm_exe_fullpath + '"' + ' ' + '"' + inputfile_fullpath + '"'

        if os.path.exists(outputfile_fullpath):
            # overwrite = input(f"The file '{outputfile_fullpath}' already exists. Overwrite? [y/n]: ").strip().lower()
            overwrite = 'n'
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
        fullfact = build_design_matrix_fullfactorial(
            tC_range=tC_range, 
            x1_lacticacid_range=x1_lacticacid_range, 
            solvents=solvents,
            )
        
        # print("Full factorial design matrix:")
        # Run the calculations for the design matrix 
        run_initialSample_calculations(fullfact)
    
    # bayesaian optimization loop:
