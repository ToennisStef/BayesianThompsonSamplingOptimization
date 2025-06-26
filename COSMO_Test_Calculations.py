# Exemplary calcualtion of liquid-liquid equilibrium calculation using COSMOtherm
# Calculation at different temperature and (water-lacticacid)composition levels
# Phase 1 is denoted as "p1" and Phase 2 as "p2"


# --- Imports ---
from COSMOtherm.Configfiles import Config
from COSMOtherm.Configfiles import COSMOthermConfig  # Importing Config_ for the input file generation
from COSMOtherm.src.COSMOtherm_functions.Inputfile_Generation import gen_TBOIL_inp_file, gen_binaryLLE_inp_file, gen_binaryActivity_inp_file, gen_PVAP_inp_file
from COSMOtherm.src.COSMOtherm_functions.Run_COSMOtherm_Calculations import run_COSMOtherm_calculations
import logging
import pandas as pd
import os
# Ensure the logs directory exists
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Configure logging in the main file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "COSMO_LLE_Calculation.log")),
        logging.StreamHandler()
    ]
)


# --- Main Execution ---
if __name__ == "__main__":
    
    logging.info("Starting new COSMOtherm LLE calculation run.")
        
    
    # Load the solvents data from the CSV file 
    solvents = pd.read_csv(Config.solvents_fullpath)
    
    Server = 'TIGER'  # or 'TIGER2'
    
    for solvent in solvents['COSMO_name'].unique():
        
        # file = gen_TBOIL_inp_file(
        #     solvent=solvent, 
        #     ctd_file=COSMOthermConfig.TIGER['ctd_file'], 
        #     cdir=COSMOthermConfig.TIGER['cdir'], 
        #     ldir=COSMOthermConfig.TIGER['ldir'], 
        #     odir=COSMOthermConfig.TIGER['odir'], 
        #     fdir=COSMOthermConfig.TIGER['fdir'],
        #     overwrite=True
        #     )
        
        # solvent_ = "ethanol"
        
        # file = gen_binaryLLE_inp_file(
        #     solvent=solvent, 
        #     ctd_file=COSMOthermConfig.TIGER['ctd_file'], 
        #     cdir=COSMOthermConfig.TIGER['cdir'], 
        #     ldir=COSMOthermConfig.TIGER['ldir'], 
        #     odir=COSMOthermConfig.TIGER['odir'], 
        #     fdir=COSMOthermConfig.TIGER['fdir'],
        #     overwrite=True
        #     )
        
        # file = gen_binaryActivity_inp_file(
        #     solvent=solvent, 
        #     ctd_file=COSMOthermConfig.TIGER['ctd_file'], 
        #     cdir=COSMOthermConfig.TIGER['cdir'], 
        #     ldir=COSMOthermConfig.TIGER['ldir'], 
        #     odir=COSMOthermConfig.TIGER['odir'], 
        #     fdir=COSMOthermConfig.TIGER['fdir'],
        #     overwrite=True
        #     )
        
        file =  gen_PVAP_inp_file(
            solvent=solvent, 
            t_start=25,
            t_end=100,
            t_steps=5,
            ctd_file=COSMOthermConfig.TIGER['ctd_file'], 
            cdir=COSMOthermConfig.TIGER['cdir'], 
            ldir=COSMOthermConfig.TIGER['ldir'], 
            odir=COSMOthermConfig.TIGER['odir'], 
            fdir=COSMOthermConfig.TIGER['fdir'],
            overwrite=True
            )
        
        
        
        run_COSMOtherm_calculations(
            COSMOtherm_exe_fullpath=COSMOthermConfig.TIGER['exe_fullpath'],
            files=[file['fullpath']]
        )
        
        break