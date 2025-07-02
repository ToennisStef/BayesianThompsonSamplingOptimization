# Exemplary calcualtion of liquid-liquid equilibrium calculation using COSMOtherm
# Calculation at different temperature and (water-lacticacid)composition levels
# Phase 1 is denoted as "p1" and Phase 2 as "p2"


# --- Imports ---
from COSMOtherm.Configfiles import Config
from COSMOtherm.src.COSMOtherm_functions.Inputfile_Generation import gen_TBOIL_inp_file, gen_binaryLLE_inp_file, gen_binaryActivity_inp_file, gen_PVAP_inp_file, gen_ternaryVLE_NRTL_inp_file
from COSMOtherm.src.COSMOtherm_functions.Run_COSMOtherm_Calculations import run_COSMOtherm_calculations
import logging
import pandas as pd
import os
# Ensure the logs directory exists
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Configure logging in the main file
log_filename = os.path.splitext(os.path.basename(__file__))[0] + ".log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(log_dir, log_filename)),
        logging.StreamHandler()
    ]
)


# --- Main Execution ---
if __name__ == "__main__":
    
    logging.info("Starting new COSMOtherm LLE calculation run.")
        
    
    # Load the solvents data from the CSV file 
    solvents = pd.read_csv(Config.solvents_fullpath)
    carrier = "h2o"
    solute = "lacticacid"
    
    Server = 'TIGER'  # or 'TIGER2'
    
    for solvent in solvents['COSMO_name'].unique():
        
        # file = gen_TBOIL_inp_file(
        #     solvent=solvent, 
        #     ctd_file=Config.TIGER['ctd_file'], 
        #     cdir=Config.TIGER['cdir'], 
        #     ldir=Config.TIGER['ldir'], 
        #     odir=Config.outputfile_dir, 
        #     fdir=Config.TIGER['fdir'],
        #     overwrite=True
        #     )
        
        # solvent_ = "ethanol"
        
        # file = gen_binaryLLE_inp_file(
        #     solvent=solvent, 
        #     ctd_file=Config.TIGER['ctd_file'], 
        #     cdir=Config.TIGER['cdir'], 
        #     ldir=Config.TIGER['ldir'], 
        #     odir=Config.outputfile_dir, 
        #     fdir=Config.TIGER['fdir'],
        #     overwrite=True
        #     )
        
        # file = gen_binaryActivity_inp_file(
        #     solvent=solvent, 
        #     ctd_file=Config.TIGER['ctd_file'], 
        #     cdir=Config.TIGER['cdir'], 
        #     ldir=Config.TIGER['ldir'], 
        #     odir=Config.outputfile_dir, 
        #     fdir=Config.TIGER['fdir'],
        #     overwrite=True
        #     )
        
        # file =  gen_PVAP_inp_file(
        #     solvent=solvent, 
        #     t_start=25,
        #     t_end=100,
        #     t_steps=5,
        #     ctd_file=Config.TIGER['ctd_file'], 
        #     cdir=Config.TIGER['cdir'], 
        #     ldir=Config.TIGER['ldir'], 
        #     odir=Config.outputfile_dir,
        #     fdir=Config.TIGER['fdir'],
        #     inputfiles_folder=Config.inputfile_dir,
        #     overwrite=True
        #     )
        
        file = gen_ternaryVLE_NRTL_inp_file(
            solvent=solvent, 
            carrier=carrier,
            solute=solute,
            tC=30.0,
            ctd_file=Config.TIGER['ctd_file'], 
            cdir=Config.TIGER['cdir'], 
            ldir=Config.TIGER['ldir'], 
            odir=Config.outputfile_dir, 
            fdir=Config.TIGER['fdir'],
            inputfiles_folder=Config.inputfile_dir,
            overwrite=True
        )

        
        run_COSMOtherm_calculations(
            COSMOtherm_exe_fullpath=Config.TIGER['exe_fullpath'],
            files=[file['fullpath']]
        )
        
        break