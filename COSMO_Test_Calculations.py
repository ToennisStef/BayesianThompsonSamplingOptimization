# Exemplary calcualtion of liquid-liquid equilibrium calculation using COSMOtherm
# Calculation at different temperature and (water-lacticacid)composition levels
# Phase 1 is denoted as "p1" and Phase 2 as "p2"


# --- Imports ---
from COSMOtherm.Configfiles import Config
from COSMOtherm.src.Inputfile_Generation import gen_TBOIL_inp_file, gen_binaryLLE_inp_file, gen_binaryActivity_inp_file, gen_PVAP_inp_file, gen_ternaryVLE_NRTL_inp_file, gen_2Phase_LIQEX_inp_file, FileGenConfig
from COSMOtherm.src.Run_COSMOtherm_Calculations import run_COSMOtherm_calculations
import logging
import pandas as pd
import os
import multiprocessing
import time
from multiprocessing.dummy import Pool as ThreadPool
from tqdm import tqdm
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


def process_solvent(solvent):
    carrier = "h2o"
    solute = "lacticacid"
    try:
        config = FileGenConfig(
            ctd_file=Config.TIGER['ctd_file_not_FINE'],
            cdir=Config.TIGER['cdir'],
            ldir=Config.TIGER['ldir'],
            odir=Config.outputfile_dir,
            fdir=Config.TIGER['fdir_not_FINE'],
            inputfiles_folder=Config.inputfile_dir,
            overwrite=True
        )
        # file = gen_PVAP_inp_file(
        #     solvent=solvent,
        #     t_start=25,
        #     t_end=200,
        #     t_steps=5,
        #     config=config
        # )

        #tboil
        # file = gen_TBOIL_inp_file(
        #     solvent=solvent,
        #     config=config
        # )

        file = gen_ternaryVLE_NRTL_inp_file(
            carrier=carrier,
            solute=solute,
            solvent=solvent,
            tC=40.0,
            config=config,
        )

        result = run_COSMOtherm_calculations(
            COSMOtherm_exe_fullpath=Config.TIGER['exe_fullpath'],
            files=[file['fullpath']]
        )
        logging.info(f"Completed calculation for solvent: {solvent}")
        return (solvent, True, None)
    except Exception as e:
        logging.error(f"Error processing solvent {solvent}: {e}")
        return (solvent, False, str(e))

# --- Main Execution ---
if __name__ == "__main__":
    logging.info("Starting new COSMOtherm PVAP calculation run.")
    solvents = pd.read_csv(Config.solvents_fullpath)
    Server = 'TIGER'  # or 'TIGER2'
    solvent_list = solvents['COSMO_name'].unique().tolist()
    # Progress bar for preparing jobs (if needed)
    logging.info(f"Preparing {len(solvent_list)} solvent jobs...")
    start_time = time.time()
    with ThreadPool(processes=400) as tpool:
        results_thread = list(tqdm(tpool.imap(process_solvent, solvent_list), total=len(solvent_list), desc="Running calculations", smoothing=0.1))
    thread_time = time.time() - start_time
    logging.info(f"ThreadPool completed in {thread_time:.2f} seconds.")

    # Log results for thread pool
    for solvent, success, error in results_thread:
        if success:
            logging.info(f"[ThreadPool] Solvent {solvent} processed successfully.")
        else:
            logging.error(f"[ThreadPool] Solvent {solvent} failed with error: {error}")