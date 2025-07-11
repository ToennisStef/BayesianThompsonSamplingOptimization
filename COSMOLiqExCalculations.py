# Automization of liquid-liquid extraction calculations using COSMOtherm
# 2-phase(liquid-liquid), 3-component(water,lacticacid,solvent) system 
# Calculation at different temperature and (water-lacticacid)composition levels
# Phase 1 is denoted as "p1" and Phase 2 as "p2"


# --- Imports ---
from COSMOtherm.Configfiles import Config
from COSMOtherm.src.Chemfuncs import calc_la_molefrac, calc_rho_h2o
from COSMOtherm.src.Design_Matrix import create_design_matrix_for_solvent_screening
from COSMOtherm.src.Inputfile_Generation import gen_2Phase_LIQEX_inp_file, FileGenConfig
from COSMOtherm.src.Run_COSMOtherm_Calculations import run_COSMOtherm_calculations
import logging
import pandas as pd
import os
import time
from multiprocessing.dummy import Pool as ThreadPool
from tqdm import tqdm

# Configure logging in the main file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("COSMOLiqExCalculation_MP.log"),
        logging.StreamHandler()
    ]
)


def process_design_row(row_dict):
    try:
        import os
        from COSMOtherm.Configfiles import Config
        from COSMOtherm.src.Inputfile_Generation import gen_2Phase_LIQEX_inp_file, FileGenConfig
        from COSMOtherm.src.Run_COSMOtherm_Calculations import run_COSMOtherm_calculations
        import logging
        # Reconstruct the row as a Series-like object
        row = row_dict
        composition_type = Config.composition_type
        p1_input = {}
        for key, value in Config.composition_types.items():
            p1_input[key] = [row[f"{value}1_1"], row[f"{value}1_2"], row[f"{value}1_3"]]
        p2_input = {}
        for key, value in Config.composition_types.items():
            p2_input[key] = [row[f"{value}2_1"], row[f"{value}2_2"], row[f"{value}2_3"]]
        components = [str(row['component1']), str(row['component2']), str(row['component3'])]
        config = FileGenConfig(
            ctd_file=Config.TIGER['ctd_file_not_FINE'],
            cdir=Config.TIGER['cdir'],
            ldir=Config.TIGER['ldir'],
            odir=Config.outputfile_dir_screening[composition_type],
            fdir=Config.TIGER['fdir_not_FINE'],
            inputfiles_folder=Config.inputfile_dir_screening[composition_type],
            overwrite=True
        )
        inputfile = gen_2Phase_LIQEX_inp_file(
            tC=float(row['tC']),
            p1_input=p1_input[composition_type],
            p2_input=p2_input[composition_type],
            components=components,
            config=config,
            composition_type=composition_type
        )
        filename = inputfile['filename']
        inputfile_path = inputfile['fullpath']
        inputfile_basename = os.path.splitext(filename)[0]
        tab_file_path = os.path.join(
            Config.outputfile_dir_screening[composition_type],
            inputfile_basename + '.tab'
        )
        if not os.path.exists(tab_file_path):
            run_COSMOtherm_calculations(
                COSMOtherm_exe_fullpath=Config.TIGER['exe_fullpath'],
                files=[inputfile_path]
            )
            logging.info(f"Completed calculation for input: {filename}")
            return (filename, True, None)
        else:
            logging.info(f"Tab file already exists for input: {filename}, skipping.")
            return (filename, True, 'exists')
    except Exception as e:
        logging.error(f"Error processing input {row_dict}: {e}")
        return (str(row_dict), False, str(e))

# --- Main Execution ---
if __name__ == "__main__":
    # Additional parameters for the calculations
    V_p1 = Config.V_p1 # Volume of the first phase in [L]
    V_p2 = Config.V_p2 # Volume of the second phase in [L] 
    tC_levels = Config.tC_levels  # [°C]
    rho_lacticacid_levels = Config.rho_lacticacid_levels  # [g/L] mass concentration of lactic acid in the solution [250]
    rho_h2o_levels = calc_rho_h2o(rho_lacticacid_levels)
    m_lacticacid_levels = [V_p1 * rho_la_level for rho_la_level in rho_lacticacid_levels]  # Mass of lactic acid in [g]
    m_h2o_levels = [V_p1 * rho_h2o_level for rho_h2o_level in rho_h2o_levels]  # Mass of water in [g]
    x_lacticacid_levels = calc_la_molefrac(
        rho_lacticacid=rho_lacticacid_levels
    )
    x_h2o_levels = [1 - x_la_level for x_la_level in x_lacticacid_levels]  # Mole fraction of water in the solution
    # Load the solvents data from the CSV file 
    solvents = pd.read_csv(Config.solvents_fullpath)
    # Progress bar for design matrix creation (if needed)
    logging.info("Creating design matrix...")
    design_matrix = create_design_matrix_for_solvent_screening(
        tC_levels=tC_levels, 
        rho_lacticacid_levels=rho_lacticacid_levels, 
        x_h2o_levels=x_h2o_levels, 
        x_lacticacid_levels=x_lacticacid_levels, 
        m_h2o_levels=m_h2o_levels, 
        m_lacticacid_levels=m_lacticacid_levels, 
        V_p2=V_p2,
        solvents=solvents
    )
    logging.info(f"Design matrix created with {len(design_matrix)} rows.")
    # Convert DataFrame rows to dicts for picklability
    design_rows = [row._asdict() if hasattr(row, '_asdict') else row.to_dict() for _, row in tqdm(design_matrix.iterrows(), total=len(design_matrix), desc="Preparing jobs")]  # progress bar for row conversion
    start_time = time.time()
    with ThreadPool(processes=32) as tpool:
        results_thread = list(tqdm(tpool.imap(process_design_row, design_rows), total=len(design_rows), desc="Running calculations", smoothing=0.1))
    thread_time = time.time() - start_time
    logging.info(f"ThreadPool completed in {thread_time:.2f} seconds.")
    # Log results for thread pool
    for filename, success, error in results_thread:
        if success and error != 'exists':
            logging.info(f"[ThreadPool] Input {filename} processed successfully.")
        elif error == 'exists':
            logging.info(f"[ThreadPool] Input {filename} already exists, skipped.")
        else:
            logging.error(f"[ThreadPool] Input {filename} failed with error: {error}")