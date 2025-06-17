# Automization of liquid-liquid extraction calculations using COSMOtherm
# 2-phase(liquid-liquid), 3-component(water,lacticacid,solvent) system 
# Calculation at different temperature and (water-lacticacid)composition levels
# Phase 1 is denoted as "p1" and Phase 2 as "p2"

# Multiprocessing version of the COSMOtherm liquid-liquid extraction calculations
# This script is designed to run COSMOtherm calculations in parallel using multiprocessing.

# --- Imports ---
from COSMOtherm.Configfiles import Config
from COSMOtherm.Configfiles import COSMOthermConfig  # Importing Config_ for the input file generation
from COSMOtherm.src.Chemfuncs import calc_la_molefrac, calc_rho_h2o
from COSMOtherm.src.Design_Matrix import create_design_matrix_for_solvent_screening
from COSMOtherm.src.COSMOtherm_functions.Inputfile_Generation import gen_2Phase_LIQEX_inp_file
from COSMOtherm.src.COSMOtherm_functions.Run_COSMOtherm_Calculations import run_COSMOtherm_calculations
import logging
import logging.handlers
import multiprocessing
import concurrent.futures



import pandas as pd
import os

# Configure logging in the main file
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     handlers=[
#         logging.FileHandler("COSMOLiqExCalculation.log"),
#         logging.StreamHandler()
#     ]
# )

def process_design_row(row, composition_type):
    import os
    from COSMOtherm.src.COSMOtherm_functions.Inputfile_Generation import gen_2Phase_LIQEX_inp_file
    from COSMOtherm.src.COSMOtherm_functions.Run_COSMOtherm_Calculations import run_COSMOtherm_calculations

    setup_logging(log_queue)
    logging.info(f"Worker PID {os.getpid()} starting row processing for {row['tC']} °C, {row['component1']}, {row['component2']}, {row['component3']}")
    
    p1_input = {}
    for key, value in COSMOthermConfig.composition_types.items():
        p1_input[key] = [row[value+'1_1'], row[value+'1_2'], row[value+'1_3']]
    p2_input = {}
    for key, value in COSMOthermConfig.composition_types.items():
        p2_input[key] = [row[value+'2_1'], row[value+'2_2'], row[value+'2_3']]
    components = [row['component1'], row['component2'], row['component3']]

    inputfile = gen_2Phase_LIQEX_inp_file(
        tC=row['tC'],
        p1_input=p1_input[composition_type],
        p2_input=p2_input[composition_type],
        components=components,
        ctd_file=COSMOthermConfig.TIGER['ctd_file'],
        cdir=COSMOthermConfig.TIGER['cdir'],
        ldir=COSMOthermConfig.TIGER['ldir'],
        fdir=COSMOthermConfig.TIGER['fdir'],
        odir=COSMOthermConfig.outputfile_dir_screening[composition_type],
        inputfiles_folder=COSMOthermConfig.inputfile_dir_screening[composition_type],
        composition_type=composition_type,
        overwrite=False,
    )

    filename = inputfile['filename']
    inputfile_path = inputfile['fullpath']
    inputfile_basename = os.path.splitext(filename)[0]
    tab_file_path = os.path.join(
        COSMOthermConfig.outputfile_dir_screening[composition_type],
        inputfile_basename + '.tab'
    )

    if not os.path.exists(tab_file_path):
        run_COSMOtherm_calculations(
            COSMOtherm_exe_fullpath=COSMOthermConfig.TIGER['exe_fullpath'],
            files=[inputfile_path]
        )
    return tab_file_path

def setup_logging(log_queue):
    handler = logging.handlers.QueueHandler(log_queue)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers = []
    root.addHandler(handler)

def listener_process(log_queue):
    handler = logging.FileHandler("COSMOLiqExCalculation_MP_Mole_fraction.log")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    listener = logging.handlers.QueueListener(log_queue, handler)
    listener.start()
    return listener

def worker_init(lq):
    global log_queue
    log_queue = lq

# --- Main Execution ---
if __name__ == "__main__":
    
    log_queue = multiprocessing.Queue()
    listener = listener_process(log_queue)
    setup_logging(log_queue)
    
    
    # Additional parameters for the calculations
    V_p1 = 1.0 # Volume of the first phase in [L]
    V_p2 = 1.0 # Volume of the second phase in [L] 
    
    tC_levels = [20, 30, 40]  # [°C]
    rho_lacticacid_levels = [5, 10, 20, 50, 100, 150, 200, 250]  # [g/L] mass concentration of lactic acid in the solution [250]
    rho_h2o_levels = calc_rho_h2o(rho_lacticacid_levels) # [g/L] mass concentration of water in the fermentation broth
    
    m_lacticacid_levels = [V_p1 * rho_la_level for rho_la_level in rho_lacticacid_levels]  # Mass of lactic acid in [g]
    m_h2o_levels = [V_p1 * rho_h2o_level for rho_h2o_level in rho_h2o_levels]  # Mass of water in [g]
    
    x_lacticacid_levels = calc_la_molefrac(
        rho_lacticacid=rho_lacticacid_levels
        )
    x_h2o_levels = [1 - x_la_level for x_la_level in x_lacticacid_levels]  # Mole fraction of water in the solution
    
    # Load the solvents data from the CSV file 
    solvents = pd.read_csv(Config.solvents_fullpath)
    
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
    
    composition_type = "Mole fraction"  # composition type for COSMOtherm calculation
    
    # Convert DataFrame rows to dicts for pickling
    rows = [row._asdict() if hasattr(row, '_asdict') else row.to_dict() for _, row in design_matrix.iterrows()]

    logging.info(f"Starting COSMOtherm calculations for {len(rows)} design matrix rows with composition type '{composition_type}'")
    with concurrent.futures.ProcessPoolExecutor(initializer=worker_init, initargs=(log_queue,)) as executor:
        futures = [executor.submit(process_design_row, row, composition_type) for row in rows]
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                logging.info(f"Processed: {result}")
            except Exception as exc:
                logging.error(f"Generated an exception: {exc}")

    listener.stop()
    
    # for index, row in design_matrix.iterrows():
    #     p1_input = {}
    #     for key, value in COSMOthermConfig.composition_types.items():
    #         p1_input[key] = [row[value+'1_1'], row[value+'1_2'], row[value+'1_3']]
    #     p2_input = {}
    #     for key, value in COSMOthermConfig.composition_types.items():
    #         p2_input[key] = [row[value+'2_1'], row[value+'2_2'], row[value+'2_3']]
    #     components = {
    #         'c1': row['component1'],
    #         'c2': row['component2'],
    #         'c3': row['component3']
    #     }
    #     components = [row['component1'], row['component2'], row['component3']]
        
    #     inputfiles = []
    #     inputfiles.append(gen_2Phase_LIQEX_inp_file(
    #         tC=row['tC'],
    #         p1_input=p1_input[composition_type],
    #         p2_input=p2_input[composition_type],
    #         components=components,
    #         ctd_file=COSMOthermConfig.TIGER['ctd_file'],
    #         cdir=COSMOthermConfig.TIGER['cdir'],
    #         ldir=COSMOthermConfig.TIGER['ldir'],
    #         fdir=COSMOthermConfig.TIGER['fdir'],
    #         odir=COSMOthermConfig.outputfile_dir_screening[composition_type],
    #         inputfiles_folder=COSMOthermConfig.inputfile_dir_screening[composition_type],
    #         composition_type=composition_type,
    #         overwrite=False,
    #     ))
        
    #     # Check if the corresponding .tab file already exists

    #     filename = inputfiles[-1]['filename']
    #     # Get the input file name without extension
    #     inputfile_path = inputfiles[-1]['fullpath']
    #     inputfile_basename = os.path.splitext(filename)[0]
    #     tab_file_path = os.path.join(
    #         COSMOthermConfig.outputfile_dir_screening[composition_type],
    #         inputfile_basename + '.tab'
    #     )

        
    #     if not os.path.exists(tab_file_path):
    #         run_COSMOtherm_calculations(
    #         COSMOtherm_exe_fullpath=COSMOthermConfig.TIGER['exe_fullpath'],
    #         files=[inputfile_path]
    #         )
        
        