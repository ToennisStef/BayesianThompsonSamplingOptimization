# Automization of liquid-liquid extraction calculations using COSMOtherm
# 2-phase(liquid-liquid), 3-component(water,lacticacid,solvent) system 
# Calculation at different temperature and (water-lacticacid)composition levels
# Phase 1 is denoted as "p1" and Phase 2 as "p2"


# --- Imports ---
from COSMOtherm.Configfiles import Config
from COSMOtherm.src.Chemfuncs import calc_la_molefrac, calc_rho_h2o
from COSMOtherm.src.Design_Matrix import create_design_matrix_for_solvent_screening
from COSMOtherm.src.COSMOtherm_functions.Inputfile_Generation import gen_2Phase_LIQEX_inp_file
from COSMOtherm.src.COSMOtherm_functions.Run_COSMOtherm_Calculations import run_COSMOtherm_calculations
import logging
import pandas as pd
import os

# Configure logging in the main file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("COSMOLiqExCalculation_MP.log"),
        logging.StreamHandler()
    ]
)


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
    
    composition_type = Config.composition_type  # composition type for COSMOtherm calculation
    
    for index, row in design_matrix.iterrows():
        p1_input = {}
        for key, value in Config.composition_types.items():
            p1_input[key] = [row[value+'1_1'], row[value+'1_2'], row[value+'1_3']]
        p2_input = {}
        for key, value in Config.composition_types.items():
            p2_input[key] = [row[value+'2_1'], row[value+'2_2'], row[value+'2_3']]
        components = {
            'c1': row['component1'],
            'c2': row['component2'],
            'c3': row['component3']
        }
        components = [row['component1'], row['component2'], row['component3']]
        
        inputfiles = []
        inputfiles.append(gen_2Phase_LIQEX_inp_file(
            tC=row['tC'],
            p1_input=p1_input[composition_type],
            p2_input=p2_input[composition_type],
            components=components,
            ctd_file=Config.TIGER['ctd_file'],
            cdir=Config.TIGER['cdir'],
            ldir=Config.TIGER['ldir'],
            fdir=Config.TIGER['fdir'],
            odir=Config.outputfile_dir_screening[composition_type],
            inputfiles_folder=Config.inputfile_dir_screening[composition_type],
            composition_type=composition_type,
            overwrite=False,
        ))
        
        # Check if the corresponding .tab file already exists

        filename = inputfiles[-1]['filename']
        # Get the input file name without extension
        inputfile_path = inputfiles[-1]['fullpath']
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
        
        