# --- Imports ---
from COSMOtherm.Configfiles import Config
from COSMOtherm.src.Chemfuncs import calc_la_molefrac
from COSMOtherm.src.Design_Matrix import build_design_matrix
from COSMOtherm.src.funcs import load_and_sort_solvents, run_initialSample_calculations
import logging

# --- Constants & Global Variables ---
Run_initialSampling = True  # Set to False if you want to skip the initial sampling step

tC_range = [20, 30, 40]  # [°C]
massconcentration_lacticacid_range = [5, 10, 20, 50, 100, 150, 200, 250]  # [g/L] mass concentration of lactic acid in the solution [250]
# Konzentrationen:
# 5[g/L] 10[g/L] 20[g/L] (Für kontinuierliche Abtrennung aus dem Fermenter relevant)
# 50[g/L] 100[g/L] 150[g/L] 200[g/L] 250[g/L] (Das ist laut Literatur die maximale Konzentration die bisher bei Fermentation erreicht wird)

composition_types = [
    "Mole fraction", 
    "Mass fraction", 
    "Masses W[g]", 
    "Mole numbers"
    ] # Types of composition to be used in the calculations

# Configure logging in the main file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("application.log"),
        logging.StreamHandler()
    ]
)
                

# --- Main Execution ---
if __name__ == "__main__":
    
    x1_lacticacid_range = calc_la_molefrac(
        massconcentration_lacticacid=massconcentration_lacticacid_range
        )
    solvents = load_and_sort_solvents(
        Config.solvents_fullpath
        )
        
    # --- Initial Sampling ---    
    if Run_initialSampling:
        # Build the design matrix using the reduced full factorial design for initial sampling
        design_matrix = build_design_matrix(
            tC_range=tC_range, 
            x1_lacticacid_range=x1_lacticacid_range, 
            solvents=solvents,
            reduction=0
            )
        
        # Run the calculations for the design matrix 
        run_initialSample_calculations(
            design_matrix,
            overwrite=False
            )