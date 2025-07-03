# General setup file for COSMOtherm parameters
# naming: dir + file + extension

import json
import os

# Load configuration from config.json
config_path = os.path.join(os.path.dirname(__file__), 'config.json')
with open(config_path, 'r') as f:
    config = json.load(f)

tC_range = config["tC_range"]  # considered temperature range for the extraction in [°C]
massconcentration_lacticacid_range = config["massconcentration_lacticacid_range"]  # considered initial mass concentration of lactic acid in the fermentation broth in [g/L]
solvents_fullpath = config["solvents_fullpath"]  # Path the solvents file (Contains information on all considered solvents)
outputfile_dir = config["outputfile_dir"]  # the generated outputfiles are stored here
initialSamples_dir = config["initialSamples_dir"]  # The outputfiles form the initial sampling are stored here
inputfile_dir = config["inputfile_dir"]  # The generated input files are stored here

# Merged from COSMOthermConfig.py
LION = config["LION"]
TIGER = config["TIGER"]
composition_types = config["composition_types"]
inputfile_dir_screening = config["inputfile_dir_screening"]
outputfile_dir_screening = config["outputfile_dir_screening"]
# Add additional parameters here if needed

#   },
#   "V_p1": 1.0,
#   "V_p2": 1.0,
#   "tC_levels": [20, 30, 40],
#   "rho_lacticacid_levels": [5, 10, 20, 50, 100, 150, 200, 250],
#   "composition_type": "Masses W[g]",
#   "N_steps": 50,
#   "Run_initialSampling": false,
#   "log_file": "BayesianOptimization.log",
#   "reduction": 4
# } 
V_p1 = config["V_p1"] 
V_p2 = config["V_p2"] 
tC_levels = config["tC_levels"]  # [°C]
rho_lacticacid_levels = config["rho_lacticacid_levels"]  # [g/L] mass concentration of lactic acid in the solution
composition_type = config["composition_type"]  # composition type for COSMOtherm calculation
N_steps = config["N_steps"]  # Number of steps for the Bayesian optimization
Run_initialSampling = config["Run_initialSampling"]  # Flag to run initial sampling
log_file = config["log_file"]  # Log file name for the calculations
reduction = config["reduction"]  # Reduction factor for the number of solvents in the screening

# cosmotherm_filepath = r""""C:\Program Files\COSMOlogic\COSMOthermX19\COSMOtherm\BIN-WINDOWS\cosmotherm.exe" """
# Screening_dir = r".\COSMOtherm\outputfiles\CompleteScreening" # Use relative path
