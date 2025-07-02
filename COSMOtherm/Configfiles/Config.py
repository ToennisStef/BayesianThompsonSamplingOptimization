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

# cosmotherm_filepath = r""""C:\Program Files\COSMOlogic\COSMOthermX19\COSMOtherm\BIN-WINDOWS\cosmotherm.exe" """
# Screening_dir = r".\COSMOtherm\outputfiles\CompleteScreening" # Use relative path
