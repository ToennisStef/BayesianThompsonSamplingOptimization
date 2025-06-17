# General setup file for COSMOtherm parameters
# naming: dir + file + extension

# --- Bayesion Optimization Configuration File ---
tC_range = [20, 40]  # considered temperature range for the extraction in [°C] 
massconcentration_lacticacid_range = [5, 250]  # considered initial mass concentration of lactic acid in the fermentation broth in [g/L] 
solvents_fullpath = r"./COSMOtherm/Datafiles/solvents+densities.csv" # Path the solvents file (Contains information on all considered solvents)

# --- 
COSMOtherm_exe_fullpath = r"C:\Program Files\COSMOlogic\COSMOthermX19\COSMOtherm\BIN-WINDOWS\cosmotherm.exe" # Use absolute path
outputfile_dir = r".\COSMOtherm\outputfiles" # the generated outputfiles are stored here 
initialSamples_dir = r".\COSMOtherm\outputfiles\CompleteScreening" # The outputfiles form the initial sampling are stored here # r"./COSMOtherm/outputfiles/init_files" # Use relative path
inputfile_dir = r"./COSMOtherm/inputfiles" # The generated input files are stored here
# Add additional parameters here if needed



# cosmotherm_filepath = r""""C:\Program Files\COSMOlogic\COSMOthermX19\COSMOtherm\BIN-WINDOWS\cosmotherm.exe" """
# Screening_dir = r".\COSMOtherm\outputfiles\CompleteScreening" # Use relative path
