# --- COSMOtherm Configuration file ---
# This file contains the configuration for COSMOtherm calculations 
ctd_file = "BP_TZVPD_FINE_19.ctd"  # COSMOtherm CTD file
cdir = r"C:\Program Files\COSMOlogic\COSMOthermX19\COSMOtherm\CTDATA-FILES"  # COSMOtherm CTDATA directory
ldir = "C:\Program Files\COSMOlogic\COSMOthermX19\licensefiles"  # COSMOtherm license files directory
fdir = r"V:\groups\COSMOTHERM-Datenbank\COSMObase-1901\BP-TZVPD-FINE"  # COSMObase directory
odir = r"Q:\Groups\eicr students\Stefan Tönnis\GIT_Repos\BayesianThompsonSamplingOptimization\COSMOtherm\outputfiles" # Output directory

# Server configuration for COSMOtherm
LION = {
    "COSMOtherm_version": r"COSMOtherm 2019",  # Version of COSMOtherm
    "exe_fullpath": r"C:\Program Files\COSMOlogic\COSMOthermX19\COSMOtherm\BIN-WINDOWS\cosmotherm.exe",  # Full path to the COSMOtherm executable
    "ctd_file": ctd_file,  # CTD file for COSMOtherm
    "cdir": cdir,  # Directory for COSMOtherm CTDATA files
    "ldir": ldir,  # Directory for COSMOtherm license files
    "fdir": fdir,  # Directory for COSMObase files
    "odir": odir  # Output directory for results
}

TIGER = {
    "COSMOtherm_version": r"COSMOtherm 2021",  # Version of COSMOtherm
    "exe_fullpath": r"C:\Program Files\BIOVIA\COSMOtherm2021\COSMOtherm\BIN-WINDOWS\cosmotherm.exe",  # Full path to the COSMOtherm executable
    "ctd_file": r"BP_TZVPD_FINE_21.ctd",  # CTD file for COSMOtherm
    "cdir": r"C:\Program Files\BIOVIA\COSMOtherm2021\COSMOtherm\CTDATA-FILES",          # Directory for COSMOtherm CTDATA files
    "ldir": r"C:\Program Files\BIOVIA\COSMOtherm2021\licensefiles",          # Directory for COSMOtherm license files
    "fdir": r"V:\groups\COSMOTHERM-Datenbank 2021\BP-TZVPD-FINE",          # Directory for COSMObase files
    "odir": r"Q:\Groups\eicr students\Stefan Tönnis\GIT_Repos\BayesianThompsonSamplingOptimization\COSMOtherm\outputfiles"           # Output directory for results
}

composition_types = {
    "Mole fraction": 'x', 
    "Mass fraction": 'c', 
    "Masses W[g]": 'W', 
    "Mole numbers": 'N',
}

inputfile_dir_screening = {
    "Mole fraction": r"./COSMOtherm/inputfiles/Screening/MoleFraction",
    "Mass fraction": r"./COSMOtherm/inputfiles/Screening/MassFraction",
    "Masses W[g]": r"./COSMOtherm/inputfiles/Screening/MassesWg",
    "Mole numbers": r"./COSMOtherm/inputfiles/Screening/MoleNumbers",
}

outputfile_dir_screening = {
    "Mole fraction": r"./COSMOtherm/outputfiles/Screening/MoleFraction",
    "Mass fraction": r"./COSMOtherm/outputfiles/Screening/MassFraction",
    "Masses W[g]": r"./COSMOtherm/outputfiles/Screening/MassesWg",
    "Mole numbers": r"./COSMOtherm/outputfiles/Screening/MoleNumbers",
}