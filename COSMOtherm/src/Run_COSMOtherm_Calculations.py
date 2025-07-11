import subprocess
import logging 

logger = logging.getLogger(__name__)

def run_COSMOtherm_calculations(
    COSMOtherm_exe_fullpath: str, 
    files: list,
    )-> list:
    """
    Run COSMOtherm calculations for the provided files using the specified executable.
    Parameters:
        COSMO_exe_fullpath (str): Full path to the COSMO executable.
        files (list): List of file paths to be processed. 
        outputfiles_dir (str): Directory where the output files will be saved.
    Returns:
        results (list): List of results from the calculations.
    """
    results = []
    for file in files:
        # Run the calculations for each file
        # subprocess_str = '"'+ COSMOtherm_exe_fullpath + '"' + ' ' + '"' + file + '"'
        subprocess_str = f'"{COSMOtherm_exe_fullpath}" "{file}" -n 4'  
        logger.info(f"Running COSMOtherm calculation for file: {file}")
        result = subprocess.run(subprocess_str, shell=True)
        results.append(result)
        logger.info(f"Calculation completed with return code: {result.returncode}")
    return results