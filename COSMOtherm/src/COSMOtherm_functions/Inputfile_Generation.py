import os
import logging

# Use a module-specific logger
logger = logging.getLogger(__name__)

def gen_LIQEX_inp_file(
    tC:float,
    x1_lacticacid:float, 
    solvent:str, 
    ctd_file:str, 
    cdir:str, 
    ldir:str, 
    odir:str, 
    fdir:str,
    file_name:str = None,  
    inputfiles_folder:str =".",
    overwrite:bool = False
    )->None:
    """
    Generates an input file for LIQEX calculations in COSMOtherm based on the provided parameters.
    This is hardcoded for the LIQEX calculation with 3 components: water, lactic acid, and a solvent.
    The input file is saved in the specified directory with the given name or a default name if none is provided.
    The function also ensures that the output directory exists before writing the file.
    The function checks if the file already exists and prompts the user for overwriting it.
    Parameters:
        tC (float): Temperature in degrees Celsius.
        x1_lacticacid (float): Mole fraction of lactic acid in initial water & lactic acid solution.
        solvent (str): Name of the solvent.
        ctd_file (str): Path to the CT-Data file.
        cdir (str): Directory for the CT-data/parameterization file.  
        ldir (str): Directory for the license file.
        odir (str): Directory for the output files.
        fdir (str): Directory for the compound data files.
        file_name (str, optional): Name of the inputfile. If None, a default name will be generated.
        inputfiles_folder (str, optional): Directory where the inputfile will be saved. Default is the current directory.
    Returns:
        str: The full path of the generated input file.
        str: The name of the generated input file.
    """
    
    # Ensure the output directory exists
    os.makedirs(inputfiles_folder, exist_ok=True)
    logger.debug(f"Ensured the directory '{inputfiles_folder}' exists.")
    
    if file_name is None:
        file_name = f"LIQEX_{solvent}_tc{tC}_x{x1_lacticacid}.inp"
    file_fullpath = os.path.join(inputfiles_folder, file_name)
    logger.debug(f"Generated file name: {file_name}, full path: {file_fullpath}")
    
    # Calculate the mole fraction of water
    x1_h2o = 1-x1_lacticacid
    
    # Define content
    content = f"""ctd={ctd_file} CDIR="{cdir}" LDIR="{ldir}" odir="{odir}" # Global command line
FDIR="{fdir}" vpfile CTAB WCONF AUTOC                                 # Global command line
!! Multi-Component-2-Phase-Equilibrium calculation                        # Comment line
f = h2o                      # Compound input (water)
f = {solvent}                # Compound input (solvent)
f = lacticacid               # Compound input (lactic acid)
tc={tC} LIQ_EX x1={{{x1_h2o} 0 {x1_lacticacid}}} x2={{0 1 0}}
# LIQ_EX: Liquid phase equilibrium calculation
# x1: Mole fraction of the first component in the input stream
# x2: Mole fraction of the second component in the input stream
# tc: Temperature in degrees Celsius
# ctd: CT-Data file name / parameterization file
# CDIR: Directory for the CT-data/parameterization file
# LDIR: Directory for the licence file
# odir: Directory for the output files
# FDIR: Directory for the compound data files
# vpfile: ???
# CTAB: some additional output configuration (see documentation)
# WCONF: some additional output configuration (see documentation)
# AUTOC: Specifies automatic conformere search & consideration for all compound
"""
        

    # Check if the output file already exists
    if os.path.exists(file_fullpath):
        logger.warning(f"File already exists: {file_fullpath}")
        if not overwrite:
            logger.info(f"Skipping inputfile generation for: '{file_name}'")
            logger.info("The file was not overwritten. Returning the existing file path and name.")
            return file_fullpath, file_name
        
    with open(file_fullpath, "w") as file:
        logger.debug(f"Generating inputfile: '{file_fullpath}'")
        file.write(content)
        logger.info(f"Input file '{file_fullpath}' has been created successfully. Returning the file path and name.")
        
    return file_fullpath, file_name

    #     if overwrite != 'y':
    #         print("Operation cancelled. The file was not overwritten. returning the existing file path and name.")
    #     else:    
    #     # Write content to file
    #         with open(file_fullpath, "w") as file:
    #             file.write(content)
    #         print(f"Input file '{file_fullpath}' has been created successfully. returning the file path and name")
    # else:
    #     with open(file_fullpath, "w") as file:
    #         file.write(content)
    #         print(f"Input file '{file_fullpath}' has been created successfully. returning the file path and name")
    # return file_fullpath, file_name
