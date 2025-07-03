import os
import logging

# Use a module-specific logger
logger = logging.getLogger(__name__)

def gen_2Phase_LIQEX_inp_file(
    tC:float,
    p1_input:list,
    p2_input:list, 
    components:list,
    ctd_file:str, 
    cdir:str, 
    ldir:str, 
    fdir:str,
    odir:str, 
    file_name:str = None,  
    inputfiles_folder:str =".",
    composition_type:str = "Mole fraction",
    overwrite:bool = False
    )->None:
    """
    Generates an input file for a general multi-component 2-phase LIQEX calculation in COSMOtherm.

    This function allows specification of arbitrary components and their compositions in two phases.
    The input file is saved in the specified directory with the given name or a default name if none is provided.
    The function ensures that the output directory exists before writing the file.
    If the file already exists, it will only be overwritten if 'overwrite' is set to True.

    Parameters:
        tC (float): Temperature in degrees Celsius.
        p1_input (list): Composition of phase 1 (e.g., mole fractions, mass fractions, etc.).
        p2_input (list): Composition of phase 2 (same format as p1_input).
        components (list): List of component names (strings).
        ctd_file (str): Path to the CT-Data file.
        cdir (str): Directory for the CT-data/parameterization file.
        ldir (str): Directory for the license file.
        fdir (str): Directory for the compound data files.
        odir (str): Directory for the output files.
        file_name (str, optional): Name of the input file. If None, a default name will be generated.
        inputfiles_folder (str, optional): Directory where the input file will be saved. Default is the current directory.
        composition_type (str, optional): Type of composition provided ("Mole fraction", "Mass fraction", "Masses W[g]", or "Mole numbers"). Default is "Mole fraction".
        overwrite (bool, optional): If True, overwrite existing files. Default is False.

    Returns:
        str: The full path of the generated input file.
        str: The name of the generated input file.
    """
    
    if composition_type == "Mole fraction":
        c_type = 'x'
    elif composition_type == "Mass fraction":
        c_type = 'c'
    elif composition_type == "Masses W[g]":
        c_type = 'W'
    elif composition_type == "Mole numbers": 
        c_type = 'N'
    
    # Normalize p1_input and p2_input if needed
    if composition_type in ["Mole fraction", "Mass fraction"]:
        sum_p1 = sum(p1_input)
        sum_p2 = sum(p2_input)
        if sum_p1 > 0 and not abs(sum_p1 - 1.0) < 1e-8:
            p1_input = [x / sum_p1 for x in p1_input]
            logger.info("Normalized p1_input to sum to 1.")
        if sum_p2 > 0 and not abs(sum_p2 - 1.0) < 1e-8:
            p2_input = [x / sum_p2 for x in p2_input]
            logger.info("Normalized p2_input to sum to 1.")
    
    # Ensure the output directory exists
    os.makedirs(inputfiles_folder, exist_ok=True)
    logger.debug(f"Ensured the directory '{inputfiles_folder}' exists.")
    
    if file_name is None:
        file_name = f"LIQEX_{components[1]}_tc{tC}_{c_type}1{p1_input[2]}.inp"
    file_fullpath = os.path.join(inputfiles_folder, file_name)
    logger.debug(f"Generated file name: {file_name}, full path: {file_fullpath}")
        
    # Define content
    content = f"""ctd={ctd_file} CDIR="{cdir}" LDIR="{ldir}" odir="{odir}" # Global command line
FDIR="{fdir}" vpfile CTAB WCONF AUTOC                                 # Global command line
!! Multi-Component-2-Phase-Equilibrium calculation                    # Comment line
"""

    # Ensure P1_input and P2_input have the same length as Components
    if len(p1_input) > len(components):
        raise ValueError("Too many entries for P1_input. Number of entires cannot exeed the number of specified Components!")
    if len(p2_input) > len(components):
        raise ValueError("Too many entries for P2_input. Number of entires cannot exeed the number of specified Components!")
    
    while len(p1_input) < len(components):
        p1_input.append(0)
    while len(p2_input) < len(components):
        p2_input.append(0)

    for component in components:
        content += f"f = {component}\n"
    
    content += f"""tc={tC} LIQ_EX {c_type}1={{{' '.join(map(str, p1_input))}}} {c_type}2={{{' '.join(map(str, p2_input))}}} maxiter=1000 """
            
    # Check if the output file already exists
    if os.path.exists(file_fullpath):
        logger.warning(f"File already exists: {file_fullpath}")
        if not overwrite:
            logger.info(f"Skipping inputfile generation for: '{file_name}'")
            logger.info("The file was not overwritten. Returning the existing file path and name.")
            return {
                "fullpath": os.path.abspath(file_fullpath), 
                "filename": file_name,
                "folder": os.path.abspath(inputfiles_folder)
            }
        
    with open(file_fullpath, "w") as file:
        logger.debug(f"Generating inputfile: '{file_fullpath}'")
        file.write(content)
        logger.info(f"Input file '{file_fullpath}' has been created successfully. Returning the file path and name.")
    
    return {
        "fullpath": os.path.abspath(file_fullpath),
        "filename": file_name,
        "folder": os.path.abspath(inputfiles_folder)
    }
    
def gen_TBOIL_inp_file(
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
    Generates an input file for TBOIL calculations in COSMOtherm for a given solvent.
    
    Parameters:
        solvent (str): Name of the solvent.
        ctd_file (str): Path to the CT-Data file.
        cdir (str): Directory for the CT-data/parameterization file.
        ldir (str): Directory for the license file.
        odir (str): Directory for the output files.
        fdir (str): Directory for the compound data files.
        file_name (str, optional): Name of the input file. If None, a default name will be generated.
        inputfiles_folder (str, optional): Directory where the input file will be saved. Default is the current directory.
        overwrite (bool, optional): If True, overwrite existing files. Default is False.

    Returns:
        str: The full path of the generated input file.
        str: The name of the generated input file.
    """
    
    # Ensure the output directory exists
    os.makedirs(inputfiles_folder, exist_ok=True)
    logger.debug(f"Ensured the directory '{inputfiles_folder}' exists.")
    
    if file_name is None:
        file_name = f"TBOIL_{solvent}.inp"
    file_fullpath = os.path.join(inputfiles_folder, file_name)
    
    # Define content
    content = f"""ctd={ctd_file} CDIR="{cdir}" LDIR="{ldir}" odir="{odir}" # Global command line
FDIR="{fdir}" efile UNIT=SI AUTOC                                 # Global command line
!! TBOIL calculation for a single solvent                          # Comment line
"""
    content += f"f = {solvent}\n"
    content += f"x={{1}} pvap=1013.2 tk=300.0 "
    
    if os.path.exists(file_fullpath):
        logger.warning(f"File already exists: {file_fullpath}")
        if not overwrite:
            logger.info(f"Skipping inputfile generation for: '{file_name}'")
            logger.info("The file was not overwritten. Returning the existing file path and name.")
            return {
                "fullpath": os.path.abspath(file_fullpath),
                "filename": file_name,
                "folder": os.path.abspath(inputfiles_folder)
            }
            
    with open(file_fullpath, "w") as file:
        logger.debug(f"Generating inputfile: '{file_fullpath}'")
        file.write(content)
        logger.info(f"Input file '{file_fullpath}' has been created successfully. Returning the file path and name.")
    
    return {
        "fullpath": os.path.abspath(file_fullpath),
        "filename": file_name,
        "folder": os.path.abspath(inputfiles_folder)
    }
    
    
def gen_binaryLLE_inp_file(
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
    Generates an input file for TBOIL calculations in COSMOtherm for a given solvent.
    
    Parameters:
        solvent (str): Name of the solvent.
        ctd_file (str): Path to the CT-Data file.
        cdir (str): Directory for the CT-data/parameterization file.
        ldir (str): Directory for the license file.
        odir (str): Directory for the output files.
        fdir (str): Directory for the compound data files.
        file_name (str, optional): Name of the input file. If None, a default name will be generated.
        inputfiles_folder (str, optional): Directory where the input file will be saved. Default is the current directory.
        overwrite (bool, optional): If True, overwrite existing files. Default is False.

    Returns:
        str: The full path of the generated input file.
        str: The name of the generated input file.
    """
    
    # Ensure the output directory exists
    os.makedirs(inputfiles_folder, exist_ok=True)
    logger.debug(f"Ensured the directory '{inputfiles_folder}' exists.")
    
    if file_name is None:
        file_name = f"binaryLLE_h2o_{solvent}.inp"
    file_fullpath = os.path.join(inputfiles_folder, file_name)
    
    # Define content
    content = f"""ctd={ctd_file} CDIR="{cdir}" LDIR="{ldir}" odir="{odir}" # Global command line
FDIR="{fdir}" efile UNIT=SI AUTOC nomix                                # Global command line
! Binary phase VLE/LLE diagram computation with phase separation                  # Comment line
"""
    content += f"f = h2o\n"  # Water is always the first component in binary LLE
    content += f"f = {solvent}\n"
    content += f"tc=25 tc2=200 tstepsize=10 BINARY LLE NRTL                                                            # Automatic binary computation and LLE search"
    
    if os.path.exists(file_fullpath):
        logger.warning(f"File already exists: {file_fullpath}")
        if not overwrite:
            logger.info(f"Skipping inputfile generation for: '{file_name}'")
            logger.info("The file was not overwritten. Returning the existing file path and name.")
            return {
                "fullpath": os.path.abspath(file_fullpath),
                "filename": file_name,
                "folder": os.path.abspath(inputfiles_folder)
            }
            
    with open(file_fullpath, "w") as file:
        logger.debug(f"Generating inputfile: '{file_fullpath}'")
        file.write(content)
        logger.info(f"Input file '{file_fullpath}' has been created successfully. Returning the file path and name.")
    
    return {
        "fullpath": os.path.abspath(file_fullpath),
        "filename": file_name,
        "folder": os.path.abspath(inputfiles_folder)
    }
    
    
def gen_binaryActivity_inp_file(
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
    Generates an input file for TBOIL calculations in COSMOtherm for a given solvent.
    
    Parameters:
        solvent (str): Name of the solvent.
        ctd_file (str): Path to the CT-Data file.
        cdir (str): Directory for the CT-data/parameterization file.
        ldir (str): Directory for the license file.
        odir (str): Directory for the output files.
        fdir (str): Directory for the compound data files.
        file_name (str, optional): Name of the input file. If None, a default name will be generated.
        inputfiles_folder (str, optional): Directory where the input file will be saved. Default is the current directory.
        overwrite (bool, optional): If True, overwrite existing files. Default is False.

    Returns:
        str: The full path of the generated input file.
        str: The name of the generated input file.
    """
    
    # Ensure the output directory exists
    os.makedirs(inputfiles_folder, exist_ok=True)
    logger.debug(f"Ensured the directory '{inputfiles_folder}' exists.")
    
    if file_name is None:
        file_name = f"binaryactivity_h2o_{solvent}.inp"
    file_fullpath = os.path.join(inputfiles_folder, file_name)
    
    # Define content
    content = f"""ctd={ctd_file} CDIR="{cdir}" LDIR="{ldir}" odir="{odir}" # Global command line
FDIR="{fdir}" AUTOC                                 # Global command line
!! Activity Coefficient computation                                               # Comment line
"""
    content += f"f = h2o\n"  # Water is always the first component in binary LLE
    content += f"f = {solvent}\n"
    content += f"tc=25 gamma xg={{{"0.24 0.76"}}}"                                                              # Automatic binary computation and LLE search"
    
    if os.path.exists(file_fullpath):
        logger.warning(f"File already exists: {file_fullpath}")
        if not overwrite:
            logger.info(f"Skipping inputfile generation for: '{file_name}'")
            logger.info("The file was not overwritten. Returning the existing file path and name.")
            return {
                "fullpath": os.path.abspath(file_fullpath),
                "filename": file_name,
                "folder": os.path.abspath(inputfiles_folder)
            }
            
    with open(file_fullpath, "w") as file:
        logger.debug(f"Generating inputfile: '{file_fullpath}'")
        file.write(content)
        logger.info(f"Input file '{file_fullpath}' has been created successfully. Returning the file path and name.")
    
    return {
        "fullpath": os.path.abspath(file_fullpath),
        "filename": file_name,
        "folder": os.path.abspath(inputfiles_folder)
    }
    

def gen_PVAP_inp_file(
    solvent:str, 
    t_start:float,
    t_end:float,    
    t_steps:int,
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
    Generates an input file for TVAP (Vapour pressure calculation) calculations in COSMOtherm for a given solvent.
    Vapour pressure calculations are executed at different temperatures. 
    When more then three temperatures are specified, the Antoine parameters are fitted to the data.
    
    Parameters:
        solvent (str): Name of the solvent.
        ctd_file (str): Path to the CT-Data file.
        cdir (str): Directory for the CT-data/parameterization file.
        ldir (str): Directory for the license file.
        odir (str): Directory for the output files.
        fdir (str): Directory for the compound data files.
        file_name (str, optional): Name of the input file. If None, a default name will be generated.
        inputfiles_folder (str, optional): Directory where the input file will be saved. Default is the current directory.
        overwrite (bool, optional): If True, overwrite existing files. Default is False.

    Returns:
        str: The full path of the generated input file.
        str: The name of the generated input file.
    """
    
    # Ensure the output directory exists
    os.makedirs(inputfiles_folder, exist_ok=True)
    logger.debug(f"Ensured the directory '{inputfiles_folder}' exists.")
    
    if file_name is None:
        file_name = f"PVAP_{solvent}.inp"
    file_fullpath = os.path.join(inputfiles_folder, file_name)
    
    # Define content
    content = f"""ctd={ctd_file} CDIR="{cdir}" LDIR="{ldir}" odir="{odir}" # Global command line
FDIR="{fdir}" AUTOC UNIT=SI                                # Global command line
!  Automatic computation of vapor pressure curve p_vap(T)                                          # Comment line
"""
    content += f"f = {solvent}\n"
    content += f"x={{1}} pvap tc={t_start} tc2={t_end} tstep={t_steps}"                                                              # Automatic binary computation and LLE search"
    
    if os.path.exists(file_fullpath):
        logger.warning(f"File already exists: {file_fullpath}")
        if not overwrite:
            logger.info(f"Skipping inputfile generation for: '{file_name}'")
            logger.info("The file was not overwritten. Returning the existing file path and name.")
            return {
                "fullpath": os.path.abspath(file_fullpath),
                "filename": file_name,
                "folder": os.path.abspath(inputfiles_folder)
            }
            
    with open(file_fullpath, "w") as file:
        logger.debug(f"Generating inputfile: '{file_fullpath}'")
        file.write(content)
        logger.info(f"Input file '{file_fullpath}' has been created successfully. Returning the file path and name.")
    
    return {
        "fullpath": os.path.abspath(file_fullpath),
        "filename": file_name,
        "folder": os.path.abspath(inputfiles_folder)
    }

# ternary={1 2 3} tc=25.0 NRTL # Ternary VLE computation
def gen_ternaryVLE_NRTL_inp_file(
    carrier:str,
    solute:str,
    solvent:str,
    tC:float,
    ctd_file:str,
    cdir:str,
    ldir:str,
    odir:str,
    fdir:str,
    file_name:str = None,
    inputfiles_folder:str = ".",
    overwrite:bool = False
    )->None:
    """
    Generates an input file for ternary VLE + NRTL calculations in COSMOtherm for a given carrier, solute, and solvent.
    
    Parameters:
        carrier (str): Name of the carrier component.
        solute (str): Name of the solute component.
        solvent (str): Name of the solvent component.
        tC (float): Temperature in degrees Celsius.
        ctd_file (str): Path to the CT-Data file.
        cdir (str): Directory for the CT-data/parameterization file.
        ldir (str): Directory for the license file.
        odir (str): Directory for the output files.
        fdir (str): Directory for the compound data files.
        file_name (str, optional): Name of the input file. If None, a default name will be generated.
        inputfiles_folder (str, optional): Directory where the input file will be saved. Default is the current directory.
        overwrite (bool, optional): If True, overwrite existing files. Default is False.

    Returns:
        str: The full path of the generated input file.
        str: The name of the generated input file.
        str: The folder where the input file is saved.
    """
    
    # Ensure the output directory exists
    os.makedirs(inputfiles_folder, exist_ok=True)
    logger.debug(f"Ensured the directory '{inputfiles_folder}' exists.")
    
    if file_name is None:
        file_name = f"ternaryVLE_NRTL_{carrier}_{solute}_{solvent}.inp"
    file_fullpath = os.path.join(inputfiles_folder, file_name)

    # Define content
    content = f"""ctd={ctd_file} CDIR="{cdir}" LDIR="{ldir}" odir="{odir}" # Global command line
FDIR="{fdir}" AUTOC UNIT=SI                                # Global command line
!  Ternary VLE computation with NRTL model for {carrier}, {solute}, and {solvent} at {tC} °C                                          # Comment line
"""
    content += f"f = {carrier}\n"
    content += f"f = {solute}\n"
    content += f"f = {solvent}\n"
    content += f"ternary={{1 2 3}} tc={tC} LLE NRTL RENORM nomix"  # Ternary VLE computation with NRTL model
    
    if os.path.exists(file_fullpath):
        logger.warning(f"File already exists: {file_fullpath}")
        if not overwrite:
            logger.info(f"Skipping inputfile generation for: '{file_name}'")
            logger.info("The file was not overwritten. Returning the existing file path and name.")
            return {
                "fullpath": os.path.abspath(file_fullpath),
                "filename": file_name,
                "folder": os.path.abspath(inputfiles_folder)
            }
            
    with open(file_fullpath, "w") as file:
        logger.debug(f"Generating inputfile: '{file_fullpath}'")
        file.write(content)
        logger.info(f"Input file '{file_fullpath}' has been created successfully. Returning the file path and name.")
    
    return {
        "fullpath": os.path.abspath(file_fullpath),
        "filename": file_name,
        "folder": os.path.abspath(inputfiles_folder)
    }

