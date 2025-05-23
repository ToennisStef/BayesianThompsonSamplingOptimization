import os
import pandas as pd
import glob
import logging

def sort_solvents_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adjusts the DataFrame by removing rows with NaN values in 'SMILES' and sorting it based on a custom sorting key.
    The sorting key is a tuple derived from the 'SMILES' string of each row. The tuple consists of:
        1. The count of 'C' atoms (including both uppercase 'C' and lowercase 'c', excluding 'Cl').
        2. The count of 'O' atoms in the SMILES string.
        3. The position of the first occurrence of 'O' in the SMILES string (or infinity if 'O' is not present).
        4. The count of 'C' atoms within parentheses, but only if the SMILES string contains '=O'.
    
    Parameters:
        df (pd.DataFrame): The DataFrame to be adjusted and sorted.
    Returns:
        pd.DataFrame: The adjusted and sorted DataFrame.
    """
    
    import math
    import re

    def parse_smiles(smiles):
        # Count 'C' atoms excluding 'Cl'
        c_count = len(re.findall(r'(?<!Cl)C', smiles)) + smiles.count('c')
        
        # Count 'O' atoms
        o_count = smiles.count('O')
        
        # Find the position of the first 'O'
        o_position = smiles.find('O') if 'O' in smiles else math.inf

        # Check if '=O' is present
        has_eq_o = '=O' in smiles

        # Count how many 'C' are within parentheses only if '=O' is present
        c_paren_count = 0
        if has_eq_o:
            paren_matches = re.findall(r'\([^()]*\)', smiles)
            for match in paren_matches:
                c_paren_count += len(re.findall(r'(?<!Cl)C', match)) + match.count('c')

        return (c_count, o_count, o_position, c_paren_count)

    df['smiles_sort_key'] = df['SMILES'].apply(parse_smiles)
    df = df.sort_values(by='smiles_sort_key').reset_index(drop=True).drop(columns='smiles_sort_key')
    return df

def load_and_sort_solvents(solvents_fullpath):
    solvents = pd.read_csv(solvents_fullpath, delim_whitespace=True)
    solvents = sort_solvents_df(solvents)
    return solvents


def ensure_full_path_to_cwd(input_path: str) -> str:
    """
    Ensures the input path is a full path by appending the current working directory (cwd)
    if any part of the cwd is not already in the input path.

    Parameters:
        input_path (str): The input path to be checked and adjusted.

    Returns:
        str: The full path.
    """
    cwd = os.getcwd()
    full_path = os.path.abspath(input_path)

    if not full_path.startswith(cwd):
        full_path = os.path.join(cwd, input_path)

    return os.path.normpath(full_path)

def list_files_with_extension(folder_path, file_extension):
        search_pattern = os.path.join(folder_path, f"*.{file_extension}")
        files = glob.glob(search_pattern)
        return files

def get_filelist_from_design_matrix(
    design_matrix: pd.DataFrame,
    output_dir: str
    )-> list:
    """
    Generates a list of file paths based on the provided DataFrame and output directory.
    The function constructs the file paths using the 'COSMO_name' column from the DataFrame.
    
    Parameters:
        design_matrix (pd.DataFrame): DataFrame containing the design matrix consisting of the following columns:
            - temperature: Temperature in degrees Celsius.
            - x1_lacticacid: Mole fraction of lactic acid.
            - solvent: ID of the solvent.
            - COSMO_name: Name of the solvent from the solvents DataFrame.
        output_dir (str): Directory where the output files are located.
    Returns:
        list: List of file paths.
    """
    
    filelist = []
    for index, row in design_matrix.iterrows():
        solvent = row['COSMO_name']
        x = row['x1_lacticacid']
        tc = row['temperature']
        file_path = os.path.join(output_dir, f"LIQEX_{solvent}_tc{tc}_x{x}.tab")
        filelist.append(file_path)
    
    return filelist
    
    
def run_initialSample_calculations(
    design_matrix: pd.DataFrame, 
    overwrite:bool=False
    )-> None:
    """
    Run COSMO-RS calculations for the initial sampling design matrix.
    Parameters:
        design_matrix (pd.DataFrame): DataFrame containing the design matrix, consting of the following columns:
            - temperature: Temperature in degrees Celsius.
            - x1_lacticacid: Mole fraction of lactic acid.
            - solvent: ID of the solvent.
            - COSMO_name: Name of the solvent from the solvents DataFrame.
        overwrite (bool): Set to "True" to overwrite existing files for both the input and output files. "False" skips the inputfile and COSMOther calculation if the file already exists.
    Notes:
        The directories for the input and output files are defined in the Config and COSMOthermConfig scripts.
    
    """
    from COSMOtherm_functions.Inputfile_Generation import gen_LIQEX_inp_file
    from COSMOtherm_functions.Run_COSMOtherm_Calculations import run_COSMOtherm_calculations
    from ..Configfiles import Config, COSMOthermConfig
    
    # Each row in the design matrix is a COSMOtherm calculation:
    for index, row in design_matrix.iterrows():
        
        temperature = row['temperature']
        x1_lacticacid = row['x1_lacticacid']
        solvent = row['COSMO_name']
        
        # Generate the input file for COSMOtherm
        inputfile_fullpath, inputfile_filename = gen_LIQEX_inp_file(
            tC=temperature,
            x1_lacticacid=x1_lacticacid,
            solvent=solvent,
            ctd_file=COSMOthermConfig.ctd_file,
            cdir=COSMOthermConfig.cdir,
            ldir=COSMOthermConfig.ldir,
            odir=COSMOthermConfig.odir,
            fdir=COSMOthermConfig.fdir,
            inputfiles_folder=Config.inputfile_dir,
            overwrite=overwrite
        )
        outputfile_fullpath = COSMOthermConfig.odir + "\\" + inputfile_filename[:-3] + "tab"
        
        if os.path.exists(outputfile_fullpath):
            logging.info("File already exists:", outputfile_fullpath)
            # print("File already exists:", outputfile_fullpath)
            if not overwrite:
                logging.info("Skipping COSMOther calculation for: %s", inputfile_filename)
                # print("Skipping COSMOther calculation for:", inputfile_filename)
                continue
    
        logging.info("Running COSMO-RS calculation for: %s", inputfile_filename)
        # print("Running COSMO-RS calculation for:", inputfile_filename)
        results = run_COSMOtherm_calculations(
                COSMOtherm_exe_fullpath=Config.COSMOtherm_exe_fullpath,
                files= [inputfile_fullpath]
            )
        logging.info("Return code: %s", results[0].returncode)
        # print("Return code:", results[0].returncode)
        


# def run_initialSample_calculations(
#     design_matrix: pd.DataFrame, 
#     overwrite:bool=False
#     )-> None:
#     """
#     Run COSMO-RS calculations for the initial sampling design matrix.
#     Parameters:
#         design_matrix (pd.DataFrame): DataFrame containing the design matrix, consting of the following columns:
#             - temperature: Temperature in degrees Celsius.
#             - x1_lacticacid: Mole fraction of lactic acid.
#             - solvent: ID of the solvent.
#             - COSMO_name: Name of the solvent from the solvents DataFrame.
#         overwrite (bool): Set to "True" to overwrite existing files. "False" skips the calculation if the file already exists.
#     """
#     for index, row in design_matrix.iterrows():
        
#         temperature = row['temperature']
#         x1_lacticacid = row['x1_lacticacid']
#         solvent = row['COSMO_name']
        
#         inputfile_fullpath, inputfile_filename = gen_LIQEX_inp_file(
#             tC=temperature,
#             x1_lacticacid=x1_lacticacid,
#             solvent=solvent,
#             ctd_file=COSMOthermConfig.ctd_file,
#             cdir=COSMOthermConfig.cdir,
#             ldir=COSMOthermConfig.ldir,
#             odir=COSMOthermConfig.odir,
#             fdir=COSMOthermConfig.fdir,
#             inputfiles_folder=Config.inputfile_dir
#         )
#         outputfile_fullpath = COSMOthermConfig.odir + "\\" + inputfile_filename[:-3] + "tab"
#         subprocess_str = '"' + Config.COSMOtherm_exe_fullpath + '"' + ' ' + '"' + inputfile_fullpath + '"'

#         if os.path.exists(outputfile_fullpath):
#             print("File already exists:", outputfile_fullpath)
#             if not overwrite:
#                 print("Skipping COSMOther calculation for:", inputfile_filename)
#                 continue
    
    
#         print("Running COSMO-RS calculation for:", inputfile_filename)
#         results = run_COSMOtherm_calculations(
#                 COSMOtherm_exe_fullpath=Config.COSMOtherm_exe_fullpath,
#                 files= [inputfile_fullpath]
#             )
#         print("Return code:", results[0].returncode)
        

# subprocess_str = '"' + Config.COSMOtherm_exe_fullpath + '"' + ' ' + '"' + inputfile_fullpath + '"'

# def sort_solvents_df(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Adjusts the DataFrame by removing rows with NaN values in 'SMILES' and sorting it based on a custom sortingkey.
#     The sorting key is a tuple derived from the 'SMILES' string of each row. The tuple consists of:
#         1. The count of 'C' atoms in the SMILES string.
#         2. The count of 'O' atoms in the SMILES string.
#         3. The position of the first occurrence of 'O' in the SMILES string (or infinity if 'O' is not present).
#         4. The count of 'C' atoms within parentheses, but only if the SMILES string contains '=O'.
    
#     Parameters:
#         df (pd.DataFrame): The DataFrame to be adjusted and sorted.
#     Returns:
#         pd.DataFrame: The adjusted and sorted DataFrame.
#     """
    
#     import math
#     import re

#     def parse_smiles(smiles):
#         c_count = smiles.count('C')
#         o_count = smiles.count('O')
#         o_position = smiles.find('O') if 'O' in smiles else math.inf

#         # Check if '=O' is present
#         has_eq_o = '=O' in smiles

#         # Count how many 'C' are within parentheses only if '=O' is present
#         c_paren_count = 0
#         if has_eq_o:
#             paren_matches = re.findall(r'\([^()]*\)', smiles)
#             for match in paren_matches:
#                 c_paren_count += match.count('C')

#         return (c_count, o_count, o_position, c_paren_count)

#     df['smiles_sort_key'] = df['SMILES'].apply(parse_smiles)
#     df = df.sort_values(by='smiles_sort_key').reset_index(drop=True).drop(columns='smiles_sort_key')
#     return df
