import os
import pandas as pd

def gen_LIQEX_inp_file(
    temperature:float,
    x1_lacticacid:float, 
    solvent:str, 
    ctd_file:str, 
    cdir:str, 
    ldir:str, 
    odir:str, 
    fdir:str,
    file_name:str = None,  
    output_folder:str =".")->None:
    """
    Generates an input file for LIQEX calculations based on the provided parameters.
    Parameters:
        temperature (float): Temperature in degrees Celsius.
        x1_lacticacid (float): Mole fraction of lactic acid.
        solvent (str): Name of the solvent.
        ctd_file (str): Path to the CT-Data file.
        cdir (str): Directory for the CT-data/parameterization file.  
        ldir (str): Directory for the license file.
        odir (str): Directory for the output files.
        fdir (str): Directory for the compound data files.
        file_name (str, optional): Name of the output file. If None, a default name will be generated.
        output_folder (str, optional): Directory where the output file will be saved. Default is the current directory.
    Returns:
        None
    """
    
    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)
    
    if file_name is None:
        file_name = f"LIQEX_{solvent}_tc{temperature}_x{x1_lacticacid}.inp"
    output_file = os.path.join(output_folder, file_name)
    
    # Ensure the output file does not already exist
    # if os.path.exists(output_file):
    #     raise FileExistsError(f"The file '{output_file}' already exists. Please choose a different name or delete the existing file.")
    
    # Calculate the mole fraction of water
    x1_h2o = 1-x1_lacticacid
    
    # Define content
    content = f"""ctd={ctd_file} CDIR="{cdir}" LDIR="{ldir}" odir="\\{odir}" # Global command line
FDIR="{fdir}" vpfile CTAB WCONF AUTOC                                 # Global command line
!! Multi-Component-2-Phase-Equilibrium calculation                        # Comment line
f = h2o                      # Compound input (water)
f = {solvent}                # Compound input (solvent)
f = lacticacid               # Compound input (lactic acid)
tc={temperature} LIQ_EX x1={{{x1_h2o} 0 {x1_lacticacid}}} x2={{0 1 0}}
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
    
    # Write content to file
    with open(output_file, "w") as file:
        file.write(content)
    
    print(f"Input file '{output_file}' has been created successfully.")


def sort_solvents_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adjusts the DataFrame by removing rows with NaN values in 'SMILES' and sorting it based on a custom sortingkey.
    The sorting key is a tuple derived from the 'SMILES' string of each row. The tuple consists of:
        1. The count of 'C' atoms in the SMILES string.
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
        c_count = smiles.count('C')
        o_count = smiles.count('O')
        o_position = smiles.find('O') if 'O' in smiles else math.inf

        # Check if '=O' is present
        has_eq_o = '=O' in smiles

        # Count how many 'C' are within parentheses only if '=O' is present
        c_paren_count = 0
        if has_eq_o:
            paren_matches = re.findall(r'\([^()]*\)', smiles)
            for match in paren_matches:
                c_paren_count += match.count('C')

        return (c_count, o_count, o_position, c_paren_count)

    df['smiles_sort_key'] = df['SMILES'].apply(parse_smiles)
    df = df.sort_values(by='smiles_sort_key').reset_index(drop=True)
    return df
