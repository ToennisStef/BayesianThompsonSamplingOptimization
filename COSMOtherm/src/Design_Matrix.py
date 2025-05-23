import pyDOE3
import pandas as pd

def build_design_matrix(
    tC_range:list, 
    x1_lacticacid_range:list, 
    solvents:pd.DataFrame,
    reduction:int = 4
    )-> pd.DataFrame:
    """
    Generate a design matrix based on the provided ranges and solvents.
    The design matrix is generated using the pyDOE3 library.
    The function returns a DataFrame with the generated design matrix.
    Parameters:
        tC_range (list): List of temperature values.
        x1_lacticacid_range (list): List of lactic acid mole fraction values.
        solvents (pd.DataFrame): DataFrame containing solvent information.
        reduction (int): Reduction factor for the design matrix. Default is 4. (0 for full factorial)
    Returns:
        pd.DataFrame: DataFrame containing the generated design matrix consisting of:
            - temperature: Temperature in degrees Celsius.
            - x1_lacticacid: Mole fraction of lactic acid.
            - solvent: ID of the solvent.
            - COSMO_name: Name of the solvent from the solvents DataFrame.
    """
    solvents_ids = solvents.index.tolist()

    if reduction == 0 or reduction == 1:
        # Full factorial design
        design = pyDOE3.fullfact([len(tC_range), len(x1_lacticacid_range), len(solvents_ids)])
    else:
        # Reduced full factorial design
        design = pyDOE3.gsd(levels=[len(tC_range), len(x1_lacticacid_range), len(solvents_ids)], reduction=reduction)
    
    design_matrix = pd.DataFrame(design, columns=['temperature', 'x1_lacticacid', 'solvent'])
    design_matrix['temperature'] = design_matrix['temperature'].map({
        i: tC_range[i] for i in range(len(tC_range))
    })
    design_matrix['x1_lacticacid'] = design_matrix['x1_lacticacid'].map({
        i: x1_lacticacid_range[i] for i in range(len(x1_lacticacid_range))
    })
    design_matrix['solvent'] = design_matrix['solvent'].map({
        i: solvents_ids[i] for i in range(len(solvents_ids))
    })
    design_matrix = design_matrix.join(solvents['COSMO_name'], on='solvent')
    return design_matrix