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


def create_design_matrix_for_solvent_screening(tC_levels, rho_lacticacid_levels, x_h2o_levels, x_lacticacid_levels, m_h2o_levels, m_lacticacid_levels, V_p2, solvents):
        """
        Create a design matrix DataFrame for the solvent screening experimental setup.

        NOTE: This function is hard coded for the specific experimental setup (solvent screening).
        It assumes a 2-phase (liquid-liquid), 3-component (water, lactic acid, solvent) system,
        and expects the input arrays/lists to match the experimental design.
        But it should return a generalized DataFrame setting for liquid-liquid extraction calculations.

        Parameters
        ----------
        design_matrix : np.ndarray
            Factorial design matrix (output of pyDOE3.fullfact).
        tC_levels : list
            List of temperature levels [°C].
        x_h2o_levels : list
            List of mole fractions of water in phase 1.
        x_lacticacid_levels : list
            List of mole fractions of lactic acid in phase 1.
        m_h2o_levels : list
            List of masses of water in phase 1 [g].
        m_lacticacid_levels : list
            List of masses of lactic acid in phase 1 [g].
        V_p2 : float
            Volume of phase 2 [L].
        solvents : pd.DataFrame
            DataFrame containing solvent information.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the full design matrix with all calculated columns for liquid-liquid extraction calculations in COSMOtherm.
        """
        design_matrix = pyDOE3.fullfact([len(tC_levels), len(rho_lacticacid_levels), len(solvents)])
    
        density_map = dict(zip(solvents['COSMO_name'], solvents['Density']))
        cas_map = dict(zip(solvents['COSMO_name'], solvents['CAS_Number']))

        df = pd.DataFrame(design_matrix, columns=['tC', 'rho1_2', 'solvent'])
        df['tC'] = df['tC'].apply(lambda x: tC_levels[int(x)])
        df['component1'] = 'h2o'
        df['component3'] = df['solvent'].apply(lambda x: solvents['COSMO_name'][int(x)])
        df['component2'] = 'lacticacid'
        
        df['cas_c1'] = '7732-18-5'
        df['cas_c3'] = df['solvent'].apply(lambda x: cas_map[solvents.iloc[int(x)]['COSMO_name']])
        df['cas_c2'] = '50-21-5'
        
        df['x1_1'] = df['rho1_2'].apply(lambda x: x_h2o_levels[int(x)])
        df['x1_3'] = 0
        df['x1_2'] = df['rho1_2'].apply(lambda x: x_lacticacid_levels[int(x)])
        
        df['x2_1'] = 0
        df['x2_3'] = 1
        df['x2_2'] = 0
        
        df['W1_1'] = df['rho1_2'].apply(lambda x: m_h2o_levels[int(x)])
        df['W1_3'] = 0
        df['W1_2'] = df['rho1_2'].apply(lambda x: m_lacticacid_levels[int(x)])
        
        df['W2_1'] = 0
        df['W2_3'] = df['component3'].map(density_map) * V_p2 * 1000
        df['W2_2'] = 0
        
        df['c1_1'] = df['W1_1'] / (df['W1_1'] + df['W1_2'] + df['W1_3'])
        df['c1_3'] = df['W1_3'] / (df['W1_1'] + df['W1_2'] + df['W1_3'])
        df['c1_2'] = df['W1_2'] / (df['W1_1'] + df['W1_2'] + df['W1_3'])
        
        df['c2_1'] = 0
        df['c2_3'] = 1
        df['c2_2'] = 0
        
        df['N1_1'] = df['W1_1'] / 18.01528
        df['N1_3'] = 0
        df['N1_2'] = df['W1_2'] / 90.078
        df['N2_1'] = 0
        df['N2_3'] = 1
        df['N2_2'] = 0 #df['m2_3'] / solvents['Molar_mass'].map(lambda x: x * 1000)  # Convert g/mol to g
        return df