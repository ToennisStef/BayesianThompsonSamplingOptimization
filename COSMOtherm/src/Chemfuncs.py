from .ChemConstants import M_h2o, M_lacticacid, c_p_h2o, c_p_lacticacid


def calc_la_molefrac(rho_lacticacid: list): 
    """
    Convert mass concentration of lactic acid to molar fraction, assuming a 
    binary solution of lactic acid and water. The conversion is based on a 
    simple linear interpolation between the pure concentrations of water and 
    lactic acid at 25°C, which serves as a crude approximation.

    Parameters:
        massc1_lacticacid_range (list, numpy array, or torch tensor): 
            Mass concentration of lactic acid.

    Returns:
        list: Molar fractions of lactic acid.
    """
    c1_lacticacid_range = [mass / M_lacticacid for mass in rho_lacticacid]
    c1_h2o_range = [
        c_p_h2o - (c_p_h2o / c_p_lacticacid) * c1_lacticacid
        for c1_lacticacid in c1_lacticacid_range
    ]
    x1_lacticacid_range = [
        c1_lacticacid / (c1_lacticacid + c1_h2o)
        for c1_lacticacid, c1_h2o in zip(c1_lacticacid_range, c1_h2o_range)
    ]
    return x1_lacticacid_range


def calc_rho_h2o(
    rho_lacticacid: list
    ) -> list:
    """
    Calculate the mass concentration of water in a binary solution of lactic acid and water.

    Parameters:
        x1_lacticacid (float): Molar fraction of lactic acid.
        massconcentration_lacticacid (float): Mass concentration of lactic acid.

    Returns:
        float: Mass concentration of water.
    """
    rho_h2o = [ M_h2o*(c_p_h2o - (c_p_h2o / c_p_lacticacid) *(cm_la / M_lacticacid)) for cm_la in rho_lacticacid]
    return rho_h2o