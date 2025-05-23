from .ChemConstants import M_lacticacid, c_p_h2o, c_p_lacticacid


def calc_la_molefrac(massconcentration_lacticacid: list): 
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
    c1_lacticacid_range = [mass / M_lacticacid for mass in massconcentration_lacticacid]
    c1_h2o_range = [
        c_p_h2o - (c_p_h2o / c_p_lacticacid) * c1_lacticacid
        for c1_lacticacid in c1_lacticacid_range
    ]
    x1_lacticacid_range = [
        c1_lacticacid / (c1_lacticacid + c1_h2o)
        for c1_lacticacid, c1_h2o in zip(c1_lacticacid_range, c1_h2o_range)
    ]
    return x1_lacticacid_range
