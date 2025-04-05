from  COSMOtherm.Configfiles import Setup
from COSMOtherm.funcs import gen_LIQEX_inp_file
import subprocess
import pandas as pd


solvents = pd.read_csv(Setup.solvents_filepath)


temperature_range = [20, 40] # [°C] Temperature range in Celsius; From Nina [20°C, 30°C, 40°C]
c1_lacticacid_ = [5, 20] # [g/L] Concentration of lactic acid in Fermentation Broth: From Nina [5g/L, 10g/L, 20g/L] concentration incrments in literature [50g/L, 100g/L, 150g/L, 200g/L, 250g/L]




gen_LIQEX_inp_file(
    temperature=20, 
    x1_lacticacid=0.25,
    solvent="hexane",
    ctd_file=Setup.ctd_file,
    cdir=Setup.cdir,
    ldir=Setup.ldir,
    odir=Setup.odir,
    fdir=Setup.fdir,
    output_folder=Setup.inputfile_path
)
    
