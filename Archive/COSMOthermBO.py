from  COSMOtherm.Configfiles import Config
from COSMOtherm.funcs import gen_LIQEX_inp_file, sort_solvents_df
import subprocess
import pandas as pd


solvents = pd.read_csv(Config.solvents_filepath)


temperature_range = [20, 40] # [°C] Temperature range in Celsius; From Nina [20°C, 30°C, 40°C]
massc1_lacticacid_range = [5, 20] # [g/L] Concentration of lactic acid in Fermentation Broth: From Nina [5g/L, 10g/L, 20g/L] concentration incrments in literature [50g/L, 100g/L, 150g/L, 200g/L, 250g/L]

M_h2o = 18.01528 # [g/mol] Molar mass of water
M_lacticacid = 90.078 # [g/mol] Molar mass of lactic acid

rho_lacticacid = 1.209 # [g/mL] Density of lactic acid
rho_h2o = 1.0 # [g/mL] Density of water

c_p_h2o = 55.5 # [mol/L] Concentration of pure water at 25°C: 
c_p_lacticacid = rho_lacticacid / (M_lacticacid*10**3) # [mol/L] Concentration of pure lactic acid at 25°C:

c1_lacticacid_range = [massc1_lacticacid_range/M_lacticacid] # [mol/L] Concentration of lactic acid in Fermentation Broth

# x1_lacticacid = 
# Konvert mass concentration to molar fraction 


solvents = pd.read_csv(Config.solvents_filepath)
solvents = sort_solvents_df(solvents)
# Access the unique row identifier (index) of each row in the DataFrame

n_solvents = len(solvents)
solvents_ids = solvents.index.tolist()



for index, row in solvents.iterrows():
    print(f"Row Index: {index}, Row Data: {row.to_dict()}")


# Find the row index of 'n-decane' in the DataFrame
n_decane_row = solvents[solvents['solvent_name'] == 'n-decane']
if not n_decane_row.empty:
    n_decane_index = n_decane_row.index[0]
    print(f"'n-decane' is in row index: {n_decane_index}")
else:
    print("'n-decane' not found in the DataFrame.")

gen_LIQEX_inp_file(
    temperature=20, 
    x1_lacticacid=0.25,
    solvent="hexane",
    ctd_file=Config.ctd_file,
    cdir=Config.cdir,
    ldir=Config.ldir,
    odir=Config.odir,
    fdir=Config.fdir,
    output_folder=Config.inputfile_path
)
    
