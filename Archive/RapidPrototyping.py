from  COSMOtherm.Configfiles import Config
from COSMOtherm.funcs import gen_LIQEX_inp_file, sort_solvents_df
# from doepy import build
import pyDOE3
import subprocess
import pandas as pd
import os




temperature_range = [20, 40] # [°C] Temperature range in Celsius; From Nina [20°C, 30°C, 40°C]
massc1_lacticacid_range = [5, 20] # [g/L] Concentration of lactic acid in Fermentation Broth: From Nina [5g/L, 10g/L, 20g/L] concentration incrments in literature [50g/L, 100g/L, 150g/L, 200g/L, 250g/L]

### Conversion of mass concentration to molar fraction
M_h2o = 18.01528 # [g/mol] Molar mass of water
M_lacticacid = 90.078 # [g/mol] Molar mass of lactic acid

rho_lacticacid = 1.209 # [g/mL] Density of lactic acid
rho_h2o = 1.0 # [g/mL] Density of water

c_p_h2o = 55.5 # [mol/L] Concentration of pure water at 25°C: 
c_p_lacticacid = rho_lacticacid / M_lacticacid*(10**3) # [mol/L] Concentration of pure lactic acid at 25°C:

c1_lacticacid_range = [massc1_lacticacid_range[0]/M_lacticacid, massc1_lacticacid_range[1]/M_lacticacid] # [mol/L] Concentration of lactic acid in Fermentation Broth
c1_h2o_range = [c_p_h2o - (c_p_h2o/c_p_lacticacid)*c1_lacticacid_range[0], c_p_h2o - (c_p_h2o/c_p_lacticacid)*c1_lacticacid_range[1]] # [mol/L] Concentration of water in Fermentation Broth

x1_lacticacid_range = [c1_lacticacid_range[0]/(c1_lacticacid_range[0]+c1_h2o_range[0]), c1_lacticacid_range[1]/(c1_lacticacid_range[1]+c1_h2o_range[1])] # [mol/mol] Mole fraction of lactic acid in Fermentation Broth
# x1_h2o_range = [c1_h2o_range[0]/(c1_lacticacid_range[0]+c1_h2o_range[0]), c1_h2o_range[1]/(c1_lacticacid_range[1]+c1_h2o_range[1])] # [mol/mol] Mole fraction of water in Fermentation Broth


# Load COSMO-solvents dataframe
solvents = pd.read_csv(Config.solvents_filepath)
solvents = sort_solvents_df(solvents)
# Access the unique row identifier (index) of each row in the DataFrame

solvents_ids = solvents.index.tolist()

# Design = {
#     'temperature': temperature_range,
#     'x1_lacticacid': x1_lacticacid_range,
#     'solvent': solvents_ids
# }

# fullfact = build.full_fact(Design)
levels = [len(temperature_range), len(x1_lacticacid_range), len(solvents_ids)]
reduced_design = pyDOE3.gsd(levels=levels,reduction=9)
reduced_fullfact = pd.DataFrame(reduced_design, columns=['temperature', 'x1_lacticacid', 'solvent'])
reduced_fullfact['temperature'] = reduced_fullfact['temperature'].map({i: temperature_range[i] for i in range(len(temperature_range))})
reduced_fullfact['x1_lacticacid'] = reduced_fullfact['x1_lacticacid'].map({i: x1_lacticacid_range[i] for i in range(len(x1_lacticacid_range))})
reduced_fullfact['solvent'] = reduced_fullfact['solvent'].map({i: solvents_ids[i] for i in range(len(solvents_ids))})

reduced_fullfact = reduced_fullfact.join(solvents['COSMO_name'], on='solvent')

# Create the initial samples according to the design matrix (reduced_fullfact)
for index, row in reduced_fullfact.iterrows():
    
    print(index)
    if index >= 1:
        break
    
    # Extract the values for each parameter
    temperature = row['temperature']
    x1_lacticacid = row['x1_lacticacid']
    solvent = row['COSMO_name']
    

    # print(solvent)
    # # Create the input file for COSMO-RS
    inputfile, filename = gen_LIQEX_inp_file(
    temperature=temperature, 
    x1_lacticacid=x1_lacticacid,
    solvent=solvent,
    ctd_file=Config.ctd_file,
    cdir=Config.cdir,
    ldir=Config.ldir,
    odir=Config.odir,
    fdir=Config.fdir,
    output_folder=Config.inputfile_path
    )
    
    output_file = Config.odir+"\\"+filename[:-3]+"tab" 
    subprocess_str = Config.cosmotherm_filepath + '"' + inputfile + '"'
    
    # print(output_file)
    if os.path.exists(output_file):
        overwrite = input(f"The file '{output_file}' already exists. Do you want to overwrite it? [y/n]: ").strip().lower()
        if overwrite != 'y':
            print("Operation cancelled. The file was not overwritten. returning the existing file path and name.")
        else:
            print("Running COSMO-RS calculation and overwriting the existing file.")
            result = subprocess.run(subprocess_str, shell=True)
            print("Return code:", result.returncode)
            print("Output:", result.stdout)
            print("Error:", result.stderr)
    else:
        print("Running COSMO-RS calculation.")
        result = subprocess.run(subprocess_str, shell=True)
        print("Return code:", result.returncode)
        print("Output:", result.stdout)
        print("Error:", result.stderr)