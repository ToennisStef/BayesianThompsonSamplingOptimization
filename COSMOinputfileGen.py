import os

def generate_inp_file(file_name, ctd_file, cdir, ldir, odir, fdir, output_folder ="."):
    # Get user inputs for directories
    # ctd_file = "BP_TZVP_19.ctd"
    
    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, file_name)
       
    # Define content
    content = f"""ctd={ctd_file} CDIR="{cdir}" LDIR="{ldir}" odir="\\{odir}" # Global command line
FDIR="{fdir}" vpfile CTAB WCONF AUTOC                                 # Global command line
!! Multi-Component-2-Phase-Equilibrium calculation                        # Comment line
f = h2o           
f = 1-octanol
f = lacticacid                                                               # Compound input
tc=25 LIQ_EX x1={{0.75 0 0.25}} x2={{0 1 0}}
"""
    
    # Write content to file
    with open(output_file, "w") as file:
        file.write(content)
    
    print(f"Input file '{output_file}' has been created successfully.")

# Example usage 
# generate_inp_file("COSMO_input.inp")