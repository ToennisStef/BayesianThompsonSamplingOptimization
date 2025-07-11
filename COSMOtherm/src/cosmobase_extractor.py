import os
import glob
from pathlib import Path

import pandas as pd
import re
import concurrent.futures

# tempfile
import tempfile
import shutil
import subprocess

# Try to import openbabel via pybel (preferred for Python), fallback to openbabel.openbabel
try:
    from openbabel import openbabel as ob
except ImportError:
    import openbabel as ob
# Use pybel from openbabel for SMILES conversion
try:
    from openbabel import pybel
except ImportError:
    pybel = None

# V:\groups\COSMOTHERM-Datenbank 2021\BP-TZVP-COSMO\h
# V:\groups\COSMOTHERM-Datenbank 2021\BP-TZVP-COSMO\i
# V:\groups\COSMOTHERM-Datenbank 2021\BP-TZVP-COSMO\j
# V:\groups\COSMOTHERM-Datenbank 2021\BP-TZVP-COSMO\k
# V:\groups\COSMOTHERM-Datenbank 2021\BP-TZVP-COSMO\l


# !V:\groups\COSMOTHERM-Datenbank 2021\BP-TZVP-COSMO\l\lacticacid_c0.cosmo

# $cosmo_info !only sometimes there 
#  Molecule name = D-2-HYDROXYPROPANOICACID
#  COSMO calculational method = BP-TZVP
#  CAS number = 010326-41-7
#  Molecular Weight = 90.08
#  Sum Formula = C3H6O3
# $info
# current prog.: ridft;A matrix vers.:1.0;cav. vers.: 1.0;;b-p;def-TZVP;COSMOlogic
#  + COSMOmic,Burscheider Str. 515,D-51381 Leverkusen
# $cosmo
#   epsilon=infinity
#   nppa= 1082
#   nspa=   92
#   disex= 10.0000
#   rsolv= 1.30
#   routf= 0.85
#   cavity closed
#   amat file=amat.cosmo
#   phsran=  0.0
#   ampran= 0.10E-04
# $cosmo_data
#   fepsi=     1.0000000
#   disex2= 3880.50
#   nsph=   32
#   nps=  477
#   npsd=  713
#   npspher=  236
#   area=  425.43
#   volume=  714.08
# $coord_rad
# #atom   x
#    1   1.48375382725252   0.10964684584791   0.15379616715080  c      2.00000
#    2   3.02028813324316   1.80289345029590   0.46402266951481  o      1.72000
#    3  -1.34806170512646   0.36281982327698   0.75826906231309  c      2.00000
#    4   2.15874078375668  -2.15483599744424  -0.78782536331937  o      1.72000
#    5  -2.57089241216019  -2.03755347639930   0.29051135472676  o      1.72000
#    6  -2.56589827124584   2.40850048234966  -0.87145740558952  c      2.00000
#    7  -1.49827633166268   0.86511752216008   2.77002921137320  h      1.30000
#    8   0.56760627332943  -3.16230907083967  -0.82401614800273  h      1.30000
#    9  -2.89653610815381  -2.85213723625939   1.90924881618494  h      1.30000
#   10  -2.47535856949483   1.89047908368315  -2.88093680479175  h      1.30000
#   11  -1.58432486915099   4.21647267223970  -0.60491963596872  h      1.30000
#   12  -4.55301297880438   2.64047427170648  -0.32486563432395  h      1.30000
# $coord_car
# !BIOSYM archive 3
# PBC=OFF
# coordinates from COSMO calculation
# !DATE
# C1       0.785168768    0.058022616    0.081385433 COSM 1      c       C   0.000
# O1       1.598267766    0.954050196    0.245550240 COSM 1      o       O   0.000
# C2      -0.713363585    0.191995996    0.401258736 COSM 1      c       C   0.000
# O2       1.142356509   -1.140290185   -0.416899258 COSM 1      o       O   0.000
# O3      -1.360457774   -1.078226943    0.153731999 COSM 1      o       O   0.000
# C3      -1.357814988    1.274523659   -0.461155433 COSM 1      c       C   0.000
# H1      -0.792853747    0.457800510    1.465836438 COSM 1      h       H   0.000
# H2       0.300364326   -1.673422015   -0.436050598 COSM 1      h       H   0.000
# H3      -1.532781009   -1.509286136    1.010331036 COSM 1      h       H   0.000
# H4      -1.309903438    1.000398521   -1.524526213 COSM 1      h       H   0.000
# H5      -0.838388676    2.231261409   -0.320109709 COSM 1      h       H   0.000
# H6      -2.409350883    1.397278911   -0.171911503 COSM 1      h       H   0.000
# end
# end
# $screening_charge
#   cosmo      =  -0.021430
#   correction =   0.020968
#   total      =  -0.000462
# $cosmo_energy


def get_smiles_from_car(car_data: str) -> dict:
    """
    Converts molecule data in CAR format to a SMILES string using Open Babel.

    Args:
        car_data: A string containing the molecule data in the .car file format.

    Returns:
        A dictionary containing the resulting SMILES string.
    """
    temp_car_file_path = None
    temp_smi_file_path = None

    try:
        # 1. Create a named temporary file for the input (.car).
        # We use 'delete=False' so we can get its name and use it after the 'with' block closes it.
        # The 'with' block ensures the file is closed and its contents are flushed to disk.
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix=".car", dir=".") as temp_car_file:
            temp_car_file_path = temp_car_file.name
            temp_car_file.write(car_data)
        
        print(f"Wrote CAR data to temporary file: {temp_car_file_path}")

        # 2. Create a named temporary file for the output (.smi).
        # We just need its name, and we ensure it's closed so obabel can write to it.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".smi", dir=".") as temp_smi_file:
            temp_smi_file_path = temp_smi_file.name

        print(f"Output SMILES file will be: {temp_smi_file_path}")

        # 3. Run the Open Babel command.
        # Now that the input file is written and closed, obabel can safely read it.
        # The output file path is also ready for obabel to write the results.
        command = [
            "obabel",
            "-i", "car", temp_car_file_path,
            "-o", "smi",
            "-O", temp_smi_file_path
        ]
        print(f"Running command: {' '.join(command)}")
        subprocess.run(command, check=True, capture_output=True, text=True)

        # 4. Read the SMILES string from the output file.
        with open(temp_smi_file_path, 'r', encoding='utf-8') as smi_file:
            smiles_output = smi_file.read().strip().split('\t')[0]  # Get the first part before any tab character
        
        print(f"Successfully generated SMILES: {smiles_output}")

    except FileNotFoundError:
        print("Error: 'obabel' command not found.")
        print("Please ensure Open Babel is installed and in your system's PATH.")
        smiles_output = "Error: Open Babel not found."  
    except subprocess.CalledProcessError as e:
        print(f"Open Babel encountered an error (Exit Code: {e.returncode}).")
        print(f"Stderr: {e.stderr}")
        smiles_output = f"Error: {e.stderr.strip()}"
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        smiles_output = f"Error: {str(e)}"
    finally:
        # 5. Clean up the temporary files.
        # This block executes whether an error occurred or not.
        if temp_car_file_path and os.path.exists(temp_car_file_path):
            os.remove(temp_car_file_path)
            print(f"Cleaned up temporary file: {temp_car_file_path}")
        if temp_smi_file_path and os.path.exists(temp_smi_file_path):
            os.remove(temp_smi_file_path)
            print(f"Cleaned up temporary file: {temp_smi_file_path}")
            
    return smiles_output

def extract_cosmobase_info(cosmo_db_folder: Path):
    """
    Recursively extract info from all .cosmo files in all subfolders of cosmo_db_folder.
    Returns a DataFrame with columns:
    ['Molecule name', 'COSMO calculational method', 'CAS number', 'Molecular Weight', 'Sum Formula', 'file_name', 'SMILES']
    """
    columns = ['COSMO_name', 'Molecule name', 'COSMO calculational method', 'CAS number', 'Molecular Weight', 'Sum Formula', 'file_name', 'SMILES']
    
    def extract_from_file(cosmo_file):
        info = {col: "" for col in columns}
        try:
            with open(cosmo_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                # Extract info fields
                for line in lines:
                    if 'Molecule name =' in line:
                        info['Molecule name'] = line.split('=', 1)[1].strip()
                    elif 'COSMO calculational method =' in line:
                        info['COSMO calculational method'] = line.split('=', 1)[1].strip()
                    elif 'CAS number =' in line:
                        info['CAS number'] = line.split('=', 1)[1].strip()
                    elif 'Molecular Weight =' in line:
                        info['Molecular Weight'] = line.split('=', 1)[1].strip()
                    elif 'Sum Formula =' in line:
                        info['Sum Formula'] = line.split('=', 1)[1].strip()
                # Extract $coord_car block
                car_block = []
                in_car = False
                for line in lines:
                    if line.strip().startswith('$coord_car'):
                        in_car = True
                        car_block = [line]
                        continue
                    if in_car:
                        car_block.append(line)
                        if line.strip() == 'end':
                            break
                car_data = ''.join(car_block)
                info['SMILES'] = get_smiles_from_car(car_data)
                # Clean up the temporary file


            info['COSMO_name'] = cosmo_file.name.split('_c0.cosmo')[0]  # Extract the name before '_c0.cosmo'
            print(f"Extracted info for {cosmo_file.name}, SMILES: {info['SMILES']}")
            return info
        
        except Exception as e:
            print(f"Error reading {cosmo_file}: {e}")
            return None

    cosmo_files = list(cosmo_db_folder.glob("**/*_c0.cosmo"))
    data = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=200) as executor:
        results = list(executor.map(extract_from_file, cosmo_files))
        data = [r for r in results if r is not None]
    df = pd.DataFrame(data, columns=columns)  # type: ignore
    return df


if __name__ == "__main__":
    db_folder = r"V:\groups\COSMOTHERM-Datenbank 2021\BP-TZVP-COSMO"
    db_folder = Path(db_folder)
    df = extract_cosmobase_info(db_folder)
    print(df)
    df.to_csv("cosmobase_info_final.csv", index=False)

    # df = pd.read_csv("cosmobase_info.csv")

    # fix this ,,,,,,{'SMILES': 'O=C([C@@]12C[C@H]3C[C@@H](C1)C[C@H](C3)C2)[C@@]12C[C@H]3C[C@@H](C1)C[C@H](C3)C2'}
    # df['SMILES'] = df['SMILES'].apply(lambda x: str(x).split("'SMILES': '")[-1].split("'}")[0]) 
    # print(df.head())
    # df.to_csv("cosmobase_info_cleaned.csv", index=False)

    # Get CAS numbers and SMILES for the first 10 molecules
    # molecule_names = df['name'].tolist()[:-10]

    # resolv_df = pura_get_cas_number_and_smiles(molecule_names)
    # print(resolv_df)