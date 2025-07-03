import os
import glob
from pathlib import Path
import pandas as pd
import re


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


def extract_cosmobase_info(cosmo_db_folder: Path):
    """
    Recursively extract info from all .cosmo files in all subfolders of cosmo_db_folder.
    Returns a DataFrame with columns:
    ['Molecule name', 'COSMO calculational method', 'CAS number', 'Molecular Weight', 'Sum Formula', 'file_name']
    """
    columns = ['Molecule name', 'COSMO calculational method', 'CAS number', 'Molecular Weight', 'Sum Formula', 'file_name']
    data = []

    # Recursively find all .cosmo files in all subfolders
    for cosmo_file in cosmo_db_folder.glob("**/*.cosmo"):
        info = {col: "" for col in columns}
        info['file_name'] = str(cosmo_file)
        try:
            with open(cosmo_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
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
            data.append(info)
        except Exception as e:
            print(f"Error reading {cosmo_file}: {e}")
            continue
    df = pd.DataFrame(data, columns=columns)
    return df


if __name__ == "__main__":
    db_folder = r"V:\groups\COSMOTHERM-Datenbank 2021\BP-TZVP-COSMO"
    db_folder = Path(db_folder)
    df = extract_cosmobase_info(db_folder)
    print(df)
    df.to_csv("cosmobase_info.csv", index=False)