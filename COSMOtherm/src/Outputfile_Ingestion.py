import os
import glob
import pandas as pd
import re
import numpy as np

def list_files_with_extension(folder_path, file_extension):
        search_pattern = os.path.join(folder_path, f"*.{file_extension}")
        files = glob.glob(search_pattern)
        return files

def parse_ternaryVLE_tab_blocks(filepath: str): 
    """
    Parses a .tab file for ternary VLE and LLE data, including NRTL fits.
    Returns:
        tuple: (df_real, df_lle, df_lle_renom, df_nrtl_params, df_nrtl_pred, df_lle_nrtl, df_lle_nrtl_renom)
    """
    import re
    from io import StringIO
    import pandas as pd
    with open(filepath, 'r') as f:
        text = f.read()
    # Split at each 'Property  job 1 :' to get all sections
    sections = re.split(r'(Property  job 1 :)', text)
    # Reconstruct sections to include the split marker
    sections = [sections[i] + sections[i+1] for i in range(1, len(sections), 2)]

    # Helper to find the first table after a section
    def extract_table(section, header_regex):
        lines = section.splitlines()
        header_idx = None
        for i, line in enumerate(lines):
            if re.match(header_regex, line):
                header_idx = i
                break
        if header_idx is None:
            return None
        header_line = lines[header_idx]
        columns = re.split(r'\s{2,}', header_line.strip())
        # Find where the data ends
        data_start = header_idx + 1
        data_end = data_start
        for i in range(data_start, len(lines)):
            if lines[i].strip() == '' or not re.match(r'^\s*\d', lines[i]):
                break
            data_end = i + 1
        data_str = '\n'.join(lines[data_start:data_end])
        if not data_str.strip():
            return None
        df = pd.read_csv(StringIO(data_str), delim_whitespace=True, names=columns)
        return df

    # Helper to extract NRTL parameter table
    def extract_nrtl_param_table(section):
        lines = section.splitlines()
        header_idx = None
        for i, line in enumerate(lines):
            if re.match(r'^\s*model\s+parameter', line):
                header_idx = i
                break
        if header_idx is None:
            return None
        header_line = lines[header_idx]
        columns = re.split(r'\s{2,}', header_line.strip())
        data_start = header_idx + 1
        data_end = data_start
        for i in range(data_start, len(lines)):
            if lines[i].strip() == '' or not re.match(r'^\s*NRTL', lines[i]):
                break
            data_end = i + 1
        data_str = '\n'.join(lines[data_start:data_end])
        if not data_str.strip():
            return None
        df = pd.read_csv(StringIO(data_str), delim_whitespace=True, names=columns)
        return df

    # Initialize DataFrames
    df_real = None
    df_lle = None
    df_lle_renom = None
    df_nrtl_params = None
    df_nrtl_pred = None
    df_lle_nrtl = None
    df_lle_nrtl_renom = None

    # Process each section
    lle_results_count = 0
    lle_renorm_count = 0
    for i, section in enumerate(sections):
        # print(f"Processing section {i+1}: {section[:100]}...")
        if 'Ternary mixture ;' in section and 'General   job 1 :' in section:
            df_real = extract_table(section, r'^\s*x1\s+x2\s+x3')
        elif 'LLE results for ternary system ;' in section:
            lle_results_count += 1
            if lle_results_count == 1:
                df_lle = extract_table(section, r'^\s*x`\(1\)\s+x`\(2\)\s+x`\(3\)')
            elif lle_results_count == 2:
                df_lle_nrtl = extract_table(section, r'^\s*x`\(1\)\s+x`\(2\)\s+x`\(3\)')
        elif 'LLE (renormalized) for ternary system ;' in section:
            lle_renorm_count += 1
            if lle_renorm_count == 1:
                df_lle_renom = extract_table(section, r'^\s*x`\(1\)\s+x`\(2\)\s+x`\(3\)')
            elif lle_renorm_count == 2:
                df_lle_nrtl_renom = extract_table(section, r'^\s*x`\(1\)\s+x`\(2\)\s+x`\(3\)')
        elif 'NRTL model parameters' in section:
            df_nrtl_params = extract_nrtl_param_table(section)
        elif 'Ternary mixture - NRTL fit ;' in section:
            df_nrtl_pred = extract_table(section, r'^\s*x1\s+x2\s+x3')
    return df_real, df_lle, df_lle_renom, df_nrtl_params, df_nrtl_pred, df_lle_nrtl, df_lle_nrtl_renom

def parse_PVAPantoine_density_tab_blocks(filepath: str):
    """
    Parses a .tab file containing:
    1. Vapor pressure table (T, PVtot, mu(Liquid), E_Gas-E_COSMO, H(Vapori))
    2. Antoine equation coefficients (A, B, C)
    3. Extended Antoine equation coefficients (A, B, C, D, E, F, G)
    4. Density/volume table (Nr, Compound, Density, Volume, ...)
    5. Compound name from the top of the file (from 'Compounds job 1 : ...')
    Returns:
        tuple: (df_pvap, df_antoine, df_antoine_ext, df_density, compound_name)
    """
    import re
    from io import StringIO
    import pandas as pd
    with open(filepath, 'r') as f:
        text = f.read()
    # 1. Vapor pressure block
    pvap_match = re.search(r"Property  job 1 : Vapor pressures ;.*?\n\s*(T.+?H\\(Vapori\\))\n((?:[ \t]*\d.+\n)+)", text, re.DOTALL)
    df_pvap = None
    if pvap_match:
        header = pvap_match.group(1)
        data = pvap_match.group(2)
        lines = [l for l in data.splitlines() if len(l.split()) == 5]
        data_clean = '\n'.join(lines)
        df_pvap = pd.read_csv(StringIO(data_clean), sep=r'\s+', names=header.split())
    # 1a. Compound name from the top (do not use density table Compound field)
    compound_name = None
    compound_name_match = re.search(r"Compounds job 1 :\s*(.+?)\s*\(\d+\)\s*;", text)
    if compound_name_match:
        compound_name = compound_name_match.group(1).strip()
    # 2. Antoine equation block
    antoine_match = re.search(r"Vapor pressure \(ln\(p\) calculated\) fitted to Antoine equation ;.*?\n\s*A.+?C\n(.+?)\n\n", text, re.DOTALL)
    df_antoine = None
    if antoine_match:
        data = antoine_match.group(1)
        lines = [l for l in data.splitlines() if len(l.split()) == 3]
        data_clean = '\n'.join(lines)
        df_antoine = pd.read_csv(StringIO(data_clean), sep=r'\s+', names=["A", "B", "C"])
    # 3. Extended Antoine equation block
    antoine_ext_match = re.search(r"Vapor pressure \(ln\(p\) calculated\) fitted to Extended Antoine equation ;.*?\n\s*A.+?G\n(.+?)\n\n", text, re.DOTALL)
    df_antoine_ext = None
    if antoine_ext_match:
        data = antoine_ext_match.group(1)
        lines = [l for l in data.splitlines() if len(l.split()) == 7]
        data_clean = '\n'.join(lines)
        df_antoine_ext = pd.read_csv(StringIO(data_clean), sep=r'\s+', names=["A", "B", "C", "D", "E", "F", "G"])
    # 4. Density/volume block
    density_match = re.search(r"Property  job 2 : Liquid density and volume ;.*?\n\s*Nr Compound.+?NRing\n((?:.+\n)+)", text, re.DOTALL)
    df_density = None
    if density_match:
        data = density_match.group(1)
        lines = [l for l in data.splitlines() if len(l.split()) >= 2]  # at least Nr and Compound
        data_clean = '\n'.join(lines)
        df_density = pd.read_csv(StringIO(data_clean), sep=r'\s+', names=["Nr", "Compound", "Density", "Volume", "Exp_Density", "Exp_Volume", "MolWeight", "COSMO_Volume", "Smom(2)", "Smom(2)^2", "NRing"])
    return df_pvap, df_antoine, df_antoine_ext, df_density, compound_name

def ingest_pvap_antoine_density_folder_parallel(folderpath: str):
    import os
    import concurrent.futures
    import pandas as pd
    from tqdm import tqdm

    def process_file(filepath):
        try:
            df_pvap, df_antoine, df_antoine_ext, df_density, compound_name = parse_PVAPantoine_density_tab_blocks(filepath)
            # Extract values or set to None if missing
            A = B = C = Ae = Be = Ce = De = Ee = Fe = Ge = Density = Volume = None
            if df_antoine is not None and not df_antoine.empty:
                A, B, C = df_antoine.iloc[0][["A", "B", "C"]]
            if df_antoine_ext is not None and not df_antoine_ext.empty:
                Ae, Be, Ce, De, Ee, Fe, Ge = df_antoine_ext.iloc[0][["A", "B", "C", "D", "E", "F", "G"]]
            if df_density is not None and not df_density.empty:
                Density = df_density.iloc[0]["Density"]
                Volume = df_density.iloc[0]["Volume"]
            return {
                "file_name": os.path.basename(filepath),
                "compound_name": compound_name,
                "A": A, "B": B, "C": C,
                "Ae": Ae, "Be": Be, "Ce": Ce, "De": De, "Ee": Ee, "Fe": Fe, "Ge": Ge,
                "Density": Density, "Volume": Volume
            }
        except Exception as e:
            return {"file_name": os.path.basename(filepath), "compound_name": None, "A": None, "B": None, "C": None, "Ae": None, "Be": None, "Ce": None, "De": None, "Ee": None, "Fe": None, "Ge": None, "Density": None, "Volume": None, "error": str(e)}

    # List all .tab files
    filepaths = [os.path.join(folderpath, f) for f in os.listdir(folderpath) if f.endswith('.tab')]
    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for result in tqdm(executor.map(process_file, filepaths), total=len(filepaths), desc="Ingesting .tab files"):
            results.append(result)
    df = pd.DataFrame(results)
    return df

def Antoine_p_from_A_B_C(A, B, C, T):
    return np.exp(A - B / (T + C))

def Antoine_T_from_p_A_B_C(p, A, B, C):
    return B / (A - np.log(p)) - C

def parse_LIQEX_tab_block(filepath: str):
    """
    Parses a LIQEX .tab file with the expected format:
    - First row: header
    - Next three rows: carrier (h2o), solute (lactic acid), solvent (e.g., benzene)
    Returns a single-row DataFrame with columns named as:
        carrier_phase_1_N, carrier_phase_1_W, carrier_phase_1_x, ...
        solute_phase_1_N, ...
        solvent_phase_1_N, ...
        Kx_1_2
    """
    import pandas as pd
    import re
    from io import StringIO

    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Remove empty lines and strip
    lines = [l.strip() for l in lines if l.strip()]
    # Find header (first line with 'Nr' and 'Compound')
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith('Nr') and 'Compound' in line:
            header_idx = i
            break
    if header_idx is None:
        raise ValueError('Header not found in LIQEX tab file.')
    header = re.split(r'\s{2,}', lines[header_idx].strip())
    data_lines = lines[header_idx+1:header_idx+4]  # Always 3 compounds
    data = '\n'.join(data_lines)
    df = pd.read_csv(StringIO(data), sep=r'\s+', names=header)

    # # Map roles
    role_map = ['carrier', 'solute', 'solvent'] 

    # solvent_name = filepath regex
    filename = os.path.basename(filepath)
    match = re.search(r'LIQEX_(.*?)_tc', filename)
    if match:
        solvent_name = match.group(1)
    else:
        solvent_name = df["Nr Compound"].to_list()[-1]

    df["Nr Compound"] = role_map

    df.set_index('Nr Compound', inplace=True)

    # Flatten the DataFrame
    flattened = df.stack()
    flattened.index = [f"{idx}_{col}" for idx, col in flattened.index]
    flattened_df = pd.DataFrame([flattened])
    flattened_df["COSMO_name"] = [solvent_name]

    return flattened_df

def ingest_LIQEX_folder_parallel(folder_path: str):
    import os
    import concurrent.futures
    import pandas as pd
    from tqdm import tqdm

    def process_file(filepath):
        try:
            df = parse_LIQEX_tab_block(filepath)
            df["file_name"] = os.path.basename(filepath)
            return df
        except Exception as e:
            # Return a DataFrame with error info for this file
            return pd.DataFrame({"file_name": [os.path.basename(filepath)], "error": [str(e)]})

    # List all .tab files
    filepaths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.tab')]
    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for result in tqdm(executor.map(process_file, filepaths), total=len(filepaths), desc="Ingesting LIQEX .tab files"):
            results.append(result)
    # Concatenate all DataFrames
    df = pd.concat(results, ignore_index=True)
    return df

if __name__ == "__main__":
    # Test for PVAP/Antoine/density tab file
    pvap_path = r"U:\\Github\\BayesianThompsonSamplingOptimization\\COSMOtherm\\outputfiles\\PVAP\\PVAP_(-)-(2r,4s)-florol.tab"
    df_pvap, df_antoine, df_antoine_ext, df_density, compound_name = parse_PVAPantoine_density_tab_blocks(pvap_path)
    print("PVAP Table:\n", df_pvap)
    print("\nAntoine coefficients:\n", df_antoine)
    print("\nExtended Antoine coefficients:\n", df_antoine_ext)
    print("\nDensity Table:\n", df_density)
    print("\nCompound Name:\n", compound_name)

    pvap_folder = r"U:\\Github\\BayesianThompsonSamplingOptimization\\COSMOtherm\\outputfiles\\PVAP"
    # df = ingest_pvap_antoine_density_folder_parallel(pvap_folder)
    df = pd.read_csv(r"U:\\Github\\BayesianThompsonSamplingOptimization\\COSMOtherm\\outputfiles\\PVAP_antoine_density_ingested.csv")

    df["T_boil"] = Antoine_T_from_p_A_B_C(101.325, df["A"], df["B"], df["C"])

    print(df.head())
    # df.to_csv(r"U:\\Github\\B\\COSMOtherm\\outputfiles\\PVAP_antoine_density_ingested.csv", index=False)

    # Test for LIQEX tab file
    liqex_path = r"U:\\Github\\BayesianThompsonSamplingOptimization\\COSMOtherm\\outputfiles\\Screening\\MassesWg\\LIQEX_benzene_tc40.0_W1100.0.tab"

    df_liqex = parse_LIQEX_tab_block(liqex_path)
    print(df_liqex.columns)

    # Test for LIQEX folder parallel ingestion
    liqex_folder = r"U:\\Github\\BayesianThompsonSamplingOptimization\\COSMOtherm\\outputfiles\\Screening\\MassesWg"
    df_liqex_all = ingest_LIQEX_folder_parallel(liqex_folder)
    print("LIQEX parallel ingestion columns:", df_liqex_all.columns)
    print(df_liqex_all.head())
    df_liqex_all.to_csv(r"U:\\Github\\BayesianThompsonSamplingOptimization\\COSMOtherm\\outputfiles\\Screening\\LIQEX_all.csv")
