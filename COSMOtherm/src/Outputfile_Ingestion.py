import os
import glob
import pandas as pd
import re
import numpy as np

def list_files_with_extension(folder_path, file_extension):
        search_pattern = os.path.join(folder_path, f"*.{file_extension}")
        files = glob.glob(search_pattern)
        return files

def get_LIQEX_output_df(
    files: list,
    return_description: bool = False,
    ) -> tuple:
    """
    Extracts and processes data from a list of files to create a pandas DataFrame 
    containing temperature, phase input/output values, and compound information.
    Args:
        files (list): A list of file paths to process. Each file is expected to 
                      have a specific format with temperature, phase data, and 
                      compound information.
    Returns:
        tuple: A tuple containing:
            - result_df (pd.DataFrame): A DataFrame with the following columns:
                - 'Temperature (°C)': Temperature in Celsius extracted from the file.
                - 'Phase 1 x(1) input': Input value for x(1) in Phase 1.
                - 'Phase 1 x(2) input': Input value for x(2) in Phase 1.
                - 'Phase 1 x(3) input': Input value for x(3) in Phase 1.
                - 'Phase 2 x(1) input': Input value for x(1) in Phase 2.
                - 'Phase 2 x(2) input': Input value for x(2) in Phase 2.
                - 'Phase 2 x(3) input': Input value for x(3) in Phase 2.
                - 'Compound 1': Name of the first compound.
                - 'Compound 2': Name of the second compound.
                - 'Compound 3': Name of the third compound.
                - 'Phase 1 x(1) output': Output value for x(1) in Phase 1.
                - 'Phase 1 x(2) output': Output value for x(2) in Phase 1.
                - 'Phase 1 x(3) output': Output value for x(3) in Phase 1.
                - 'Phase 2 x(1) output': Output value for x(1) in Phase 2.
                - 'Phase 2 x(2) output': Output value for x(2) in Phase 2.
                - 'Phase 2 x(3) output': Output value for x(3) in Phase 2.
                - 'Warning msg': A warning message from COSMOtherm if applicable.
            - description (str): A textual description of the DataFrame columns.
    Notes:
        - The function assumes a specific file format and structure, including 
          temperature information on the third line and phase data in subsequent lines.
        - The DataFrame does not natively support metadata for descriptions. The 
          description is returned as a separate string.
    """
    
    data_list = []
    for file in files:
        # Load the data from the file
        with open(file, 'r') as f:
            for i, line in enumerate(f):
                if i == 2:  # Line 3 (0-based index)
                    settings_line = line.strip()
                    break

        # Extract Temperature (T) and x(3) value
        temperature_match = re.search(r'T= (\d+\.\d+) K', settings_line)
        
        temperature = float(temperature_match.group(1)) if temperature_match else None
        
        # Split the settings line by ";" to extract the relevant sections
        sections = settings_line.split(";")
        
        # Extract Phase 1 and Phase 2 sections
        phase_1_section = sections[1].strip() if len(sections) > 0 else ""
        phase_2_section = sections[2].strip() if len(sections) > 1 else ""

        # print(f"Phase 1 section: {phase_1_section}")
        # print(f"Phase 2 section: {phase_2_section}")
        
        # Initialize default values for Phase 1
        P1_x1_in, P1_x2_in, P1_x3_in = 0, 0, 0
        # Extract values for Phase 1
        for match in re.finditer(r'x\((\d+)\)= ([\d\.E\-]+)', phase_1_section):
            index, value = int(match.group(1)), float(match.group(2))
            if index == 1:
                P1_x1_in = value
            elif index == 2:
                P1_x2_in = value
            elif index == 3:
                P1_x3_in = value

        # Initialize default values for Phase 2
        P2_x1_in, P2_x2_in, P2_x3_in = 0, 0, 0
        # Extract values for Phase 2
        for match in re.finditer(r'x\((\d+)\)= ([\d\.E\-]+)', phase_2_section):
            index, value = int(match.group(1)), float(match.group(2))
            if index == 1:
                P2_x1_in = value
            elif index == 2:
                P2_x2_in = value
            elif index == 3:
                P2_x3_in = value

        
        # Determine if the descriptor is 4 or 5 lines long
        with open(file, 'r') as f:
            lines = [next(f) for _ in range(6)]
        # Check if the 5th line (index 4) contains column headers (e.g., 'Compound')
        header_line = lines[4].strip()
        if re.match(r'Compound', header_line):
            skiprows = 4
        else:
            skiprows = 5
            
        warning_msg = ""
        if skiprows == 5:
            warning_line = lines[4].strip()
            warning_match = re.search(r'WARNING:\s*(.*)', warning_line)
            if warning_match:
                warning_msg = warning_match.group(1)
        else:
            warning_msg = ""

            
        # Read the file data
        C = []
        P1_x_out = []
        P2_x_out = []
        file_data = pd.read_csv(file, sep=r'\s+', skiprows=skiprows)
        
        # file_data = pd.read_csv(file, sep=r'\s+', skiprows=4)
        for idx, row in file_data.iterrows():
            C.append(row['Compound'])
            P1_x_out.append(row['phase_1_x'])
            P2_x_out.append(row['phase_2_x'])
            
        tC = temperature - 273.15 if temperature else None  # Convert to Celsius

        # Append the extracted data to the list
        data_list.append({
            "Temperature (°C)": tC,
            "Phase 1 x(1) input": P1_x1_in,
            "Phase 1 x(2) input": P1_x2_in,
            "Phase 1 x(3) input": P1_x3_in,
            "Phase 2 x(1) input": P2_x1_in,
            "Phase 2 x(2) input": P2_x2_in,
            "Phase 2 x(3) input": P2_x3_in,
            "Compound 1": C[0],
            "Compound 2": C[1],
            "Compound 3": C[2],
            "Phase 1 x(1) output": P1_x_out[0],
            "Phase 1 x(2) output": P1_x_out[1],
            "Phase 1 x(3) output": P1_x_out[2],
            "Phase 2 x(1) output": P2_x_out[0],
            "Phase 2 x(2) output": P2_x_out[1],
            "Phase 2 x(3) output": P2_x_out[2],
            "Warning msg": warning_msg
        })

    # Create a DataFrame from the collected data
    result_df = pd.DataFrame(data_list)

    # Add a description
    description = (
        "This DataFrame contains the following columns:\n"
        "- 'Temperature (°C)': Temperature in Celsius extracted from the file.\n"
        "- 'Phase 1 x(1) input': Input value for x(1) in Phase 1.\n"
        "- 'Phase 1 x(2) input': Input value for x(2) in Phase 1.\n"
        "- 'Phase 1 x(3) input': Input value for x(3) in Phase 1.\n"
        "- 'Phase 2 x(1) input': Input value for x(1) in Phase 2.\n"
        "- 'Phase 2 x(2) input': Input value for x(2) in Phase 2.\n"
        "- 'Phase 2 x(3) input': Input value for x(3) in Phase 2.\n"
        "- 'Compound 1': Name of the first compound.\n"
        "- 'Compound 2': Name of the second compound.\n"
        "- 'Compound 3': Name of the third compound.\n"
        "- 'Phase 1 x(1) output': Output value for x(1) in Phase 1.\n"
        "- 'Phase 1 x(2) output': Output value for x(2) in Phase 1.\n"
        "- 'Phase 1 x(3) output': Output value for x(3) in Phase 1.\n"
        "- 'Phase 2 x(1) output': Output value for x(1) in Phase 2.\n"
        "- 'Phase 2 x(2) output': Output value for x(2) in Phase 2.\n"
        "- 'Phase 2 x(3) output': Output value for x(3) in Phase 2."
    )

    if return_description:
        return result_df, description
    else:
        return result_df
    
    
def parse_ternaryVLE_tab_blocks(filepath: str): 
    """
    Splits the .tab file at each 'General   job 1 :' and parses the three blocks:
    1. Real prediction (first block)
    2. NRTL parameters (second block)
    3. NRTL predictions (third block)
    Returns:
        tuple: (df_real, df_nrtl_params, df_nrtl_pred)
    """
    import re
    from io import StringIO
    with open(filepath, 'r') as f:
        text = f.read()
    # Split at each 'General   job 1 :' (keep the split points)
    blocks = re.split(r'(General   job 1 :)', text)
    # Reconstruct blocks to include the split marker
    blocks = [blocks[i] + blocks[i+1] for i in range(1, len(blocks), 2)]
    # There should be three blocks: real, NRTL params, NRTL pred
    if len(blocks) < 3:
        raise ValueError('Expected at least three blocks in the file.')
    # Helper to find the first table after a block
    def extract_table(block, header_regex):
        lines = block.splitlines()
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
    # 1. Real prediction block
    df_real = extract_table(blocks[0], r'^\s*x1\s+x2\s+x3')
    # 2. NRTL parameters block
    # Look for the NRTL parameter table (header: 'model  parameter')
    def extract_nrtl_param_table(block):
        lines = block.splitlines()
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
    df_nrtl_params = extract_nrtl_param_table(blocks[1])
    # 3. NRTL predictions block
    df_nrtl_pred = extract_table(blocks[2], r'^\s*x1\s+x2\s+x3')
    return df_real, df_nrtl_params, df_nrtl_pred

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
    df.to_csv(r"U:\\Github\\BayesianThompsonSamplingOptimization\\COSMOtherm\\outputfiles\\PVAP_antoine_density_ingested.csv", index=False)
