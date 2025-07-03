import os
import glob
import pandas as pd
import re

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

if __name__ == "__main__":

    df_real, df_nrtl_params, df_nrtl_pred = parse_ternaryVLE_tab_blocks(
        r"C:\Users\kabe02-lokal\Documents\Github\BayesianThompsonSamplingOptimization\COSMOtherm\outputfiles\ternaryVLE_h2o_lacticacid_n-undecane.tab"
    )
    print(df_real.head())
    print(df_nrtl_params)
    print(df_nrtl_pred.head())