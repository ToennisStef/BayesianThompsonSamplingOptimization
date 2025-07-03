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
    
def get_ternaryVLE_NRTL_output_df(
    files: list,
    return_description: bool = False,
    ) -> tuple:
    """
    Extracts and processes data from a list of files to create a pandas DataFrame 
    containing temperature, phase input/output values, and compound information.
    """
    # EXAMPLE file

    #  Property  job 1 : Ternary mixture ;
    #  Compounds job 1 : h2o (1) ; lacticacid (2) ; n-undecane (3) ;
    #  Settings  job 1 : T= 303.15 K ;
    #  Units     job 1 : Energies in kJ/mol ; Pressure in kPa ; Area in nm^2 ; Temperature in K ; Molecular weights in g/mol ; Concentrations x : mole fraction ;
    #  General   job 1 : Molecular weights 18.0153 (1) 90.0783 (2) 156.3093 (3) ; Surface areas 0.4307 (1) 1.1968 (2) 2.5175 (3) ;
    
    #           x1          x2          x3             H^E             G^E            ptot    mu1+RTln(x1)    mu2+RTln(x2)    mu3+RTln(x3)      ln(gamma1)      ln(gamma2)      ln(gamma3)          y1          y2          y3
    #  0.000998004 0.000998004 0.998003992      0.05835114      0.03461745      2.09785273     13.52537457     17.91562133    -13.69832344      5.98606583      7.25499471      0.00052061 0.942820411 0.002009701 0.055169888
    #  0.000999001 0.049950050 0.949050949      0.82359720      0.75673467      0.32515368      7.57112326     22.26380382    -13.71085347      3.62276463      5.06708065      0.04584421 0.573033842 0.072781724 0.354184433
    #  0.000999001 0.099900100 0.899100899      1.27993355      1.32800778      0.24222153      6.12817811     21.89488942    -13.67815365      3.05028741      4.22756956      0.11288483 0.433942808 0.084398208 0.481658985
    #  0.000999001 0.149850150 0.849150849      1.62270566      1.80438032      0.20861703      5.16485730     21.47978481    -13.61655615      2.66809740      3.65741494      0.19448157 0.343805226 0.083113672 0.573081102

    # ....

    #  0.899100899 0.099900100 0.000999001     -0.47183633      0.20729401    112.80528689     15.67990466     12.11779211      3.54206504      0.03746555      0.34858256     13.74726687 0.041219849 0.000003746 0.958776405
    #  0.949050949 0.000999001 0.049950050      0.77267448      1.63657975     73.95562239     16.20349186     -1.25381759      2.38098757      0.19112747     -0.35132897      9.37459545 0.077389184 0.000000028 0.922610787
    #  0.949050949 0.049950050 0.000999001     -0.26280666      0.15084429    676.74518204     15.75640931     11.06836290      8.14611477      0.01375096      0.62537690     15.57388765 0.007082599 0.000000412 0.992916989
    #  0.998003992 0.000998004 0.000998004      0.01081313      0.04905197   9233.91911396     15.84909772      2.34013544     14.74968164      0.00022957      1.07554233     18.19479973 0.000538521 0.000000000 0.999461479

    #  Property  job 1 : NRTL model parameters for the activity coefficients of the TERNARY mixture of h2o (1) + lacticacid (2) + n-undecane (3) ;
    #  Settings  job 1 : T= 303.15 K ;
    #  Units     job 1 : rms in ln(gamma) ;
    #  General   job 1 :WARNING: NRTL fit did not converge - use parameters with caution ;
    
    #    model  parameter                 value
    #     NRTL        rms               0.14541
    #     NRTL    Alpha21           11079.44078
    #     NRTL    Alpha31              -0.91374
    #     NRTL    Alpha32              -0.49250
    #     NRTL      Tau12               3.6E-05
    #     NRTL      Tau13               1.13739
    #     NRTL      Tau21 -1.04628763936280E-04
    #     NRTL      Tau23               1.70744
    #     NRTL      Tau31               2.20114
    #     NRTL      Tau32               1.99529

    #  Property  job 1 : Ternary mixture - NRTL fit ;
    #  Compounds job 1 : h2o (1) ; lacticacid (2) ; n-undecane (3) ;
    #  Settings  job 1 : T= 303.15 K ;
    #  Units     job 1 : Energies in kJ/mol ; Pressure in kPa ; Area in nm^2 ; Temperature in K ; Molecular weights in g/mol ; Concentrations x : mole fraction ;
    #  General   job 1 : Molecular weights 18.0153 (1) 90.0783 (2) 156.3093 (3) ; Surface areas 0.4307 (1) 1.1968 (2) 2.5175 (3) ;
    
    #           x1          x2          x3            ptot      ln(gamma1)      ln(gamma2)      ln(gamma3)          y1          y2          y3
    #  0.000998004 0.000998004 0.998003992      1.19376348      5.37818968      5.91430830      0.00003902 0.902170090 0.000924135 0.096905775
    #  0.000999001 0.049950050 0.949050949      0.58927110      4.51223245      5.04844843      0.02297780 0.769564047 0.039418854 0.191017099
    

def ternaryVLE_tab_to_df(filepath: str) -> pd.DataFrame:
    """
    Converts a COSMOtherm ternary VLE .tab output file to a pandas DataFrame.
    Args:
        filepath (str): Path to the .tab file.
    Returns:
        pd.DataFrame: DataFrame containing the parsed data with appropriate column names.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Find the header line (first non-empty line starting with whitespace and then column names)
    header_idx = None
    for i, line in enumerate(lines):
        if re.match(r'^\s*x1\s+x2\s+x3', line):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("Could not find data header in the file.")
    
    # Extract column names
    header_line = lines[header_idx]
    columns = re.split(r'\s{2,}', header_line.strip())
    
    # Find where the data ends (either next empty line or end of file)
    data_start = header_idx + 1
    data_end = data_start
    for i in range(data_start, len(lines)):
        if lines[i].strip() == '' or not re.match(r'^\s*\d', lines[i]):
            break
        data_end = i + 1
    
    # Read the data into a DataFrame
    from io import StringIO
    data_str = ''.join(lines[data_start:data_end])
    df = pd.read_csv(StringIO(data_str), delim_whitespace=True, names=columns)
    return df
    
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

    files = list_files_with_extension(r"C:\Users\kabe02-lokal\Documents\Github\BayesianThompsonSamplingOptimization\COSMOtherm\outputfiles", "tab")
    df = ternaryVLE_tab_to_df(files[0])
    print(df)

    df_real, df_nrtl_params, df_nrtl_pred = parse_ternaryVLE_tab_blocks(
        r"C:\Users\kabe02-lokal\Documents\Github\BayesianThompsonSamplingOptimization\COSMOtherm\outputfiles\ternaryVLE_h2o_lacticacid_n-undecane.tab"
    )
    print(df_real.head())
    print(df_nrtl_params)
    print(df_nrtl_pred.head())