import os
import glob
import pandas as pd
import re

def list_files_with_extension(folder_path, file_extension):
        search_pattern = os.path.join(folder_path, f"*.{file_extension}")
        files = glob.glob(search_pattern)
        return files

def get_output_df(
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

        

        # Read the file data
        C = []
        P1_x_out = []
        P2_x_out = []
        file_data = pd.read_csv(file, sep=r'\s+', skiprows=4)
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