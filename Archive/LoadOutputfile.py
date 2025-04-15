import pandas as pd

# Use a relative path with forward slashes
tab_file_path = "COSMOtherm/outputfiles/ExtractionT25.tab"

df = pd.read_csv(tab_file_path, sep="\t", header=0)
print(df.head())