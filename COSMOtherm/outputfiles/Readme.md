# COSMOtherm Liquid Extraction Calculations - Output Files

This contains the output files generated from COSMOtherm Liquid Extraction calculations. The calculations were performed to evaluate the extraction efficiency of various solvents for a system consisting of:

- **Water (H₂O)**  
- **Solvent**  
- **Lactic acid**

## CompleteScreening

The Folder **Complete Screening** contains outputfiles from a first screening of 42 Solvents. The individual solvents are listed under Solvents.
The following parameters were varied during the calculations:

1. **Temperature Levels**:  

3 different temperature levels were tested.
The considered temperature levels are as follows:
- 20 [°C]
- 30 [°C]
- 40 [°C] 

2. **Initial Concentration Levels**:  

8 concentration levels of the initial concentration of the lactic acid & water solution were evaluated.
The considered initial lactic acid concentration levels are as follows 
- 5 [g/L], 0.0010032800176357408 [mol/mol]
- 10 [g/L], 0.0020129013383111492 [mol/mol]
- 20 [g/L], 0.004051349137288097 [mol/mol]
- 50 [g/L], 0.010325086834736085 [mol/mol]
- 100 [g/L], 0.021340983896365212 [mol/mol] 
- 150 [g/L], 0.03311941974040262 [mol/mol]
- 200 [g/L], 0.04574240928256 [mol/mol]
- 250 [g/L], 0.05930416628155502 [mol/mol]

The mass concentration was converted to molar fraction of the initial lactic acid & water solution for the input in COSMOtherm.
The corresponsind molar fractions are second values behind the mass concentration.

3. **Solvents**:  

A total of 42 solvents were analyzed.
The considered solvents are listed here:
- n-undecane,
- hexane,
- n-heptane,
- n-decane,
- dodecane,
- octane,
- n-nonane,
- 3-heptanol
- dodecanol,
- 1-decanol,
- 2-octanol,
- 3-hexanol,
- octanol,
- 5-nonanol,
- 1-hexanol,
- 3-decanol,
- 1-heptanol,
- 4-heptanol,
- 2-hexanol,
- 4-decanol,
- 1-undecanol,
- 2-heptanol,
- 2-nonanol,
- 3-octanol,
- 1-nonanol,
- 2-undecanol
- 3-octanone,
- 4-nonanone,
- 2-hexanone,
- 4-decanone,
- 4-heptanone
- 4-octanone,
- 2-undecanone
- 3-heptanone,
- 3-hexanone,
- 2-heptanone,
- 2-octanone,
- 2-nonanone,
- 5-nonanone,
- 2-decanone,
- 3-nonanone,
- 3-decanone,
- 4-methyl-2-pentanone

## File Structure

Each output file corresponds to a specific combination of solvent, temperature, and initial concentration. The naming convention of the files reflects these parameters for easy identification.

Two main files are created by the COSMOtherm Calculations:
- .out: contains log information of the COSMOtherm calculation
- .tab: contains the main result in a tabular form

## Purpose

The results in this folder are intended to support the analysis and optimization of liquid-liquid extraction processes using COSMOtherm. The data can be used to identify the most effective solvents for extracting lactic acid from water under various conditions.

## Notes

- Ensure that you have the necessary tools to interpret COSMOtherm output files.
- For further details on the calculations or methodology, refer to the main project documentation.


## Visualization
A first jupyter notebook for loading and visualizing the results is "viz.ipynb"