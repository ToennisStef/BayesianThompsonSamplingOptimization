import subprocess
from COSMOinputfileGen import generate_inp_file

T_init = 25 # Initial Test Temperature - 25°

# Directory setup for COSMO-therm
ctd_file = "BP_TZVP_19.ctd"
ctd_file = "BP_TZVPD_FINE_19.ctd"
cdir="C:/Program Files/COSMOlogic/COSMOthermX19/COSMOtherm/CTDATA-FILES"
ldir="C:/Program Files/COSMOlogic/COSMOthermX19/licensefiles"
odir="\\avt.rwth-aachen.de\home$\student\stto01\.AVT-UserConfig\Desktop\COSMOthemOutput"
# fdir="C:/Program Files/COSMOlogic/COSMOthermX19/COSMOtherm/DATABASE-COSMO/BP-TZVP-COSMO"
fdir="V:\groups\COSMOTHERM-Datenbank 2022\COSMObase2022\BP-TZVP-COSMO"
fdir="V:\groups\COSMOTHERM-Datenbank\COSMObase-1901\BP-TZVPD-FINE"

inputfile_path = f"Q:\Groups\eicr students\Stefan Tönnis\GIT_Repos\BayesianThompsonSamplingOptimization\COSMOtherm\inputfiles"
inputfile_name = f"ExtractionT{T_init}.inp"

generate_inp_file(inputfile_name,
                  ctd_file=ctd_file,
                  cdir=cdir,
                  ldir=ldir,
                  odir=odir,
                  fdir=fdir, 
                  output_folder=inputfile_path)

inputfile_fullpath = f'"{inputfile_path}\\{inputfile_name}"'
cosmotherm_fullpath = r""""C:\Program Files\COSMOlogic\COSMOthermX19\COSMOtherm\BIN-WINDOWS\cosmotherm.exe" """
#inputfile_fullpath = r""""Q:\Groups\eicr students\Stefan Tönnis\GIT_Repos\BayesianThompsonSamplingOptimization\COSMOtherm\inputfiles\VLE-OCTANE-ACETICACID-IEI.inp" """

subprocess_str = cosmotherm_fullpath+inputfile_fullpath

print(subprocess_str)

subprocess.run(subprocess_str, shell=True)

