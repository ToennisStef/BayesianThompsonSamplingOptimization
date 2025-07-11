import os
import logging
from dataclasses import dataclass
from typing import Optional, List, Dict

# Use a module-specific logger
logger = logging.getLogger(__name__)

@dataclass
class FileGenConfig:
    ctd_file: str
    cdir: str
    ldir: str
    odir: str
    fdir: str
    inputfiles_folder: str = "."
    file_name: Optional[str] = None
    overwrite: bool = False


def ensure_dir_exists(path: str) -> None:
    """Ensure a directory exists."""
    os.makedirs(path, exist_ok=True)
    logger.debug(f"Ensured the directory '{path}' exists.")


def write_input_file(file_fullpath: str, content: str, overwrite: bool) -> bool:
    """Write content to a file, handling overwrite logic. Returns True if written, False if skipped."""
    if os.path.exists(file_fullpath):
        logger.warning(f"File already exists: {file_fullpath}")
        if not overwrite:
            logger.info(f"Skipping inputfile generation for: '{os.path.basename(file_fullpath)}'")
            logger.info("The file was not overwritten. Returning the existing file path and name.")
            return False
        else:
            logger.info(f"Overwriting inputfile: '{file_fullpath}'")
    with open(file_fullpath, "w") as file:
        logger.debug(f"Generating inputfile: '{file_fullpath}'")
        file.write(content)
        logger.info(f"Input file '{file_fullpath}' has been created successfully.")
    return True


def gen_2Phase_LIQEX_inp_file(
    tC: float,
    p1_input: List[float],
    p2_input: List[float],
    components: List[str],
    config: FileGenConfig,
    composition_type: str = "Mole fraction",
) -> Dict[str, str]:
    """
    Generates an input file for a general multi-component 2-phase LIQEX calculation in COSMOtherm.
    """
    c_type_map = {
        "Mole fraction": 'x',
        "Mass fraction": 'c',
        "Masses W[g]": 'W',
        "Mole numbers": 'N',
    }
    c_type = c_type_map.get(composition_type, 'x')

    # Normalize p1_input and p2_input if needed
    if composition_type in ["Mole fraction", "Mass fraction"]:
        for idx, p_input in enumerate([p1_input, p2_input]):
            s = sum(p_input)
            if s > 0 and not abs(s - 1.0) < 1e-8:
                if idx == 0:
                    p1_input = [x / s for x in p1_input]
                    logger.info("Normalized p1_input to sum to 1.")
                else:
                    p2_input = [x / s for x in p2_input]
                    logger.info("Normalized p2_input to sum to 1.")

    ensure_dir_exists(config.inputfiles_folder)

    file_name = config.file_name or f"LIQEX_{components[2]}_tc{tC}_{c_type}1{p1_input[1]}.inp"
    file_fullpath = os.path.join(config.inputfiles_folder, file_name)

    if len(p1_input) > len(components) or len(p2_input) > len(components):
        raise ValueError("Too many entries for P1_input or P2_input. Number of entries cannot exceed the number of specified Components!")
    p1_input = p1_input + [0] * (len(components) - len(p1_input))
    p2_input = p2_input + [0] * (len(components) - len(p2_input))

    content = f"""ctd={config.ctd_file} CDIR=\"{config.cdir}\" LDIR=\"{config.ldir}\" odir=\"{config.odir}\" # Global command line
FDIR=\"{config.fdir}\" vpfile CTAB WCONF AUTOC                                 # Global command line
!! Multi-Component-2-Phase-Equilibrium calculation                    # Comment line
"""
    for component in components:
        content += f"f = {component}\n"
    content += f"tc={tC} LIQ_EX {c_type}1={{{' '.join(map(str, p1_input))}}} {c_type}2={{{' '.join(map(str, p2_input))}}} maxiter=1000 pr_K "

    written = write_input_file(file_fullpath, content, config.overwrite)
    return {
        "fullpath": os.path.abspath(file_fullpath),
        "filename": file_name,
        "folder": os.path.abspath(config.inputfiles_folder)
    }


def gen_TBOIL_inp_file(
    solvent: str,
    config: FileGenConfig,
) -> Dict[str, str]:
    """
    Generates an input file for TBOIL calculations in COSMOtherm for a given solvent.
    """
    tboil_input_folder = os.path.join(config.inputfiles_folder, "TBOIL")
    tboil_output_folder = os.path.join(config.odir, "TBOIL")  
    ensure_dir_exists(tboil_input_folder)
    ensure_dir_exists(tboil_output_folder)
    file_name = config.file_name or f"TBOIL_{solvent}.inp"
    file_fullpath = os.path.join(tboil_input_folder, file_name)
    content = f"""ctd={config.ctd_file} CDIR=\"{config.cdir}\" LDIR=\"{config.ldir}\" odir=\"{tboil_output_folder}\" # Global command line
FDIR=\"{config.fdir}\" efile UNIT=SI AUTOC                                 # Global command line
!! TBOIL calculation for a single solvent                          # Comment line
f = {solvent}\nx={{1}} pvap=1013.2 tk=300.0 """
    write_input_file(file_fullpath, content, config.overwrite)
    return {
        "fullpath": os.path.abspath(file_fullpath),
        "filename": file_name,
        "folder": os.path.abspath(config.inputfiles_folder)
    }


def gen_binaryLLE_inp_file(
    solvent: str,
    config: FileGenConfig,
) -> Dict[str, str]:
    """
    Generates an input file for binary LLE calculations in COSMOtherm for a given solvent.
    """
    ensure_dir_exists(config.inputfiles_folder)
    file_name = config.file_name or f"binaryLLE_h2o_{solvent}.inp"
    file_fullpath = os.path.join(config.inputfiles_folder, file_name)
    content = f"""ctd={config.ctd_file} CDIR=\"{config.cdir}\" LDIR=\"{config.ldir}\" odir=\"{config.odir}\" # Global command line
FDIR=\"{config.fdir}\" efile UNIT=SI AUTOC nomix                                # Global command line
! Binary phase VLE/LLE diagram computation with phase separation                  # Comment line
f = h2o\nf = {solvent}\ntc=25 tc2=200 tstepsize=10 BINARY LLE NRTL                                                            # Automatic binary computation and LLE search"""
    write_input_file(file_fullpath, content, config.overwrite)
    return {
        "fullpath": os.path.abspath(file_fullpath),
        "filename": file_name,
        "folder": os.path.abspath(config.inputfiles_folder)
    }


def gen_binaryActivity_inp_file(
    solvent: str,
    config: FileGenConfig,
) -> Dict[str, str]:
    """
    Generates an input file for binary activity coefficient calculations in COSMOtherm for a given solvent.
    """
    ensure_dir_exists(config.inputfiles_folder)
    file_name = config.file_name or f"binaryactivity_h2o_{solvent}.inp"
    file_fullpath = os.path.join(config.inputfiles_folder, file_name)
    content = f"""ctd={config.ctd_file} CDIR=\"{config.cdir}\" LDIR=\"{config.ldir}\" odir=\"{config.odir}\" # Global command line
FDIR=\"{config.fdir}\" AUTOC                                 # Global command line
!! Activity Coefficient computation                                               # Comment line
f = h2o\nf = {solvent}\ntc=25 gamma xg={{0.24 0.76}}"""
    write_input_file(file_fullpath, content, config.overwrite)
    return {
        "fullpath": os.path.abspath(file_fullpath),
        "filename": file_name,
        "folder": os.path.abspath(config.inputfiles_folder)
    }


def gen_PVAP_inp_file(
    solvent: str,
    t_start: float,
    t_end: float,
    t_steps: int,
    config: FileGenConfig,
) -> Dict[str, str]:
    """
    Generates an input file for PVAP (Vapour pressure calculation) calculations in COSMOtherm for a given solvent.
    """
    pvap_input_folder = os.path.join(config.inputfiles_folder, "PVAP")
    pvap_output_folder = os.path.join(config.odir, "PVAP")
    ensure_dir_exists(pvap_input_folder)
    ensure_dir_exists(pvap_output_folder)
    file_name = config.file_name or f"PVAP_{solvent}.inp"
    file_fullpath = os.path.join(pvap_input_folder, file_name)
    content = f"""ctd={config.ctd_file} CDIR=\"{config.cdir}\" LDIR=\"{config.ldir}\" odir=\"{pvap_output_folder}\" # Global command line
FDIR=\"{config.fdir}\" AUTOC UNIT=SI                                # Global command line
!  Automatic computation of vapor pressure curve p_vap(T)                                          # Comment line
f = {solvent}\nx={{1}} pvap tc={t_start} tc2={t_end} tstep={t_steps}
tc={t_start} DENSITY"""
    write_input_file(file_fullpath, content, config.overwrite)
    return {
        "fullpath": os.path.abspath(file_fullpath),
        "filename": file_name,
        "folder": os.path.abspath(pvap_input_folder)
    }


def gen_ternaryVLE_NRTL_inp_file(
    carrier: str,
    solute: str,
    solvent: str,
    tC: float,
    config: FileGenConfig,
) -> Dict[str, str]:
    """
    Generates an input file for ternary VLE + NRTL calculations in COSMOtherm for a given carrier, solute, and solvent.
    """
    task = "NRTL"
    input_folder = os.path.join(config.inputfiles_folder, task)
    output_folder = os.path.join(config.odir, task)
    ensure_dir_exists(input_folder)
    ensure_dir_exists(output_folder)
    file_name = config.file_name or f"{task}_{solvent}.inp"
    file_fullpath = os.path.join(input_folder, file_name)
    content = f"""ctd={config.ctd_file} CDIR=\"{config.cdir}\" LDIR=\"{config.ldir}\" odir=\"{output_folder}\" # Global command line
FDIR=\"{config.fdir}\" AUTOC UNIT=SI                                # Global command line
!  Ternary VLE computation with NRTL model for {carrier}, {solute}, and {solvent} at {tC} °C                                          # Comment line
f = {carrier}\nf = {solute}\nf = {solvent}\nternary={{1 2 3}} tc={tC} LLE NRTL RENORM nomix"""
    write_input_file(file_fullpath, content, config.overwrite)
    return {
        "fullpath": os.path.abspath(file_fullpath),
        "filename": file_name,
        "folder": os.path.abspath(config.inputfiles_folder)
    }

