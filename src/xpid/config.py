"""
config.py
Configuration constants, atom definitions, and geometric thresholds.
"""
import gemmi
import os
import json
from pathlib import Path
from typing import Dict, Set, Optional

# --- Configuration Management ---
CONFIG_FILE = Path.home() / ".xpid_config.json"

def load_saved_mon_lib() -> Optional[str]:
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r') as f:
                data = json.load(f)
                return data.get("monomer_library_path", None)
        except Exception:
            return None
    return None

def save_mon_lib_path(path: str) -> str:
    data = {}
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r') as f:
                data = json.load(f)
        except Exception:
            data = {}
    
    abs_path = str(Path(path).resolve())
    data["monomer_library_path"] = abs_path
    with open(CONFIG_FILE, 'w') as f:
        json.dump(data, f, indent=4)
    return abs_path

# --- Defaults ---
_env_path = os.environ.get("GEMMI_MON_LIB_PATH", None)
_saved_path = load_saved_mon_lib()
DEFAULT_MON_LIB_PATH = _env_path if _env_path else _saved_path
DEFAULT_H_CHANGE = 4 

# --- Atom Definitions ---
RING_ATOMS: Dict[str, Set[str]] = {
    'TRP': {'CD2', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'},
    'TYR': {'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'CG'},
    'PTR': {'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'CG'},
    'PHE': {'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'CG'},
    'HIS': {'CE1', 'ND1', 'NE2', 'CG', 'CD2'}
}

TRP_A_ATOMS: Dict[str, Set[str]] = {
    'TRP': {'CD1', 'CD2', 'NE1', 'CG', 'CE2'}
}

TARGET_ELEMENTS_X = {gemmi.Element('C'), gemmi.Element('N'), gemmi.Element('O'), gemmi.Element('S')}
TARGET_ELEMENTS_H = {gemmi.Element('H'), gemmi.Element('D')}

# --- Rotatable Donor Definitions (Cone Logic) ---
# Format: { 'RES': { 'ATOM': 'PARENT_ATOM' } }
ROTATABLE_MAPPING: Dict[str, Dict[str, str]] = {
    # Methyls (C-C axis)
    'ALA': {'CB': 'CA'},
    'VAL': {'CG1': 'CB', 'CG2': 'CB'},
    'LEU': {'CD1': 'CG', 'CD2': 'CG'},
    'ILE': {'CD1': 'CG1', 'CG2': 'CB'}, 
    'MET': {'CE': 'SD'},
    'MSE': {'CE': 'SE'}, # 硒原子替代了硫原子 SD
    'THR': {'CG2': 'CB'},
    # Hydroxyls (C-O axis)
    'SER': {'OG': 'CB'},
    'THR': {'OG1': 'CB'},
    'TYR': {'OH': 'CZ'},
    # 'PTR': {'OH': 'CZ'}, 
    # Thiols (C-S axis)
    'CYS': {'SG': 'CB'},
    # Amines (C-N axis, Lysine NH3+)
    'LYS': {'NZ': 'CE'}
}

# --- Cone Geometry Parameters ---
CONE_PARAMS = {
    'METHYL': {'angle': 70.5, 'tolerance': 25.0}, # Methyls (Fast rotation)
    'OH_SH':  {'angle': 70.5, 'tolerance': 15.0}, # OH/SH (Stricter)
    'AMINE':  {'angle': 70.5, 'tolerance': 20.0}  # Lysine
}

def get_cone_params(atom_element: str) -> Dict[str, float]:
    """Returns (angle, tolerance) based on atom type."""
    if atom_element == 'C': return CONE_PARAMS['METHYL']
    if atom_element in ('O', 'S'): return CONE_PARAMS['OH_SH']
    if atom_element == 'N': return CONE_PARAMS['AMINE']
    return {'angle': 70.5, 'tolerance': 20.0} 

# --- Geometric Thresholds ---
DIST_SEARCH_LIMIT = 6.0
THRESHOLDS = {
    'N': 4.3,
    'O': 4.3,
    'C': 4.5,
    'S': 4.8, 
    'default': 4.5
}
DIST_CUTOFF_H = 1.3