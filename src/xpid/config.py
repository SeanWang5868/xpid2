"""
config.py
Configuration constants, atom definitions, and geometric thresholds.
"""
import gemmi
import os
import json
from pathlib import Path
from typing import Dict, Set, Optional, List

from collections import defaultdict

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

# --- Fallback Manual Ring Definitions (kept as safety net for cases with no monomer library) ---
FALLBACK_RINGS: Dict[str, List[Set[str]]] = {
    # 'TRP': [
    #     {'CD2', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'},  # 6-ring
    #     {'CD1', 'CD2', 'NE1', 'CG', 'CE2'}           # 5-ring
    # ],
    # 'TYR': [{'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'CG'}],
    # 'PTR': [{'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'CG'}],
    # 'PHE': [{'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'CG'}],
    # 'HIS': [{'CE1', 'ND1', 'NE2', 'CG', 'CD2'}],
    # 'BER': [
    #     {'C1', 'N1', 'C3', 'C6', 'C8', 'C12'},
    #     {'C8', 'C12', 'C13', 'C15', 'C16', 'C18'},
    #     {'C2', 'C4', 'C5', 'C9', 'C11', 'C14'},
    # ],
    # '4PO': [
    #     {'N2', 'C6', 'C7', 'C8', 'C9', 'C10'}
    # ]
}

# --- Cache ---
AROMATIC_RINGS_CACHE: Dict[str, List[Set[str]]] = {}

# --- Robust Aromatic Ring Detection (strictly following your direct CIF parsing + DFS logic) ---
def get_aromatic_rings(res_name: str, mon_lib_path: Optional[str] = None) -> List[Set[str]]:
    """
    Detect all possible 5/6-membered aromatic rings by direct CIF parsing.
    Priority:
      1. Plane restraints (most accurate when defined)
      2. Your exact aromatic-bond DFS cycle search
      3. Manual fallback only if no library/no rings found
    Fully cached.
    """
    if res_name in AROMATIC_RINGS_CACHE:
        return AROMATIC_RINGS_CACHE[res_name]
    
    rings: List[Set[str]] = []
    seen: Set[frozenset] = set()
    
    monlib_path_to_use = mon_lib_path or DEFAULT_MON_LIB_PATH
    
    if monlib_path_to_use:
        root_path = Path(monlib_path_to_use)
        res_upper = res_name.upper()
        first_letter = res_upper[0].lower()
        
        # Possible CIF locations (covers CCD style, gzipped, flat, etc.)
        possible_paths = [
            root_path / first_letter / f"{res_upper}.cif",
            root_path / first_letter / f"{res_upper}.cif.gz",
            root_path / f"{res_upper}.cif",
            root_path / f"{res_upper}.cif.gz",
        ]
        
        cif_path: Optional[Path] = None
        for p in possible_paths:
            if p.exists():
                cif_path = p
                break
        
        if cif_path:
            try:
                doc = gemmi.cif.read_file(str(cif_path))  # Handles .gz automatically
                
                block_name = f"comp_{res_upper}"
                block = doc.find_block(block_name)
                if block is None:
                    # Fallback to sole block if naming differs
                    if len(doc.blocks) == 1:
                        block = doc[0]
                
                if block:
                    # === 1. Plane restraints (grouped reliably) ===
                    plane_atoms: Dict[str, List[str]] = defaultdict(list)
                    for plane_id_item, atom_item in block.find(['_chem_comp_plane_atom.plane_id',
                                                               '_chem_comp_plane_atom.atom_id']):
                        atom = atom_item.strip()
                        if atom and plane_id_item:
                            plane_atoms[plane_id_item].append(atom)
                    
                    for atoms_in_plane in plane_atoms.values():
                        if len(atoms_in_plane) in (5, 6):
                            ring_set = frozenset(atoms_in_plane)
                            if ring_set not in seen:
                                rings.append(set(atoms_in_plane))
                                seen.add(ring_set)
                    
                    # === 2. Your exact aromatic-bond DFS logic ===
                    aromatic_edges = []
                    # We include both aromatic='y'/'yes' and (as robustness) type='arom' if column missing
                    for row in block.find(['_chem_comp_bond.atom_id_1',
                                           '_chem_comp_bond.atom_id_2',
                                           '_chem_comp_bond.aromatic']):
                        a1, a2, arom = row
                        a1_stripped = a1.strip()
                        a2_stripped = a2.strip()
                        if a1_stripped and a2_stripped and str(arom).lower() in {'y', 'yes'}:
                            aromatic_edges.append((a1_stripped, a2_stripped))
                    
                    if aromatic_edges and not rings:  # Only if no planes found (priority)
                        graph = defaultdict(set)
                        for a, b in aromatic_edges:
                            graph[a].add(b)
                            graph[b].add(a)
                        
                        found_rings = set()
                        
                        def dfs(path: List[str], start: str):
                            if len(path) > 8:
                                return
                            cur = path[-1]
                            for nb in graph[cur]:
                                if nb == start and len(path) in {5, 6}:
                                    ring_tuple = tuple(sorted(path))
                                    found_rings.add(ring_tuple)
                                elif nb not in path:
                                    dfs(path + [nb], start)
                        
                        for atom in list(graph.keys()):
                            dfs([atom], atom)
                        
                        for ring_tuple in found_rings:
                            ring_set = frozenset(ring_tuple)
                            if ring_set not in seen:
                                rings.append(set(ring_tuple))
                                seen.add(ring_set)
                                
            except Exception as e:
                print(f"Warning: Failed to parse CIF for {res_name} at {cif_path}: {str(e)}")
    
    # === 3. Final safety fallback (only if nothing found and no library coverage) ===
    if not rings and res_name in FALLBACK_RINGS:
        rings = FALLBACK_RINGS[res_name]
    
    AROMATIC_RINGS_CACHE[res_name] = rings
    return rings

# --- Rest completely unchanged ---
TARGET_ELEMENTS_X = {gemmi.Element('C'), gemmi.Element('N'), gemmi.Element('O'), gemmi.Element('S')}
TARGET_ELEMENTS_H = {gemmi.Element('H'), gemmi.Element('D')}

ROTATABLE_MAPPING: Dict[str, Dict[str, str]] = {
    'ALA': {'CB': 'CA'},
    'VAL': {'CG1': 'CB', 'CG2': 'CB'},
    'LEU': {'CD1': 'CG', 'CD2': 'CG'},
    'ILE': {'CD1': 'CG1', 'CG2': 'CB'}, 
    'MET': {'CE': 'SD'},
    'MSE': {'CE': 'SE'},
    'THR': {'CG2': 'CB'},
    'SER': {'OG': 'CB'},
    'THR': {'OG1': 'CB'},
    'TYR': {'OH': 'CZ'},
    'CYS': {'SG': 'CB'},
    'LYS': {'NZ': 'CE'}
}

# 阵营 B: 具有诱导契合能力的柔性基团 (低旋转势垒，约 1-2 kcal/mol，允许 360° 连续扫描)
FLEXIBLE_DONORS = {'OG', 'OG1', 'OH', 'SG'}  

# 阵营 A: 受限于交错态的刚性转子 (高三重扭转势垒，约 3 kcal/mol，仅允许 3 态离散扫描)
RIGID_DONORS = {'CB', 'CG1', 'CG2', 'CD1', 'CD2', 'CE', 'NZ'} 

# 祖父原子映射字典 (Grandparent Mapping) 
# 必需！用于定义参考平面，从而计算出甲基/铵基那 3 个绝对准确的交错态位置
GRANDPARENT_MAPPING: Dict[str, Dict[str, str]] = {
    'ALA': {'CB': 'N'},           # CA 的母体，用骨架 N 作为参考相
    'VAL': {'CG1': 'CA', 'CG2': 'CA'},
    'LEU': {'CD1': 'CB', 'CD2': 'CB'},
    'ILE': {'CD1': 'CB', 'CG2': 'CA'},
    'MET': {'CE': 'CG'},
    'MSE': {'CE': 'CG'},
    'THR': {'CG2': 'CA', 'OG1': 'CA'},
    'SER': {'OG': 'CA'},
    'TYR': {'OH': 'CE1'},         # 实际上分在柔性组，这里备用
    'CYS': {'SG': 'CA'},
    'LYS': {'NZ': 'CD'}
}


BOND_LENGTHS = {
    'C': 1.09, 
    'N': 1.01, 
    'O': 0.96, 
    'S': 1.33
}
TETRAHEDRAL_ANGLE = 109.5  # sp3 杂化的典型 Parent-X-H 角度

# CONE_PARAMS = {
#     'METHYL': {'angle': 70.5, 'tolerance': 25.0},
#     'OH_SH':  {'angle': 70.5, 'tolerance': 15.0},
#     'AMINE':  {'angle': 70.5, 'tolerance': 20.0}
# }

# def get_cone_params(atom_element: str) -> Dict[str, float]:
#     if atom_element == 'C': return CONE_PARAMS['METHYL']
#     if atom_element in ('O', 'S'): return CONE_PARAMS['OH_SH']
#     if atom_element == 'N': return CONE_PARAMS['AMINE']
#     return {'angle': 70.5, 'tolerance': 20.0}

DIST_SEARCH_LIMIT = 6.0
THRESHOLDS = {
    'N': 4.3,
    'O': 4.3,
    'C': 4.5,
    'S': 4.8, 
    'default': 4.5
}
DIST_CUTOFF_H = 1.3