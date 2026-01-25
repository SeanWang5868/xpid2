"""
prep.py
Handles structure preparation and Hydrogen addition.
"""
import gemmi
import os
import logging
from typing import Optional

logger = logging.getLogger("xpid.prep")
_CACHED_MONLIB: Optional[gemmi.MonLib] = None
_CACHED_LIB_PATH: Optional[str] = None

def _get_shared_monlib(mon_lib_path: Optional[str]) -> gemmi.MonLib:
    global _CACHED_MONLIB, _CACHED_LIB_PATH
    if _CACHED_MONLIB is None or _CACHED_LIB_PATH != mon_lib_path:
        monlib = gemmi.MonLib()
        _CACHED_MONLIB = monlib
        _CACHED_LIB_PATH = mon_lib_path
    return _CACHED_MONLIB

def add_hydrogens_memory(structure: gemmi.Structure, 
                         mon_lib_path: Optional[str] = None, 
                         h_change_val: int = 4) -> Optional[gemmi.Structure]:
    try:
        if h_change_val == 0: return structure

        all_codes = set()
        for model in structure:
            for chain in model:
                for residue in chain: all_codes.add(residue.name)
        
        monlib = _get_shared_monlib(mon_lib_path)
        missing = [c for c in list(all_codes) if c not in monlib.monomers]
        if missing and mon_lib_path:
            try: monlib.read_monomer_lib(mon_lib_path, missing)
            except: pass

        gemmi.prepare_topology(structure, monlib, model_index=0, h_change=h_change_val, reorder=False, ignore_unknown_links=True)
        structure.setup_cell_images()
        return structure
    except Exception as e:
        logger.error(f"Topology failed: {e}")
        return None