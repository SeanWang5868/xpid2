"""
prep.py
Handles structure preparation and Hydrogen addition.
"""
import gemmi
import os
import re
import logging
from typing import Optional

logger = logging.getLogger("xpid.prep")
_CACHED_MONLIB: Optional[gemmi.MonLib] = None
_CACHED_LIB_PATH: Optional[str] = None

def _get_shared_monlib(mon_lib_path: Optional[str], residue_names: set) -> gemmi.MonLib:
    global _CACHED_MONLIB, _CACHED_LIB_PATH
    if _CACHED_MONLIB is None or _CACHED_LIB_PATH != mon_lib_path:
        _CACHED_MONLIB = gemmi.MonLib()
        _CACHED_LIB_PATH = mon_lib_path
    monlib = _CACHED_MONLIB
    if mon_lib_path:
        missing = [c for c in residue_names if c not in monlib.monomers]
        if missing:
            try:
                monlib.read_monomer_lib(mon_lib_path, missing)
            except Exception:
                pass
    return monlib

def add_hydrogens_memory(structure: gemmi.Structure, 
                         mon_lib_path: Optional[str] = None, 
                         h_change_val: int = 4) -> Optional[gemmi.Structure]:
    try:
        if h_change_val == 0: return structure

        all_codes = set()
        for model in structure:
            for chain in model:
                for residue in chain: all_codes.add(residue.name)
        
        monlib = _get_shared_monlib(mon_lib_path, all_codes)

        max_attempts = 10  # Maximum auto-removal retries for problematic residues
        for attempt in range(max_attempts):
            try:
                # Build topology and add hydrogens
                gemmi.prepare_topology(structure, monlib, model_index=0, h_change=h_change_val, reorder=False, ignore_unknown_links=True)
                break  # Success: exit retry loop
                
            except Exception as topo_err:
                err_msg = str(topo_err)
                
                # Strategy 1: Clear explicit links on link-related errors
                if "link" in err_msg.lower():
                    logger.warning("  -> Bad explicit link detected. Clearing connections and retrying...")
                    structure.connections.clear()
                    continue
                    
                # Strategy 2: Surgically remove distorted residues
                # Parse error format, e.g.: "bonded to V/NAG 2/O4 failed"
                match = re.search(r"bonded to ([^/]+)/([^ ]+) ([^/]+)/([^ ]+) failed", err_msg)
                if match:
                    bad_chain = match.group(1)
                    bad_seqid = match.group(3)
                    
                    removed = False
                    for model in structure:
                        for chain in model:
                            if chain.name == bad_chain:
                                for i in range(len(chain)):
                                    if str(chain[i].seqid).strip() == bad_seqid.strip():
                                        del chain[i]  # Remove the problematic residue
                                        removed = True
                                        logger.warning(f"  -> Removed twisted residue {bad_chain}/{bad_seqid} to save the rest of the structure.")
                                        break
                            if removed: break
                        if removed: break
                        
                    if removed:
                        continue  # Retry after successful removal
                
                # Strategy 3: Give up if error is unrecognized
                logger.warning(f"  -> Topology incomplete after {attempt} retries: {err_msg}.")
                break  # Continue with partial hydrogen placement
        # ---------------------------------------------------------

        structure.setup_cell_images()
        return structure
        
    except Exception as e:
        logger.error(f"Critical error in prep: {e}")
        # Never return None here — return the original structure so the pipeline continues
        return structure