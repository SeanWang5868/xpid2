"""
topology.py
Manages chemical topology information via Gemmi.
Hybrid approach: Hardcoded standard residues (for speed/safety) + Dynamic MonLib lookup (for ligands).
"""
import gemmi
from typing import List, Dict, Optional, Set

STANDARD_AROMATICS = {
    'PHE': [['CG', 'CD1', 'CE1', 'CZ', 'CE2', 'CD2']],
    'TYR': [['CG', 'CD1', 'CE1', 'CZ', 'CE2', 'CD2']],
    'TRP': [
        ['CG', 'CD1', 'NE1', 'CE2', 'CD2'],      
        ['CD2', 'CE2', 'CZ2', 'CH2', 'CZ3', 'CE3'] 
    ],
    'HIS': [['CG', 'ND1', 'CE1', 'NE2', 'CD2']],
    'ADE': [['N1', 'C6', 'N6', 'C5', 'N7', 'C8', 'N9', 'C4', 'N3', 'C2']], 
    'A':   [['N1', 'C6', 'N6', 'C5', 'N7', 'C8', 'N9', 'C4', 'N3', 'C2']],
}

class TopologyManager:
    def __init__(self, structure: gemmi.Structure = None, mon_lib_path: str = None):
        self.mon_lib_path = mon_lib_path
        # res_name -> List[List[atom_name]]
        self.cache: Dict[str, List[List[str]]] = dict(STANDARD_AROMATICS)

    def get_aromatic_rings(self, res_name: str) -> List[List[str]]:

        if res_name in self.cache:
            return self.cache[res_name]

        if not self.mon_lib_path:
            self.cache[res_name] = [] 
            return []

        rings = self._load_and_analyze_ligand(res_name)
        
        self.cache[res_name] = rings
        return rings

    def _load_and_analyze_ligand(self, res_name: str) -> List[List[str]]:
        try:
            temp_lib = gemmi.read_monomer_lib(self.mon_lib_path, [res_name], ignore_missing=True)
            
            if res_name not in temp_lib.monomers:
                return []
            
            chemcomp = temp_lib.monomers[res_name]
            return self._find_rings_in_chemcomp(chemcomp)
            
        except Exception as e:
            # print(f"[Debug] Failed to load definition for {res_name}: {e}")
            return []

    def _find_rings_in_chemcomp(self, chemcomp: gemmi.ChemComp) -> List[List[str]]:
        adj = {}
        atom_elements = {} 
        
        for atom in chemcomp.atoms:
            adj[atom.id] = []
            atom_elements[atom.id] = atom.el.name.upper()

        for bond in chemcomp.bonds:
            a1 = bond.id1.atom if hasattr(bond.id1, 'atom') else str(bond.id1)
            a2 = bond.id2.atom if hasattr(bond.id2, 'atom') else str(bond.id2)
            if a1 in adj and a2 in adj:
                adj[a1].append(a2)
                adj[a2].append(a1)

        found_rings = []
        visited_fingerprints = set() # frozenset(atoms)

        def dfs(start, curr, path, depth):
            if depth > 6: return
            
            for neighbor in adj.get(curr, []):
                if neighbor == start:
  
                    if depth in [5, 6]:
                        fingerprint = frozenset(path)
                        if fingerprint not in visited_fingerprints:
                            visited_fingerprints.add(fingerprint)
                            found_rings.append(list(path))
                    continue
                
                if neighbor not in path:
                    dfs(start, neighbor, path + [neighbor], depth + 1)

        for atom_id in adj.keys():
            dfs(atom_id, atom_id, [atom_id], 1)

        valid_rings = []
        valid_elements = {'C', 'N', 'O', 'S'}
        
        for ring in found_rings:
            is_aromatic_candidate = True
            for at in ring:
                if atom_elements.get(at, 'X') not in valid_elements:
                    is_aromatic_candidate = False
                    break
            if is_aromatic_candidate:
                valid_rings.append(ring)

        return valid_rings