"""
residue_ss.py (Formerly ss.py)
Secondary Structure helper.
"""
import gemmi
from typing import Dict, List, Tuple

def build_index(structure: gemmi.Structure) -> Dict[str, List[Tuple[int, int, str, int]]]:
    ss_index = {}
    uid = 1
    def add(chain, start, end, code, u):
        if chain not in ss_index: ss_index[chain] = []
        ss_index[chain].append((start, end, code, u))

    if hasattr(structure, 'helices'):
        for h in structure.helices:
            try:
                code = 'H'
                if h.pdb_helix_class == 5: code = 'G'
                elif h.pdb_helix_class == 3: code = 'I'
                add(h.start.chain_name, h.start.res_id.seqid.num, h.end.res_id.seqid.num, code, uid)
                uid += 1
            except: pass
            
    if hasattr(structure, 'sheets'):
        for s in structure.sheets:
            for st in s.strands:
                try:
                    add(st.start.chain_name, st.start.res_id.seqid.num, st.end.res_id.seqid.num, 'E', uid)
                    uid += 1
                except: pass
    return ss_index

def get_info(chain_name: str, res_num: int, ss_index: Dict) -> Tuple[str, int]:
    if chain_name not in ss_index: return ('C', -1)
    for start, end, code, uid in ss_index[chain_name]:
        if start <= res_num <= end: return (code, uid)
    return ('C', -1)