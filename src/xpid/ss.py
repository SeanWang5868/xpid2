import bisect
import gemmi
from typing import Dict, List, Tuple

def build_index(structure: gemmi.Structure) -> Dict[str, List[Tuple[int, int, str, str]]]:
    """Build secondary structure index from PDB HELIX/SHEET records.
    Returns {chain: [(start, end, code, uid), ...]} sorted by start residue number."""
    ss_index = {}
    helix_uid = 1
    
    def add(chain, start, end, code, u):
        if chain not in ss_index: ss_index[chain] = []
        ss_index[chain].append((start, end, code, u))

    if hasattr(structure, 'helices'):
        for h in structure.helices:
            try:
                code = h.pdb_helix_class.name
                add(h.start.chain_name, h.start.res_id.seqid.num, h.end.res_id.seqid.num, code, str(helix_uid))
                helix_uid += 1
            except (AttributeError, ValueError):
                pass
            
    if hasattr(structure, 'sheets'):
        for sheet in structure.sheets:
            for strand in sheet.strands:
                try:
                    packed_uid = str(sheet.name) + "_" + str(strand.name)
                    
                    sense = getattr(strand, 'sense', 0)
                    if sense == 1:
                        code = 'Ep'
                    elif sense == -1:
                        code = 'Ea'
                    else:
                        code = 'E0'
                    
                    add(strand.start.chain_name, 
                        strand.start.res_id.seqid.num, 
                        strand.end.res_id.seqid.num, 
                        code, 
                        packed_uid)
                except (AttributeError, ValueError):
                    pass

    # Sort by start residue for binary search
    for chain in ss_index:
        ss_index[chain].sort(key=lambda x: x[0])

    return ss_index

def get_info(chain_name: str, res_num: int, ss_index: Dict) -> Tuple[str, str]:
    if chain_name not in ss_index: return ('C', '-1')
    entries = ss_index[chain_name]
    # Binary search: find rightmost entry where start <= res_num
    idx = bisect.bisect_right(entries, (res_num,)) - 1
    # Check from idx backwards (handles overlapping ranges)
    for i in range(max(0, idx), -1, -1):
        start, end, code, uid = entries[i]
        if start > res_num:
            continue
        if start <= res_num <= end:
            return (code, uid)
        if start < res_num:
            break  # Sorted, so earlier entries won't match either
    return ('C', '-1')