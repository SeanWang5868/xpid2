import gemmi
from typing import Dict, List, Tuple

def build_index(structure: gemmi.Structure) -> Dict[str, List[Tuple[int, int, str, str]]]:
    ss_index = {}
    helix_uid = 1
    
    def add(chain, start, end, code, u):
        if chain not in ss_index: ss_index[chain] = []
        ss_index[chain].append((start, end, code, u))

    if hasattr(structure, 'helices'):
        for h in structure.helices:
            try:
                code = h.pdb_helix_class
                add(h.start.chain_name, h.start.res_id.seqid.num, h.end.res_id.seqid.num, code, str(helix_uid))
                helix_uid += 1
            except: pass
            
    if hasattr(structure, 'sheets'):
        for sheet in structure.sheets:
            for strand in sheet.strands:
                try:
                    packed_uid = str(sheet.name) + str("_") + str(strand.name) 
                    
                    sense = getattr(strand, 'sense', 0)
                    # 根据 sense 赋予不同的 code
                    if sense == 1:
                        code = 'Ep'  # Parallel (平行)
                    elif sense == -1:
                        code = 'Ea'  # Antiparallel (反平行)
                    else:
                        code = 'E0'  # First strand (第一条链)
                    
                    add(strand.start.chain_name, 
                        strand.start.res_id.seqid.num, 
                        strand.end.res_id.seqid.num, 
                        code, 
                        packed_uid)
                except: pass
                
    return ss_index

def get_info(chain_name: str, res_num: int, ss_index: Dict) -> Tuple[str, str]:
    if chain_name not in ss_index: return ('C', '-1')
    for start, end, code, uid in ss_index[chain_name]:
        if start <= res_num <= end: return (code, uid)
    return ('C', '-1')