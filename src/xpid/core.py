"""
core.py
Core logic for detecting XH-pi interactions using Dual-Track logic.
Track 1: Implicit/Cone (Optional via use_cone=True)
Track 2: Explicit H (Default)
"""
import gemmi
import numpy as np
from typing import List, Dict, Any, Optional, Union
from . import config
from . import geometry
from . import residue_ss

def detect_interactions_in_structure(structure: gemmi.Structure, 
                                     pdb_name: str,
                                     filter_pi: Optional[List[str]] = None,
                                     filter_donor: Optional[List[str]] = None,
                                     filter_donor_atom: Optional[List[str]] = None,
                                     model_mode: Union[str, int] = 0,
                                     use_cone: bool = False) -> List[Dict[str, Any]]:
    """
    detects interactions. use_cone=True enables implicit detection for rotatable groups.
    """
    results = []
    if not structure or len(structure) == 0: return []

    models_with_ids = [] 
    if model_mode == 'all':
        for i, m in enumerate(structure):
            m_name = getattr(m, 'name', str(i+1))
            models_with_ids.append((m, m_name))
    else:
        try:
            idx = int(model_mode)
            if 0 <= idx < len(structure):
                m = structure[idx]
                m_name = getattr(m, 'name', str(idx+1))
                models_with_ids = [(m, m_name)]
            else: pass 
        except ValueError:
            m = structure[0]
            m_name = getattr(m, 'name', "1")
            models_with_ids = [(m, m_name)]

    resolution = structure.resolution if structure.resolution else 0.0

    for model, model_id in models_with_ids:
        ns = gemmi.NeighborSearch(model, structure.cell, config.DIST_SEARCH_LIMIT)
        ns.populate(include_h=True)
        ss_index = residue_ss.build_index(structure)
        
        for chain in model:
            for residue in chain:
                res_name = residue.name
                if filter_pi and res_name not in filter_pi: continue

                if res_name in config.RING_ATOMS:
                    results.extend(_detect_residue(
                        pdb_name, resolution, model, model_id, chain, residue, ns, ss_index,
                        config.RING_ATOMS[res_name], 'main', filter_donor, filter_donor_atom, use_cone
                    ))
                
                if res_name in config.TRP_A_ATOMS:
                    results.extend(_detect_residue(
                        pdb_name, resolution, model, model_id, chain, residue, ns, ss_index,
                        config.TRP_A_ATOMS[res_name], 'trpA', filter_donor, filter_donor_atom, use_cone
                    ))
    return results
# 定义常见的金属离子，用于排除金属配位的 CYS
# 这些原子如果在 SG 附近，说明 S 已经配位，没有 H
BLOCKING_METALS = {
    'ZN', 'FE', 'CU', 'MN', 'MG', 'CO', 'NI', 'CA', 'CD', 'HG'
}

def _is_chemically_blocked(x_atom, model, ns) -> bool:
    """
    检查 CYS 的 SG 是否参与了二硫键或金属配位。
    如果是，则它没有 H，不能作为供体。
    """
    # 设定搜索半径: S-S 键长约 2.05A，S-Metal 约 2.3-2.5A。
    # 2.8A 是一个非常安全的上限，能囊括稍微扭曲的键，但不会误伤范德华接触。
    radius = 2.8 
    
    # 搜索附近的原子
    neighbors = ns.find_atoms(x_atom.pos, radius=radius)
    
    for mark in neighbors:
        # 获取邻居原子的元素类型
        # 注意: mark.element 是 Element 对象，需要 .name 获取字符串
        el_name = mark.element.name.upper()
        
        # 1. 排除自己 (距离为 0)
        if mark.image_idx == 0 and mark.chain_idx == -1 and mark.atom_idx == -1: 
             if mark.pos.dist(x_atom.pos) < 0.1:
                 continue
                 
        # 2. 也是 S (硫) -> 且不是自己 -> 认为是二硫键
        # (还要排除掉比如 MET 的 SD 离得很近的情况，但在 2.5A 内通常就是成键)
        if el_name == 'S':
            # 还需要再次确认不是自己(通过内存地址或序号)
            # 在 Gemmi 中，最稳妥的是转成 CRA 比较，或者简单的距离判断
            dist = mark.pos.dist(x_atom.pos)
            if dist > 0.1: # 只要距离大于 0.1 且小于 2.8 的 S，就是二硫键伙伴
                return True # Blocked!
                
        # 3. 是金属 -> 金属配位
        if el_name in BLOCKING_METALS:
            return True # Blocked!
            
    return False # Not blocked (Free Thiol)

def _detect_residue(pdb_name, resolution, model, model_id, chain, residue, ns, ss_index, 
                    target_atoms, mode, filter_donor, filter_donor_atom, use_cone):
    hits = []
    
    # 1. Setup Pi System
    pi_atoms = [atom for atom in residue if atom.name in target_atoms]
    if len(pi_atoms) != len(target_atoms): return []

    pi_occs = [atom.occ for atom in pi_atoms]
    avg_pi_occ = sum(pi_occs) / len(pi_occs)
    pi_alt = pi_atoms[0].altloc # 假设环内原子altloc一致，这在PDB中通常成立
    
    pi_center, pi_center_arr, pi_normal, pi_b_mean = geometry.get_pi_info(pi_atoms)
    alt_pi = pi_atoms[0].altloc
    
    # 2. Search Neighbors (Heavy Atoms X)
    x_candidates = ns.find_atoms(pi_center, alt=alt_pi, radius=config.DIST_SEARCH_LIMIT)
    
    for x_mark in x_candidates:
        x_cra = x_mark.to_cra(model)
        x_atom = x_cra.atom
        x_res = x_cra.residue
        x_res_name = x_res.name
        
        # --- [NEW] 1. Occupancy & Altloc Filters ---
        # A. Low Occupancy Filter (噪音过滤)
        if x_atom.occ < 0.10: continue

        # B. Conformer Consistency Check (物理互斥性检查)
        # 规则: 
        # 1. 如果两者都没有 altloc (都是空字符串)，兼容。
        # 2. 如果其中一个是空字符串 (主干/无构象)，兼容。
        # 3. 如果两者都有 altloc，必须相等 (A对A, B对B)。A不能对B。
        if pi_alt and x_atom.altloc:
            if pi_alt != x_atom.altloc:
                continue # 构象冲突：A构象的环不可能遇到B构象的供体
        
        # 计算组合占有率 (用于后续统计权重)
        # 两个独立事件同时发生的概率 P(A)*P(B)
        # 但在晶体中通常是完全相关(1.0)或完全互斥。
        # 使用 min() 更符合晶体学逻辑：瓶颈在于占有率较低的那个
        combined_occ = min(avg_pi_occ, x_atom.occ)

        # -------------------------------------------

        # Basic Filters
        if x_atom.element not in config.TARGET_ELEMENTS_X: continue
        if filter_donor and x_res_name not in filter_donor: continue
        if filter_donor_atom and x_atom.element.name not in filter_donor_atom: continue

        if x_res.name == 'CYS' and x_atom.name == 'SG':
            if _is_chemically_blocked(x_atom, model, ns):
                continue # 跳过，因为它没有 H

        x_pos_arr = np.array(x_atom.pos.tolist())

        max_dist = config.THRESHOLDS.get(x_atom.element.name , config.THRESHOLDS['default'])
        dist_x_pi = geometry.calculate_distance(x_pos_arr, pi_center_arr)
        
        if dist_x_pi > max_dist: 
            continue

        # Common Geometrics
        xpcn_angle = geometry.calculate_xpcn_angle(x_pos_arr, pi_center_arr, pi_normal)
        proj_threshold = None
        if mode == 'trpA': proj_threshold = 1.6
        elif residue.name == 'HIS': proj_threshold = 1.6
        elif residue.name in ['TRP', 'TYR', 'PHE']: proj_threshold = 2.0
        
        proj_dist = None
        if proj_threshold is not None:
            proj_dist = geometry.calculate_projection_dist(pi_normal, pi_center_arr, x_pos_arr)

        detected_mode = None 
        final_h_name = ""
        final_theta = None
        final_xh_pi_angle = None
        cone_delta = None

        # Check rotatability for potential Cone Logic
        is_rotatable = False
        parent_atom_name = None
        if x_res_name in config.ROTATABLE_MAPPING:
            mapping = config.ROTATABLE_MAPPING[x_res_name]
            if x_atom.name in mapping:
                is_rotatable = True
                parent_atom_name = mapping[x_atom.name]

        # --- TRACK 1: IMPLICIT / CONE (OPTIONAL) ---
        # Only enter if switch is ON and atom is rotatable
        if use_cone and is_rotatable:
            parent_atom = None
            for at in x_res:
                if at.name == parent_atom_name:
                    if at.altloc and pi_alt and at.altloc != pi_alt: continue
                    parent_atom = at
                    break
            
            if parent_atom:
                parent_pos_arr = np.array(parent_atom.pos.tolist())
                cone_cfg = config.get_cone_params(x_atom.element.name)
                
                cone_pass, cone_angle, delta = geometry.calculate_cone_alignment(
                    parent_pos_arr, x_pos_arr, pi_center_arr, 
                    cone_cfg['angle'], cone_cfg['tolerance']
                )

                if cone_pass:
                    is_plevin = (dist_x_pi < max_dist and xpcn_angle < 25.0)
                    is_hudson = False
                    if (dist_x_pi <= max_dist and 
                        proj_dist is not None and proj_dist <= proj_threshold):
                        is_hudson = True

                    if is_plevin or is_hudson:
                        detected_mode = "Implicit/Cone"
                        final_h_name = "(virt)"
                        final_theta = 0.0
                        final_xh_pi_angle = 180.0
                        cone_delta = round(delta, 1)
                        
                        _record_hit(hits, pdb_name, model_id, resolution, chain, residue, 
                                    x_cra, x_atom, final_h_name, dist_x_pi, 
                                    int(is_plevin), int(is_hudson), mode, 
                                    pi_center_arr, pi_b_mean, x_pos_arr, ss_index, 
                                    final_theta, final_xh_pi_angle, xpcn_angle, proj_dist, 
                                    detected_mode, cone_delta)
                        continue # Done with this atom, skip explicit check

        # --- TRACK 2: EXPLICIT H (DEFAULT & FALLBACK) ---
        h_candidates = ns.find_atoms(x_atom.pos, alt=x_atom.altloc, radius=config.DIST_CUTOFF_H)
        
        for h_mark in h_candidates:
            h_cra = h_mark.to_cra(model)
            h_atom = h_cra.atom
            if h_atom.element not in config.TARGET_ELEMENTS_H: continue
            
            if h_atom.altloc and x_atom.altloc and h_atom.altloc != x_atom.altloc:
                continue
            h_pos_arr = np.array(h_atom.pos.tolist())
            
            xh_pi_angle = geometry.calculate_xh_picenter_angle(pi_center_arr, x_pos_arr, h_pos_arr)
            theta = geometry.calculate_hudson_theta(pi_center_arr, x_pos_arr, h_pos_arr, pi_normal)
            
            if xh_pi_angle is None or theta is None or xpcn_angle is None: continue

            plevin = 0
            if (dist_x_pi < max_dist and 
                xh_pi_angle > 120.0 and 
                xpcn_angle < 25.0):
                plevin = 1
            
            hudson = 0
            if (proj_dist is not None and 
                theta <= 40.0 and 
                dist_x_pi <= max_dist and 
                proj_dist <= proj_threshold):
                hudson = 1
            
            if plevin == 1 or hudson == 1:
                _record_hit(hits, pdb_name, model_id, resolution, chain, residue, 
                            x_cra, x_atom, h_atom.name, dist_x_pi, 
                            plevin, hudson, mode, 
                            pi_center_arr, pi_b_mean, x_pos_arr, ss_index, 
                            theta, xh_pi_angle, xpcn_angle, proj_dist, 
                            "Explicit", None)
    return hits
def _record_hit(hits, pdb, mid, res, pi_chain, pi_res, x_cra, x_atom, h_name, dist, 
                is_plevin, is_hudson, mode, pi_cen, pi_b, x_pos, ss_index, 
                theta, xh_ang, xpcn, proj, method, cone_delta, 
                combined_occ=1.0): # <--- [NEW] 新增参数，默认为 1.0
    
    pi_ss_type, pi_ss_uid = residue_ss.get_info(pi_chain.name, pi_res.seqid.num, ss_index)
    x_ss_type, x_ss_uid = residue_ss.get_info(x_cra.chain.name, x_cra.residue.seqid.num, ss_index)

    seq_sep = 0
    if pi_chain.name == x_cra.chain.name:
        try: seq_sep = pi_res.seqid.num - x_cra.residue.seqid.num
        except: pass

    remark_parts = []
    if pi_res.name == "TRP": remark_parts.append("6-ring" if mode == "main" else "5-ring")
    if method == "Implicit/Cone": remark_parts.append(f"Cone(d={cone_delta})")
    
    # [Optional] 如果是低占有率，也可以在 remark 里标记一下，方便人眼看
    if combined_occ < 1.0:
        remark_parts.append(f"Occ={combined_occ:.2f}")

    hits.append({
        'pdb': pdb,
        'model': mid,
        'resolution': res,
        'pi_chain': pi_chain.name,
        'pi_res': pi_res.name,
        'pi_id': pi_res.seqid.num,
        'X_chain': x_cra.chain.name,
        'X_res': x_cra.residue.name,
        'X_id': x_cra.residue.seqid.num,
        'X_atom': x_atom.name,
        'H_atom': h_name,
        'dist_X_Pi': round(dist, 3),
        'method': method,
        'is_plevin': is_plevin,
        'is_hudson': is_hudson,
        'remark': ", ".join(remark_parts),
        'occupancy': round(combined_occ, 2), # <--- [NEW] 保存占有率
        'pi_ss_type': pi_ss_type,
        'pi_ss_id': pi_ss_uid,
        'X_ss_type': x_ss_type,
        'X_ss_id': x_ss_uid,
        'pi_avg_b': round(pi_b, 2),
        'pi_center_x': round(pi_cen[0], 3),
        'pi_center_y': round(pi_cen[1], 3),
        'pi_center_z': round(pi_cen[2], 3),
        'X_b': round(x_atom.b_iso, 2),
        'X_xyz_x': round(x_pos[0], 3),
        'X_xyz_y': round(x_pos[1], 3),
        'X_xyz_z': round(x_pos[2], 3),
        'seq_sep': seq_sep,
        'theta': round(theta, 2) if theta is not None else 0,
        'angle_XH_Pi': round(xh_ang, 2) if xh_ang is not None else 180,
        'angle_XPCN': round(xpcn, 2) if xpcn is not None else None,
        'proj_dist': round(proj, 3) if proj is not None else None,
    })