"""
core.py
Core logic for detecting XH-pi interactions using Dual-Track logic.
Track 1: Implicit/Cone (Optional via use_cone=True)
Track 2: Explicit H (Default)
"""
import gemmi
import numpy as np
from typing import List, Dict, Any, Optional, Union, Set
from . import config
from . import geometry
from . import ss

BLOCKING_METALS = {
    'ZN', 'FE', 'CU', 'MN', 'MG', 'CO', 'NI', 'CA', 'CD', 'HG',
    'NA', 'K', 'PT', 'AU', 'AG', 'FE2', 'FE3'
}

def detect_interactions_in_structure(structure: gemmi.Structure, 
                                     pdb_name: str,
                                     filter_pi: Optional[List[str]] = None,
                                     filter_donor: Optional[List[str]] = None,
                                     filter_donor_atom: Optional[List[str]] = None,
                                     model_mode: Union[str, int] = 0,
                                     use_cone: bool = False,
                                     min_occ: float = 0.0) -> List[Dict[str, Any]]:
    results = []
    if not structure or len(structure) == 0: return []

    structure.remove_alternative_conformations()
    structure.remove_empty_chains()

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
            else: 
                return []
        except ValueError:
            m = structure[0]
            m_name = getattr(m, 'name', "1")
            models_with_ids = [(m, m_name)]

    resolution = structure.resolution if structure.resolution else 0.0

    ss_index = ss.build_index(structure)
        
    for model, model_id in models_with_ids:
        ns = gemmi.NeighborSearch(model, structure.cell, config.DIST_SEARCH_LIMIT)
        ns.populate(include_h=True)
        
        for chain in model:
            for residue in chain:
                res_name = residue.name
                if filter_pi and res_name not in filter_pi: continue

                rings = config.get_aromatic_rings(res_name)
                if not rings: continue

                for ring_idx, target_atoms in enumerate(rings):
                    ring_size = len(target_atoms)
                    mode = f'ring{ring_idx+1}'
                    
                    results.extend(_detect_residue(
                        pdb_name, resolution, model, model_id, chain, residue, ns, ss_index,
                        target_atoms, mode, filter_donor, filter_donor_atom, use_cone, ring_size,
                        min_occ
                    ))
    return results

def _is_donor_blocked(x_atom: gemmi.Atom, model: gemmi.Model, ns: gemmi.NeighborSearch) -> bool:
    radius = 2.6
    neighbors = ns.find_atoms(x_atom.pos, radius=radius)
    
    x_elem = x_atom.element.name.upper()
    
    for mark in neighbors:
        dist = mark.pos.dist(x_atom.pos)
        if dist < 0.01: continue
        
        cra = mark.to_cra(model)
        neighbor_atom = cra.atom
        neighbor_el = neighbor_atom.element.name.upper()
        
        if x_elem == 'S' and neighbor_el == 'S' and neighbor_atom.name == 'SG':
            if 1.8 <= dist <= 2.2:
                return True
        
        if neighbor_el in BLOCKING_METALS:
            if dist <= 2.6:
                return True
    
    return False

def _detect_residue(pdb_name, resolution, model, model_id, chain, residue, ns, ss_index, 
                    target_atoms: Set[str], mode: str, filter_donor: Optional[List[str]], 
                    filter_donor_atom: Optional[List[str]], use_cone: bool, ring_size: int,
                    min_occ: float):
    hits = []
    
    pi_atoms = [atom for atom in residue if atom.name in target_atoms]
    if len(pi_atoms) != len(target_atoms): return []

    max_planar_dev = geometry.calculate_planarity_deviation(pi_atoms)
    if max_planar_dev > 0.5:
        return []

    pi_occs = [atom.occ for atom in pi_atoms]
    avg_pi_occ = sum(pi_occs) / len(pi_occs)
    pi_alt = pi_atoms[0].altloc if pi_atoms else ''
    
    pi_center, pi_center_arr, pi_normal, pi_b_mean = geometry.get_pi_info(pi_atoms)
    
    x_candidates = ns.find_atoms(pi_center, alt=pi_alt, radius=config.DIST_SEARCH_LIMIT)
    
    for x_mark in x_candidates:
        x_cra = x_mark.to_cra(model)
        x_atom = x_cra.atom
        x_res = x_cra.residue
        x_res_name = x_res.name
        
        if filter_donor and x_res_name not in filter_donor: continue
        if filter_donor_atom and x_atom.name not in filter_donor_atom: continue

        if (x_res_name in ('ASP', 'GLU') and x_atom.name in ('OD1', 'OD2', 'OE1', 'OE2')) or \
           (x_atom.name == 'OXT'):
            continue
            
        allow_cone_scan = True
        if (x_res_name == 'ARG' and x_atom.name in ('NH1', 'NH2', 'NE')) or \
           (x_res_name in ('ASN', 'GLN') and x_atom.name in ('ND2', 'NE2')) or \
           (x_res_name == 'HIS' and x_atom.name in ('ND1', 'NE2')) or \
           (x_res_name == 'TRP' and x_atom.name == 'NE1'):
            allow_cone_scan = False

        if _is_donor_blocked(x_atom, model, ns):
            continue

        if x_atom.occ < 0.10: continue

        if pi_alt and x_atom.altloc and pi_alt != x_atom.altloc:
            continue
        
        combined_occ = min(avg_pi_occ, x_atom.occ)

        x_pos_arr = np.array(x_atom.pos.tolist())
        dist_x_pi = geometry.calculate_distance(x_pos_arr, pi_center_arr)
        
        x_elem = x_atom.element.name.upper()
        max_dist = config.THRESHOLDS.get(x_elem, config.THRESHOLDS['default'])
        
        if dist_x_pi > max_dist: continue
        
        xpcn_angle = geometry.calculate_xpcn_angle(x_pos_arr, pi_center_arr, pi_normal)
        proj_dist = geometry.calculate_projection_dist(pi_normal, pi_center_arr, x_pos_arr)
        
        proj_threshold = 2.0 if ring_size == 6 else 1.6
        
        found_explicit_interaction = False
        orig_h_positions = [] # 用于安全存储真正的显式氢坐标，供后面圆锥算法使用

        # ---------------------------------------------------------------------
        # 🔵 第一梯队：显式精确几何算法 
        # ---------------------------------------------------------------------
        h_candidates = ns.find_atoms(x_atom.pos, alt=x_atom.altloc, radius=config.DIST_CUTOFF_H)

        for h_mark in h_candidates:
            h_cra = h_mark.to_cra(model)
            h_atom = h_cra.atom
            
            # 💡 修复 Bug 1 & 3: 使用纯字符串集合判断，确保只捞出真实的 H/D 原子
            if h_atom.element.name.upper() not in {'H', 'D'}: 
                continue
                
            # 保存真正的氢原子坐标，避免捞到供体原子自己
            h_pos_arr = np.array(h_atom.pos.tolist())
            orig_h_positions.append(h_pos_arr)
            
            if h_atom.altloc and x_atom.altloc and h_atom.altloc != x_atom.altloc:
                continue
            
            h_combined_occ = min(combined_occ, h_atom.occ)
            
            xh_pi_angle = geometry.calculate_xh_picenter_angle(pi_center_arr, x_pos_arr, h_pos_arr)
            theta = geometry.calculate_hudson_theta(pi_center_arr, x_pos_arr, h_pos_arr, pi_normal)
            
            if xh_pi_angle is None or theta is None or xpcn_angle is None: continue

            plevin = 0
            if (dist_x_pi < max_dist and xh_pi_angle >= 120.0 and xpcn_angle < 25.0):
                plevin = 1
            
            hudson = 0
            if (proj_dist is not None and theta <= 40.0 and dist_x_pi <= max_dist and proj_dist <= proj_threshold):
                hudson = 1
            
            if plevin == 1 or hudson == 1:
                found_explicit_interaction = True
                _record_hit(hits, pdb_name, model_id, resolution, chain, residue, 
                            x_cra, x_atom, h_atom.name, dist_x_pi, 
                            plevin, hudson, mode, 
                            pi_center_arr, pi_b_mean, x_pos_arr, ss_index, 
                            theta, xh_pi_angle, xpcn_angle, proj_dist, 
                            "Explicit", None, h_combined_occ, ring_size, min_occ)

        # ---------------------------------------------------------------------
        # 🔴 第二梯队：热力学限制的诱导契合与构象救援算法 (圆锥算法)
        # 💡 修复 Bug 2: 删除了这里原本错误嵌套的 for x_mark in x_candidates 循环
        # ---------------------------------------------------------------------
        if not found_explicit_interaction and use_cone and allow_cone_scan and x_res_name in config.ROTATABLE_MAPPING:
            parent_name = config.ROTATABLE_MAPPING[x_res_name].get(x_atom.name)
            
            if parent_name:
                parent_atom = next((a for a in x_res if a.name == parent_name), None)
                
                if parent_atom:
                    parent_pos_arr = np.array(parent_atom.pos.tolist())
                    
                    # 1. 提取局部重原子与极性原子 (用于位阻和氢键锁)
                    search_radius = 4.0
                    neighbors = ns.find_atoms(x_atom.pos, radius=search_radius)
                    
                    env_coords_list = []
                    acceptor_coords_list = []
                    
                    for n_mark in neighbors:
                        if n_mark.pos.dist(x_atom.pos) < 0.01: continue
                        n_cra = n_mark.to_cra(model)
                        
                        # 排除供体自身残基
                        if n_cra.residue.seqid == x_res.seqid and n_cra.chain.name == x_cra.chain.name:
                            continue
                            
                        n_elem = n_cra.atom.element.name.upper()
                        if n_elem in ('H', 'D', ''): continue
                        
                        n_pos_arr = np.array(n_cra.atom.pos.tolist())
                        dist = np.linalg.norm(n_pos_arr - x_pos_arr)
                        
                        if dist <= 4.0:
                            env_coords_list.append(n_pos_arr)
                        if dist <= 3.5 and n_elem in ('O', 'N', 'S'):
                            acceptor_coords_list.append(n_pos_arr)
                            
                    env_coords = np.array(env_coords_list) if env_coords_list else np.array([])
                    acceptor_coords = np.array(acceptor_coords_list) if acceptor_coords_list else np.array([])
                    
                    # 2. 强氢键锁定判断 (直接使用我们上面提取干净的 orig_h_positions)
                    is_locked = geometry.check_hbond_locked(x_pos_arr, orig_h_positions, acceptor_coords)
                    
                    h_candidates_cone = []
                    
                    if not is_locked:
                        flexible_donors = {('SER', 'OG'), ('THR', 'OG1'), ('TYR', 'OH'), ('CYS', 'SG')}
                        
                        if (x_res_name, x_atom.name) in flexible_donors:
                            h_candidates_cone = geometry.generate_rotated_hydrogens(
                                parent_pos_arr, x_pos_arr, x_elem, 
                                env_coords=env_coords, clash_cutoff=2.0, num_samples=72
                            )
                        else:
                            axis = x_pos_arr - parent_pos_arr
                            axis_norm = np.linalg.norm(axis)
                            if axis_norm > 1e-5:
                                axis = axis / axis_norm
                                wobble_angles_deg = [angle for angle in range(-20, 21, 5) if angle != 0]
                                
                                for h_pos_orig in orig_h_positions:
                                    vec_xh = h_pos_orig - x_pos_arr
                                    for angle_deg in wobble_angles_deg:
                                        theta_rad = np.radians(angle_deg)
                                        cos_theta = np.cos(theta_rad)
                                        sin_theta = np.sin(theta_rad)
                                        
                                        cross_prod = np.cross(axis, vec_xh)
                                        dot_prod = np.dot(axis, vec_xh)
                                        
                                        vec_xh_rotated = (vec_xh * cos_theta + 
                                                          cross_prod * sin_theta + 
                                                          axis * dot_prod * (1 - cos_theta))
                                        
                                        h_pos_wobbled = x_pos_arr + vec_xh_rotated
                                        
                                        if len(env_coords) > 0:
                                            min_dist = np.min(np.linalg.norm(env_coords - h_pos_wobbled, axis=1))
                                            if min_dist < 2.0: continue
                                                
                                        h_candidates_cone.append(h_pos_wobbled)

                    # 3. 对候选氢原子进行评分筛选最优解
                    best_hit = None
                    best_xh_angle = -1.0  
                    
                    for h_pos_np in h_candidates_cone:
                        theta = geometry.calculate_hudson_theta(pi_center_arr, x_pos_arr, h_pos_np, pi_normal)
                        xh_pi_angle = geometry.calculate_xh_picenter_angle(pi_center_arr, x_pos_arr, h_pos_np)
                        
                        if theta is None or xh_pi_angle is None: continue
                            
                        is_plevin_cand = (dist_x_pi < max_dist and xpcn_angle < 25.0 and xh_pi_angle >= 120.0)
                        is_hudson_cand = (dist_x_pi <= max_dist and proj_dist is not None and proj_dist <= proj_threshold and theta <= 40.0)
                        
                        if is_plevin_cand or is_hudson_cand:
                            if xh_pi_angle > best_xh_angle:
                                best_xh_angle = xh_pi_angle
                                best_hit = {
                                    'theta': theta,
                                    'xh_pi_angle': xh_pi_angle,
                                    'is_plevin': 1 if is_plevin_cand else 0,
                                    'is_hudson': 1 if is_hudson_cand else 0,
                                    'h_pos_np': h_pos_np
                                }

                    if best_hit is not None:
                        detected_mode = "Implicit/Cone_Rescue"
                        final_theta = best_hit['theta']
                        final_xh_pi_angle = best_hit['xh_pi_angle']
                        
                        _record_hit(hits, pdb_name, model_id, resolution, chain, residue, 
                                    x_cra, x_atom, "(virt)", dist_x_pi, 
                                    int(best_hit['is_plevin']), int(best_hit['is_hudson']), mode, 
                                    pi_center_arr, pi_b_mean, x_pos_arr, ss_index, 
                                    final_theta, final_xh_pi_angle, xpcn_angle, proj_dist, 
                                    detected_mode, 0.0, combined_occ, ring_size, min_occ)
    return hits

def _record_hit(hits: List[Dict[str, Any]], pdb: str, mid: str, res: float, pi_chain, pi_res, 
                x_cra, x_atom, h_name: str, dist: float, 
                is_plevin: int, is_hudson: int, mode: str, 
                pi_cen: np.ndarray, pi_b: float, x_pos: np.ndarray, ss_index: Dict, 
                theta: Optional[float], xh_ang: Optional[float], xpcn: Optional[float], 
                proj: Optional[float], method: str, cone_delta: Optional[float], 
                combined_occ: float = 1.0, ring_size: int = 0, min_occ: float = 0.0):
    
    if combined_occ < min_occ:
        return
    
    pi_ss_type, pi_ss_uid = ss.get_info(pi_chain.name, pi_res.seqid.num, ss_index)
    x_ss_type, x_ss_uid = ss.get_info(x_cra.chain.name, x_cra.residue.seqid.num, ss_index)

    seq_sep = 0
    if pi_chain.name == x_cra.chain.name:
        try: seq_sep = pi_res.seqid.num - x_cra.residue.seqid.num
        except: pass

    remark_parts = []
    
    remark_parts.append(f"{ring_size}-ring")
    
    if method == "Implicit/Cone" and cone_delta is not None:
        remark_parts.append(f"Cone(d={cone_delta})")
    
    if combined_occ < 0.5:
        remark_parts.append(f"LowOcc({combined_occ:.2f})")
    
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
        'occupancy': round(combined_occ, 2),
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