"""
geometry.py
Geometric calculations including Plevin, Hudson, and Cone alignment.
"""
import numpy as np
import gemmi
from typing import Tuple, Optional, List
from . import config

def get_pi_info(atoms: List[gemmi.Atom]) -> Tuple[gemmi.Position, np.ndarray, np.ndarray, float]:
    positions = np.array([atom.pos.tolist() for atom in atoms])
    center_array = np.mean(positions, axis=0)
    pi_center = gemmi.Position(*center_array)
    b_mean = sum(atom.b_iso for atom in atoms) / len(atoms)
    
    centered_pos = positions - center_array
    _, _, vh = np.linalg.svd(centered_pos)
    normal_vector = vh[2, :] 
    
    return pi_center, center_array, normal_vector, b_mean

def calculate_planarity_deviation(atoms: List[gemmi.Atom]) -> float:
    """Calculate maximum deviation from the best-fit plane (in Å)."""
    if len(atoms) < 3:
        return 999.0
    
    _, center_arr, normal, _ = get_pi_info(atoms)
    norm_normal = np.linalg.norm(normal)
    if norm_normal == 0:
        return 999.0
    
    normal = normal / norm_normal  # Normalize
    
    max_dev = 0.0
    for atom in atoms:
        pos_arr = np.array(atom.pos.tolist())
        vec = pos_arr - center_arr
        dev = np.abs(np.dot(normal, vec))
        if dev > max_dev:
            max_dev = dev
    
    return max_dev

def calculate_distance(pos1_array: np.ndarray, pos2_array: np.ndarray) -> float:
    return np.linalg.norm(pos1_array - pos2_array)

def calculate_xpcn_angle(x_pos: np.ndarray, pi_center: np.ndarray, pi_normal: np.ndarray) -> Optional[float]:
    v_x_pi = pi_center - x_pos
    norm_v = np.linalg.norm(v_x_pi)
    norm_n = np.linalg.norm(pi_normal)
    if norm_v == 0 or norm_n == 0: return None

    dot_product = np.dot(v_x_pi, pi_normal)
    cos_theta = np.clip(dot_product / (norm_v * norm_n), -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_theta))
    
    if angle > 90: angle = 180 - angle
    return angle

def calculate_xh_picenter_angle(pi_center: np.ndarray, x_pos: np.ndarray, h_pos: np.ndarray) -> Optional[float]:
    v_hx = x_pos - h_pos
    v_hc = pi_center - h_pos 
    norm_hx = np.linalg.norm(v_hx)
    norm_hc = np.linalg.norm(v_hc)
    if norm_hx == 0 or norm_hc == 0: return None
    
    cos_theta = np.clip(np.dot(v_hx, v_hc) / (norm_hx * norm_hc), -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))

def calculate_hudson_theta(pi_center: np.ndarray, x_pos: np.ndarray, h_pos: np.ndarray, normal: np.ndarray) -> Optional[float]:
    v_x_pi = pi_center - x_pos
    v_xh = h_pos - x_pos
    norm_xpi = np.linalg.norm(v_x_pi)
    
    if norm_xpi == 0: return None

    # Projection check: H must point towards ring
    proj_len = np.dot(v_xh, v_x_pi) / norm_xpi
    
    if proj_len > 0:
        norm_n = np.linalg.norm(normal)
        norm_xh = np.linalg.norm(v_xh)
        if norm_n == 0 or norm_xh == 0: return None
        
        cos_angle = np.clip(np.dot(normal, v_xh) / (norm_n * norm_xh), -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        if angle >= 90: angle = 180 - angle
        return angle
    return None

def calculate_projection_dist(normal: np.ndarray, pi_center: np.ndarray, x_pos: np.ndarray) -> Optional[float]:
    numerator = np.dot(normal, pi_center - x_pos)
    denominator = np.dot(normal, normal)
    if denominator == 0: return None
    t = numerator / denominator
    projection_point = x_pos + t * normal
    return np.linalg.norm(projection_point - pi_center)

def check_hbond_locked(x_pos: np.ndarray, 
                       orig_h_positions: list, 
                       acceptor_coords: np.ndarray, 
                       dist_cutoff: float = 3.5, 
                       angle_cutoff_deg: float = 120.0) -> bool:
    """
    检查原始极性氢是否已经被周围的强氢键网络“锁定”。
    如果存在 D-H...A 夹角 > 120° 且 H...A 距离 < 3.5 Å 的情况，返回 True。
    """
    if len(acceptor_coords) == 0 or len(orig_h_positions) == 0:
        return False
        
    for h_pos in orig_h_positions:
        vec_xh = h_pos - x_pos
        norm_xh = np.linalg.norm(vec_xh)
        if norm_xh == 0: continue
        vec_xh_norm = vec_xh / norm_xh
        
        # 计算氢原子到所有潜在受体的距离
        dists = np.linalg.norm(acceptor_coords - h_pos, axis=1)
        valid_acceptors = acceptor_coords[dists <= dist_cutoff]
        
        for acc in valid_acceptors:
            vec_ha = acc - h_pos
            norm_ha = np.linalg.norm(vec_ha)
            if norm_ha == 0: continue
            
            # 计算 D-H...A 夹角 (通过向量点乘)
            cos_theta = np.dot(-vec_xh_norm, vec_ha / norm_ha)
            angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
            
            if angle >= angle_cutoff_deg:
                return True # 被强氢键锁定！
                
    return False

def generate_rotated_hydrogens(parent_pos: np.ndarray, 
                               x_pos: np.ndarray, 
                               element: str, 
                               env_coords: np.ndarray = None, 
                               clash_cutoff: float = 2.0,
                               num_samples: int = 72) -> list:
    """
    带有环境感知(位阻检查)的圆锥氢原子生成器。
    """
    axis = x_pos - parent_pos
    norm_axis = np.linalg.norm(axis)
    if norm_axis == 0: 
        return []
    u = axis / norm_axis 
    
    arbitrary_vec = np.array([1.0, 0.0, 0.0])
    if np.abs(np.dot(u, arbitrary_vec)) > 0.99:
        arbitrary_vec = np.array([0.0, 1.0, 0.0])
    
    v = np.cross(u, arbitrary_vec)
    v = v / np.linalg.norm(v)
    w = np.cross(u, v)
    
    # 获取键长与键角 (需确保你的 config 中有这些字典)
    bond_length = config.BOND_LENGTHS.get(element, 1.09) # 默认 O-H 或 N-H 键长
    theta_rad = np.radians(config.TETRAHEDRAL_ANGLE)
    
    h_proj_u = bond_length * np.cos(np.pi - theta_rad)
    h_radius = bond_length * np.sin(np.pi - theta_rad)
    
    h_positions = []
    angles = np.linspace(0, 2 * np.pi, num_samples, endpoint=False)
    
    for phi in angles:
        displacement = h_radius * np.cos(phi) * v + h_radius * np.sin(phi) * w
        h_pos = x_pos + (h_proj_u * u) + displacement
        
        # 💥 核心防线：空间位阻检查 (Clash Check)
        if env_coords is not None and len(env_coords) > 0:
            # 使用 numpy 向量化计算该虚拟氢原子与周围所有重原子的距离
            min_dist = np.min(np.linalg.norm(env_coords - h_pos, axis=1))
            if min_dist < clash_cutoff:
                continue # 发生物理碰撞，直接丢弃该构象！
                
        h_positions.append(h_pos)
        
    return h_positions