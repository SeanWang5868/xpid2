"""
topology.py
Manages chemical topology information via Gemmi.
Hybrid approach: Hardcoded standard residues (for speed/safety) + Dynamic MonLib lookup (for ligands).
"""
import gemmi
from typing import List, Dict, Optional, Set

# === 1. 安全底网：标准芳香族氨基酸定义 ===
# 无论外部库是否加载成功，这些必须能工作。
STANDARD_AROMATICS = {
    'PHE': [['CG', 'CD1', 'CE1', 'CZ', 'CE2', 'CD2']],
    'TYR': [['CG', 'CD1', 'CE1', 'CZ', 'CE2', 'CD2']],
    'TRP': [
        ['CG', 'CD1', 'NE1', 'CE2', 'CD2'],       # 5-ring
        ['CD2', 'CE2', 'CZ2', 'CH2', 'CZ3', 'CE3'] # 6-ring
    ],
    'HIS': [['CG', 'ND1', 'CE1', 'NE2', 'CD2']],
    # 常见核苷酸 (DNA/RNA)
    'ADE': [['N1', 'C6', 'N6', 'C5', 'N7', 'C8', 'N9', 'C4', 'N3', 'C2']], # 简化处理，通常 gemmi 会拆分，这里仅作示例
    'A':   [['N1', 'C6', 'N6', 'C5', 'N7', 'C8', 'N9', 'C4', 'N3', 'C2']],
}

class TopologyManager:
    def __init__(self, structure: gemmi.Structure = None, mon_lib_path: str = None):
        self.mon_lib_path = mon_lib_path
        # 缓存：res_name -> List[List[atom_name]]
        # 初始化时直接装入标准定义
        self.cache: Dict[str, List[List[str]]] = dict(STANDARD_AROMATICS)
        
        # 我们不再预加载整个库，而是按需加载 (Lazy Load)
        # 这种方式对于包含未知配体的 PDB 更安全

    def get_aromatic_rings(self, res_name: str) -> List[List[str]]:
        """
        获取残基的芳香环定义。
        """
        # 1. 命中缓存 (标准氨基酸或已处理过的配体)
        if res_name in self.cache:
            return self.cache[res_name]

        # 2. 未知残基，且没有提供库路径 -> 放弃
        if not self.mon_lib_path:
            self.cache[res_name] = [] # 标记为无环，避免重复检查
            return []

        # 3. 尝试从库中加载该残基的定义
        rings = self._load_and_analyze_ligand(res_name)
        
        # 4. 更新缓存
        self.cache[res_name] = rings
        return rings

    def _load_and_analyze_ligand(self, res_name: str) -> List[List[str]]:
        """
        尝试加载单个残基的定义并分析环。
        """
        try:
            # gemmi.read_monomer_lib 可以只读一个残基，加上 ignore_missing 防止报错
            # 注意：这会返回一个新的 MonLib 对象，虽然频繁 IO 有一点开销，
            # 但对于只有几个不同配体的 PDB，这比加载整个库要稳健得多。
            temp_lib = gemmi.read_monomer_lib(self.mon_lib_path, [res_name], ignore_missing=True)
            
            # 检查是否成功加载
            if res_name not in temp_lib.monomers:
                return []
            
            chemcomp = temp_lib.monomers[res_name]
            return self._find_rings_in_chemcomp(chemcomp)
            
        except Exception as e:
            # 任何加载错误都只影响这一个残基，不影响全局
            # print(f"[Debug] Failed to load definition for {res_name}: {e}")
            return []

    def _find_rings_in_chemcomp(self, chemcomp: gemmi.ChemComp) -> List[List[str]]:
        """
        使用 DFS 在化学组件中寻找 5元和 6元环。
        """
        # 1. 构建邻接表
        adj = {}
        atom_elements = {} # 用于后续元素检查
        
        for atom in chemcomp.atoms:
            adj[atom.id] = []
            atom_elements[atom.id] = atom.el.name.upper()

        for bond in chemcomp.bonds:
            # 兼容性处理：id1 可能是对象也可能是字符串
            a1 = bond.id1.atom if hasattr(bond.id1, 'atom') else str(bond.id1)
            a2 = bond.id2.atom if hasattr(bond.id2, 'atom') else str(bond.id2)
            if a1 in adj and a2 in adj:
                adj[a1].append(a2)
                adj[a2].append(a1)

        # 2. 搜索环
        found_rings = []
        visited_fingerprints = set() # 用 frozenset(atoms) 去重

        def dfs(start, curr, path, depth):
            if depth > 6: return
            
            for neighbor in adj.get(curr, []):
                if neighbor == start:
                    # 找到闭环
                    if depth in [5, 6]:
                        # 去重：不管顺序，只要组成原子一样就算同一个环
                        fingerprint = frozenset(path)
                        if fingerprint not in visited_fingerprints:
                            visited_fingerprints.add(fingerprint)
                            found_rings.append(list(path))
                    continue
                
                if neighbor not in path:
                    dfs(start, neighbor, path + [neighbor], depth + 1)

        # 遍历每个原子作为起点
        for atom_id in adj.keys():
            dfs(atom_id, atom_id, [atom_id], 1)

        # 3. 过滤非芳香环 (简单规则：必须只包含 C, N, O, S)
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