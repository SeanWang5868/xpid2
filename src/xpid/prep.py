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

        # ---------------------------------------------------------
        # 🌟 修复区：增加错误重试机制 (Fallback Mechanism)
        # ---------------------------------------------------------
        try:
            # 第一次尝试：保留所有原始连接信息进行拓扑构建
            gemmi.prepare_topology(structure, monlib, model_index=0, h_change=h_change_val, reorder=False, ignore_unknown_links=True)
        except Exception as topo_err:
            err_msg = str(topo_err).lower()
            # 扩大捕获范围：只要报错信息里包含 "link" 这个词，统统进入抢救流程
            if "link" in err_msg:
                # 打印具体的报错原因，方便你查看
                logger.warning(f"  -> Bad explicit link detected: '{str(topo_err)}'. Clearing connections and retrying...")
                # 清除 CIF 文件中携带的所有 explicit links
                structure.connections.clear()
                
                # 第二次尝试：依赖距离和标准氨基酸字典重新构建拓扑加氢
                gemmi.prepare_topology(structure, monlib, model_index=0, h_change=h_change_val, reorder=False, ignore_unknown_links=True)
            else:
                # 如果是其他严重的底层错误，抛出交给外层处理
                raise topo_err
        # ---------------------------------------------------------

        structure.setup_cell_images()
        return structure
        
    except Exception as e:
        logger.error(f"Topology failed: {e}")
        return None