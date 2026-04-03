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


        max_attempts = 10  # 最多允许自动切除 10 个坏死残基
        for attempt in range(max_attempts):
            try:
                # 尝试构建拓扑与加氢
                gemmi.prepare_topology(structure, monlib, model_index=0, h_change=h_change_val, reorder=False, ignore_unknown_links=True)
                break  # 如果毫无报错，直接跳出重试循环！
                
            except Exception as topo_err:
                err_msg = str(topo_err)
                
                # 策略 1：如果是未知 Link 报错，清空连接信息并重试
                if "link" in err_msg.lower():
                    logger.warning(f"  -> Bad explicit link detected. Clearing connections and retrying...")
                    structure.connections.clear()
                    continue  # 继续下一次尝试
                    
                # 策略 2：几何扭曲报错 (精准切除)
                # 解析报错格式，例如: "bonded to V/NAG 2/O4 failed"
                match = re.search(r"bonded to ([^/]+)/([^ ]+) ([^/]+)/([^ ]+) failed", err_msg)
                if match:
                    bad_chain = match.group(1)   # 例如 'V'
                    bad_seqid = match.group(3)   # 例如 '2'
                    
                    removed = False
                    for model in structure:
                        for chain in model:
                            if chain.name == bad_chain:
                                for i in range(len(chain)):
                                    # 匹配序列号，剥离空格
                                    if str(chain[i].seqid).strip() == bad_seqid.strip():
                                        del chain[i]  # 核心：直接把这个残基从链中删除！
                                        removed = True
                                        logger.warning(f"  -> Removed twisted residue {bad_chain}/{bad_seqid} to save the rest of the structure.")
                                        break
                            if removed: break
                        if removed: break
                        
                    if removed:
                        continue  # 成功切除后，重新启动加氢进程！
                
                # 策略 3：如果正则没匹配上，或者超过最大重试次数
                logger.warning(f"  -> Topology incomplete after {attempt} retries: {err_msg}.")
                break  # 停止重试，带着现有（部分加氢）的残基继续前进
        # ---------------------------------------------------------

        structure.setup_cell_images()
        return structure
        
    except Exception as e:
        logger.error(f"Critical error in prep: {e}")
        # ⚠️ 绝对不能 return None！如果全盘崩溃，至少返回原始结构，保证主流程不中断
        return structure