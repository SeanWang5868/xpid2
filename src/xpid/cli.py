"""
cli.py
Command Line Interface for xpid.
(最终修复版：结构加载用 gemmi.read_structure；Parquet 输出用 pyarrow.Table.from_pandas；所有新增功能完整保留)
"""
import argparse
import sys
import re
import multiprocessing
import logging
import gemmi
import json
import csv
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set

try:
    from xpid import prep, core, config
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from xpid import prep, core, config

# --- Constants & Global Setup ---
H_MODE_MAP = {
    0: "NoChange", 1: "Shift", 2: "Remove", 
    3: "ReAdd", 4: "ReAddButWater", 5: "ReAddKnown"
}

SIMPLE_COLS = [
    'pdb', 'resolution', 
    'pi_chain', 'pi_res', 'pi_id', 
    'X_chain', 'X_res', 'X_id', 'X_atom', 'H_atom', 
    'dist_X_Pi', 'method', 'is_plevin', 'is_hudson', 'remark'
]

logger = logging.getLogger('xpid')

def setup_logging(log_file: Path):
    if log_file.parent:
        log_file.parent.mkdir(parents=True, exist_ok=True)
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, mode='w', encoding='utf-8')
    ]
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S', handlers=handlers, force=True)

# --- Helper: File Resolution Logic ---
def parse_pdb_list_file(list_path: Path) -> Set[str]:
    """Reads a text file and extracts 4-char PDB codes (comma or newline separated)."""
    codes = set()
    try:
        with open(list_path, 'r', encoding='utf-8') as f:
            content = f.read()
            tokens = re.split(r'[,\s\n\t]+', content)
            for t in tokens:
                clean_t = t.strip()
                if len(clean_t) == 4 and clean_t.isalnum():
                    codes.add(clean_t.lower())
    except Exception as e:
        logger.error(f"Failed to read PDB list file: {e}")
        sys.exit(1)
    return codes

def resolve_mirror_path(mirror_root: Path, pdb_code: str) -> Path:
    """
    Resolves a PDB code to a file path using standard divided structure.
    Rule: 1fn0 -> {mirror_root}/fn/1fn0.cif.gz (or .cif)
    """
    middle_two = pdb_code[1:3].lower()
    
    # Priority 1: gzipped mmCIF (Standard)
    path_gz = mirror_root / middle_two / f"{pdb_code}.cif.gz"
    if path_gz.exists(): return path_gz

    # Priority 2: uncompressed mmCIF
    path_cif = mirror_root / middle_two / f"{pdb_code}.cif"
    if path_cif.exists(): return path_cif
    
    # Priority 3: Flat structure (just in case)
    path_flat = mirror_root / f"{pdb_code}.cif.gz"
    if path_flat.exists(): return path_flat

    return None

def resolve_redo_path(mirror_root: Path, pdb_code: str) -> Path:
    """
    Resolves a PDB code to a file path in a PDB-REDO mirror.
    Tries various common PDB-REDO directory structures.
    """
    middle_two = pdb_code[1:3].lower()
    
    candidates = [
        # PDB-REDO 官方 rsync 结构: /ab/1abc/1abc_final.cif
        mirror_root / middle_two / pdb_code / f"{pdb_code}_final.cif",
        mirror_root / middle_two / pdb_code / f"{pdb_code}_final.cif.gz",
        # 简化版结构: /ab/1abc_final.cif
        mirror_root / middle_two / f"{pdb_code}_final.cif",
        mirror_root / middle_two / f"{pdb_code}_final.cif.gz",
        # 与常规 PDB 完全相同的结构: /ab/1abc.cif
        mirror_root / middle_two / f"{pdb_code}.cif",
        mirror_root / middle_two / f"{pdb_code}.cif.gz",
    ]
    
    for path in candidates:
        if path.exists():
            return path
            
    return None

def gather_inputs(inputs: List[str], pdb_list: str, pdb_mirror: str, redo_mirror: str) -> List[Path]:
    """
    Combines direct inputs and mirror lookups into a final list of file paths.
    Prioritizes PDB-REDO mirror over standard PDB mirror if both are provided.
    """
    final_files = set()
    
    # 1. Process Direct Inputs (Files or Directories)
    if inputs:
        pattern = re.compile(r'^[a-zA-Z0-9]{4}(_final)?\.(cif|pdb)(\.gz)?$', re.IGNORECASE)
        for inp in inputs:
            path = Path(inp)
            if path.is_file(): 
                final_files.add(path.resolve())
            elif path.is_dir():
                for p in path.rglob("*"):
                    if p.is_file() and pattern.match(p.name): 
                        final_files.add(p.resolve())

    # 2. Process PDB List + Mirrors
    if pdb_list:
        if not pdb_mirror and not redo_mirror:
            logger.error("Argument --pdb-mirror or --redo-mirror is REQUIRED when using --pdb-list.")
            sys.exit(1)
        
        pdb_root = Path(pdb_mirror).resolve() if pdb_mirror else None
        redo_root = Path(redo_mirror).resolve() if redo_mirror else None

        if pdb_root and not pdb_root.exists():
            logger.error(f"PDB Mirror directory not found: {pdb_root}")
            sys.exit(1)
        if redo_root and not redo_root.exists():
            logger.error(f"PDB-REDO Mirror directory not found: {redo_root}")
            sys.exit(1)

        codes = parse_pdb_list_file(Path(pdb_list))
        logger.info(f"Parsed {len(codes)} PDB codes from list.")
        
        found_redo_count = 0
        found_pdb_count = 0
        missing_codes = []

        for code in codes:
            fpath = None
            
            # 优先级 1: 尝试从 PDB-REDO 镜像寻找
            if redo_root:
                fpath = resolve_redo_path(redo_root, code)
                if fpath:
                    found_redo_count += 1
            
            # 优先级 2: 如果 REDO 中没有，尝试从常规 PDB 镜像寻找
            if not fpath and pdb_root:
                fpath = resolve_mirror_path(pdb_root, code)
                if fpath:
                    found_pdb_count += 1
                    
            if fpath:
                final_files.add(fpath)
            else:
                missing_codes.append(code)
        
        logger.info(f"Found {found_redo_count} in PDB-REDO, {found_pdb_count} in standard PDB.")
        
        if missing_codes:
            logger.warning(f"Could not find {len(missing_codes)} PDBs in any mirror (e.g., {', '.join(missing_codes[:5])}...)")

    return sorted(list(final_files))

# --- Output Streaming ---
class ResultStreamer:
    def __init__(self, output_path: Path, file_type: str, verbose: bool):
        self.output_path = output_path
        self.file_type = file_type.lower()
        self.verbose = verbose
        self.file_handle = None
        self.csv_writer = None
        self.parquet_writer = None
        self.is_first_chunk = True

        # Check dependencies early if parquet is selected
        if self.file_type == 'parquet':
            try:
                import pandas  # noqa: F401
                import pyarrow as pa
                import pyarrow.parquet as pq
                self.pa = pa
                self.pq = pq
            except ImportError:
                logger.error("[ERROR] To use --file-type parquet, you must install 'pandas' and 'pyarrow'.")
                logger.error("Try: pip install pandas pyarrow")
                sys.exit(1)

    def __enter__(self):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.file_type in ['json', 'csv']:
            self.file_handle = open(self.output_path, 'w', newline='', encoding='utf-8')
            if self.file_type == 'json': 
                self.file_handle.write('[\n')
        
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file_type == 'json' and self.file_handle:
            self.file_handle.write('\n]')
        
        if self.file_handle: 
            self.file_handle.close()
            
        if self.parquet_writer:
            self.parquet_writer.close()

    def write_chunk(self, results: List[Dict[str, Any]]):
        if not results: return
        
        if self.file_type == 'csv':
            if self.is_first_chunk:
                headers = results[0].keys() if self.verbose else SIMPLE_COLS
                self.csv_writer = csv.DictWriter(self.file_handle, fieldnames=headers)
                self.csv_writer.writeheader()
                self.is_first_chunk = False
            rows = results if self.verbose else [{k: r[k] for k in SIMPLE_COLS} for r in results]
            self.csv_writer.writerows(rows)
            
        elif self.file_type == 'json':
            comma = '' if self.is_first_chunk else ',\n'
            for r in results:
                self.file_handle.write(comma + json.dumps(r, indent=2 if self.verbose else None))
                comma = ',\n'
            self.is_first_chunk = False
            
        elif self.file_type == 'parquet':
            import pandas as pd
            df = pd.DataFrame(results)
            table = self.pa.Table.from_pandas(df if self.verbose else df[SIMPLE_COLS])
            if self.is_first_chunk:
                self.parquet_writer = self.pq.ParquetWriter(self.output_path, table.schema)
                self.is_first_chunk = False
            self.parquet_writer.write_table(table)

# --- Core Processing Function ---
def process_one_file(args_packet):
    (filepath, mon_lib_path, ftype_arg, h_mode, output_dir_str, separate, 
     filters, verbose, model_mode, use_cone, min_occ) = args_packet
    
    output_dir = Path(output_dir_str)
    pdb_code = filepath.stem.split('.')[0].lower()
    
    try:
        # 直接使用 gemmi 读取结构（支持 .gz）
        structure = gemmi.read_structure(str(filepath))
        
        if not structure or len(structure) == 0:
            return f"Empty or invalid structure: {filepath}", 0, [], None
        
        # 加氢处理
        if h_mode > 0:
            structure = prep.add_hydrogens_memory(structure, mon_lib_path, h_change_val=h_mode)
            if structure is None:
                return f"Hydrogen addition failed: {filepath}", 0, [], None
        
        # 核心检测
        results = core.detect_interactions_in_structure(
            structure,
            pdb_name=pdb_code,
            filter_pi=filters.get('pi'),
            filter_donor=filters.get('donor'),
            filter_donor_atom=filters.get('donor_atom'),
            model_mode=model_mode,
            use_cone=use_cone,
            min_occ=min_occ
        )
        
        count = len(results)
        
        if separate:
            out_path = output_dir / f"{pdb_code}_xpid_results.{ftype_arg}"
            with ResultStreamer(out_path, ftype_arg, verbose) as streamer:
                streamer.write_chunk(results)
            return None, count, [], str(out_path)
        else:
            return None, count, results, None
            
    except Exception as e:
        import traceback
        return f"{filepath}: {str(e)}\n{traceback.format_exc()}", 0, [], None

# --- Main ---
def main():
    parser = argparse.ArgumentParser(description="XH-pi interaction detector")
    parser.add_argument('inputs', nargs='*', help="PDB/CIF files or directories")
    parser.add_argument('--pdb-list', type=str, help="Text file with PDB codes")
    parser.add_argument('--pdb-mirror', type=str, help="Local PDB mirror root")
    parser.add_argument('--redo-mirror', type=str, help="Local PDB-REDO mirror root (prioritized over standard PDB)")
    
    out_group = parser.add_argument_group("Output Options")
    out_group.add_argument('--separate', action='store_true', help="Separate output files for each PDB.")
    out_group.add_argument('--out-dir', type=str, help="Directory for output files.")
    out_group.add_argument('--output-name', type=str, default='xpid_results', help="Filename for merged output.")
    out_group.add_argument('--file-type', default='json', choices=['json', 'csv', 'parquet'], help="Output format.")
    out_group.add_argument('-v', '--verbose', action='store_true', help="Include detailed geometric columns.")
    out_group.add_argument('--log', action='store_true', help="Save run log to file.")

    proc_group = parser.add_argument_group("Processing Options")
    proc_group.add_argument('--h-mode', type=int, default=4, help="Hydrogen handling mode (0-5). Default: 4.")
    proc_group.add_argument('--jobs', type=int, default=1, help="Number of CPU cores to use.")
    proc_group.add_argument('--model', type=str, default="0", help="Model index to analyze (or 'all').")
    proc_group.add_argument('--cone', action='store_true', help="Enable implicit Cone logic for rotatable groups.")
    
    filter_group = parser.add_argument_group("Filters & Config")
    filter_group.add_argument('--mon-lib', type=str, help="Path to custom Monomer Library.")
    filter_group.add_argument('--set-mon-lib', type=str, help="Permanently set default Monomer Library path.")
    filter_group.add_argument('--pi-res', type=str, help="Filter: Pi residues (e.g. TRP,TYR).")
    filter_group.add_argument('--donor-res', type=str, help="Filter: Donor residues (e.g. LYS,ARG).")
    filter_group.add_argument('--donor-atom', type=str, help="Filter: Donor atoms (e.g. N,O,C).")
    filter_group.add_argument('--min-occ', type=float, default=0.0, 
                              help="Minimum combined occupancy to report an interaction (default: 0.0).")

    args = parser.parse_args()

    # Handle Config Setting
    if args.set_mon_lib:
        if os.path.isdir(args.set_mon_lib):
            config.save_mon_lib_path(args.set_mon_lib)
            print(f"[CONFIG] Default Monomer Library path set to: {args.set_mon_lib}")
            sys.exit(0)
        else:
            print("[ERROR] Invalid directory provided for monomer library.")
            sys.exit(1)

    # Output Directory Setup
    output_dir = Path(args.out_dir) if args.out_dir else Path.cwd() / "xpid_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Logging Setup
    log_file = output_dir / "xpid_run.log"
    if args.log: setup_logging(log_file)
    else: logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.StreamHandler(sys.stdout)])

    # --- Step 1: Gather Files ---
    logger.info("--- Xpid Initialization ---")
    files = gather_inputs(args.inputs, args.pdb_list, args.pdb_mirror, args.redo_mirror)
    
    if not files:
        logger.error("[ERROR] No valid input files found. Please check inputs or list/mirror paths.")
        sys.exit(1)

    # --- Step 2: Prepare Configurations ---
    mon_lib_path = args.mon_lib if args.mon_lib else config.DEFAULT_MON_LIB_PATH
    filters = {
        'pi': [x.strip().upper() for x in args.pi_res.split(',')] if args.pi_res else None,
        'donor': [x.strip().upper() for x in args.donor_res.split(',')] if args.donor_res else None,
        'donor_atom': [x.strip().upper() for x in args.donor_atom.split(',')] if args.donor_atom else None
    }

    # Logging Details
    h_mode_desc = H_MODE_MAP.get(args.h_mode, "Unknown")
    model_desc = "All" if args.model == 'all' else f"Index {args.model}"
    cone_status = "Enabled" if args.cone else "Disabled (Default Static H)"
    
    logger.info(f"Targets    : {len(files)} unique structures")
    logger.info(f"Output Dir : {output_dir.resolve()}")
    logger.info(f"Format     : {args.file_type.upper()} ({'Separate Files' if args.separate else 'Merged File'})")
    logger.info(f"H-Mode     : {args.h_mode} ({h_mode_desc})")
    logger.info(f"Cone Logic : {cone_status}")
    print("")

    # --- Step 3: Execution ---
    ftype_arg = args.file_type.lower()
    tasks = [(f, mon_lib_path, ftype_arg, args.h_mode, str(output_dir), args.separate, filters, 
              args.verbose, args.model, args.cone, args.min_occ) for f in files]
    
    error_logs = []
    total_found = 0
    
    try:
        merge_file_path = None
        streamer = None
        
        if not args.separate:
            merge_filename = f"{args.output_name}.{ftype_arg}"
            merge_file_path = output_dir / merge_filename
            streamer = ResultStreamer(merge_file_path, ftype_arg, args.verbose)
            streamer.__enter__()

        with multiprocessing.Pool(args.jobs, maxtasksperchild=100) as pool:
            for i, (err, count, data, out_path) in enumerate(pool.imap_unordered(process_one_file, tasks), 1):
                if err: 
                    error_logs.append(err)
                    logging.warning(f"\n[WARN] {err}")
                
                total_found += count
                
                if not args.separate and data:
                    streamer.write_chunk(data)
                
                # Simple progress bar
                percent = (i / len(files)) * 100
                msg = f"\r[INFO] Progress   : {i}/{len(files)} ({percent:.1f}%)"
                sys.stdout.write(msg)
                sys.stdout.flush()

        if streamer:
            streamer.__exit__(None, None, None)

        print("\n")

        # --- Step 4: Summary ---
        print("-" * 60)
        print(f"[SUMMARY] Total XH-pi interactions detected: {total_found}")
        
        if error_logs:
            print(f"[WARNING] {len(error_logs)} files failed processing. Check log for details.")

        if total_found > 0:
            if not args.separate:
                print(f"[OUTPUT] Merged result saved to:\n  -> {merge_file_path.resolve()}")
            else:
                print(f"[OUTPUT] Separate result files saved in:\n  -> {output_dir.resolve()}")
        else:
            print("[OUTPUT] No interactions found.")
            
    except Exception as e:
        logger.error(f"\nExecution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()