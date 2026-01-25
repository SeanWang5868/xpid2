"""
cli.py
Command Line Interface for xpid.
Handles argument parsing, file discovery (Direct/Recursive/Mirror), and process orchestration.
Supports CSV, JSON, and Parquet output formats.
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

def gather_inputs(inputs: List[str], pdb_list: str, pdb_mirror: str) -> List[Path]:
    """
    Combines direct inputs and mirror lookups into a final list of file paths.
    """
    final_files = set()
    
    # 1. Process Direct Inputs (Files or Directories)
    if inputs:
        pattern = re.compile(r'^[a-zA-Z0-9]{4}\.(cif|pdb)(\.gz)?$', re.IGNORECASE)
        for inp in inputs:
            path = Path(inp)
            if path.is_file(): 
                final_files.add(path.resolve())
            elif path.is_dir():
                for p in path.rglob("*"):
                    if p.is_file() and pattern.match(p.name): 
                        final_files.add(p.resolve())

    # 2. Process PDB List + Mirror
    if pdb_list:
        if not pdb_mirror:
            logger.error("Argument --pdb-mirror is REQUIRED when using --pdb-list.")
            sys.exit(1)
        
        mirror_root = Path(pdb_mirror).resolve()
        if not mirror_root.exists():
            logger.error(f"Mirror directory not found: {mirror_root}")
            sys.exit(1)

        codes = parse_pdb_list_file(Path(pdb_list))
        logger.info(f"Parsed {len(codes)} PDB codes from list.")
        
        found_count = 0
        missing_codes = []

        for code in codes:
            fpath = resolve_mirror_path(mirror_root, code)
            if fpath:
                final_files.add(fpath)
                found_count += 1
            else:
                missing_codes.append(code)
        
        if missing_codes:
            logger.warning(f"Could not find {len(missing_codes)} PDBs in mirror (e.g., {', '.join(missing_codes[:5])}...)")

    return sorted(list(final_files))

# --- Output Streaming ---

class ResultStreamer:
    def __init__(self, output_path: Path, file_type: str, verbose: bool):
        self.output_path = output_path
        self.file_type = file_type.lower()
        self.verbose = verbose
        self.file_handle = None
        self.csv_writer = None
        self.parquet_writer = None # Handle for ParquetWriter
        self.is_first_chunk = True

        # Check dependencies early if parquet is selected
        if self.file_type == 'parquet':
            try:
                import pandas
                import pyarrow
                import pyarrow.parquet
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
        
        # Note: Parquet file handle is managed by pyarrow internally, opened on first chunk
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
            self._write_csv(results)
        elif self.file_type == 'json':
            self._write_json(results)
        elif self.file_type == 'parquet':
            self._write_parquet(results)

    def _write_csv(self, results):
        if self.is_first_chunk:
            keys = list(results[0].keys())
            fieldnames = keys if self.verbose else [k for k in SIMPLE_COLS if k in keys]
            self.csv_writer = csv.DictWriter(self.file_handle, fieldnames=fieldnames, extrasaction='ignore')
            self.csv_writer.writeheader()
            self.is_first_chunk = False
        self.csv_writer.writerows(results)

    def _write_json(self, results):
        keys_to_keep = None
        if not self.verbose and self.is_first_chunk:
            sample_keys = results[0].keys()
            keys_to_keep = set([k for k in SIMPLE_COLS if k in sample_keys])
        
        for item in results:
            clean_item = item
            if not self.verbose:
                if keys_to_keep is None:
                    keys_to_keep = set([k for k in SIMPLE_COLS if k in item.keys()])
                clean_item = {k: v for k, v in item.items() if k in keys_to_keep}
            
            if not self.is_first_chunk: self.file_handle.write(',\n')
            else: self.is_first_chunk = False
            json.dump(clean_item, self.file_handle, indent=2)

    def _write_parquet(self, results):
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq

        df = pd.DataFrame(results)
        
        # Filter columns if not verbose
        if not self.verbose:
            existing_cols = [c for c in SIMPLE_COLS if c in df.columns]
            df = df[existing_cols]

        table = pa.Table.from_pandas(df)

        if self.is_first_chunk:
            # Initialize the writer with the schema from the first chunk
            self.parquet_writer = pq.ParquetWriter(self.output_path, table.schema)
            self.is_first_chunk = False
        
        # Write this chunk as a Row Group
        if self.parquet_writer:
            self.parquet_writer.write_table(table)

# --- Worker Function ---

def process_one_file(args_packet):
    # Unpack including use_cone flag
    filepath, mon_lib, ftype, hmode, output_dir, separate_mode, filters, verbose, model_mode, use_cone = args_packet
    
    pdb_name = filepath.stem.split('.')[0] 
    match = re.search(r'([0-9][a-zA-Z0-9]{3})', filepath.name)
    if match:
        pdb_name = match.group(1).lower()

    try:
        try: structure = gemmi.read_structure(str(filepath))
        except Exception as e: return (f"Read Error ({pdb_name}): {e}", 0, None, None)

        structure = prep.add_hydrogens_memory(structure, mon_lib, h_change_val=hmode)
        if not structure: return (f"AddH Failed ({pdb_name})", 0, None, None)

        results = core.detect_interactions_in_structure(
            structure, pdb_name, 
            filter_pi=filters['pi'], filter_donor=filters['donor'],
            filter_donor_atom=filters['donor_atom'], model_mode=model_mode,
            use_cone=use_cone 
        )
        count = len(results)
        if count > 0:
            if separate_mode:
                out_path = Path(output_dir) / f"{pdb_name}_xpid.{ftype}"
                with ResultStreamer(out_path, ftype, verbose) as s: s.write_chunk(results)
                return (None, count, None, str(out_path.parent))
            else:
                return (None, count, results, None)
        else: return (None, 0, None, None)
    except Exception as e:
        return (f"Critical Error ({pdb_name}): {e}", 0, None, None)

# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(prog="xpid", description="xpid: XH-/pi detector with Dual-Track logic.")
    
    # Input Group
    input_group = parser.add_argument_group("Input Sources")
    input_group.add_argument('inputs', nargs='*', help="Direct input file(s) or folder(s).")
    input_group.add_argument('--pdb-list', type=str, help="Text file containing PDB codes (comma/newline separated).")
    input_group.add_argument('--pdb-mirror', type=str, help="Root path to local PDB mirror (structure: <mid>/<pdb>.cif.gz).")

    # Output Group
    out_group = parser.add_argument_group("Output Options")
    out_group.add_argument('--separate', action='store_true', help="Separate output files for each PDB.")
    out_group.add_argument('--out-dir', type=str, help="Directory for output files.")
    out_group.add_argument('--output-name', type=str, default='xpid_results', help="Filename for merged output.")
    # UPDATED: Added 'parquet' to choices
    out_group.add_argument('--file-type', default='json', choices=['json', 'csv', 'parquet'], help="Output format.")
    out_group.add_argument('-v', '--verbose', action='store_true', help="Include detailed geometric columns.")
    out_group.add_argument('--log', action='store_true', help="Save run log to file.")

    # Processing Options
    proc_group = parser.add_argument_group("Processing Options")
    proc_group.add_argument('--h-mode', type=int, default=4, help="Hydrogen handling mode (0-5). Default: 4.")
    proc_group.add_argument('--jobs', type=int, default=1, help="Number of CPU cores to use.")
    proc_group.add_argument('--model', type=str, default="0", help="Model index to analyze (or 'all').")
    proc_group.add_argument('--cone', action='store_true', help="Enable implicit Cone logic for rotatable groups.")
    
    # Filtering & Config
    filter_group = parser.add_argument_group("Filters & Config")
    filter_group.add_argument('--mon-lib', type=str, help="Path to custom Monomer Library.")
    filter_group.add_argument('--set-mon-lib', type=str, help="Permanently set default Monomer Library path.")
    filter_group.add_argument('--pi-res', type=str, help="Filter: Pi residues (e.g. TRP,TYR).")
    filter_group.add_argument('--donor-res', type=str, help="Filter: Donor residues (e.g. LYS,ARG).")
    filter_group.add_argument('--donor-atom', type=str, help="Filter: Donor atoms (e.g. N,O,C).")

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
    files = gather_inputs(args.inputs, args.pdb_list, args.pdb_mirror)
    
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
    # Convert file_type to lower case for consistency
    ftype_arg = args.file_type.lower()
    tasks = [(f, mon_lib_path, ftype_arg, args.h_mode, str(output_dir), args.separate, filters, args.verbose, args.model, args.cone) for f in files]
    
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

        # Multiprocessing Loop
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