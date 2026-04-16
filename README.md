# Xpid

[![PyPI version](https://img.shields.io/pypi/v/xpid)](https://pypi.org/project/xpid/)
[![Python 3.9+](https://img.shields.io/pypi/pyversions/xpid)](https://pypi.org/project/xpid/)
[![License](https://img.shields.io/github/license/SeanWang5868/xpid2)](https://github.com/SeanWang5868/xpid2/blob/main/LICENSE)

**Xpid** is a [Gemmi](https://gemmi.readthedocs.io/)-based tool for detecting XH&#8211;&#960; interactions in protein structures from PDB/mmCIF files.

- **Source Code**: [https://github.com/SeanWang5868/xpid2](https://github.com/SeanWang5868/xpid2)
- **PyPI**: [https://pypi.org/project/xpid/](https://pypi.org/project/xpid/)

---

## Installation

Requires **Python 3.9+**.

```bash
pip install xpid
```

## Configuration

The detection of XH&#8211;&#960; interactions depends on the position of H atoms.
To add hydrogen atoms before detection, specify the path to a monomer library (e.g. CCP4 monomer library):

```bash
xpid --set-mon-lib /path/to/monomers
```

## Quick Start

Scan a directory or a single PDB/mmCIF file:

```bash
xpid 1abc.cif --file-type csv
```

> **Output**: `./xpid_output/xpid_results.json`

Use a PDB code list with a local mirror:

```bash
xpid --pdb-list codes.txt --pdb-mirror /path/to/pdb/mirror
```

Use PDB-REDO structures (prioritized over standard PDB):

```bash
xpid --pdb-list codes.txt --redo-mirror /path/to/pdb-redo/mirror
```

Export results in different formats:

```bash
xpid ./data --file-type csv
xpid ./data --file-type parquet
```

## Python API

Xpid can also be used as a library:

```python
from xpid import detect

results = detect("structure.cif", mon_lib_path="/path/to/monomers")
for hit in results:
    print(hit["pdb"], hit["pi_res"], hit["X_res"], hit["dist_X_Pi"])
```

## Geometric Criteria

**Definitions**: C&#960; (Ring Centroid), **n** (Ring Normal), X (Donor Heavy Atom), Xp (Projection of X onto the &#960; plane), H (Hydrogen).

### [Hudson System](https://doi.org/10.1021/jacs.5b08424)

| Parameter | Threshold |
| :--- | :--- |
| d(X&#8211;C&#960;) | &le; 4.5 &#8491; |
| &ang;(X&#8211;H&#8211;**n**) | &le; 40&deg; |
| d(Xp&#8211;C&#960;) | &le; 1.6 &#8491; (His, Trp-A); &le; 2.0 &#8491; (Phe, Trp-B, Tyr) |

### [Plevin System](https://doi.org/10.1038/nchem.650)

| Parameter | Threshold |
| :--- | :--- |
| d(X&#8211;C&#960;) | &le; 4.5 &#8491; |
| &ang;(X&#8211;H&#8211;C&#960;) | &ge; 120&deg; |
| &ang;(X&#8211;C&#960;&#8211;**n**) | &ge; 25&deg; |

## Command-Line Options

### Input

| Argument | Description |
| :--- | :--- |
| `inputs` | PDB/CIF file(s) or directory path(s). |
| `--pdb-list` | Text file containing PDB codes (comma or newline separated). |
| `--pdb-mirror` | Path to a local PDB mirror (divided layout). |
| `--redo-mirror` | Path to a local PDB-REDO mirror (prioritized over `--pdb-mirror`). |

### Output

| Argument | Description |
| :--- | :--- |
| `--out-dir` | Custom output directory (default: `./xpid_output`). |
| `--output-name` | Filename for merged output (default: `xpid_results`). |
| `--separate` | Write separate output files per PDB (default: merged). |
| `--file-type` | Output format: `json` (default), `csv`, or `parquet`. |
| `-v`, `--verbose` | Include detailed geometric metrics in output. |
| `--log` | Save run log to file. |

### Processing

| Argument | Description |
| :--- | :--- |
| `--jobs N` | Number of CPU cores (default: 1). |
| `--h-mode N` | Hydrogen handling mode: 0=NoChange, 1=Shift, 2=Remove, 3=ReAdd, 4=ReAddButWater (default), 5=ReAddKnown. |
| `--model ID` | Model index to analyze (default: `0`; use `all` for NMR ensembles). |
| `--cone` | Enable implicit cone logic for rotatable groups. |
| `--sym-contacts` | Detect interactions across crystallographic symmetry mates. |
| `--include-water` | Include water molecules as potential donors. |
| `--max-b N` | Maximum B-factor filter (default: 0 = no filter). |

### Filters

| Argument | Description |
| :--- | :--- |
| `--pi-res` | Limit acceptor residues (e.g., `TRP,TYR`). |
| `--donor-res` | Limit donor residues (e.g., `LYS,ARG`). |
| `--donor-atom` | Limit donor element types (e.g., `N,O,C`). |
| `--min-occ N` | Minimum combined occupancy to report (default: 0.0). |
| `--mon-lib` | Path to a custom monomer library for this run. |
| `--set-mon-lib` | Permanently save a default monomer library path. |

## Output Data

**Simple Mode** (default)

- PDB ID, Resolution
- Chain, Residue Name, Residue ID for both donor (X) and acceptor (&#960;) residues
- Distance d(X&#8211;C&#960;)
- Hudson / Plevin classification flags
- Symmetry operation (if applicable)

**Verbose Mode** (`-v`)

All simple fields, plus:

- **Secondary structure**: Type (H/G/I/E/C) and region IDs
- **Coordinates**: x, y, z for &#960;-center and X-atom
- **Geometric parameters**: &ang;(X&#8211;H&#8211;**n**), &ang;(X&#8211;H&#8211;C&#960;), &ang;(X&#8211;C&#960;&#8211;**n**), d(Xp&#8211;C&#960;)
- **B-factors**: Average B-factor for ring atoms and X-atom

## Dependencies

- [gemmi](https://gemmi.readthedocs.io/) &mdash; macromolecular crystallography library
- [numpy](https://numpy.org/) &mdash; numerical computing
- [pandas](https://pandas.pydata.org/) + [pyarrow](https://arrow.apache.org/docs/python/) &mdash; required only for Parquet output

---

## Contact

**Sean Wang** &mdash; sean.wang@york.ac.uk

York Structural Biology Laboratory (YSBL), University of York
