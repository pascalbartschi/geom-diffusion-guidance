from __future__ import annotations

from pathlib import Path
import argparse
import subprocess
import re
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback
from typing import List, Dict
from itertools import combinations

from string import ascii_uppercase

import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
from rdkit import Chem
import gemmi

from posebusters.modules.intermolecular_distance import check_intermolecular_distance
from posebusters import PoseBusters

from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns

import pyKVFinder

from scipy.stats import chisquare, kruskal, f_oneway, wasserstein_distance, chi2_contingency, pearsonr, kendalltau
# https://docs.scipy.org/doc/scipy-1.16.1/reference/generated/scipy.stats.chisquare.html
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f_oneway.html
from statsmodels.stats.contingency_tables import cochrans_q
# https://www.statsmodels.org/dev/generated/statsmodels.stats.contingency_tables.cochrans_q.html
from scikit_posthocs import posthoc_dunn
# https://scikit-posthocs.readthedocs.io/en/latest/generated/scikit_posthocs.posthoc_dunn.html
import pandas as pd

import matplotlib as mpl
from matplotlib.colors import Normalize, LogNorm

from moldiff.constants import aa_decoder3

pd.set_option("display.width", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_colwidth", None)


os.environ["KMP_WARNINGS"] = "0"



def make_relative_path(path: Path, base: Path) -> Path:
    """
    Return `path` expressed relative to directory `base`.
    
    If `path` is inside `base`, use Path.relative_to() for a clean relative path.
    Otherwise fall back to os.path.relpath().

    Parameters
    ----------
    path : Path
        The absolute or arbitrary filesystem path to convert.
    base : Path
        The reference directory (e.g., cwd of a subprocess).

    Returns
    -------
    Path
        A relative path from `base` to `path`.
    """
    path = Path(path)
    base = Path(base)

    try:
        # Cleanest form (no "../.."), only works when inside base
        return path.relative_to(base)
    except ValueError:
        # General fallback: produce a valid relative path even across directories
        return Path(os.path.relpath(path, base))


def classify_residue(resname: str) -> str:
    r = resname.upper()
    if r in {"ARG", "LYS", "HIS"}:
        return "charged_pos"
    if r in {"ASP", "GLU"}:
        return "charged_neg"
    if r in {"SER", "THR", "ASN", "GLN"}:
        return "polar"
    if r in {"PHE", "TYR", "TRP"}:
        return "aromatic"
    if r in {"ALA", "VAL", "LEU", "ILE", "MET", "PRO"}:
        return "hydrophobic"
    if r in {"CYS"}:
        return "special"
    return "other"

def build_protein_atom_table_gemmi(pocket_pdb: Path) -> pd.DataFrame:
    """
    Build a table mapping protein_atom_id -> residue + atom metadata,
    including distance to the residue CA atom.

    Adds:
      - protein_sidechain_weight: sum of atomic weights of sidechain atoms
        (non-backbone atoms: not in {N, CA, C, O}) for the residue.

    Assumes atom order matches what was used for clash detection.
    """
    st = gemmi.read_structure(str(pocket_pdb))
    model = st[0]  # single model

    records = []
    atom_idx = 0

    backbone_names = {"N", "CA", "C", "O"}

    # --- 1) cache CA coordinates per residue ---
    ca_coords = {}  # key: (chain_id, resid) -> np.array([x,y,z])

    for chain in model:
        for res in chain:
            for atom in res:
                if atom.name.strip() == "CA":
                    ca_coords[(chain.name, res.seqid.num)] = np.array(
                        [atom.pos.x, atom.pos.y, atom.pos.z]
                    )
                    break  # one CA per residue

    # --- 2) cache sidechain weight per residue ---
    sidechain_weight = {}  # key: (chain_id, resid) -> float

    for chain in model:
        chain_id = chain.name
        for res in chain:
            resid = res.seqid.num
            w = 0.0
            for atom in res:
                atom_name = atom.name.strip()
                if atom_name in backbone_names:
                    continue

                # Exclude hydrogens explicitly
                if atom.element.name.strip() == "H":
                    continue

                # gemmi Element exposes atomic weight via `weight`
                w += float(atom.element.weight)

            sidechain_weight[(chain_id, resid)] = w

    # --- 3) build atom table with CA distances + residue sidechain weight ---
    for chain in model:
        chain_id = chain.name
        for res in chain:
            resname = res.name
            resid = res.seqid.num
            ca = ca_coords.get((chain_id, resid), None)
            sc_w = sidechain_weight.get((chain_id, resid), 0.0)

            for atom in res:
                atom_name = atom.name.strip()
                atom_element = atom.element.name.strip()
                is_backbone = atom_name in backbone_names

                atom_pos = np.array([atom.pos.x, atom.pos.y, atom.pos.z])
                dist_to_ca = np.nan if ca is None else float(np.linalg.norm(atom_pos - ca))

                records.append(
                    {
                        "protein_atom_id": atom_idx,
                        "protein_atom_name": atom_name,
                        "protein_atom_element": atom_element,
                        "protein_resname": resname,
                        "protein_resid": resid,
                        "protein_chain": chain_id,
                        "protein_is_backbone": is_backbone,
                        "protein_region": "backbone" if is_backbone else "sidechain",
                        "protein_res_class": classify_residue(resname),
                        "protein_dist_to_ca": dist_to_ca,
                        "protein_sidechain_weight": sc_w,
                    }
                )
                atom_idx += 1

    return pd.DataFrame.from_records(records)



def build_dpocket_ready_complex(
    rec_pdb: Path, 
    lig_sdf: Path, 
    out_root: Path,
    # crossdocked_root: Path,
    # pocket_id: str,
    lig_resname: str = None,
    lig_chain: str = "Z",
    lig_resid: int = 1
) -> Path:
    """
    Fuse CrossDocked receptor (pocket_id.pdb) and ligand (pocket_id.sdf)
    into a single PDB suitable for dpocket.

    Returns path to the complex PDB.
    """

    sample_id = lig_sdf.name.split("_")[0]
    pocket_id = lig_sdf.parent.name

    if not rec_pdb.exists():
        raise FileNotFoundError(f"Receptor PDB not found: {rec_pdb}")
    if not lig_sdf.exists():
        raise FileNotFoundError(f"Ligand SDF not found: {lig_sdf}")

    out_pdb = out_root / f"{pocket_id}_{sample_id}_complex_for_dpocket.pdb"

    if out_pdb.exists():
        return out_pdb

    # --- read receptor PDB as text ---
    with open(rec_pdb, "r") as f:
        rec_lines = [
            l.rstrip("\n") for l in f
            if l.startswith(("ATOM", "HETATM", "TER", "END"))
        ]

    # --- max atom serial ---
    rec_serials = [
        int(l[6:11]) for l in rec_lines
        if l.startswith(("ATOM", "HETATM")) and l[6:11].strip().isdigit()
    ]
    start_serial = max(rec_serials) if rec_serials else 0

    # --- ligand from SDF ---
    suppl = Chem.SDMolSupplier(str(lig_sdf), sanitize=False)
    lig = suppl[0]
    if lig is None:
        raise ValueError(f"Could not read ligand from {lig_sdf}")

    if lig_resname is None:
        # heuristic: take 4th field from pocket_id "14gs-A-rec-20gs-cbd-lig-tt-min"
        parts = pocket_id.split("-")
        if len(parts) >= 5:
            lig_resname = parts[4][:3].upper()
        else:
            lig_resname = "LIG"

    lig_res = lig_resname[:3].upper()

    # --- assign PDB residue info to ligand atoms ---
    for atom in lig.GetAtoms():
        info = Chem.AtomPDBResidueInfo()
        info.SetResidueName(lig_res)
        info.SetChainId(lig_chain)
        info.SetResidueNumber(lig_resid)
        info.SetName(atom.GetSymbol().rjust(4))
        atom.SetPDBResidueInfo(info)

    lig_block = Chem.MolToPDBBlock(lig)
    lig_raw_lines = [
        l.rstrip("\n")
        for l in lig_block.splitlines()
        if l.startswith(("ATOM", "HETATM"))
    ]

    # --- renumber ligand atoms ---
    lig_lines = []
    serial = start_serial
    for line in lig_raw_lines:
        serial += 1
        new_line = f"{line[:6]}{serial:5d}{line[11:]}"
        lig_lines.append(new_line)

    # --- write complex ---
    with open(out_pdb, "w") as f:
        for l in rec_lines:
            if not l.startswith("END"):
                f.write(l + "\n")
        for l in lig_lines:
            f.write(l + "\n")
        f.write("END\n")

    return out_pdb



def run_dpocket(dpocket_input_file: Path, dpocket_processing_dir: Path, dpocket_detection_radius:int = 4):
    """
    Run dpocket only once and ensure the dpocket_input_path
    is correctly referenced relative to output_root.
    """

    # print(f"[INFO] dpocket function received prefix {prefix}")


    # explicit_out = dpocket_processing_dir / f"{prefix}_exp.txt"

    # # print(dpocket_input_path)
    # # print(output_root)
    # # print(explicit_out)

    # # # Skip if dpocket already ran
    # if explicit_out.exists():
    #     print("[OK] dpocket results already exist, skipping.")
    #     return

    # Compute the path relative to output_root (subprocess cwd)
    dpocket_input_rel = make_relative_path(dpocket_input_file, dpocket_processing_dir)
    
    # print(dpocket_input_rel)

    print(f"[INFO] Running dpocket with input: {dpocket_input_rel}")
    # print("[INFO] dpocket ARGV =", ["dpocket", "-f", str(dpocket_input_rel),  "-d", str(dpocket_detection_radius)])
    # assert isinstance(prefix, str) and prefix and not prefix.startswith("-"), f"prefix is {prefix}"

    subprocess.run(
        ["dpocket", "-f", str(dpocket_input_rel), "-E", "-d", str(dpocket_detection_radius)],
        cwd=str(dpocket_processing_dir),
        check=True
    )

    # print(f"[OK] dpocket executed successfully for pocket_id {prefix}.")

def run_pyKVFinder(pocket_path:str, kv_args:Dict, return_df:bool = True):

    results = pyKVFinder.run_workflow(
        input=pocket_path, 
        # ligand="0_ligand.pdb", 
        **kv_args,
        include_depth=True, 
        include_hydropathy=True, 
        hydrophobicity_scale='EisenbergWeiss'
        )
    if results is None:
        return
    
    if return_df:
        residue_class_meaning_mapper = {
            "R1": "alipathic-apolar", 
            "R2": "aromatic", 
            "R3": "polar-uncharged", 
            "R4": "negatively-charged", 
            "R5": "positively-charged", 
            "RX": "Non-standard"
        }
        # Find biggest cavity and construct df
        pocket_id = max(results.volume, key=results.volume.get)
        row = {
            # "sample_id": f"{Path(pocket_path).parent.name}_{Path(pocket_path).stem[0]}" ,
            "pyKV_volume": results.volume.get(pocket_id),
            "pyKV_surface_area": results.area.get(pocket_id),
            "pyKV_avg_depth": results.avg_depth.get(pocket_id),
            "pyKV_max_depth": results.max_depth.get(pocket_id),
            "pyKV_avg_hydropathy": results.avg_hydropathy.get(pocket_id),
        } | {
            f"pyKV_{key}_{residue_class_meaning_mapper.get(subkey, subkey)}": value for key, item in results.frequencies[pocket_id].items() for subkey, value in item.items()
        }

        
        return row
    else:
        return results


def evaluate_class_worker(pocket_dir: Path, dpocket_complex_dir: Path, dpocket_detection_radius:int, pykv_args:Dict = {}) -> pd.DataFrame:
            

        if not pocket_dir.is_dir():
            print(f"[WARNING] returning None for {pocket_dir} with working directory {os.getcwd()}")
            return 

        pocket_id = pocket_dir.name


        lig_rows = []

        # # filtering ligand files that pass all filters prior to the clash filter
        # ligand_files = sorted(pocket_dir.glob("*_ligand.sdf"))
        # pocket_file = sorted(pocket_dir.glob("*_pocket.pdb"))[0]
        # metrics_file = Path(pocket_dir.parents[1]) / (pocket_dir.parent.name.split("_")[0] + "_metrics") / "metrics_detailed.csv"
        # prior_filters = ["sanitization", "all_atoms_connected", "aromatic_ring_flatness", "bond_angles", "bond_lengths", "double_bond_flatness", "internal_steric_clash"]

        # if metrics_file.exists():
        #     bust_result = pd.read_csv(metrics_file)
        #     rel_tail = ["/".join(p.parts[-2:]) for p in ligand_files]
        #     abs_tail = bust_result["sdf_file"].apply(lambda p: "/".join(Path(p).parts[-2: ]))
        #     mask = abs_tail.isin(rel_tail)
        #     # order absolute paths *in the order they appear in list_rel*
        #     order_idx = abs_tail[mask].map(
        #         {k: i for i, k in enumerate(rel_tail)}
        #     ).sort_values().index

        #     # ordered result
        #     updated_prior_filters = [f"posebusters.{s}" for s in prior_filters]
        #     filter_mask = bust_result.loc[order_idx, updated_prior_filters].all(axis=1).values
            
        # else:
        #     print("[INFO] No metrics file found, running PoseBusters")
        #     buster = PoseBusters(config="mol")
        #     bust_result = buster.bust(mol_pred=ligand_files, mol_cond=pocket_file)
        #     filter_mask = bust_result[prior_filters].all(axis=1).values

        # filtered_ligand_files = np.array(ligand_files)[filter_mask]

        metrics_file = Path(pocket_dir.parents[1]) / f"{pocket_dir.parent.name.removesuffix('_samples')}_metrics" / "metrics_detailed.csv"
        filtered_ligand_files = sorted(pocket_dir.glob("*_ligand.sdf"))



        # run the dpocket evaluation of all valid ligand/pocket complexes
        dpocket_lines = []
        dpocket_wd = dpocket_complex_dir / pocket_id # parallel running dpocket should run in seperate folders
        dpocket_wd.mkdir(exist_ok = True)

        # fetch pyKV descriptors
        first_pocket_pdb = pocket_dir / "0_pocket.pdb" # assuming all pockets in the directory are the same
        pyKV_results = run_pyKVFinder(str(first_pocket_pdb), pykv_args)


        # Evaluate all sample ligands
        for lig_file in filtered_ligand_files:
            # print(lig_file)
            m = re.match(r"(\d+)_ligand\.sdf", lig_file.name)
            if not m:
                print(f"[WARN] Skipped {lig_file}: lacking regex match")
                continue
            sample_idx = int(m.group(1))
            pocket_pdb = pocket_dir / f"{sample_idx}_pocket.pdb"

            # run dpocket

            dpocket_pdb_complex_path = build_dpocket_ready_complex(pocket_pdb, lig_file, out_root = dpocket_wd)
            # extract ligand name
            parts = pocket_id.split("-")
            lig_resname =  parts[4].upper() if len(parts) >= 5 else "LIG"
            # Compute the path relative to output_root (subprocess cwd)
            complex_pdb_rel = make_relative_path(dpocket_pdb_complex_path, dpocket_wd)
            dpocket_lines.append(f"{complex_pdb_rel}\t{lig_resname}")


            # get per atom steric clashes
            ligand = Chem.SDMolSupplier(str(lig_file), sanitize=False)[0] # TODO: is this loading only a small fragment?
            pocket_mol = Chem.MolFromPDBFile(str(pocket_pdb), sanitize=False, removeHs=False)

            if ligand is None or pocket_mol is None:
                print(f"[WARN] Skipped {lig_file}: failed ligand or pocket loading")
                continue

            result = check_intermolecular_distance(mol_pred=ligand, mol_cond=pocket_mol) #, ignore_types = {"hydrogens"})
            details = result["details"]
            clashes = details.copy() # details[details["clash"]].copy()

            # annotate with gemmi
            atom_table = build_protein_atom_table_gemmi(pocket_pdb)
            clashes = clashes.merge(atom_table, on="protein_atom_id", how="left")

            # summary info
            for k, v in result["results"].items():
                clashes[f"clash_summary_{k}"] = v

            # unique sample id
            clashes["sample_id"] = f"{pocket_id}_{sample_idx}"
            clashes["pocket_id"] = pocket_id
            clashes["sample_idx"] = sample_idx

            if pyKV_results is not None:
                for k, v in pyKV_results.items():
                    clashes[k] = v
            
            lig_rows.append(clashes)
        
        lig_df = pd.concat(lig_rows, axis=0, ignore_index=True)
        cols_to_fill = [
            c for c in lig_df.columns
            if c.startswith("RESIDUES") or c.startswith("CLASS")
        ]
        lig_df[cols_to_fill] = lig_df[cols_to_fill].fillna(0) # for residues that where NANs

        dpocket_input_file = dpocket_wd / f"{pocket_id}_dpocket_input.txt"

        with open(dpocket_input_file, "w") as f:
            f.write("\n".join(dpocket_lines))

        run_dpocket(dpocket_input_file, dpocket_wd, dpocket_detection_radius=dpocket_detection_radius)

        # merge the descriptors resulting from the explicit volumen calculated by dpocket
        dpocket_explicit_data = pd.read_csv(f"{dpocket_wd}/dpout_explicitp.txt", sep="\s+")
        dpocket_explicit_data.rename(columns={old_name: f"fpocket_explicit_{old_name}" for old_name in dpocket_explicit_data.columns}, inplace=True)
        dpocket_explicit_data["sample_id"] = dpocket_explicit_data["fpocket_explicit_pdb"].apply(lambda x: x.split("_complex")[0])

        lig_rows = lig_df.merge(dpocket_explicit_data, how="left", on="sample_id")

        # merge the descriptors resulting from the top fpocket detected cavity
        fpocket_data = pd.read_csv(f"{dpocket_wd}/dpout_fpocketp.txt", sep="\s+")
        fpocket_data.rename(columns={old_name: f"fpocket_cavity_{old_name}" for old_name in fpocket_data.columns}, inplace=True)
        fpocket_data["sample_id"] = fpocket_data["fpocket_cavity_pdb"].apply(lambda x: x.split("_complex")[0])

        lig_rows = lig_rows.merge(fpocket_data, how="left", on="sample_id")

        # if available, merge metrics information from drugflow pipeline
        if metrics_file.exists():
            print("[INFO] Merging metrics data")
            metrics_data = pd.read_csv(metrics_file)
            metrics_data["sample_id"] = metrics_data["sdf_file"].apply(lambda x: Path(x).parent.name) + "_" + metrics_data["sample"].astype(str)
            print(metrics_data["sample_id"][0])
            # print(str(metrics_data["sample"]))
            lig_rows = lig_rows.merge(metrics_data, how="left", on="sample_id")
        else:
            print(f"[WARNING] No metrics file exists at {metrics_file}")


        # print(type(lig_rows))
        
        return lig_rows

def get_allowed_cpus(requested=None):
    # CPUs granted by SLURM
    total_cpus = os.cpu_count()
    slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", total_cpus))

    if slurm_cpus is None:
        raise RuntimeError("No CPUs available")

    # Allow user to request fewer
    if requested is None:
        return slurm_cpus
    
    if requested > slurm_cpus:
        print(f"[INFO] Number of parallel processes reduced to {slurm_cpus}")

    return min(requested, slurm_cpus)

def full_pipeline(
    ours_samples_root: Path,
    # crossdocked_root: Path,
    # output_root: Path, 
    dpocket_detection_radius: int = 4,
    pykv_args: Dict = {}, 
    overwrite_existing: bool = False,
    requested_workers: int|None = None # None uses max
):
    output_root = ours_samples_root.parent /  f"{ours_samples_root.parent.name.removesuffix('_samples')}_metrics" / "detailed_clash_evaluation" # .split("_")[0]
    if overwrite_existing:
        try:
            shutil.rmtree(output_root)
            print(f"[INFO] Deleted existing {output_root}")
        except FileNotFoundError:
            print(f"{output_root} doesn't exist and will therefore not be deleted. Creating directory...")
    
    output_root.mkdir(parents=True, exist_ok=True)

    all_rows = []
    pdb_paths = sorted(ours_samples_root.iterdir())
    available_workers = get_allowed_cpus(requested_workers)
        

    with ProcessPoolExecutor(max_workers=available_workers) as pool:

        futures = [pool.submit(evaluate_class_worker, p, output_root,  dpocket_detection_radius) for p in pdb_paths]

        for i, fut in tqdm(enumerate(as_completed(futures)), total=len(futures)):

            try:
                clash_result = fut.result()
            except Exception as e:
                print(f"[ERROR] processing path {pdb_paths[i]}:", e)
                traceback.print_exc()
            else:
                all_rows.append(clash_result)

    if not all_rows:
        return pd.DataFrame()

    print(type(all_rows[0]))

    final_df = pd.concat(all_rows, axis=0, ignore_index=True)

    final_df.to_csv(output_root.parent / "detailed_clash_evaluation.csv", index=False)
    print(f"[OK] Final dataframe saved in {output_root}.csv")

    return final_df


# ============================================================
# Core analysis helpers (largely unchanged)
# ============================================================

def calc_means_classes(df, classes):
    df = df.copy()
    df["clash"] = df["clash"].astype(bool)

    return (
        df
        .groupby(classes)
        .agg(
            n_pairs=("clash", "size"),
            n_clashes=("clash", "sum"),
            clash_rate=("clash", "mean"),
            mean_dist=("distance", "mean"),
            mean_rel_dist=("relative_distance", "mean"),
            n_prot_atoms=("protein_atom_id", "nunique"),
            n_lig_atoms=("ligand_atom_id", "nunique"),
        )
        .sort_values("clash_rate", ascending=False)
    )


def clash_matrix(
    df,
    value="clash",
    aggfunc="mean",
    protein_axis="protein_element",
):
    return (
        df.pivot_table(
            index="ligand_element",
            columns=protein_axis,
            values=value,
            aggfunc=aggfunc,
            fill_value=0.0,
        )
    )


def chi_square_clash_test(df, variable):
    contingency = pd.crosstab(df[variable], df["clash"])
    chi2, p, _, _ = chi2_contingency(contingency)
    return chi2, p


def kruskal_wallis_and_posthoc_dunn(df, cat_var, num_var):
    data = df[[cat_var, num_var]].dropna()

    groups = [g[num_var].values for _, g in data.groupby(cat_var)]
    kw = kruskal(*groups)

    dunn = posthoc_dunn(
        data,
        val_col=num_var,
        group_col=cat_var,
        p_adjust="holm",
    )

    return kw, dunn

def _sanitize_filename(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^a-z0-9_\-\.]+", "", name)
    name = re.sub(r"_+", "_", name)
    return name[:200] if len(name) > 200 else name


def _save_fig(fig: plt.Figure, out_dir: Path, figtitle: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{_sanitize_filename(figtitle)}.png"
    out_file = out_dir / fname
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_file


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.1, 1.05, label,
        transform=ax.transAxes,
        va="top", ha="left",
        # fontname="Arial",
        fontsize=14,
        fontweight="bold",
    )


def analyze_and_plot_clashes(
    final_df: pd.DataFrame,
    save_path: Path,
    features_to_corrplot: List|None = None
):
    """
    Saves grouped report figures (instead of one png per plot) into:
        save_path / "detailed_clash_analysis_graphics"

    Figures:
      - Figure 1: filter-level histograms (single plot)
      - Figure 2: Wasserstein distance table (single table)
      - Figure 3: 3x4 grid (A–L): [freq bars | heatmaps | barplots | Dunn heatmaps] × 3 prot axes
      - Figure 4: Tables P(clash | prot_col) top 20 (1x3)
      - Figure 5: Test results text panels (1x3)
      - Figure 6: Correlations (Pearson R vs Kendall τ) (1x2, A–B)
    """
    # clean all the protein_atom_name = H, as those are malinformed rows where PBD filled an H but PoseBusters interpret a C -> don't know what to do with those
    final_df = final_df[(final_df["protein_atom_name"] != "H") & (final_df["protein_atom_name"] != "HG")]
    final_df["protein_atom_name_exact"] = final_df["protein_atom_name"]
    final_df["protein_atom_name"] = final_df["protein_atom_name"].apply(lambda s: s[:2]) 

    # =========================================================
    # Output folder
    # =========================================================
    out_dir = save_path / "detailed_clash_analysis_graphics"
    out_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []

    # =========================================================
    # Filters
    # =========================================================
    prior_posebusters_filters = [
        f"posebusters.{f}"
        for f in [
            "sanitization",
            "all_atoms_connected",
            "aromatic_ring_flatness",
            "bond_angles",
            "bond_lengths",
            "double_bond_flatness",
            "internal_steric_clash",
        ]
    ]
    prior_medchem_filters = [f"medchem.{f}" for f in ["valid", "connected"]]

    df0 = final_df.copy()
    medchem_mask = df0[prior_medchem_filters].all(axis=1)
    posebusters_mask = df0[prior_posebusters_filters].all(axis=1)

    df_filtered = df0[medchem_mask & posebusters_mask].copy()
    df_filtered["clash_binary"] = df_filtered["clash"].astype(int)

    total_mask_agreement = (medchem_mask == posebusters_mask).mean()
    connected_agreement = (
        df0["posebusters.all_atoms_connected"] == df0["medchem.connected"]
    ).mean()

    dataframes_by_filterlevel = {
        "none": df0,
        "medchem": df0[medchem_mask],
        "posebusters": df0[posebusters_mask],
        "both": df_filtered,
    }

    # =========================================================
    # Wasserstein distances (per-sample)
    # =========================================================
    def per_sample(series, df):
        return df.groupby("sample_id")[series].mean().values

    wass_rows = []
    for (k1, df1), (k2, df2) in combinations(dataframes_by_filterlevel.items(), 2):
        wass_rows.append(
            (
                k1,
                k2,
                wasserstein_distance(per_sample("clash", df1), per_sample("clash", df2)),
                wasserstein_distance(
                    per_sample("relative_distance", df1),
                    per_sample("relative_distance", df2),
                ),
            )
        )

    wasserstein_df = pd.DataFrame(
        wass_rows,
        columns=["filter_a", "filter_b", "W_clash_rate", "W_relative_dist"],
    )

    # =========================================================
    # Protein axes
    # =========================================================
    prot_cols = ["protein_element", "protein_atom_name", "protein_resname"]

    # =========================================================
    # Correlation descriptors
    # =========================================================
    pocket_descriptors = [
        c
        for c in df_filtered.columns
        if (
            (c.startswith("pyKV_") or c.startswith("fpocket_explicit_") or c.startswith("fpocket_cavity_"))
            and is_numeric_dtype(df_filtered[c])
        )
    ]
    ligand_eval_descriptors = ["medchem.size"]
    x_vars = pocket_descriptors + ligand_eval_descriptors + ["protein_dist_to_ca", "protein_sidechain_weight"]
    x_vars = [
        var for var in x_vars if all([aa not in var for aa in aa_decoder3])
        ] #  + ["CLASS"]

    per_sample_agg = (
        df_filtered.groupby("sample_id")
        .agg(
            clash_rate=("clash_binary", "mean"),
            relative_distance=("relative_distance", "mean"),
            **{k: (k, "first") for k in x_vars},
        )
    )

    # =========================================================
    # Figure 1: filter-level histograms (single figure)
    # =========================================================
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    for name, df in dataframes_by_filterlevel.items():
        ax1.hist(
            df.groupby("sample_id")["clash"].mean(),
            bins=40,
            alpha=0.5,
            label=name,
        )
    ax1.legend()
    ax1.set_title("")  # no title
    fig1.tight_layout()
    saved.append(_save_fig(fig1, out_dir, "figure1_per_sample_clash_rate_distributions"))

    # =========================================================
    # Figure 2: Wasserstein distance table (single figure)
    # =========================================================
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.axis("off")
    table = ax2.table(
        cellText=[
            [a, b, f"{w1:.4f}", f"{w2:.4f}"]
            for a, b, w1, w2 in wasserstein_df.itertuples(index=False)
        ],
        colLabels=wasserstein_df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)
    ax2.set_title(
        "Wasserstein distances\n"
        f"Mask agreement: {total_mask_agreement:.2%}\n"
        f"Connected agreement: {connected_agreement:.2%}"
    )
    fig2.tight_layout()
    saved.append(_save_fig(fig2, out_dir, "figure2_wasserstein_distances_table"))

    # =========================================================
    # Shared per-sample class-aggregates + tests per prot_col
    # =========================================================
    ps_by_prot: dict[str, pd.DataFrame] = {}
    chi_by_prot: dict[str, tuple[float, float]] = {}
    kw_by_prot: dict[str, tuple[float, float]] = {}
    dunn_by_prot: dict[str, pd.DataFrame] = {}

    for prot_col in prot_cols:
        ps = (
            df_filtered.groupby(["sample_id", prot_col])
            .agg(
                clash_rate=("clash_binary", "mean"),
                relative_distance=("relative_distance", "mean"),
            )
            .reset_index()
        )
        ps_by_prot[prot_col] = ps

        # Chi-square (binary presence of clash per (sample, class))
        ps["has_clash"] = (ps["clash_rate"] > 0).astype(int)
        contingency = pd.crosstab(ps[prot_col], ps["has_clash"])
        chi2, p_chi, _, _ = chi2_contingency(contingency)
        chi_by_prot[prot_col] = (chi2, p_chi)

        # Kruskal–Wallis + Dunn on relative_distance
        groups = [g["relative_distance"].values for _, g in ps.groupby(prot_col)]
        kw_res = kruskal(*groups) if len(groups) >= 2 else None
        if kw_res is None:
            kw_by_prot[prot_col] = (np.nan, np.nan)
        else:
            kw_by_prot[prot_col] = (float(kw_res.statistic), float(kw_res.pvalue))

        try:
            dunn = posthoc_dunn(
                ps,
                val_col="relative_distance",
                group_col=prot_col,
                p_adjust="holm",
            )
        except Exception:
            dunn = pd.DataFrame()
        dunn_by_prot[prot_col] = dunn

    # =========================================================
    # Figure 3: 3x4 grid (A–L) — main report figure
    # Row 1: frequency barplots (3)            (ONLY categories present in Row 3; taller)
    # Row 2: heatmaps P(clash | ligand×prot)   (shared colorbar; shorter)
    # Row 3: barplots mean P(clash) by class   (taller)
    # Row 4: Dunn posthoc heatmaps (p-values)  (shared colorbar; shorter; p=0 red, p=1 grey)
    # =========================================================

    height_ratios = [1.35, 1.0, 1.35, 1.0]

    fig3, axes = plt.subplots(
        nrows=4, ncols=3,
        figsize=(18, 20),
        gridspec_kw={"height_ratios": height_ratios},
        constrained_layout=True
    )

    # --- helper: categories present in Row 3 (derived from df_filtered / ps_by_prot) ---
    # Row 3 uses prot_cols: ["protein_element", "protein_atom_name", "protein_resname"]
    row3_categories = {
        prot_col: set(ps_by_prot[prot_col][prot_col].dropna().unique())
        for prot_col in prot_cols
    }

    # ---------- Row 1: frequency barplots (ONLY categories present in Row 3) ----------
    root_dir = Path(save_path.parents[0]) / ("_".join(save_path.name.split("_")[:-1]) + "_samples")
    all_atom_tables = pd.concat(
        [build_protein_atom_table_gemmi(pocket_dir / "0_pocket.pdb") for pocket_dir in root_dir.iterdir()],
        ignore_index=True,
    )

    frequency_data = {
        "protein_atom_element": all_atom_tables["protein_atom_element"],
        "protein_atom_name": all_atom_tables["protein_atom_name"],
        "protein_resname": all_atom_tables[["protein_resname", "protein_resid"]].drop_duplicates()["protein_resname"],
    }

    # Map Row-1 keys -> the corresponding Row-3 prot_col whose categories we want to keep
    # NOTE: element naming mismatch: Row 3 uses "protein_element", Row 1 uses "protein_atom_element"
    row1_to_row3 = {
        "protein_atom_element": "protein_element",
        "protein_atom_name": "protein_atom_name",
        "protein_resname": "protein_resname",
    }

    for j, (key, array) in enumerate(frequency_data.items()):
        ax = axes[0, j]

        s = pd.Series(array).dropna()

        # Keep only categories that actually appear in Row 3 barplots for the matching axis
        ref_col = row1_to_row3[key]
        allowed = row3_categories.get(ref_col, set())
        if len(allowed) > 0:
            s = s[s.isin(allowed)]

        counts = s.value_counts()
        if counts.empty:
            ax.axis("off")
            ax.text(0.02, 0.98, f"{key}\n\nNo overlap with Row 3 categories", va="top")
            continue

        fractions = counts / counts.sum() * 100
        fractions = fractions[fractions > 0]  # explicit

        ax.barh(fractions.index.astype(str), fractions.values, color="purple")
        ax.invert_yaxis()
        ax.set_xlabel("Fraction (%)")
        ax.set_ylabel("")
        ax.xaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)

    # ---------- Row 2: heatmaps P(clash | ligand × prot_col) + shared colorbar ----------
    row2_mats = []
    for prot_col in prot_cols:
        row2_mats.append(
            clash_matrix(
                df_filtered,
                value="clash",
                aggfunc="mean",
                protein_axis=prot_col,
            )
        )

    norm_row2 = Normalize(vmin=0.0, vmax=1.0)
    cmap_row2 = plt.get_cmap("viridis")

    for j, mat in enumerate(row2_mats):
        ax = axes[1, j]
        sns.heatmap(
            mat,
            cmap=cmap_row2,
            norm=norm_row2,
            ax=ax,
            cbar=False,
        )
        ax.set_xticks(np.arange(len(mat.columns)) + 0.5)
        ax.set_xticklabels(mat.columns, rotation=90)
        ax.set_xlabel(None)
        ax.set_ylabel(None)

    sm2 = mpl.cm.ScalarMappable(norm=norm_row2, cmap=cmap_row2)
    sm2.set_array([])
    cb2 = fig3.colorbar(sm2, ax=axes[1, :], fraction=0.02, pad=0.02)
    cb2.set_label("P(clash)")

    # ---------- Row 3: barplots mean P(clash) by class ----------
    for j, prot_col in enumerate(prot_cols):
        ax = axes[2, j]
        ps = ps_by_prot[prot_col]
        mat2 = ps.groupby(prot_col)["clash_rate"].mean().sort_values(ascending=False)

        ax.barh(mat2.index.astype(str), mat2.values, color="purple")
        ax.invert_yaxis()
        ax.set_xlabel("P(clash)")
        ax.set_ylabel("")
        ax.xaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)

    # ---------- Row 4: Dunn posthoc heatmaps + shared colorbar (0 red -> 1 grey) ----------
    norm_row4 = Normalize(vmin=0.0, vmax=1.0)
    cmap_row4 = plt.get_cmap("Greys")

    for j, prot_col in enumerate(prot_cols):
        ax = axes[3, j]
        dunn = dunn_by_prot[prot_col]

        if dunn is None or dunn.empty:
            ax.axis("off")
            ax.text(0.02, 0.98, f"{prot_col}\n\nDunn posthoc unavailable", va="top")
            continue

        sns.heatmap(
            dunn,
            cmap=cmap_row4,
            norm=norm_row4,
            ax=ax,
            cbar=False,
        )
        ax.set_xticks(np.arange(len(dunn.columns)) + 0.5)
        ax.set_xticklabels(dunn.columns, rotation=90)
        ax.set_yticks(np.arange(len(dunn.index)) + 0.5)
        ax.set_yticklabels(dunn.index, rotation=0)
        ax.set_xlabel(None)
        ax.set_ylabel(None)

    sm4 = mpl.cm.ScalarMappable(norm=norm_row4, cmap=cmap_row4)
    sm4.set_array([])
    cb4 = fig3.colorbar(sm4, ax=axes[3, :], fraction=0.02, pad=0.02)
    cb4.set_label("Dunn posthoc p-value")

    # ---------- Panel labels A–L ----------
    labels = iter(ascii_uppercase)
    for r in range(4):
        for c in range(3):
            _panel_label(axes[r, c], next(labels))

    saved.append(_save_fig(fig3, out_dir, "figure3_clash_analysis_grid_3x4"))



    # =========================================================
    # Figure 4: Tables P(clash | prot_col) top 20 (1x3, A–C)
    # =========================================================
    fig4, axes4 = plt.subplots(nrows=1, ncols=3, figsize=(18, 6), constrained_layout=True)
    for j, prot_col in enumerate(prot_cols):
        ax = axes4[j]
        ax.axis("off")
        stats = calc_means_classes(df_filtered, [prot_col]).head(20)

        t = ax.table(
            cellText=np.round(stats.values, 3),
            rowLabels=stats.index,
            colLabels=stats.columns,
            loc="center",
            cellLoc="center",
        )
        t.auto_set_font_size(False)
        t.set_fontsize(8)
        t.scale(1, 1.4)

        _panel_label(ax, ascii_uppercase[j])

    saved.append(_save_fig(fig4, out_dir, "figure4_tables_p_clash_by_protein_axis_top20"))

    # =========================================================
    # Figure 5: Test results panels (1x3, A–C)
    # =========================================================
    fig5, axes5 = plt.subplots(nrows=1, ncols=3, figsize=(18, 4.5), constrained_layout=True)
    for j, prot_col in enumerate(prot_cols):
        ax = axes5[j]
        ax.axis("off")

        chi2, p_chi = chi_by_prot[prot_col]
        H, p_kw = kw_by_prot[prot_col]

        stats_text = (
            f"{prot_col}\n\n"
            f"Chi-square (per-sample clash presence)\n"
            f"χ² = {chi2:.3f}\n"
            f"p  = {p_chi:.2e}\n\n"
            f"Kruskal–Wallis (relative distance)\n"
            f"H  = {H:.3f}\n"
            f"p  = {p_kw:.2e}"
        )
        ax.text(0.02, 0.98, stats_text, va="top", fontsize=11)

        _panel_label(ax, ascii_uppercase[j])

    saved.append(_save_fig(fig5, out_dir, "figure5_test_results_by_protein_axis"))

    # =========================================================
    # Figure 6 & 7: correlations (Pearson R vs Kendall τ) (1x2, A–B)
    # =========================================================
    # --- correlations + p-values ---
    def _corr_and_pvals(df: pd.DataFrame, x_vars: list[str], y: str):
        rows = []
        yv = df[y]
        for x in x_vars:
            xv = df[x]

            # drop NaNs pairwise
            m = xv.notna() & yv.notna()
            if m.sum() < 3:
                rows.append((x, np.nan, np.nan, np.nan, np.nan))
                continue

            r, p_r = pearsonr(xv[m], yv[m])
            tau, p_tau = kendalltau(xv[m], yv[m], nan_policy="omit")

            rows.append((x, r, p_r, tau, p_tau))

        out = pd.DataFrame(
            rows,
            columns=["feature", f"{y}_R", f"{y}_p_R", f"{y}_tau", f"{y}_p_tau"],
        ).set_index("feature")
        return out

    corr_clash = _corr_and_pvals(per_sample_agg, x_vars, "clash_rate")
    corr_dist  = _corr_and_pvals(per_sample_agg, x_vars, "relative_distance")

    corr_all = corr_clash.join(corr_dist, how="outer")

    # score as before (but now using the correlation columns)
    score = (
        corr_all["clash_rate_R"].abs()
        + corr_all["clash_rate_tau"].abs()
        + corr_all["relative_distance_R"].abs()
        + corr_all["relative_distance_tau"].abs()
    ).fillna(0)

    top_features = score.sort_values(ascending=False).head(150).index
    corr_top = corr_all.loc[top_features].copy()
    corr_top = corr_top.loc[score.loc[top_features].sort_values(ascending=False).index]

    def _p_to_stars(p: float) -> str:
        if pd.isna(p):
            return "nan"
        if p < 1e-3:
            return "***"
        if p < 1e-2:
            return "**"
        if p < 5e-2:
            return "*"
        return "o"

    def _barh_corr(ax: plt.Axes, target: str, features_to_corrplot: list[str] | None = None, legend:bool = True):
        if target == "clash_rate":
            cols = ["clash_rate_R", "clash_rate_tau"]
            pcols = ["clash_rate_p_R", "clash_rate_p_tau"]
            metric_labels = ["R", "τ"]
        elif target == "relative_distance":
            cols = ["relative_distance_R", "relative_distance_tau"]
            pcols = ["relative_distance_p_R", "relative_distance_p_tau"]
            metric_labels = ["R", "τ"]
        else:
            raise ValueError("target must be 'clash_rate' or 'relative_distance'")

        # long df with correlations and p-values (aligned)
        wide = corr_top[cols + pcols].copy()
        wide.columns = [*metric_labels, *(f"{m}_p" for m in metric_labels)]

        long_df = (
            wide.reset_index()
            .melt(id_vars="feature", value_vars=metric_labels, var_name="metric", value_name="correlation")
        )

        p_long = (
            wide.reset_index()
            .melt(id_vars="feature", value_vars=[f"{m}_p" for m in metric_labels],
                var_name="metric_p", value_name="p_value")
        )
        p_long["metric"] = p_long["metric_p"].str.replace("_p", "", regex=False)
        p_long = p_long.drop(columns="metric_p")

        long_df = long_df.merge(p_long, on=["feature", "metric"], how="left")

        # filter for features to plot
        if features_to_corrplot is not None:
            print("Selecting features to plot...")
            long_df = long_df.loc[long_df["feature"].isin(features_to_corrplot), :]

        # plot
        sns.barplot(
            data=long_df,
            y="feature",
            x="correlation",
            hue="metric",
            ax=ax,
            orient="h",
        )

        ax.axvline(0, linewidth=1)
        ax.set_xlabel("Correlation")
        ax.set_ylabel("")
        xticks = np.arange(-0.25, 0.25 + 1e-9, 0.05)  # avoids floating precision cutoff
        ax.set_xticks(xticks)
        ax.set_xlim(-0.3, 0.3)
        ax.xaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_title("")
        ax.invert_yaxis()

        # ---- annotate stars next to each bar (ROBUST) ----
        # Build a lookup: (feature, metric) -> (corr, p)
        lookup = {
            (r.feature, r.metric): (r.correlation, r.p_value)
            for r in long_df.itertuples(index=False)
        }

        # Get y tick positions + labels, sorted bottom->top in data coords
        yticks = ax.get_yticks()
        ylabels = [t.get_text() for t in ax.get_yticklabels()]
        y_map = sorted(zip(yticks, ylabels), key=lambda t: t[0])  # bottom->top

        # x-offset in data units
        x0, x1 = ax.get_xlim()
        dx = 0.012 * (x1 - x0)  # a bit more padding than before

        # seaborn stores bars per hue level in ax.containers
        # Container i corresponds to hue_order[i] (as used in the plot)
        for metric, container in zip(metric_labels, ax.containers):
            # sort bars bottom->top to align with y_map
            bars = sorted(container.patches, key=lambda b: b.get_y())

            for bar, (yt, feature) in zip(bars, y_map):
                corr, p = lookup.get((feature, metric), (np.nan, np.nan))
                stars = _p_to_stars(p)
                if not stars or pd.isna(corr):
                    continue

                x_end = bar.get_x() + bar.get_width()
                y_mid = bar.get_y() + bar.get_height() / 2

                if corr < 0:
                    ax.text(
                        x_end - dx, y_mid, stars,
                        va="center", ha="right", fontsize=9,
                        clip_on=False
                    )
                else:
                    ax.text(
                        x_end + dx, y_mid, stars,
                        va="center", ha="left", fontsize=9,
                        clip_on=False
                    )
        # ---- legend: keep R/τ + add star meaning ----
        handles, labels = ax.get_legend_handles_labels()
        # Add dummy handles for star meaning
        star_handles = [
            mlines.Line2D([], [], color="none", label="o p ≥ 0.05"),
            mlines.Line2D([], [], color="none", label="*  p < 0.05"),
            mlines.Line2D([], [], color="none", label="** p < 0.01"),
            mlines.Line2D([], [], color="none", label="*** p < 0.001"),
            
        ]
        if legend:
            ax.legend(handles + star_handles, labels + [h.get_label() for h in star_handles],
                    title="", loc="best", frameon=True)
        else:
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()


    fig6, axes6 = plt.subplots(nrows=1, ncols=1, figsize=(8, 8), constrained_layout=True)
    _barh_corr(axes6, "clash_rate", features_to_corrplot)
    fig7, axes7 = plt.subplots(nrows=1, ncols=1, figsize=(8, 8), constrained_layout=True)
    _barh_corr(axes7, "relative_distance", features_to_corrplot)

    # _panel_label(axes6[0], "A")
    # _panel_label(axes6[1], "B")

    saved.append(_save_fig(fig6, out_dir, "figure6_correlations_R_vs_tau_clashes"))
    saved.append(_save_fig(fig7, out_dir, "figure7_correlations_R_vs_tau_distances"))

    print(f"Saved {len(saved)} figures → {out_dir}")
    for p in saved:
        print(f" - {p}")
    
    # =========================================================
    # More info for figure captions, discussions etc    
    # =========================================================
   

    cols = [
        "sdf_file",
        "pyKV_volume",
        "fpocket_explicit_pock_vol",
        "fpocket_cavity_pock_vol",
        "fpocket_explicit_lig_vol",
        # "fpocket_cavity_lig_vol",
        "pyKV_surface_area", 
        "pyKV_avg_depth", 
        "pyKV_avg_hydropathy",
        "pyKV_max_depth",
        "medchem.size"

    ]

    # Keep relevant columns
    df_sub = final_df[cols].copy()

    # Add n_comparisons per sdf_file
    df_sub["n_comparisons"] = (
        df_sub.groupby("sdf_file")["sdf_file"]
        .transform("count")
    )

    # Reduce to unique sdf_file rows (one row per ligand)
    df_unique = df_sub.drop_duplicates(subset="sdf_file")

    # Compute Kendall correlations on numeric columns
    other_correlations = (
        df_unique
        .select_dtypes(include="number")
        .corr(method="kendall")
    )

    print("Supplementary Information:")
    print(other_correlations)





    return saved



    




# ============================================================
# __main__ orchestration (exactly one call)
# ============================================================

if __name__ == "__main__":
    # run data extraction
    # kv_args = {"step":0.6, "probe_in":0.4, "probe_out":4.0, "removal_distance":2.4}
    # input_folder = Path("../benchmarks/ours_guidanceBL/ours_guidanceBL_samples/")
    # final_df = full_pipeline(input_folder, dpocket_detection_radius = 10, pykv_args = kv_args, overwrite_existing=True, requested_workers=None)
    
    # run analysis

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_path",
        type=Path,
        default=Path("../benchmarks/ours_guidanceBL/ours_guidanceBL_metrics/"),
        help="Path to metrics folder"
    )

    args = parser.parse_args()

    save_path = args.save_path
    final_df = pd.read_csv(save_path / "detailed_clash_evaluation.csv")
    # # sanity checks
    # print("Shape:", final_df.shape)
    # print("Index unique:", final_df.index.is_unique)
    # print("Columns unique:", final_df.columns.is_unique)
    # print(final_df.info())
    # print(final_df.describe().T)
    # na_col_frac = final_df.isna().mean().sort_values(ascending=False)
    # print("Na columns fraction:\n", na_col_frac)
    # full_na_cols = final_df.columns[final_df.isna().all()]
    # print("Fully NA columns:", list(full_na_cols))
    # na_row_frac = final_df.isna().mean(axis=1)
    # print("Na rows fraction:\n", na_row_frac.describe())
    # full_na_rows = final_df.index[final_df.isna().all(axis=1)]
    # print("Fully NA rows:", list(full_na_rows))
    # print(final_df[["pyKV_volume", "fpocket_cavity_pock_vol", "fpocket_explicit_pock_vol"]].corr(method="kendall"))


    analyze_and_plot_clashes(
        final_df=final_df,
        save_path=save_path,
        features_to_corrplot = [
            "fpocket_cavity_crit4",
            # "fpocket_explicit_crit4",
            "fpocket_cavity_flex",
            "fpocket_explicit_flex",
            "fpocket_explicit_lig_vol",
            # "fpocket_cavity_lig_vol", 
            "fpocket_explicit_volume_score", 
            "fpocket_explicit_charge_score", 
            "fpocket_explicit_hydrophobicity_score", 
            "fpocket_cavity_volume_score", 
            "fpocket_cavity_charge_score", 
            "fpocket_cavity_hydrophobicity_score", 
            "fpocket_explicit_pock_vol", 
            "fpocket_cavity_pock_vol",
            "pyKV_volume",
            "pyKV_surface_area", 
            "pyKV_avg_depth", 
            "pyKV_avg_hydropathy",
            "pyKV_max_depth",
            "protein_dist_to_ca", 
            "protein_sidechain_weight", 
            "medchem.size"
        ]
    )


