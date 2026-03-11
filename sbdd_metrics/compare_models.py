from pathlib import Path
import argparse
import json
from typing import Dict, List, Union, Optional
import pandas as pd


import re
import pandas as pd
import numpy as np

from moldiff.constants import bond_decoder, atom_decoder
from moldiff.atom_bond_ring_distr_barplots import atom_bond_ring_distributions_from_smiles

_MEAN_STD_RE = re.compile(
    r"""^\s*
        (?P<mean>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)   # mean
        \s*\(\s*
        (?P<std>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)    # std
        \s*\)\s*$
    """,
    re.VERBOSE,
)

def split_mean_std_columns(
    df: pd.DataFrame,
    *,
    sample_size: int = 50,
    min_match_fraction: float = 0.6,
    decimal_comma: bool = True,
    drop_original: bool = False,
) -> pd.DataFrame:
    """
    Split columns containing values like "mean (std)" into two numeric columns:
      <col>.mean and <col>.std

    - Leaves word/text columns untouched.
    - Only splits a column if enough non-null entries match the pattern.
    - Returns a new DataFrame (does not mutate input unless drop_original=True with copy=False).

    Args:
        df: input dataframe
        sample_size: number of non-null values to sample per column for deciding whether to split
        min_match_fraction: fraction of sampled non-null values that must match pattern to split
        decimal_comma: if True, also accept comma decimals like "12,3 (4,1)"
        drop_original: if True, drop the original mixed column after splitting

    Returns:
        A new DataFrame with split columns added (and optionally originals dropped).
    """
    out = df.copy()

    for col in list(out.columns):
        s = out[col]
        if not isinstance(s, pd.Series):
            continue

        # Work on string view for detection/parsing, but keep NaNs
        s_str = s.astype("string")

        # Take a sample of non-null values for decision
        non_null = s_str.dropna()
        if non_null.empty:
            continue

        sample = non_null.head(sample_size)

        # Optional: normalize decimal comma for detection/parsing
        if decimal_comma:
            sample_norm = sample.str.replace(",", ".", regex=False)
        else:
            sample_norm = sample

        # Count pattern matches in sample
        matches = sample_norm.str.match(_MEAN_STD_RE)
        match_fraction = float(matches.mean()) if len(matches) else 0.0

        if match_fraction < min_match_fraction:
            continue  # treat as non-mean/std column; leave untouched

        # Parse full column
        parse_src = s_str
        if decimal_comma:
            parse_src = parse_src.str.replace(",", ".", regex=False)

        extracted = parse_src.str.extract(_MEAN_STD_RE)
        # Convert to numeric; invalid becomes NaN
        mean_vals = pd.to_numeric(extracted["mean"], errors="coerce")
        std_vals = pd.to_numeric(extracted["std"], errors="coerce")

        out[f"{col}.mean"] = mean_vals
        out[f"{col}.std"] = std_vals

        if drop_original:
            out.drop(columns=[col], inplace=True)

    return out


def collect_metric_files(base_dir: Union[str, Path]) -> Dict[str, List[str]]:
    """
    Scan base_dir for structure: base_dir/<model_name>/<model_name>_metrics/
    or any subdirectory of <model_name> that ends with '_metrics'.
    Returns a dict: { model_name: [relative/path/to/file1.csv, ...] }
    Only .csv and .pkl files are collected (case-insensitive).
    Paths in lists are relative to base_dir (POSIX-style).
    """
    base = Path(base_dir)
    if not base.is_dir():
        raise NotADirectoryError(f"{base} is not a directory")

    result: Dict[str, List[str]] = {}

    for model_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        model = model_dir.name
        # prefer exact match <model_name>_metrics, otherwise any *_metrics
        metrics_dir: Optional[Path] = model_dir / f"{model}_metrics"
        if not metrics_dir.exists() or not metrics_dir.is_dir():
            metrics_dir = next((d for d in model_dir.iterdir() if d.is_dir() and d.name.endswith("_metrics")), None)
        if metrics_dir is None:
            continue

        files: List[str] = []
        for p in metrics_dir.iterdir(): #rglob("*"):
            if p.is_file() and p.suffix.lower() in {".csv", ".pkl"}:
                files.append(p.relative_to(base).as_posix())

        if files:
            result[model] = sorted(files)

    return result

def stack_and_save_metrics(base_dir: Union[str, Path],
                            files_map: Optional[Dict[str, List[str]]] = None,
                            out_dir: Optional[Union[str, Path]] = None) -> pd.DataFrame:
        """
        Build stacked DataFrames across models:
          - a processed "aggregated" table where each model is one row and
            columns are metrics with values "mean (std)"
          - a "distances" table where each model is one row and columns are distance metrics
          - a concatenated table with both parts side-by-side

        Uses files_map if provided (output of collect_metric_files), otherwise collects files itself.
        Writes three CSVs into out_dir (default: base_dir):
          - metrics_aggregated_stacked.csv
          - metrics_distances_stacked.csv
          - metrics_combined_stacked.csv

        Returns the combined DataFrame (models x (n_agg + n_dist) columns).
        """
        base = Path(base_dir)
        out_dir = Path(out_dir) if out_dir is not None else base

        if files_map is None:
            files_map = collect_metric_files(base)

        def _read(path: Path) -> pd.DataFrame:
            if not path.exists():
                raise FileNotFoundError(path)
            if path.suffix.lower() == ".pkl":
                return pd.read_pickle(path)
            # try csv
            return pd.read_csv(path, index_col=None)

        def _process_aggregated(df: pd.DataFrame) -> Dict[str, str]:
            # normalize columns: look for metric name, mean, std
            cols = {c.lower(): c for c in df.columns}
            # try known column names
            if "mean" in cols and "std" in cols:
                mean_col = cols["mean"]
                std_col = cols["std"]
                if "metric" in cols:
                    metric_col = cols["metric"]
                    names = df[metric_col].astype(str)
                else:
                    # maybe first column is metric names
                    if df.shape[1] > 2:
                        metric_col = df.columns[0]
                        names = df[metric_col].astype(str)
                    else:
                        names = df.index.astype(str)
                means = df[mean_col]
                stds = df[std_col]
            elif df.shape[1] == 3:
                names = df.iloc[:, 0].astype(str)
                means = df.iloc[:, 1]
                stds = df.iloc[:, 2]
            else:
                # fallback: try index as metric names and first numeric cols as mean/std
                names = df.index.astype(str)
                numeric = df.select_dtypes("number")
                if numeric.shape[1] >= 2:
                    means = numeric.iloc[:, 0]
                    stds = numeric.iloc[:, 1]
                elif numeric.shape[1] == 1:
                    means = numeric.iloc[:, 0]
                    stds = pd.Series([pd.NA] * len(means), index=means.index)
                else:
                    # cannot parse, return empty
                    return {}
            out: Dict[str, str] = {}
            for n, m, s in zip(names, means, stds):
                try:
                    m_fmt = f"{float(m):g}"
                except Exception:
                    m_fmt = "" if pd.isna(m) else str(m)
                try:
                    s_fmt = f"{float(s):g}"
                except Exception:
                    s_fmt = "" if pd.isna(s) else str(s)
                val = f"{m_fmt} ({s_fmt})" if s_fmt not in ["nan", ""] else m_fmt
                out[str(n)] = val
            return out

        def _process_distances(df: pd.DataFrame) -> Dict[str, Union[float, str]]:
            # If single column with metric names as index -> use index as keys
            if df.shape[1] == 1:
                col = df.columns[0]
                if df.index.is_unique and not any(str(i).startswith("Unnamed") for i in df.index):
                    return {str(idx): df.iloc[i, 0] for i, idx in enumerate(df.index)}
                else:
                    # single row probably
                    return {str(col): df.iloc[0, 0]}
            # If single row -> use columns as keys
            if df.shape[0] == 1:
                row = df.iloc[0]
                return {str(c): row[c] for c in df.columns}
            # general case: flatten index x columns -> "index__col" keys
            out = {}
            for idx in df.index:
                for col in df.columns:
                    key = f"{idx}__{col}"
                    out[str(key)] = df.at[idx, col]
            return out

        agg_rows: List[Dict[str, str]] = []
        dist_rows: List[Dict[str, Union[float, str]]] = []
        models: List[str] = []

        for model, relpaths in sorted(files_map.items()):
            models.append(model)
            model_dir = base / model
            # convert to Path objects
            paths = [base / p for p in relpaths]

            # heuristics to find aggregated and distances files
            agg_path = next((p for p in paths if "aggreg" in p.name.lower()), None)
            dist_path = next((p for p in paths if "dist" in p.name.lower()), None)

            agg_dict: Dict[str, str] = {}
            dist_dict: Dict[str, Union[float, str]] = {}

            if agg_path is not None and agg_path.exists():
                try:
                    df_agg = _read(agg_path)
                    agg_dict = _process_aggregated(df_agg)
                    print("Included File: ", agg_path)
                except Exception:
                    agg_dict = {}
            if dist_path is not None and dist_path.exists():
                try:
                    df_dist = _read(dist_path)
                    dist_dict = _process_distances(df_dist)
                    print("Included File: ", dist_path)
                except Exception:
                    dist_dict = {}

            agg_rows.append(agg_dict)
            dist_rows.append(dist_dict)

        # build DataFrames, align columns across models
        df_agg = pd.DataFrame(agg_rows, index=models).sort_index()
        df_dist = pd.DataFrame(dist_rows, index=models).sort_index()

        # ensure model name as column (optional) - keep index as model
        df_agg.index.name = "model"
        df_dist.index.name = "model"

        # write separate stacked files
        out_dir.mkdir(parents=True, exist_ok=True)
        agg_out = out_dir / "metrics_aggregated_stacked.csv"
        dist_out = out_dir / "metrics_distances_stacked.csv"
        combined_out = out_dir / "metrics_combined_stacked.csv"

        df_agg.to_csv(agg_out)
        df_dist.to_csv(dist_out)

        # concat side-by-side (columns union)
        df_combined = pd.concat([df_agg, df_dist], axis=1)
        df_combined = split_mean_std_columns(df_combined, drop_original=True)
        df_combined.to_csv(combined_out)

        return df_combined









def main():
    parser = argparse.ArgumentParser(description="Collect .csv and .pkl metric files per model.")
    parser.add_argument("base_dir", nargs="?", default=".", help="Base directory containing model subfolders")
    parser.add_argument("--out", "-o", help="Optional output JSON file to save the dict")
    args = parser.parse_args()

    mapping = collect_metric_files(args.base_dir)
    # print(mapping)
    combined_df = stack_and_save_metrics(args.base_dir, mapping, args.base_dir)

    print("Stacked DataFrame saved at ", args.base_dir)
    
    
    


if __name__ == "__main__":
    # main()
    #  python -m sbdd_metrics.compare_models ../benchmarks/

    import pickle

    # IMPORT to test
    with open("../PDBbind_train/trainingPDB_discrete_distributions.pkl", "rb") as f:
        PDB_distribution_data = pickle.load(f)
    # use this for testing of PDB counts
    atom_counts = dict(zip(atom_decoder, PDB_distribution_data["atom_types"]))
    bond_counts = dict(zip(atom_decoder, PDB_distribution_data["bond_types"]))
    
    with open("../benchmarks/ours_guidanceBL/ours_guidanceBL_metrics/samples_discrete_distributions.pkl", "rb") as f:
        ours_distribution_data = pickle.load(f) # use this for testing by extracting same way as above for ours_guidanceBL


    # extract data
    folders_of_interest = [
        "ours_guidanceBL", # test and report deviance
        "targetdiff", 
        "diffsbdd", 
        "pocket2mol", 
        "drugflow"
    ]
    
    for folder in folders_of_interest:

        folder_data = pd.read_csv(f"../benchmarks/{folder}/{folder}_metrics/metrics_detailed.csv")
        smiles_list = folder_data["representation.smiles"]
        atom_counts, bond_counts, ring_counts = atom_bond_ring_distributions_from_smiles(smiles_list)
    
    # for training data
    smiles_list = np.load(f"../benchmarks/preprocessed_crossdocked/train_smiles.npy", allow_pickle=False)
    atom_counts, bond_counts, ring_counts = atom_bond_ring_distributions_from_smiles(smiles_list)

    # for training data
    smiles_list = np.load(f"../PDB_train/train_smiles.npy", allow_pickle=False)
    atom_counts, bond_counts, ring_counts = atom_bond_ring_distributions_from_smiles(smiles_list) # test and report deviance


    
    # TODO: 
    # - barplot the three distributions with 1, 3 subplots labelled iwth letter A-C:
    # ax.text(
    #     -0.1, 1.05, label,
    #     transform=ax.transAxes,
    #     va="top", ha="left",
    #     # fontname="Arial",
    #     fontsize=14,
    #     fontweight="bold",
    # )
    # - color ours_guidanceBL/PDB in blueish colors and targetdiff, crossdocked etc in redish colors
    


