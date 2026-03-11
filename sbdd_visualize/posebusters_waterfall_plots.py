import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import argparse


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def extract_pocket_name(path_str: str) -> str:
    """
    Given path like: some/dir/.../<pocket_name>/3_ligand.sdf
    Return pocket_name (the directory above the ligand file).
    """
    p = Path(path_str)
    return p.parent.name


def compute_fail_count(row, pb_cols):
    """
    Given PoseBusters boolean columns (True=pass, False=fail),
    compute fail_count = number of False entries.
    """
    values = row[pb_cols].values
    # ensure boolean
    values = values.astype(bool)
    passed = np.sum(values)
    total = len(values)
    return total - passed


def plot_posebusters_metric_waterfall(df, pb_cols, savepath, title="PoseBusters Breakdown"):
    """
    PoseBusters-style sequential metric waterfall.
    Each ligand is removed at the first metric it fails.
    If a metric eliminates 0% (loss=0), draw a green bar from 0 to remaining%.
    """

    N = len(df)
    if N == 0:
        return

    survivors = np.ones(N, dtype=bool)
    remaining_vals = [100.0]
    labels = ["All predictions"]

    fig, ax = plt.subplots(figsize=(14, 5))

    for col in pb_cols:
        passes = df[col].astype(bool).values

        # compute failures among current survivors
        fail_mask = survivors & (~passes)
        lost_count = fail_mask.sum()
        loss_pct = (lost_count / N) * 100.0

        prev_rem = remaining_vals[-1]
        curr_rem = max(0.0, prev_rem - loss_pct)
        remaining_vals.append(curr_rem)

        # Pretty label
        nice_label = col.replace("posebusters.", "").replace("posebusters_", "").replace("_", " ")
        labels.append(nice_label)

        # Plot bar
        idx = len(remaining_vals) - 1

        if loss_pct > 0:  
            # normal orange drop
            ax.bar(idx, prev_rem - curr_rem, bottom=curr_rem, color="#d95f02")
            ax.text(idx, prev_rem + 1, f"-{loss_pct:.0f}%", ha="center", va="bottom", fontsize=8)

        else:
            # zero-loss → green bar upward from zero to remaining%
            ax.bar(idx, curr_rem, bottom=0, color="#1b9e77")
            ax.text(idx, curr_rem + 1, f"{curr_rem:.0f}%", ha="center", va="bottom", fontsize=8)

        survivors &= passes

    # Initial annotation (100%)
    ax.text(0, remaining_vals[0] + 1, "100%", ha="center", va="bottom", fontsize=8)

    # Axis formatting
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=70, ha='right')
    ax.set_ylabel("Remaining %")
    ax.set_ylim(0, 105)
    ax.set_title(title)

    # Remove frame borders
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(savepath, dpi=200)
    plt.close()


def plot_posebusters_metric_waterfall_multi(
    dfs,
    pb_cols,
    savepath,
    labels=None,
    title="PoseBusters Breakdown – Comparison",
):
    """
    Extended PoseBusters waterfall comparing up to 4 models.

    Changes:
    - Removed initial 'All predictions' bar.
    - Added final 'overall survival' bar (green), showing % that passed all filters.
    - Loss bars: red tones.
    - No-loss bars: green tones.
    - Each model gets a unique hatch pattern shared across its bars.
    - Legend uses neutral colored rectangles with hatch patterns to denote models.
    """

    import matplotlib.pyplot as plt
    import numpy as np

    n_paths = len(dfs)
    assert 1 <= n_paths <= 6, "Supports 1–6 paths."

    if labels is None:
        labels = [f"Path {i+1}" for i in range(n_paths)]

    # fontsizes
    textfontsize = 10
    labelfontsize = 12

    # Model hatch patterns
    hatch_patterns = ['/', '\\', 'x', '.', 'o', '+'][:n_paths]

    # Colors
    RED = "#e41a1c"
    GREEN = "#4daf4a"

    # Compute waterfalls
    all_remaining = []
    all_losses = []

    for df in dfs:
        N = len(df)
        survivors = np.ones(N, dtype=bool)
        remaining_vals = [100.0]
        losses = []

        for col in pb_cols:
            passes = df[col].astype(bool).values
            fail_mask = survivors & (~passes)
            loss_pct = 100.0 * fail_mask.sum() / N

            prev = remaining_vals[-1]
            curr = max(0.0, prev - loss_pct)

            remaining_vals.append(curr)
            losses.append(loss_pct)

            survivors &= passes

        all_remaining.append(np.array(remaining_vals))  # includes initial 100%
        all_losses.append(np.array(losses))

    # We drop the first step "All predictions"
    # So metric indices start at 1→len(pb_cols)
    n_metrics = len(pb_cols)

    # +1 for the final "overall survival" bar
    n_positions = n_metrics + 1

    x = np.arange(n_positions)  # 0..M-1 metrics, + final slot
    bar_width = 0.8 / n_paths

    fig, ax = plt.subplots(figsize=(16, 6))

    # -----------------------------------------------------
    # Plot each model
    # -----------------------------------------------------
    for p in range(n_paths):
        hatch = hatch_patterns[p]
        remaining = all_remaining[p]
        losses = all_losses[p]

        # -----------------------------------------------------
        # Plot metric bars (indices 1..M mapped to x 0..M-1)
        # -----------------------------------------------------
        for m in range(n_metrics):
            loss = losses[m]                  # loss for pb_cols[m]
            prev = remaining[m]               # prev step survival %
            curr = remaining[m+1]             # after applying this filter
            xpos = m + (p - (n_paths - 1)/2) * bar_width

            # color direction
            color = RED if loss > 0 else GREEN

            height = prev - curr if loss > 0 else curr

            ax.bar(
                xpos,
                height,
                bottom=(curr if loss > 0 else 0),
                width=bar_width,
                color=color,
                hatch=hatch,
                edgecolor="black",
                alpha=0.9
            )

            txt = f"-{loss:.0f}%" if loss > 0 else f"{curr:.0f}%"
            ax.text(
                xpos,
                prev + 2,
                txt,
                ha="center",
                va="bottom",
                fontsize=textfontsize,
                rotation=90 if n_paths > 1 else 0
            )

        # -----------------------------------------------------
        # Final "overall survival"
        # -----------------------------------------------------
        final_survival = remaining[-1]
        xpos_final = (n_metrics) + (p - (n_paths - 1)/2) * bar_width

        ax.bar(
            xpos_final,
            final_survival,
            bottom=0,
            width=bar_width,
            color=GREEN,     # always green
            hatch=hatch,
            edgecolor="black",
            alpha=1.0
        )

        ax.text(
            xpos_final,
            final_survival + 2,
            f"{final_survival:.0f}%",
            ha="center",
            va="bottom",
            fontsize=textfontsize,
            rotation=90 if n_paths > 1 else 0
        )

    # -----------------------------------------------------
    # X-axis labels
    # -----------------------------------------------------
    xticks = list(range(n_positions))
    xtick_labels = [
        col.replace("posebusters.", "").replace("_", " ")
        for col in pb_cols
    ] + ["overall survival"]

    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels, rotation=70, fontsize=labelfontsize, ha="right")

    ax.set_ylim(0, 110)
    ax.set_ylabel("% survived", fontsize=labelfontsize)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0", "25", "50", "75", "100"], fontsize=labelfontsize)
    # ax.set_title(title)

    # -----------------------------------------------------
    # Legend
    # -----------------------------------------------------
    legend_handles = []
    for p, lbl in enumerate(labels):
        hatch = hatch_patterns[p]
        patch = plt.Rectangle(
            (0, 0), 1, 1,
            facecolor="lightgray",
            edgecolor="black",
            hatch=hatch
        )
        legend_handles.append(patch)

    ax.legend(legend_handles, labels, fontsize=textfontsize, loc=(0.97, 0.5))
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    for s in ax.spines.values():
        s.set_visible(False)

    plt.tight_layout()
    plt.savefig(savepath, dpi=200)
    plt.close()

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--df", type=str, required=True,
#                         help="Path to dataframe containing PoseBusters columns")
#     parser.add_argument("--out", type=str, required=True,
#                         help="Output directory")
#     parser.add_argument("--global_only", action="store_true",
#                         help="Only generate global waterfall plot")
#     args = parser.parse_args()

#     df_path = Path(args.exp_name)
#     out_dir = Path(args.out)
#     out_dir.mkdir(exist_ok=True, parents=True)

#     # Load dataframe
#     df = pd.read_csv(df_path)

#     # Hard-code PoseBusters column order here (must match df columns)
#     pb_cols_order = [
#         # example – replace with the exact columns you want, in the order you want:
#         # "posebusters.mol_cond_loaded",
#         "posebusters.mol_pred_loaded",
#         "posebusters.sanitization",
#         "posebusters.all_atoms_connected",
#         "posebusters.aromatic_ring_flatness",
#         "posebusters.bond_angles",
#         "posebusters.bond_lengths",
#         "posebusters.double_bond_flatness",
#         "posebusters.inchi_convertible",
#         "posebusters.internal_energy",
#         "posebusters.internal_steric_clash",
#         "posebusters.minimum_distance_to_inorganic_cofactors",
#         "posebusters.minimum_distance_to_organic_cofactors",
#         "posebusters.minimum_distance_to_protein",
#         "posebusters.minimum_distance_to_waters",
#         "posebusters.protein-ligand_maximum_distance",
#         "posebusters.volume_overlap_with_inorganic_cofactors",
#         "posebusters.volume_overlap_with_organic_cofactors",
#         "posebusters.volume_overlap_with_protein",
#         "posebusters.volume_overlap_with_waters",
#         "posebusters.all"
#     ]

#     # Verify columns exist
#     for col in pb_cols_order:
#         if col not in df.columns:
#             raise ValueError(f"PoseBusters column missing in df: {col}")

#     # Required file columns
#     if "sdf_file" not in df.columns:
#         raise ValueError("Missing column 'sdf_file'")
#     if "pdb_file" not in df.columns:
#         raise ValueError("Missing column 'pdb_file'")

#     # Extract pocket names
#     print("Extracting pocket names...")
#     df["pocket_name"] = df["sdf_file"].apply(extract_pocket_name)

#     # ------------------------------------------------------------------
#     # Per-pocket waterfalls (unless --global_only is set)
#     # ------------------------------------------------------------------
#     if not args.global_only:
#         print("Generating per-pocket PoseBusters metric waterfalls...")
#         for pocket, group in tqdm(df.groupby("pocket_name")):
#             savepath = out_dir / f"{pocket}_posebusters_metric_breakdown.png"
#             plot_posebusters_metric_waterfall(
#                 group,
#                 pb_cols_order,
#                 savepath,
#                 title=f"PoseBusters Breakdown – {pocket}"
#             )

#     # ------------------------------------------------------------------
#     # Global waterfall
#     # ------------------------------------------------------------------
#     print("Generating global PoseBusters metric waterfall...")
#     plot_posebusters_metric_waterfall(
#         df,
#         pb_cols_order,
#         out_dir / "global_posebusters_metric_breakdown.png",
#         title="PoseBusters Breakdown – Global"
#     )

#     print("Done.")

    # call like:
    # python -m sbdd_metrics.posebusters_waterfall_plots --df ../benchmarks/ours/ours_metrics/metrics_detailed.csv --out ../benchmarks/ours/posebusters_waterfall_plots

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_name",
        type=str,
        nargs="+",
        required=True,
        help="One or more dataframes (max 5) for side-by-side comparison"
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        help="Custom labels for each path (must match number of df files)"
    )
    parser.add_argument("--out", type=str, required=True,
                        help="Output directory")
    parser.add_argument(
        "--global_only",
        action="store_true",
        help="Only generate global multi-path waterfall plot"
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Validate input
    # ------------------------------------------------------------------
    if len(args.exp_name) > 6:
        raise ValueError("Maximum of 6 paths supported.")

    if args.labels and len(args.labels) != len(args.exp_name):
        raise ValueError("Number of --labels must match number of --df files.")

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True, parents=True)

    # ------------------------------------------------------------------
    # Load all dataframes
    # ------------------------------------------------------------------
    dfs = []
    for df in args.exp_name:
        # ./benchmarks/ours_guidanceBL/ours_guidanceBL_metrics/metrics_detailed.csv
        df_path = f"../benchmarks/{df}/{df}_metrics/metrics_detailed.csv"
        df = pd.read_csv(df_path)
        dfs.append(df)

    # ------------------------------------------------------------------
    # PoseBusters column order
    # ------------------------------------------------------------------
    pb_cols_order = [
        # "posebusters.mol_pred_loaded",
        "posebusters.sanitization",
        "posebusters.all_atoms_connected",
        "posebusters.internal_steric_clash",
        "posebusters.aromatic_ring_flatness",
        "posebusters.bond_angles",
        "posebusters.bond_lengths",
        "posebusters.double_bond_flatness",
        # "posebusters.inchi_convertible",
        # "posebusters.internal_energy",
        # "posebusters.minimum_distance_to_inorganic_cofactors",
        # "posebusters.minimum_distance_to_organic_cofactors",
        # "posebusters.minimum_distance_to_waters",
        "posebusters.volume_overlap_with_protein",
        # "posebusters.protein-ligand_maximum_distance",
        "posebusters.minimum_distance_to_protein"
        # "posebusters.volume_overlap_with_inorganic_cofactors",
        # "posebusters.volume_overlap_with_organic_cofactors",
        # "posebusters.volume_overlap_with_protein",
        # "posebusters.volume_overlap_with_waters",
        # "posebusters.all"
    ]

    # Verify columns exist in every DF
    for df_idx, df in enumerate(dfs):
        for col in pb_cols_order:
            if col not in df.columns:
                raise ValueError(f"Missing PB column {col} in df[{df_idx}]")
        if "sdf_file" not in df:
            raise ValueError(f"Missing 'sdf_file' in df[{df_idx}]")
        if "pdb_file" not in df:
            raise ValueError(f"Missing 'pdb_file' in df[{df_idx}]")

    # ------------------------------------------------------------------
    # Extract pocket_name per DF
    # ------------------------------------------------------------------
    print("Extracting pocket names for each dataframe…")
    for df in dfs:
        df["pocket_name"] = df["sdf_file"].apply(extract_pocket_name)

    # ------------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------------
    if args.labels:
        labels = args.labels
    else:
        labels = [f"Path{i+1}" for i in range(len(dfs))]

    # ------------------------------------------------------------------
    # Per-pocket multi-path waterfalls (unless global_only)
    # ------------------------------------------------------------------
    if not args.global_only:
        print("Generating per-pocket multi-path PoseBusters waterfalls...")

        # Use intersection of pocket names across all paths
        common_pockets = set(dfs[0]["pocket_name"])
        for df in dfs[1:]:
            common_pockets = common_pockets & set(df["pocket_name"])

        for pocket in tqdm(sorted(common_pockets)):
            groups = [df[df["pocket_name"] == pocket] for df in dfs]

            savepath = out_dir / f"{pocket}_posebusters_multi_breakdown.png"

            plot_posebusters_metric_waterfall_multi(
                groups,
                pb_cols_order,
                savepath,
                labels=labels,
                title=f"PoseBusters Breakdown – {pocket}"
            )

    # ------------------------------------------------------------------
    # Global multi-path waterfall
    # ------------------------------------------------------------------
    print("Generating global multi-path PoseBusters waterfall...")
    plot_posebusters_metric_waterfall_multi(
        dfs,
        pb_cols_order,
        out_dir / "global_posebusters_multi_breakdown.png",
        labels=labels,
        title="PoseBusters Breakdown – Global"
    )

    print(f"Saved in {out_dir / 'global_posebusters_multi_breakdown.png'}")

# python -m sbdd_metrics.posebusters_waterfall_plots --exp_name ours_guidanceBL  ours_guidance_reworked_1cutoff_relu ours_guidance_reworked_2cutoff_relu ours_guidance_reworked_5cutoff_relu ours_guidance_reworked_10cutoff_relu --labels BL 1 2 5 10 --out ../benchmarks/posebusters_waterfall_plots --global_only
# python -m sbdd_metrics.posebusters_waterfall_plots --exp_name ours_guidanceBL  ours_guidance_reworked_2topk ours_guidance_reworked_5topk ours_guidance_reworked_10topk --labels BL 2 5 10 --out ../benchmarks/posebusters_waterfall_plots --global_only
# python -m sbdd_metrics.posebusters_waterfall_plots --exp_name ours_guidanceBL  ours_guidance_reworked_1cutoff_relu ours_guidance_reworked_2cutoff_relu ours_guidance_reworked_5cutoff_relu ours_guidance_reworked_10cutoff_relu --labels BL 1 2 5 10 --out ../benchmarks/posebusters_waterfall_plots --global_only
# python -m sbdd_metrics.posebusters_waterfall_plots --exp_name ours_guidanceBL  ours_guidance_reworked_2neg_logsumexp ours_guidance_reworked_10neg_logsumexp ours_guidance_reworked_15neg_logsumexp ours_guidance_reworked_50neg_logsumexp --labels BL 2 10 15 30 --out ../benchmarks/posebusters_waterfall_plots --global_only


# for report
# python -m sbdd_metrics.posebusters_waterfall_plots --exp_name ours_guidanceBL  ours_guidance_reworked_10cutoff_relu ours_guidance_reworked_2topk ours_guidance_reworked_2neg_logsumexp --labels BL "Cutoff ReLU (10)" "TopK (2)" "Neg. LogSumExp (2)" --out ../benchmarks/posebusters_waterfall_plots --global_only
# python -m sbdd_metrics.posebusters_waterfall_plots --exp_name ours_guidanceBL  ours_guidance_reworked_2neg_logsumexp ours_guidance_reworked_10neg_logsumexp ours_guidance_reworked_15neg_logsumexp ours_guidance_reworked_50neg_logsumexp --labels BL 2 10 15 30 --out ../benchmarks/posebusters_waterfall_plots --global_only
# python -m sbdd_metrics.posebusters_waterfall_plots --exp_name diffsbdd_reeval drugflow pocket2mol_reeval targetdiff_reeval ours_guidanceBL PDBbind_train --labels DiffSBDD DrugFlow Pocket2Mol TargetDiff Ours PDBbind --out ../benchmarks/posebusters_waterfall_plots --global_only