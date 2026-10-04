#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on January 29 08:24:51 2025
@author: alcantar
modifed by J. Hambalek on 18 July 2025
imported changes from jupyter notebook '09nb_...' 01 August 2025
example run 1: python 14_neutral_drift_analysis_input.py \
-i ../../minibinders_orthorep_data/minibinders_orthorep_outputs/jh_008/mutation_dfs/*mutation_analysis.csv \
-m ../../minibinders_orthorep_data/ngs_raw/jh_008/demultiplex/jh_008_metadata.txt \
-r ../../minibinders_orthorep_data/ngs_raw/jh_008/references/pmaa23_nbonly.fasta \
-c 25 -t 4 -f False
"""
# activate virtual enviroment before running script
# source activate minibinders

import argparse
import json
import math
import re
from itertools import combinations

# import matplotlib
# matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import tqdm
from Bio import SeqIO
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from sklearn.mixture import GaussianMixture
from utils_nd import *


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", nargs="+", help="input mutation analysis csv files")
    parser.add_argument("-m", help="path to metadata for declaration of csvs to bins")
    parser.add_argument("-r", help="Path to reference fasta")
    parser.add_argument(
        "-k",
        "--knee_sensitivity",
        default=None,
        type=int,
        help=(
            "KneeLocator S value used when generating input CSVs with script 02c "
            "(e.g. 0.2). When supplied, only CSVs whose filename contains "
            "'S{value}' are used — allowing multiple CSV sets at different "
            "sensitivities to coexist in the same folder. Omit to use all "
            "passed CSVs regardless of sensitivity."
        ),
    )
    parser.add_argument("-t", help="count threshold for filtering")
    parser.add_argument(
        "-f",
        default="False",
        help="boolean value for applying score dissimilarity removal",
    )
    parser.add_argument(
        "-s",
        default="False",
        help="boolean value for choosing high or low replicate correlation for threshold stringency",
    )
    parser.add_argument(
        "-c",
        "--concentration",
        default=0,
        type=float,
        help="concentration value to select subset of CSVs for scoring",
    )

    args = parser.parse_args()

    ref_fasta_path = (
        args.r
    )  #'../../minibinders_data/ngs_raw/jh_001/references/pjh3_nbonly.fasta' # input
    expt_id = ref_fasta_path.split("/")[
        -3
    ]  # experiment id is the entry past the 4th slash
    nanobody_id = ref_fasta_path.split("/")[-1].replace("_nbonly.fasta", "")
    print(f"experiment id: {expt_id}")

    # out_dir = f'../../minibinders_orthorep_data/minibinders_orthorep_outputs/{expt_id}/neutral_drift_library/'
    # make_dir(out_dir)
    # out_dir_figs = out_dir+'plots/'
    # make_dir(out_dir_figs)
    # out_dir_reps = out_dir+'rep_scores/'
    # make_dir(out_dir_reps)

    # define parent / best sequence
    ref_fasta = SeqIO.read(ref_fasta_path, "fasta")
    parent_dna_seq = ref_fasta.seq
    parent_dna_id = ref_fasta.id
    # define parent / best sequence

    knee_sensitivity = args.knee_sensitivity  # None means "use all passed CSVs"
    concentration = args.concentration
    # Format concentration as an integer string when it is a whole number
    # (e.g. 16.0 -> "16", 3.2 -> "3.2") so it matches filenames like "16nM" not "16.0nM"
    conc_str = (
        str(int(concentration))
        if concentration == int(concentration)
        else str(concentration)
    )

    metadata_path = args.m
    metadata_df = pd.read_csv(metadata_path, sep="\t", index_col=None)
    metadata_df["antigen_concentration_nM"] = metadata_df[
        "antigen_concentration_nM"
    ].astype(float)
    metadata_df["bin"] = metadata_df["bin"].astype(int)
    # print(metadata_df)
    rep_total = max(metadata_df["replicate"])
    bin_total = max(metadata_df["bin"])
    # print(f"{str(rep_total)} total replicates with {str(bin_total + 1)} bins each")

    all_bin_seqs = args.i
    if knee_sensitivity is not None and concentration is not None:
        keyword1 = f"{conc_str}nM"
        # keyword2 = f"S{knee_sensitivity}"
        bin_seq_paths = [
            path for path in all_bin_seqs if isinstance(path, str) and keyword1 in path
        ]
        print(
            f"filtering CSVs to concentration {concentration}nM and knee sensitivity S={knee_sensitivity} ({len(bin_seq_paths)} matched)"
        )
    else:
        bin_seq_paths = [path for path in all_bin_seqs if isinstance(path, str)]
        print(
            f"no knee sensitivity filter applied — using all {len(bin_seq_paths)} passed CSVs"
        )
    # print(bin_seq_paths)

    # dynamic architecture placing neutral drift library wherever mutation_dfs are sourced

    # print(bin_seq_paths[0])
    dir_prefix = "/".join(bin_seq_paths[0].split("/")[:-2])
    print(f"output directory: {dir_prefix}")
    # copy of exports to be used in next script in pipeline
    export_dir = dir_prefix + "/library_dfs/"
    make_dir(export_dir)
    # out_dir = f'../../minibinders_orthorep_data/minibinders_orthorep_outputs/{expt_id}/neutral_drift_library/'
    out_dir = dir_prefix + f"/ts_library_{conc_str}nM_sens{knee_sensitivity}/"
    # print(out_dir)
    make_dir(out_dir)
    out_dir_figs = out_dir + "plots/"
    make_dir(out_dir_figs)
    out_dir_reps = out_dir + "rep_scores/"
    make_dir(out_dir_reps)

    replicate_declare_dict = {}
    metadata_conc = metadata_df[
        metadata_df["antigen_concentration_nM"] == concentration
    ].copy()
    # print(metadata_conc)

    for rep in list(range(rep_total + 1))[1:]:
        try:
            metadata_replicate = metadata_conc[metadata_conc["replicate"] == rep].copy()
            # print(f'replicate {rep}')
            # print(metadata_replicate)
            rep_bin_list = []
            # print(list(range(bin_total+1)))
            for binny in list(range(bin_total + 1)):
                metadata_slice = metadata_replicate[
                    metadata_replicate["bin"] == binny
                ].copy()
                sample_name = metadata_slice["binder_name"].tolist()[0]
                # CSV filenames drop underscores from the binder name
                # (e.g. "55_b0_3.2nM" -> "55b0_3.2nM"), so normalise before matching
                sample_name_norm = sample_name.replace("_", "", 1)
                # print(
                #     f"  [DEBUG] rep={rep} bin={binny} | sample_name='{sample_name}' | sample_name_norm='{sample_name_norm}'"
                # )
                # print(f"  [DEBUG] bin_seq_paths ({len(bin_seq_paths)} total):")
                # for p in bin_seq_paths:
                #     print(f"    {p}")
                rep_bin_samp = [
                    path2
                    for path2 in bin_seq_paths
                    if isinstance(path2, str)
                    and (sample_name in path2 or sample_name_norm in path2)
                ]
                # print(f"  [DEBUG] matched paths: {rep_bin_samp}")
                rep_bin_list.append(rep_bin_samp[0])
            replicate_declare_dict[rep] = rep_bin_list
        except Exception as e:
            print(f"replicate {rep} and bin {binny} not found in metadata! \n{e}")

    # print(replicate_declare_dict)

    count_threshold = int(args.t)
    reps_dict = {}
    reptotal_counts = []
    for replicate in replicate_declare_dict.keys():
        # derive cell fractions by finding all cell counts associated with the replicate
        metadata_rep = metadata_conc[metadata_conc["replicate"] == replicate].copy()
        cell_counts_rep = metadata_rep["cell_count"].tolist()
        sum_cell_counts = np.sum(cell_counts_rep)
        # print(cell_counts_rep)
        # print(sum_cell_counts)

        score_init = []

        bin_list = replicate_declare_dict[replicate]
        # print(bin_list)
        for scorenum, sample in enumerate(bin_list):
            bin_df = pd.read_csv(sample, index_col=0)
            bin_df_agg = aggregate_by_aa_sequence(bin_df)
            counts_dict = dict(zip(bin_df_agg["aa_sequence"], bin_df_agg["read_count"]))
            # strip the thresh/S suffix produced by script 02c to recover the bare
            # sample name; falls back gracefully for older cpm-style filenames
            raw_fname = sample.split("/")[-1]
            import re as _re

            # normalize the filename by stripping the trailing `_mutation_analysis.csv`
            raw_base = _re.sub(r"_mutation_analysis\.csv$", "", raw_fname)
            # remove diagnostic infixes or thresh/cpm-style suffixes (flexible single-regex)
            sample_name = _re.sub(r"(_diagnostic.*|_?thresh\d+.*|_?cpm[\d.]+.*)$", "", raw_base)

            # print(sample_name)
            # sample_name from the filename has no underscore between replicate and bin
            # (e.g. "55b0_3.2nM") but binder_name in metadata is "55_b0_3.2nM".
            # Re-insert the underscore after the leading digits to recover the original.
            # create a metadata-style binder name (insert underscore where digits+"b" occur)
            sample_name_meta = _re.sub(r"^(\d+)(b)", r"\1_\2", sample_name)

            # flexible matching: normalize binder names by removing non-alphanumerics and lowercasing
            def _normalize_binder(s):
                return _re.sub(r"[^0-9a-zA-Z\.]+", "", str(s).lower())

            cand_norms = {_normalize_binder(sample_name), _normalize_binder(sample_name_meta)}

            # try exact normalized match first
            metadata_rep = metadata_rep.copy()
            metadata_rep["binder_name_norm"] = metadata_rep["binder_name"].apply(_normalize_binder)
            metadata_samp = metadata_rep[metadata_rep["binder_name_norm"].isin(cand_norms)].copy()

            # fallback: try substring match of normalized candidates within normalized binder_name
            if metadata_samp.empty:
                metadata_samp = metadata_rep[
                    metadata_rep["binder_name_norm"].apply(
                        lambda x: any(c in x for c in cand_norms)
                    )
                ].copy()

            # if still empty, raise informative error to help debugging
            if metadata_samp.empty:
                raise ValueError(
                    f"No metadata entry found for sample '{raw_fname}' (normalized candidates: {cand_norms})."
                )

            # proceed with matched metadata
            bin_value = int(metadata_samp["bin"].item())
            cell_count_value = int(metadata_samp["cell_count"].item())
            cell_frac = cell_count_value / sum_cell_counts
            # print(cell_frac)
            # cell_counts.append(cell_count_value)
            # bin_df_agg = bin_df_agg.rename(columns={'read_count': f'bin_{bin_value}'},inplace=True)
            bin_df_agg[f"bin_{str(bin_value)}"] = (
                bin_df_agg["aa_sequence"].map(counts_dict).fillna(0).astype(int)
            )

            # debug module for determining mapping
            # debug_seq = []
            # for seq in bin_df_agg['aa_sequence']:
            #     if bin_df_agg[bin_df_agg['aa_sequence'] == seq][f'bin_{str(bin_value)}'].item() != bin_df_agg[bin_df_agg['aa_sequence'] == seq]['read_count'].item():
            #         debug_seq.append(bin_df_agg[seq].index)
            # print(f'{len(debug_seq)} sequences mapped incorrectly')

            # print(bin_df_agg[bin_df_agg[f'bin_{str(bin_value)}'] >= count_threshold])

            # test subset of sequences to figure out the issue
            idxs = list(range(len(bin_df_agg)))
            # idxs = list(range(8))

            # print(idxs[:3])
            for idx in idxs:
                # if bin_df_agg.loc[idx, f'bin_{str(bin_value)}'] >= count_threshold:
                score_init.append(
                    {
                        "aa_sequence": bin_df_agg.loc[idx, "aa_sequence"],
                        "dna_sequence": bin_df_agg.loc[idx, "dna_sequence"],
                        "aa_mutations": bin_df_agg.loc[idx, "aa_mutations"],
                        "dna_mutations": bin_df_agg.loc[idx, "dna_mutations"],
                        "number_aa_mutations": bin_df_agg.loc[
                            idx, "number_aa_mutations"
                        ],
                        f"bin_{str(bin_value)}": bin_df_agg.loc[
                            idx, f"bin_{str(bin_value)}"
                        ],
                        f"bin_{str(bin_value)}_norm": (
                            bin_df_agg.loc[idx, f"bin_{str(bin_value)}"] * cell_frac
                        ),
                    }
                )

                # print(score_init)

        rep_df = pd.DataFrame(score_init).fillna(0)
        # print(rep_df[:5])
        # Identify bin columns robustly
        bin_cols = [
            c for c in rep_df.columns if re.match(r"^bin_?\d+(?:_norm)?$", str(c))
        ]
        # print(bin_cols)

        # Ensure numeric, treat NaN as 0
        rep_df[bin_cols] = (
            rep_df[bin_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        )

        # Clean sequence key (avoid hidden dupes from whitespace/case)
        rep_df["aa_sequence"] = rep_df["aa_sequence"].astype(str).str.strip()

        # everything else (except the key)
        meta_cols = [c for c in rep_df.columns if c not in ["aa_sequence"] + bin_cols]

        agg_map = {**{c: "sum" for c in bin_cols}, **{c: "first" for c in meta_cols}}

        # Collapse: sum all bin columns per sequence
        rep_df_merged = rep_df.groupby("aa_sequence", as_index=False).agg(agg_map)

        bin_cols_norm = [
            c for c in rep_df.columns if re.match(r"^bin_?\d+(?:_norm)$", str(c))
        ]

        # ensure numeric and drop sequences not read enough
        rep_df_merged[bin_cols] = (  # type: ignore
            rep_df_merged[bin_cols].apply(pd.to_numeric, errors="coerce").fillna(0)  # type: ignore
        )

        bin_cols_raw = [c for c in bin_cols if c not in bin_cols_norm]

        read_total = rep_df_merged[bin_cols_raw].sum(axis=1)
        print(f"read total: {len(read_total)}")
        reptotal_counts.append(len(read_total))

        # rep_df_sorted = rep_df_merged.assign(read_sum=

        # Drop rows below threshold
        rep_df_merged = rep_df_merged.loc[read_total >= count_threshold].reset_index(
            drop=True
        )

        for binny in list(range(bin_total + 1)):
            # rep_df_merged[f'bin_{binny}_norm'] = rep_df_merged[f'bin_{binny}_norm']/np.sum(cell_counts) # cell fraction normalization
            # print(int(np.sum(rep_df_merged[f'bin_{binny}'])))
            rep_df_merged[f"bin_{binny}_norm"] = rep_df_merged[
                f"bin_{binny}_norm"
            ] / int(np.sum(rep_df_merged[f"bin_{binny}"]))  # read count normalization

        # row wise normalization

        rep_df_merged[bin_cols_norm] = rep_df_merged[bin_cols_norm].div(
            rep_df_merged[bin_cols_norm].sum(axis=1), axis=0
        )

        # add normqlized score
        # norm_counts = rep_df_merged[bin_cols_norm]
        bin_weights = [i / bin_total for i in range(bin_total + 1)]
        # print(bin_weights)
        rep_df_merged["norm_score"] = rep_df_merged[bin_cols_norm].dot(
            np.asarray(bin_weights, dtype=float)
        )

        # print(rep_df_merged)
        # print(len(rep_df), len(rep_df_merged))
        # print(rep_df_merged[rep_df_merged['aa_mutations'] == '[]'])
        reps_dict[f"replicate_{replicate}"] = rep_df_merged

        rep_out_name = (
            out_dir_reps
            + f"replicate_{replicate}_{expt_id}_countthresh{count_threshold}"
            + ".csv"
        )

        rep_df_merged.to_csv(rep_out_name)
    # print(reps_dict)

    # bringing all dfs together into one dataframe

    scored_df_long = pd.concat(
        [df.assign(rep_key=key) for key, df in reps_dict.items()], ignore_index=True
    )
    # print(scored_df_long[:5])
    scored_df_agg = (
        scored_df_long.groupby("aa_sequence")[bin_cols].agg(list).reset_index()
    )

    # check duplicates
    scored_df_dupes = scored_df_long.duplicated(
        subset=["aa_sequence", "rep_key"], keep=False
    )

    # build dataframes compiling scores for a given sequence into one row by replicate
    scored_df_wide = scored_df_long.pivot(
        index="aa_sequence", columns="rep_key", values="norm_score"
    ).reset_index()

    scored_df_wide.fillna("na").to_csv(
        out_dir_reps + f"{expt_id}_allscores_countthresh{count_threshold}" + ".csv"
    )

    scored_df_wide.columns.name = "."

    # print(scored_df_reps[:4])
    rep_cols = list(reps_dict.keys())
    mask_all_present = (
        scored_df_wide[rep_cols].notna().all(axis=1)
    )  # keep only rows with no NaNs in rep_cols
    scored_df_reps = scored_df_wide.loc[mask_all_present].reset_index(drop=True)
    # print(scored_df_reps[:4])

    # add back the rest of the metadata
    scored_meta_cols = [
        col
        for col in scored_df_long.columns
        if col not in ["norm_score", "rep_key"] + rep_cols + bin_cols
    ]
    scored_meta = scored_df_long[scored_meta_cols].drop_duplicates("aa_sequence")
    scored_bins = (
        scored_df_agg[["aa_sequence"] + bin_cols]
        .drop_duplicates(subset="aa_sequence", keep="first")
        .reset_index(drop=True)
    )

    scored_df_all = scored_df_reps.merge(
        scored_meta, on="aa_sequence", how="inner"
    ).merge(scored_bins, on="aa_sequence", how="inner")
    # print(scored_df_all[:5])

    # identify replicates with drastically different scores
    diff_reps = []
    diff_val = math.ceil(1 / 3 * 100) / 100
    for k, row in tqdm.tqdm(scored_df_all.iterrows(), total=scored_df_all.shape[0]):
        replicate_diff = row[rep_cols].std() * 2
        if replicate_diff > diff_val:
            diff_reps.append(k)

    # print(reptotal_counts[0])
    # print(reptotal_counts[1])
    # print(len(scored_df_all))
    venn = simple_venn(reptotal_counts[0], reptotal_counts[1], len(scored_df_all))

    venn_out_name = out_dir + f"library_overlap_{expt_id}"

    venn.savefig(venn_out_name + ".png", dpi=400)

    plt.clf()

    scored_df_uniform = scored_df_all.copy()
    apply_filt = args.f
    # print(apply_filt)
    if apply_filt.lower() == "true":
        print("applying filtering")
        for k2 in diff_reps:
            scored_df_uniform = scored_df_uniform.drop(k2)
        filtered = True
    else:
        filtered = False
    print(
        f"identified {len(diff_reps)} sequences with dissimilar replicate scores (|diff| > {diff_val})"
    )

    # --- Validate X before GMM fit ---
    # Confirm rep_cols exist in scored_df_uniform and contain only norm_score values
    missing = [c for c in rep_cols if c not in scored_df_uniform.columns]
    if missing:
        raise KeyError(
            f"rep_cols missing from scored_df_uniform after merges: {missing}"
        )

    X_df = scored_df_uniform[rep_cols].apply(pd.to_numeric, errors="coerce")
    n_nan = X_df.isna().sum().sum()
    if n_nan > 0:
        print(f"  [WARN] {n_nan} NaN values in X — dropping affected rows before GMM")
        X_df = X_df.dropna()
        scored_df_uniform = scored_df_uniform.loc[X_df.index].reset_index(drop=True)
        X_df = X_df.reset_index(drop=True)

    print(f"scoring matrix: {X_df.shape[0]} sequences x {X_df.shape[1]} replicates")
    for col in X_df.columns:
        print(
            f"  {col}: min={X_df[col].min():.4f}  max={X_df[col].max():.4f}  mean={X_df[col].mean():.4f}"
        )

    X = X_df.values

    # --- BIC-optimal GMM capped at n=3 ---
    # There are only 3 biological categories (Inactive / Active / Best Activity),
    # so searching beyond n=3 models intra-category density variation rather than
    # biologically meaningful groups — BIC keeps improving but the extra components
    # split one zone into two, inflating one category at the expense of others.
    mean_score = X_df.mean(axis=1).values

    best_gmm = None
    best_bic = np.inf
    best_n = 1
    for n in range(1, 4):  # max 3 components — one per biological category
        g = GaussianMixture(n_components=n, covariance_type="full", random_state=42)
        g.fit(X)
        bic = g.bic(X)
        print(f"  GMM n={n}: BIC={bic:.2f}  converged={g.converged_}")
        if bic < best_bic:
            best_bic = bic
            best_gmm = g
            best_n = n

    print(f"GMM selected n_components={best_n} (lowest BIC={best_bic:.2f})")

    gmm_labels = best_gmm.predict(X)
    component_means = best_gmm.means_
    component_weights = best_gmm.weights_

    # Sort components by their scalar mean score (low -> high)
    sorted_idx = np.argsort(component_means.mean(axis=1))
    sorted_means = component_means.mean(axis=1)[sorted_idx]
    for rank, ci in enumerate(sorted_idx):
        print(
            f"  component {rank}: mean={component_means[ci].mean():.4f}  weight={component_weights[ci]:.4f}  ({component_weights[ci] * len(X):.0f} seqs)"
        )

    # Merge components that fall in the same biological zone before labelling.
    # Two components both below 0.15 both represent Inactive; two above 0.85
    # both represent Best Activity — collapsing them prevents one zone from
    # being over-represented when BIC favours splitting a dense region.
    INACTIVE_THRESHOLD = 0.15
    BEST_ACT_THRESHOLD = 0.85

    raw_labels = np.empty(len(X), dtype=object)
    for comp_idx in sorted_idx:
        mask = gmm_labels == comp_idx
        comp_mean = component_means[comp_idx].mean()
        if comp_mean <= INACTIVE_THRESHOLD:
            raw_labels[mask] = "Inactive"
        elif comp_mean >= BEST_ACT_THRESHOLD:
            raw_labels[mask] = "Best Activity"
        else:
            raw_labels[mask] = "Active"

    # --- Fallback 1: no Inactive component found ---
    # Entire library scored high — use lower tertile as Inactive
    if "Inactive" not in raw_labels:
        print(
            "  [WARN] No Inactive component found — assigning lower tertile as Inactive"
        )
        t33 = np.percentile(mean_score, 33)
        raw_labels[mean_score <= t33] = "Inactive"

    # --- Fallback 2: no Active or Best Activity found ---
    # Entire library scored near zero — force percentile-based split
    if "Active" not in raw_labels and "Best Activity" not in raw_labels:
        print(
            "  [WARN] All sequences Inactive — forcing percentile split for Active/Best Activity"
        )
        t90 = np.percentile(mean_score, 90)
        t95 = np.percentile(mean_score, 95)
        raw_labels[mean_score >= t90] = "Active"
        raw_labels[mean_score >= t95] = "Best Activity"

    # --- Fallback 3: Active exists but no Best Activity ---
    # Split Active pool at its own median
    if "Best Activity" not in raw_labels and "Active" in raw_labels:
        print("  [WARN] No Best Activity component — splitting Active pool at median")
        active_mask = raw_labels == "Active"
        split_threshold = np.median(mean_score[active_mask])
        raw_labels[(raw_labels == "Active") & (mean_score >= split_threshold)] = (
            "Best Activity"
        )

    # --- Fallback 4: no Active (only Inactive + Best Activity, nothing in between) ---
    # Split Best Activity at its own lower quartile to recover an Active group
    if "Active" not in raw_labels:
        print(
            "  [WARN] No Active component — splitting Best Activity lower quartile into Active"
        )
        best_mask = raw_labels == "Best Activity"
        split_threshold = np.percentile(mean_score[best_mask], 25)
        raw_labels[(raw_labels == "Best Activity") & (mean_score < split_threshold)] = (
            "Active"
        )

    # --- Post-GMM hard gates ---
    # The GMM boundary is a probabilistic surface that can misplace sequences
    # sitting in dense transition regions. These deterministic guards override
    # the GMM label when per-replicate score evidence is unambiguous.

    rep1 = X_df.iloc[:, 0].values
    rep2 = X_df.iloc[:, 1].values

    # Gate 1: both replicates below the inactive threshold → force Inactive
    # Catches sequences the GMM boundary pulled into Active whose actual bin
    # profiles are entirely in bins 0+1 (no signal in either replicate).
    both_inactive = (rep1 <= INACTIVE_THRESHOLD) & (rep2 <= INACTIVE_THRESHOLD)
    n_gate1 = ((raw_labels != "Inactive") & both_inactive).sum()
    if n_gate1 > 0:
        print(
            f"  [GATE 1] {n_gate1} sequences forced to Inactive (both reps ≤ {INACTIVE_THRESHOLD})"
        )
    raw_labels[both_inactive] = "Inactive"

    # Gate 2: either replicate below the Active midpoint → downgrade Best Activity to Active
    # Catches high-disagreement sequences where one replicate has bin_0 contamination
    # pulling its score well below the Best Activity range, meaning the evidence
    # for Best Activity is one-sided and unreliable.
    active_midpoint = (INACTIVE_THRESHOLD + BEST_ACT_THRESHOLD) / 2
    either_low = (rep1 < active_midpoint) | (rep2 < active_midpoint)
    n_gate2 = ((raw_labels == "Best Activity") & either_low).sum()
    if n_gate2 > 0:
        print(
            f"  [GATE 2] {n_gate2} sequences downgraded Best Activity → Active (one rep < {active_midpoint:.2f})"
        )
    raw_labels[(raw_labels == "Best Activity") & either_low] = "Active"

    categories = raw_labels
    print("category assignment:")
    for cat in ["Inactive", "Active", "Best Activity"]:
        n_cat = (categories == cat).sum()
        pct = 100 * n_cat / len(categories)
        print(f"  {cat}: {n_cat} sequences ({pct:.1f}%)")

    # Assign to dataframe
    scored_df_uniform["predicted_category"] = categories

    # Map categories to colors for visualization
    category_colors = {
        "Inactive": "black",
        "Active": "#3B71B2",
        "Best Activity": "orange",
    }
    scored_df_uniform["color"] = scored_df_uniform["predicted_category"].map(
        category_colors
    )

    # print(scored_df_uniform)
    out_name = (
        out_dir
        + f"dyn2_library_scores_{expt_id}_countthresh{count_threshold}"
        + ("filtered" if filtered == True else "")
        + ".csv"
    )

    export_name = (
        export_dir
        + f"{conc_str}nmS{knee_sensitivity}T{count_threshold}"
        + ("filtered" if filtered == True else "")
        + ".csv"
    )

    scored_df_uniform["dna_sequence"] = scored_df_uniform["dna_sequence"].apply(
        json.dumps
    )

    # # Read back and restore to list (if needed to process downstream load json
    # df2 = pd.read_csv('out.csv', converters={'seq_list': json.loads})
    # print(scored_df_uniform[scored_df_uniform['aa_mutations'] == '[]'])
    scored_df_uniform.drop("color", axis=1).to_csv(out_name)
    scored_df_uniform.drop("color", axis=1).to_csv(export_name)

    scored_df_reps1 = scored_df_uniform[
        scored_df_uniform["replicate_1"] > scored_df_uniform["replicate_2"]
    ]
    scored_df_reps2 = scored_df_uniform[
        scored_df_uniform["replicate_2"] > scored_df_uniform["replicate_1"]
    ]

    if len(scored_df_reps1) < len(scored_df_reps2):
        out_name_rep = (
            out_dir
            + f"rep1over2_{expt_id}_countthresh{count_threshold}"
            + ("filtered" if filtered == True else "")
            + ".csv"
        )
        scored_df_reps1.drop("color", axis=1).to_csv(out_name_rep)
    else:
        out_name_rep = (
            out_dir
            + f"rep2over1_{expt_id}_countthresh{count_threshold}"
            + ("filtered" if filtered == True else "")
            + ".csv"
        )
        scored_df_reps2.drop("color", axis=1).to_csv(out_name_rep)

    # initialize figure generation
    fig, ax = plt.subplots(figsize=(10, 8))
    comp_cols = [
        c for c in rep_cols if pd.api.types.is_numeric_dtype(scored_df_uniform[c])
    ]
    corr = scored_df_uniform[comp_cols].corr(method="pearson")
    # ignore diagonal and duplicate pairs by masking upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    corr_masked = corr.mask(mask)

    stringency = args.s

    if stringency.lower() == "true":
        # option to use min value for read count stringency
        corr_idx = corr_masked.stack().idxmin()
        print(corr_idx)
    else:
        # otherwise take find max value of correlations (default)
        corr_idx = corr_masked.stack().idxmax()

    best_i, best_j = corr_idx
    best_score = corr_masked.loc[best_i, best_j]

    if stringency.lower() == "true":
        # read out the lowest correlation (could be bad)
        print(f"Worst pair: ({best_i},{best_j}) (|pearson|={best_score:.3f})")
    else:
        # standard best print statement
        print(f"Best pair: ({best_i},{best_j}) (|pearson|={best_score:.3f})")

    # plot scatter of best reps
    ax.scatter(
        scored_df_uniform[best_i],
        scored_df_uniform[best_j],
        c=scored_df_uniform["color"],
        edgecolors="black",
        s=100,
        alpha=0.75,
        label="Predicted categories",
    )

    # Highlight parent DNA sequence
    parent_mask = scored_df_uniform["aa_mutations"] == "[]"

    ax.scatter(
        scored_df_uniform[best_i][parent_mask],
        scored_df_uniform[best_j][parent_mask],
        c="orange",
        edgecolors="black",
        s=750,
        linewidth=2,
        label="Parent DNA sequence",
    )

    # Get the range for the y=x line
    min_val = min(min(scored_df_uniform[best_i]), min(scored_df_uniform[best_j]))
    max_val = max(max(scored_df_uniform[best_i]), max(scored_df_uniform[best_j]))

    # Plot y=x line
    plt.plot([min_val, max_val], [min_val, max_val], "b--", label="y=x")

    # Add labels, legend, and title
    ax.set_xlabel(f"Normalized Score {best_i}", fontsize=16)
    ax.set_ylabel(f"Normalized Score {best_j}", fontsize=16)
    ax.set_title(
        f"GMM clustering of FluA (F9) nanobody activity library {nanobody_id}",
        fontsize=16,
    )
    ax.legend(fontsize=12, loc="best")

    textstr = f"Pearson's r: {best_score:.2f}"
    plt.gca().text(
        0.01,
        0.8,
        textstr,
        transform=plt.gca().transAxes,
        fontsize=16,
        verticalalignment="top",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    out_name_fig = (
        out_dir_figs
        + f"dyn2_library_scores_{expt_id}_countthresh{count_threshold}"
        + ("filtered" if filtered == True else "")
    )

    # save plot
    plt.savefig(out_name_fig + ".png", dpi=400)
    # plt.savefig(out_name_fig+'.pdf', dpi=400)
    plt.savefig(out_name_fig + ".svg", dpi=400)
    plt.close()

    total_seqs = scored_df_uniform.shape[0]
    num_best_seqs = (scored_df_uniform["predicted_category"] == "Best Activity").sum()
    num_active_seqs = (scored_df_uniform["predicted_category"] == "Active").sum()
    active_best_seqs = num_best_seqs + num_active_seqs
    num_unactive_sequences = (
        scored_df_uniform["predicted_category"] == "Inactive"
    ).sum()

    percent_best = num_best_seqs / total_seqs * 100
    percent_active = num_active_seqs / total_seqs * 100

    percent_best_active = (num_best_seqs + num_active_seqs) / total_seqs * 100

    percent_inactive = num_unactive_sequences / total_seqs * 100

    text_file_name = out_dir + "results_summary.txt"
    with open(text_file_name, "w") as file:
        file.write(
            f"{percent_best:.2f}% of sequences ({num_best_seqs}/{total_seqs}) are Best Activity.\n"
        )
        file.write(
            f"{percent_best_active:.2f}% of sequences ({num_best_seqs + num_active_seqs}/{total_seqs}) are active.\n"
        )
        file.write(
            f"{percent_inactive:.2f}% of sequences ({num_unactive_sequences}/{total_seqs}) are inactive."
        )

    # plot hamming distances
    # plot all hamming
    # plot_hamming_distances(annotated_counts_df=scored_df_uniform,
    #                        out_name_1='hamming_vs_parent_all' ,
    #                        out_name_2='pairwise_hamming_all',
    #                        expt_id=expt_id)

    best_annotated_counts_df = scored_df_uniform.copy()[
        scored_df_uniform["predicted_category"] == "Best Activity"
    ]
    active_annotated_counts_df = scored_df_uniform.copy()[
        scored_df_uniform["predicted_category"] == "Active"
    ]
    inactive_annotated_counts_df = scored_df_uniform.copy()[
        scored_df_uniform["predicted_category"] == "Inactive"
    ]

    plot_hamming_distances(
        annotated_counts_df=best_annotated_counts_df,
        out_dir=out_dir_figs,
        out_name_1="hamming_vs_parent_Best",
        out_name_2="pairwise_hamming_Best",
        expt_id=expt_id,
        color="orange",
    )

    plot_hamming_distances(
        annotated_counts_df=active_annotated_counts_df,
        out_dir=out_dir_figs,
        out_name_1="hamming_vs_parent_active",
        out_name_2="pairwise_hamming_active",
        expt_id=expt_id,
        color="#3B71B2",
    )

    plot_hamming_distances(
        annotated_counts_df=inactive_annotated_counts_df,
        out_dir=out_dir_figs,
        out_name_1="hamming_vs_parent_inactive",
        out_name_2="pairwise_hamming_inactive",
        expt_id=expt_id,
        color="grey",
    )

    # plot versus parent
    num_sequences = scored_df_uniform.shape[0]
    # Set the Seaborn style and context
    sns.set(style="white")

    # Set global font size and family
    plt.rcParams.update({"font.size": 12, "font.family": "Arial"})
    # Set the Seaborn style
    sns.set_style("ticks")

    best_mean = np.mean(best_annotated_counts_df["number_aa_mutations"])
    best_median = np.median(best_annotated_counts_df["number_aa_mutations"])

    active_mean = np.mean(active_annotated_counts_df["number_aa_mutations"])
    active_median = np.median(active_annotated_counts_df["number_aa_mutations"])

    inactive_mean = np.mean(inactive_annotated_counts_df["number_aa_mutations"])
    inactive_median = np.median(inactive_annotated_counts_df["number_aa_mutations"])

    # histogram plots and KDE lines (match colors)
    fig, ax = plt.subplots(figsize=(7, 5))

    # Combined domain (mutations are non-negative integers)
    combined = [
        best_annotated_counts_df["number_aa_mutations"].dropna().to_numpy(),
        active_annotated_counts_df["number_aa_mutations"].dropna().to_numpy(),
        inactive_annotated_counts_df["number_aa_mutations"].dropna().to_numpy(),
    ]
    xmin = 0
    xmax = int(
        np.max([a.max() for a in combined if a.size])
    )  # observed max across groups

    sns.histplot(
        data=best_annotated_counts_df,
        x="number_aa_mutations",
        binwidth=1,
        stat="density",
        edgecolor="black",
        color="orange",
        linewidth=0.5,
        alpha=0.3,
        label=f"Best Activity (n={best_annotated_counts_df.shape[0]} [{best_mean:.1f}|{best_median:.1f}])",
    )

    sns.histplot(
        data=active_annotated_counts_df,
        x="number_aa_mutations",
        binwidth=1,
        stat="density",
        edgecolor="black",
        color="#3B71B2",
        linewidth=0.5,
        alpha=0.3,
        label=f"Active (n={active_annotated_counts_df.shape[0]} [{active_mean:.1f}|{active_median:.1f}])",
    )

    sns.histplot(
        data=inactive_annotated_counts_df,
        x="number_aa_mutations",
        binwidth=1,
        stat="density",
        edgecolor="black",
        color="grey",
        linewidth=0.5,
        alpha=0.3,
        label=f"Inactive (n={inactive_annotated_counts_df.shape[0]}[{inactive_mean:.1f}|{inactive_median:.1f}])",
    )

    sns.kdeplot(
        data=best_annotated_counts_df,
        x="number_aa_mutations",
        color="orange",
        lw=1.5,
        fill=False,
        cut=0,
        clip=(xmin, xmax),
        bw_adjust=1.5,
        ax=ax,
        zorder=5,
        label="Best Activity KDE",
    )

    sns.kdeplot(
        data=active_annotated_counts_df,
        x="number_aa_mutations",
        color="#3B71B2",
        lw=1.5,
        fill=False,
        cut=0,
        clip=(xmin, xmax),
        bw_adjust=1.5,
        ax=ax,
        zorder=5,
        label="Active KDE",
    )

    sns.kdeplot(
        data=inactive_annotated_counts_df,
        x="number_aa_mutations",
        color="grey",
        lw=1.5,
        fill=False,
        cut=0,
        clip=(xmin, xmax),
        bw_adjust=1.5,
        ax=ax,
        zorder=5,
        label="Inactive KDE",
    )

    # ax = plt.gca()
    # Adjust the ticks
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ticks = ax.get_xticks()
    if len(ticks) > 1:
        new_ticks = ticks[:-1] + 0.5
        ax.set_xticks(new_ticks)
        ax.set_xticklabels([str(int(t)) for t in new_ticks])
        ax.set_xlim([new_ticks[0] + 0.5, new_ticks[-1] - 0.5])

    ax.set_xlabel("amino acid mutations")
    ax.set_ylabel(f"density (n={num_sequences} sequences)")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_linewidth(0.5)
    ax.tick_params(width=0.5)

    plt.legend(loc="upper right")
    plt.tight_layout()

    # outname_hamming_dir = f'../../minibinders_orthorep_data/minibinders_orthorep_outputs/{expt_id}/neutral_drift_library/plots/'
    outname_histogram_all_classes = (
        out_dir_figs
        + "hamming_all_classes"
        + f"_countthresh{count_threshold}"
        + ("filtered" if filtered == True else "")
    )

    plt.savefig(outname_histogram_all_classes + ".png", dpi=400)
    # plt.savefig(outname_histogram_all_classes + '.pdf', dpi=400)
    plt.savefig(outname_histogram_all_classes + ".svg", dpi=400)
    plt.close()


if __name__ == "__main__":
    main()
