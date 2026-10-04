#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on January 15 12:16:54 2025
@author: alcantar; modified by yy and JH
example run: python 09_create_normalized_tite_seq_scores.py \
  -i ../../data/outputs/yy_009/normalized_tite_seq_counts/ \
  -m ../../data/raw/jh_001/demultiplex/jh_001_metadata.txt \
  -n fmean
n will be either 'simple' or 'fmean'; depending on which normalization score method the user wants to use

"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from utils_ts import make_dir


def load_fluorescence_metadata(
    metadata_path: Path,
) -> Dict[str, Dict[str, List[float]]]:
    """
    Parses the experiment metadata TSV and returns a nested dictionary of
    log10-transformed mean fluorescence values per replicate and concentration:
        { 'rep1': { '0.64nM': [log10_bin0, log10_bin1, ...], ... }, 'rep2': { ... } }

    The concentration keys match the column-name prefixes produced by script 08
    (e.g. '0.64nM', '3.2nM', '16nM').

    The metadata TSV must contain columns: replicate, antigen_concentration_nM,
    bin, mean_fluorescence.

    Rows where mean_fluorescence is NA are assigned log10(0.01) as a floor value.
    """
    meta_df = pd.read_csv(metadata_path, sep="\t")

    required_cols = {
        "replicate",
        "antigen_concentration_nM",
        "bin",
        "mean_fluorescence",
    }
    missing = required_cols - set(meta_df.columns)
    if missing:
        raise ValueError(
            f"Metadata file {metadata_path.name} is missing columns: {missing}"
        )

    def fmt_conc(c: float) -> str:
        return f"{int(c)}nM" if c == int(c) else f"{c}nM"

    replicate_means: Dict[str, Dict[str, List[float]]] = {}

    for rep_num, rep_df in meta_df.groupby("replicate"):
        rep_key = f"rep{rep_num}"
        conc_dict: Dict[str, List[float]] = {}
        for conc_val, conc_df in rep_df.groupby("antigen_concentration_nM"):
            raw_values = conc_df.sort_values("bin")["mean_fluorescence"].tolist()
            log10_values = [
                np.log10(v) if pd.notna(v) and v > 0 else np.log10(0.01)
                for v in raw_values
            ]
            conc_dict[fmt_conc(float(conc_val))] = log10_values
        replicate_means[rep_key] = conc_dict

    print(
        f"Loaded fluorescence data for {len(replicate_means)} replicates from metadata."
    )
    return replicate_means


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-i", "--input", required=True, help="Directory containing normalized counts"
    )
    parser.add_argument(
        "-m",
        "--metadata",
        required=True,
        help="Path to experiment metadata TSV file",
    )
    parser.add_argument(
        "-n",
        "--method",
        choices=["simple", "fmean"],
        default="fmean",
        help="Normalization method: simple (0-1) or fmean (geometric means)",
    )

    args = parser.parse_args()

    path_input = Path(args.input)
    norm_method = args.method

    # 1. Load fluorescence means from metadata
    try:
        replicate_means_map = load_fluorescence_metadata(Path(args.metadata).resolve())
    except Exception as e:
        print(f"Error loading fluorescence data from metadata: {e}")
        sys.exit(1)

    # 2. Setup Output
    output_dir = path_input.parent / path_input.name.replace(
        "normalized_tite_seq_counts", "normalized_tite_seq_scores"
    )
    make_dir(str(output_dir))

    # 3. Process each replicate identified in metadata
    for rep_id, geom_means in replicate_means_map.items():
        # Find normalized count file matching rep ID
        matches = [
            f for f in path_input.glob("*.csv") if rep_id.lower() in f.name.lower()
        ]
        if not matches:
            print(f"Warning: No normalized count file found for {rep_id}. Skipping.")
            continue

        norm_path = matches[0]
        print(f"Scoring {rep_id} using {norm_path.name}...")
        df = pd.read_csv(norm_path)

        # Handle simple normalization method
        if norm_method == "simple":
            simple_weights = [0, (1 / 3), (2 / 3), (1.0)]
            # Overwrite the geom_means with simple 0-1 scaling for all concentrations
            geom_means = {conc: simple_weights for conc in geom_means}

        # 4. Calculate Scores for each concentration
        for conc, weights in geom_means.items():
            conc_pattern = rf"^{re.escape(conc)}_bin.*_normed_counts$"
            bin_cols = df.filter(regex=conc_pattern, axis=1)

            if bin_cols.shape[1] < len(weights):
                print(
                    f"  Warning: Expected {len(weights)} bins for {conc}, found {bin_cols.shape[1]}. Skipping."
                )
                continue

            # Weighted sum of bins: Score = Sum(Fraction_in_Bin * Weight_of_Bin)
            score_col = f"score_{conc}"
            df[score_col] = 0
            for i in range(len(weights)):
                df[score_col] += bin_cols.iloc[:, i] * weights[i]

        # 5. Save Output
        out_name = norm_path.name.replace("normalized", f"{norm_method}_norm_scores")
        df.to_csv(output_dir / out_name, index=False)
        print(f"  Saved: {out_name}")


if __name__ == "__main__":
    main()
