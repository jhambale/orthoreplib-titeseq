#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Normalize raw Tite-Seq counts for two replicates (rep1, rep2).

Usage:
  python 08_create_normalized_tite_seq_counts.py \
  -i ../../data/outputs/yy_009/raw_tite_seq_counts/ \
  -m ../../data/raw/yy_009/demultiplex/yy_009_metadata.txt

The script:
  - Finds raw count CSVs containing "rep1" and "rep2" in the given directory.
  - Normalizes per concentration/bin for cell-sorting fractions and sequencing depth.
  - Writes normalized CSVs alongside the input directory, with names replacing “raw” -> “normalized”.
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
from utils_ts import make_dir


def load_cell_counts_metadata(metadata_path: Path) -> Dict[str, Dict[str, List[int]]]:
    """
    Parses the experiment metadata TSV and returns a nested dictionary:
        { 'rep1': { '0.64nM': [bin0_count, bin1_count, ...], ... }, 'rep2': { ... } }

    The concentration keys match the column-name prefixes produced by script 07
    (e.g. '0.64nM', '3.2nM', '16nM').

    The metadata TSV must contain columns: replicate, antigen_concentration_nM,
    bin, cell_count.
    """
    meta_df = pd.read_csv(metadata_path, sep="\t")

    required_cols = {"replicate", "antigen_concentration_nM", "bin", "cell_count"}
    missing = required_cols - set(meta_df.columns)
    if missing:
        raise ValueError(
            f"Metadata file {metadata_path.name} is missing columns: {missing}"
        )

    def fmt_conc(c: float) -> str:
        return f"{int(c)}nM" if c == int(c) else f"{c}nM"

    replicate_metadata: Dict[str, Dict[str, List[int]]] = {}

    for rep_num, rep_df in meta_df.groupby("replicate"):
        rep_key = f"rep{rep_num}"
        conc_dict: Dict[str, List[int]] = {}
        for conc_val, conc_df in rep_df.groupby("antigen_concentration_nM"):
            counts = conc_df.sort_values("bin")["cell_count"].astype(int).tolist()
            conc_dict[fmt_conc(float(conc_val))] = counts
        replicate_metadata[rep_key] = conc_dict

    return replicate_metadata


def normalize_replicate(
    raw_df: pd.DataFrame, cell_counts: Dict[str, List[int]]
) -> Dict[str, pd.Series]:
    """
    Performs normalization and returns a dictionary of normalized columns:
    { '10^-5M_bin1_normed_counts': pd.Series([...]), ... }
    """
    all_normed_cols = {}

    for conc, gate_counts in cell_counts.items():
        # 1. Regex to find the 4 bins for this concentration
        conc_pattern = rf"^{re.escape(conc)}(?:[nN]?M)?_bin.*_raw_counts$"
        bin_cols = raw_df.filter(regex=conc_pattern, axis=1)

        if bin_cols.empty:
            continue

        # 2. Gate fractions
        total_cells = sum(gate_counts)
        gate_fractions = [
            c / total_cells if total_cells > 0 else 0 for c in gate_counts
        ]

        # 3. Depth and Gate Correction
        depths = bin_cols.sum()
        conc_temp_cols = {}

        for i, col_name in enumerate(bin_cols.columns):
            normed_name = col_name.replace("raw_counts", "normed_counts")
            if depths.iloc[i] == 0:
                conc_temp_cols[normed_name] = pd.Series(0, index=raw_df.index)
            else:
                conc_temp_cols[normed_name] = (
                    bin_cols[col_name] * gate_fractions[i]
                ) / depths.iloc[i]

        # 4. Renormalize so rows sum to 1 for this concentration
        conc_df = pd.DataFrame(conc_temp_cols)
        row_sums = conc_df.sum(axis=1)
        conc_df = conc_df.div(row_sums, axis=0).fillna(0)

        # Add to the master dictionary
        for col in conc_df.columns:
            all_normed_cols[col] = conc_df[col]

    return all_normed_cols


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", help="Directory containing raw counts", required=True
    )
    parser.add_argument(
        "-m",
        "--metadata",
        required=True,
        help="Path to experiment metadata TSV file",
    )
    args = parser.parse_args()

    input_dir = Path(args.input)

    # 1. Load metadata as a dictionary
    metadata_dict = load_cell_counts_metadata(Path(args.metadata).resolve())

    # 2. Setup output
    output_dir = input_dir.parent / input_dir.name.replace("raw", "normalized")
    make_dir(str(output_dir))

    # 3. Match raw files to metadata keys
    for rep_id, cell_counts in metadata_dict.items():
        print(f"Searching in {input_dir} for {rep_id}")
        matches = [
            f for f in input_dir.glob("*.csv") if rep_id.lower() in f.name.lower()
        ]
        print(f"Processing {matches}")
        if not matches:
            continue

        raw_path = matches[0]
        raw_df = pd.read_csv(raw_path)

        raw_count_cols = [
            column for column in raw_df.columns if column.endswith("_raw_counts")
        ]
        if not raw_count_cols:
            raise ValueError(f"No raw-count columns found in {raw_path}")

        no_read_any_conc = pd.Series(False, index=raw_df.index)
        no_read_all_bins = raw_df[raw_count_cols].sum(axis=1).eq(0)
        for conc, gate_counts in cell_counts.items():
            conc_pattern = rf"^{re.escape(conc)}(?:[nN]?M)?_bin.*_raw_counts$"
            conc_cols = raw_df.filter(regex=conc_pattern, axis=1).columns.tolist()
            if len(conc_cols) != len(gate_counts):
                raise ValueError(
                    f"Expected {len(gate_counts)} raw-count bins for {conc} in "
                    f"{raw_path}, found {len(conc_cols)}."
                )
            no_read_conc = raw_df[conc_cols].sum(axis=1).eq(0)
            no_read_any_conc |= no_read_conc
            print(
                f"  {conc}: {int(no_read_conc.sum())} sequences have zero reads "
                "across its bins"
            )

        removed_sequences = int(no_read_any_conc.sum())
        zero_all_bins = int(no_read_all_bins.sum())
        zero_some_conc_only = int((no_read_any_conc & ~no_read_all_bins).sum())
        raw_df = raw_df.loc[~no_read_any_conc].copy()
        print(
            f"Removed {removed_sequences} sequences from {raw_path.name}: "
            f"{zero_all_bins} had zero reads across all bins; "
            f"{zero_some_conc_only} had reads overall but zero reads at one or more "
            "concentrations."
        )

        # Get normalized columns as a dictionary
        normed_cols_dict = normalize_replicate(raw_df, cell_counts)

        # Reconstruct final dataframe starting with metadata
        final_df = raw_df.iloc[:, :3].copy()
        for col_name, col_data in normed_cols_dict.items():
            final_df[col_name] = col_data

        final_df.to_csv(
            output_dir / raw_path.name.replace("raw", "normalized"), index=False
        )


if __name__ == "__main__":
    main()
