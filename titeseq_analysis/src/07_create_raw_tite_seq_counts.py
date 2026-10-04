#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on January 14 11:29:53 2025
@author: alcantar, modified by yy and JH
example run: python 07_create_raw_tite_seq_counts.py \
-i ../../data/outputs/yy_009/mutation_dfs/ \
--library ../../data/outputs/yy_009/library_dfs/ \
--metadata ../../data/raw/jh_004/demultiplex/jh_004_metadata.csv \
--min-reads 20 \
--threshold 50 \
--reference ../../titeseq_data/raw/jh_001/references/pmaa23_nbonly.fasta

"""

# activate virtual enviroment before running script
# source activate minibinders
import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from Bio import SeqIO
from parse_ts import parse_filename
from utils_nd import find_mutations, make_dir, translate_sequence


def load_aa_library(lib_dir: Path, output_dir: Path) -> List[str]:
    """Scan all CSVs in lib_dir, return amino-acid sequences present in every CSV
    (intersection across files), and write a summary CSV to lib_dir listing which
    source files each sequence was observed in.

    Each CSV must contain an 'aa_sequence' column (the format produced by script 06).
    """
    csv_files = sorted(lib_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in library directory: {lib_dir}")

    # Build a mapping {aa_sequence -> set of filenames where it appears}
    seq_to_files: dict = {}
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        if "aa_sequence" not in df.columns:
            print(f"  [WARN] Skipping {csv_path.name}: no 'aa_sequence' column found.")
            continue
        seqs = df["aa_sequence"].dropna().astype(str).str.strip().unique()
        for seq in seqs:
            seq_to_files.setdefault(seq, set()).add(csv_path.name)

    all_file_names = {p.name for p in csv_files}
    # Keep only sequences that appear in every CSV
    common_seqs = [
        seq for seq, files in seq_to_files.items() if files >= all_file_names
    ]

    if not common_seqs:
        raise ValueError(
            f"No amino-acid sequences are shared across all {len(csv_files)} CSVs in {lib_dir}."
        )

    print(
        f"Library: {len(common_seqs)} sequences found in all {len(csv_files)} CSVs "
        f"(checked: {[p.name for p in csv_files]})"
    )

    # Write a summary CSV: one row per sequence, columns for each source file
    summary_rows = []
    for seq in common_seqs:
        row: dict = {"aa_sequence": seq}
        for csv_path in csv_files:
            row[csv_path.name] = csv_path.name in seq_to_files.get(seq, set())
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "library_intersection_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Library intersection summary written to: {summary_path}")

    return common_seqs


def initialize_mutations_df(
    aa_library: List[str], parent_dna_seq: str, conc_bins: List[str]
) -> pd.DataFrame:

    # Choose a parental baseline (usually the first in the list)
    parental_aa = translate_sequence(parent_dna_seq)

    # confirm parental sequence is observed in the library
    if parental_aa not in aa_library:
        raise ValueError(f"Parental sequence {parental_aa} not found in library")

    # Initialize dataframe containing all nb variants with mutation annotations
    aa_mutations = []
    no_aa_mutations = []

    for aa_var in aa_library:
        aa_mutations.append(find_mutations(parental_aa, aa_var))
        no_aa_mutations.append(len(aa_mutations[-1]))

    mutations_core_df = pd.DataFrame(
        {
            "aa_sequence": aa_library,
            "aa_mutations": aa_mutations,
            "number_aa_mutations": no_aa_mutations,
        }
    )

    # add new columns and initialize with zeros
    for col in conc_bins:
        mutations_core_df[col] = 0

    return mutations_core_df


def merge_sample_counts(
    master_df: pd.DataFrame, fname: str, column_name: str
) -> pd.DataFrame:
    """Reads a sample CSV, groups by AA sequence, and updates the master DataFrame."""
    sample_df = pd.read_csv(fname)

    # Clean and aggregate counts (fixing the AA collision bug)
    clean_df = (
        sample_df.drop(columns=["Unnamed: 0"], errors="ignore")[
            ["aa_sequence", "read_count"]
        ]
        .groupby("aa_sequence")["read_count"]
        .sum()
        .reset_index()
        .rename(columns={"read_count": column_name})
    )

    # Use a merge/update approach to fill the master dataframe
    # We use 'left' join to keep only variants present in our theoretical library
    updated_master = master_df.merge(
        clean_df, on="aa_sequence", how="left", suffixes=("", "_new")
    )

    # Move new counts into the correct column and cleanup
    if f"{column_name}_new" in updated_master.columns:
        updated_master[column_name] = (
            updated_master[f"{column_name}_new"].fillna(0).astype(int)
        )
        updated_master.drop(columns=[f"{column_name}_new"], inplace=True)

    return updated_master


def extract_and_validate_design(
    metadata_path: Path,
) -> Tuple[List[str], List[str], dict]:
    """
    Parses the experiment metadata TSV to derive concentration names, bin names,
    and a replicate-ID map, all needed downstream in script 07.

    The metadata file must contain columns: antigen_concentration_nM, bin,
    replicate, cell_count, binder_name.

    Returns
    -------
    conc_names : List[str]
        Unique concentrations in ascending order, formatted as '{value}nM'
        (e.g. ['0.64nM', '3.2nM', '16nM', '80nM', '400nM', '2000nM']).
    bin_names : List[str]
        Unique bin labels in ascending order, formatted as 'bin{n}'
        (e.g. ['bin0', 'bin1', 'bin2', 'bin3']).
    rep_id_map : dict
        Maps the numeric replicate-ID prefix found in TiteSeq filenames
        (e.g. '55') to the canonical rep label (e.g. 'rep1').
    """
    meta_df = pd.read_csv(metadata_path, sep="\t")

    required_cols = {
        "antigen_concentration_nM",
        "bin",
        "replicate",
        "cell_count",
        "binder_name",
    }
    missing = required_cols - set(meta_df.columns)
    if missing:
        raise ValueError(
            f"Metadata file {metadata_path.name} is missing columns: {missing}"
        )

    conc_vals = sorted(meta_df["antigen_concentration_nM"].astype(float).unique())
    bin_vals = sorted(meta_df["bin"].astype(int).unique())

    # Format concentration: drop trailing '.0' for whole numbers
    def fmt_conc(c):
        return f"{int(c)}nM" if c == int(c) else f"{c}nM"

    conc_names = [fmt_conc(c) for c in conc_vals]
    bin_names = [f"bin{b}" for b in bin_vals]

    # Build rep_id_map: extract prefix from binder_name (e.g. '96rpt' or '102') -> 'rep1'
    rep_id_map = (
        meta_df[["binder_name", "replicate"]]
        .drop_duplicates()
        .assign(rep_id=lambda df: df["binder_name"].str.extract(r"^(.+?)_b")[0])
        .dropna(subset=["rep_id"])
        .set_index("rep_id")["replicate"]
        .apply(lambda r: f"rep{r}")
        .to_dict()
    )

    print(
        f"Metadata parsed: {len(conc_names)} concentrations x {len(bin_names)} bins "
        f"({meta_df['replicate'].nunique()} replicates)"
    )
    print(f"  Concentrations : {conc_names}")
    print(f"  Bins           : {bin_names}")
    print(f"  Replicate ID map : {rep_id_map}")

    return conc_names, bin_names, rep_id_map


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", help="Directory containing mutation_dfs")
    parser.add_argument(
        "--library",
        required=True,
        help=(
            "Path to a directory containing CSV files produced by script 06. "
            "Amino-acid sequences present in ALL CSVs in the directory are used "
            "as the library. A summary CSV is written to the same directory."
        ),
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=int,
        default=100,
        help="Read threshold across bins per concentration for identifying library (run from script 06)",
    )
    parser.add_argument(
        "-m",
        "--metadata",
        required=True,
        help="Path to experiment metadata TSV file",
    )
    parser.add_argument(
        "--min-reads",
        type=int,
        default=20,
        help="Minimum total reads across bins per concentration",
    )
    parser.add_argument(
        "-r",
        "--reference",
        type=str,
        required=True,
        help="Path to reference sequence file",
    )
    args = parser.parse_args()

    mutation_df_dir = Path(args.i)
    lib_path = Path(args.library).resolve()

    # If the provided mutation_df_dir doesn't exist or contains no matching files,
    # attempt to locate a suitable mutation_dfs directory automatically using
    # the experiment name from the metadata path.
    if not mutation_df_dir.exists() or not any(mutation_df_dir.glob("*mutation_analysis*.csv")):
        # Try extracting experiment id (e.g., 'jh_006') from the metadata path
        meta_path = Path(args.metadata)
        exp_candidates = [p for p in meta_path.parts if p.startswith("jh_") or p.startswith("jh-")]
        exp_name = exp_candidates[0] if exp_candidates else None
        if exp_name:
            print(f"Warning: provided mutation directory '{mutation_df_dir}' not found or empty.\nSearching for mutation_dfs directories for experiment '{exp_name}'...")
            # Search under current working directory for matching mutation_dfs directories
            found = list(Path.cwd().glob(f"**/{exp_name}/**/*mutation_dfs*"))
            if found:
                # choose the first candidate that contains mutation_analysis CSVs
                chosen = None
                for cand in found:
                    cand_path = Path(cand)
                    if cand_path.is_dir() and any(cand_path.glob("*mutation_analysis*.csv")):
                        chosen = cand_path
                        break
                if chosen:
                    mutation_df_dir = chosen
                    print(f"Using mutation directory: {mutation_df_dir}")
                else:
                    print("No suitable mutation_dfs directory with mutation_analysis CSVs found under repository.")
            else:
                print("No mutation_dfs directories found for the experiment under the current workspace.")
        else:
            print(f"Warning: provided mutation directory '{mutation_df_dir}' not found and experiment name could not be inferred from metadata.")

    # directory and path prefix for output data
    output_dir = Path(str(mutation_df_dir).replace("mutation_dfs", "raw_tite_seq_counts"))
    make_dir(output_dir)

    # Tite-Seq experimental design: concentration bins we expect in file names
    try:
        conc_names, bin_names, rep_id_map = extract_and_validate_design(Path(args.metadata))
    except Exception as e:
        print(f"Error parsing metadata file: {e}")
        sys.exit(1)

    conc_bins = [f"{c}_{b}_raw_counts" for c in conc_names for b in bin_names]

    # load AA library — intersect all CSVs in the provided directory
    tcdbnb_combo_lib_aa = load_aa_library(lib_path, output_dir)

    # define parent / best sequence
    ref_fasta_path = args.reference
    ref_fasta = SeqIO.read(ref_fasta_path, "fasta")
    parent_dna_seq = ref_fasta.seq
    # parent_dna_id = ref_fasta.id
    # define parent / best sequence

    mutations_core_df = initialize_mutations_df(
        tcdbnb_combo_lib_aa, parent_dna_seq, conc_bins
    )

    # Group mutation_dfs by replicate
    all_mutation_dfs = mutation_df_dir.glob(
        "*mutation_analysis*.csv"
    )  # look in mutation_dfs directory and Finds EVERY file whose name contains "mutation_analysis" and ends with .csv
    rep_groups = {}  # create empty dictionary
    for fname in all_mutation_dfs:
        try:
            rep, column_name = parse_filename(str(fname), rep_id_map=rep_id_map)
            rep_groups.setdefault(rep, []).append((str(fname), column_name))
        except ValueError as e:
            print(f"Skipping {fname.name}: {e}")

    processed_files = 0
    for rep, files in rep_groups.items():
        print(f"Processing {rep} ({len(files)} files)...")
        # Fill the template matrix with counts per concentration/bin for this rep
        rep_df = mutations_core_df.copy()
        for fname, col_name in files:
            if col_name not in conc_bins:
                print(f"  Warning: {col_name} not in expected design. Skipping file.")
                continue
            rep_df = merge_sample_counts(rep_df, fname, col_name)
            processed_files += 1

        # Filter low-total variants per concentration by zeroing their bins
        for conc in conc_names:
            conc_cols = [f"{conc}_{b}_raw_counts" for b in bin_names]
            present_cols = [c for c in conc_cols if c in rep_df.columns]
            if not present_cols:
                continue
            low_mask = rep_df[present_cols].sum(axis=1) < args.min_reads
            if low_mask.any():
                rep_df.loc[low_mask, present_cols] = 0

        # Save output
        out_path = output_dir / f"tite_seq_raw_{rep}.csv"
        rep_df.to_csv(out_path, index=False)
        print(f"  Saved: {out_path}")

    print(
        f"Done! Processed {processed_files} files across {len(rep_groups)} replicates."
    )


if __name__ == "__main__":
    main()
