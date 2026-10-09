#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on January 14 11:29:53 2025
@author: alcantar, modified by yy and JH
example run: python 07b_create_raw_tite_seq_counts_whitelist.py \
-i ../../titeseq_data/outputs/jh_010/mutation_dfs_whitelist/ \
-w ../../titeseq_data/outputs/jh_010/seqthresh/final_min_count_whitelist.csv \
-m ../../titeseq_data/raw/jh_010/demultiplex/jh_010_metadata.txt \
--min-reads 20 \
-r ../../titeseq_data/raw/jh_010/references/pmaa23_nbonly.fasta

"""

# activate virtual enviroment before running script
# source activate minibinders
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd
from Bio import SeqIO
from parse_ts import parse_filename
from utils_nd import find_mutations, translate_sequence


EXCEL_SAFE_CELL_LIMIT = 30000


def encode_sequence_lists(
    keys: List[str], sequences_by_key: Dict[str, Set[str]]
) -> pd.DataFrame:
    """Encode full DNA lists into JSON cells small enough for Excel."""
    encoded_rows = []
    max_chunks = 1
    for key in keys:
        sequences = sorted(sequences_by_key.get(key, set()))
        chunks = []
        current_chunk = []
        for sequence in sequences:
            candidate = current_chunk + [sequence]
            if (
                current_chunk
                and len(json.dumps(candidate)) > EXCEL_SAFE_CELL_LIMIT
            ):
                chunks.append(current_chunk)
                current_chunk = [sequence]
            else:
                current_chunk = candidate
            if len(json.dumps(current_chunk)) > EXCEL_SAFE_CELL_LIMIT:
                raise ValueError(
                    f"DNA sequence for {key!r} exceeds the Excel-safe cell limit."
                )
        chunks.append(current_chunk)
        encoded_rows.append([json.dumps(chunk) for chunk in chunks])
        max_chunks = max(max_chunks, len(chunks))

    columns = {}
    for chunk_index in range(max_chunks):
        column = (
            "dna_sequences"
            if chunk_index == 0
            else f"dna_sequences_{chunk_index + 1}"
        )
        columns[column] = [
            row[chunk_index] if chunk_index < len(row) else "[]"
            for row in encoded_rows
        ]
    return pd.DataFrame(columns)


def load_aa_whitelist(
    whitelist_path: Path, output_dir: Path
) -> List[str]:
    """Translate DNA whitelist sequences and save distinct DNA encodings by AA."""
    whitelist_df = pd.read_csv(whitelist_path)
    if "sequence" not in whitelist_df.columns:
        raise ValueError(f"Whitelist {whitelist_path} must contain a 'sequence' column.")

    aa_to_dna: Dict[str, set] = {}
    for raw_sequence in whitelist_df["sequence"].dropna():
        dna_sequence = str(raw_sequence).strip().upper()
        if not dna_sequence or set(dna_sequence) - set("ACGT"):
            raise ValueError(f"Invalid DNA sequence in whitelist: {dna_sequence!r}")
        complete_codon_length = len(dna_sequence) - len(dna_sequence) % 3
        if complete_codon_length == 0:
            raise ValueError(f"Whitelist DNA sequence is shorter than one codon: {dna_sequence}")
        aa_sequence = translate_sequence(dna_sequence[:complete_codon_length])
        aa_to_dna.setdefault(aa_sequence, set()).add(dna_sequence)

    if not aa_to_dna:
        raise ValueError(f"No DNA sequences found in whitelist {whitelist_path}.")

    aa_to_dna_sorted = {
        aa_sequence: sorted(dna_sequences)
        for aa_sequence, dna_sequences in aa_to_dna.items()
    }
    aa_sequences = sorted(aa_to_dna_sorted)
    aa_whitelist_path = output_dir / "amino_acid_whitelist.csv"
    aa_dna_columns = encode_sequence_lists(aa_sequences, aa_to_dna_sorted)
    aa_dna_columns.columns = [
        column.replace("dna_sequences", "degenerate_dna_sequences")
        for column in aa_dna_columns.columns
    ]
    aa_whitelist_df = pd.concat(
        [
            pd.DataFrame({"aa_sequence": aa_sequences}),
            aa_dna_columns,
        ],
        axis=1,
    )
    aa_whitelist_df.to_csv(aa_whitelist_path, index=False)
    print(
        f"Amino-acid whitelist ({len(aa_sequences)} sequences) written to: "
        f"{aa_whitelist_path}"
    )
    return aa_sequences


def mutations_from_reference(reference_sequence: str, sequence: str) -> List[str]:
    """Annotate substitutions and any terminal length difference to the reference."""
    shared_length = min(len(reference_sequence), len(sequence))
    mutations = find_mutations(
        reference_sequence[:shared_length], sequence[:shared_length]
    )
    if len(sequence) > shared_length:
        mutations.append(f"ins{shared_length + 1}_{sequence[shared_length:]}")
    elif len(reference_sequence) > shared_length:
        mutations.append(
            f"del{shared_length + 1}_{reference_sequence[shared_length:]}"
        )
    return mutations


def initialize_mutations_df(
    aa_library: List[str], parent_dna_seq: str, conc_bins: List[str]
) -> pd.DataFrame:

    # Use the reference as the mutation baseline, even if it is not whitelisted.
    parental_aa = translate_sequence(parent_dna_seq)

    # Initialize dataframe containing all nb variants with mutation annotations
    aa_mutations = []
    no_aa_mutations = []

    for aa_var in aa_library:
        aa_mutations.append(mutations_from_reference(parental_aa, aa_var))
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
    master_df: pd.DataFrame,
    fname: str,
    column_name: str,
    represented_dna_by_aa: Dict[str, set],
) -> pd.DataFrame:
    """Reads a sample CSV, groups by AA sequence, and updates the master DataFrame."""
    sample_df = pd.read_csv(fname)
    required_columns = {"dna_sequence", "aa_sequence", "read_count"}
    missing_columns = required_columns.difference(sample_df.columns)
    if missing_columns:
        raise ValueError(
            f"Mutation CSV {fname} is missing column(s): "
            + ", ".join(sorted(missing_columns))
        )
    for aa_sequence, dna_sequence in sample_df[
        ["aa_sequence", "dna_sequence"]
    ].dropna().itertuples(index=False, name=None):
        represented_dna_by_aa.setdefault(str(aa_sequence), set()).add(
            str(dna_sequence).strip().upper()
        )

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
    parser.add_argument(
        "-i",
        required=True,
        help="Directory containing mutation_dfs_whitelist CSV files",
    )
    parser.add_argument(
        "-w",
        "--whitelist",
        required=True,
        help="DNA whitelist CSV produced by script 04b",
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
    if not mutation_df_dir.is_dir():
        parser.error(f"Mutation CSV directory not found: {mutation_df_dir}")
    mutation_csvs = sorted(mutation_df_dir.glob("*_whitelist_mutation_analysis.csv"))
    if not mutation_csvs:
        parser.error(
            f"No '*_whitelist_mutation_analysis.csv' files found in {mutation_df_dir}"
        )
    whitelist_path = Path(args.whitelist).resolve()
    if not whitelist_path.is_file():
        parser.error(f"Whitelist CSV not found: {whitelist_path}")

    output_dir = mutation_df_dir.parent / "raw_tite_seq_counts_whitelist"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Tite-Seq experimental design: concentration bins we expect in file names
    try:
        conc_names, bin_names, rep_id_map = extract_and_validate_design(Path(args.metadata))
    except Exception as e:
        print(f"Error parsing metadata file: {e}")
        sys.exit(1)

    conc_bins = [f"{c}_{b}_raw_counts" for c in conc_names for b in bin_names]

    # Translate the DNA whitelist to the AA library and retain all DNA encodings.
    tcdbnb_combo_lib_aa = load_aa_whitelist(whitelist_path, output_dir)

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
    all_mutation_dfs = mutation_csvs
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
        represented_dna_by_aa: Dict[str, set] = {}
        for fname, col_name in files:
            if col_name not in conc_bins:
                print(f"  Warning: {col_name} not in expected design. Skipping file.")
                continue
            rep_df = merge_sample_counts(
                rep_df, fname, col_name, represented_dna_by_aa
            )
            processed_files += 1

        count_columns = [
            column for column in rep_df.columns if column.endswith("_raw_counts")
        ]
        zero_read_sequences = int(rep_df[count_columns].sum(axis=1).eq(0).sum())

        # Filter low-total variants per concentration by zeroing their bins
        for conc in conc_names:
            conc_cols = [f"{conc}_{b}_raw_counts" for b in bin_names]
            present_cols = [c for c in conc_cols if c in rep_df.columns]
            if not present_cols:
                continue
            low_mask = rep_df[present_cols].sum(axis=1) < args.min_reads
            if low_mask.any():
                rep_df.loc[low_mask, present_cols] = 0

        dna_sequence_columns = encode_sequence_lists(
            rep_df["aa_sequence"].tolist(), represented_dna_by_aa
        )
        rep_df = pd.concat(
            [rep_df.reset_index(drop=True), dna_sequence_columns], axis=1
        )
        rep_df["sum_reads"] = rep_df[count_columns].sum(axis=1).astype(int)
        rep_df.sort_values(
            ["sum_reads", "aa_sequence"],
            ascending=[False, True],
            inplace=True,
            ignore_index=True,
        )

        # Save output
        out_path = output_dir / f"tite_seq_raw_{rep}.csv"
        rep_df.to_csv(out_path, index=False)
        print(f"  Saved: {out_path}")
        print(
            f"  Amino-acid sequences with no reads in any bins: "
            f"{zero_read_sequences}"
        )

    print(
        f"Done! Processed {processed_files} files across {len(rep_groups)} replicates."
    )


if __name__ == "__main__":
    main()
