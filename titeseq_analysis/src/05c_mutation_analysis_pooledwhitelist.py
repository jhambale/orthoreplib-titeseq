#!/usr/bin/env python3

"""
Created on May 16 16:27:44 2024
@author: alcantar
modified by jhambale on 09 OCT 2026
Example:
    python 05c_mutation_analysis_pooledwhitelist.py \
        -i ../../titeseq_data/outputs/jh_010/alignments/*.bam \
        -m ../../titeseq_data/raw/jh_010/demultiplex/jh_010_metadata.txt \
        -r ../../titeseq_data/raw/jh_010/references/ \
        -w ../../titeseq_data/outputs/jh_010/seqthresh/final_whitelist.csv

"""


import argparse
from collections import Counter
from pathlib import Path

import pandas as pd
import pysam
from Bio import SeqIO
from utils_nd import find_mutations, translate_sequence


def extract_sequence(read, limits):
    """
    Given a pysam read and the index limits of a desired region, extract out the
    sequence of the desired region of the read, adjusting for any insertions or
    deletions in the alignment. written by V. Hu. Adapted by M. Alcantar
    to work with versions below python 3.10 (i.e., removed the match operator)

    PARAMETERS
    -----------
    read: pysam AlignedSegment object
        basically a record from a .bam file
    limits: 2-d list
        a list of two ints denoting index limits for the extraction

    RETURNS
    --------
        a string denoting the sequence extracted from the reads
    """
    index = 0  # Pointer to position on the query
    q_start, q_end = limits  # start and end positions on the reference
    q_start -= read.reference_start
    q_end -= read.reference_start
    for operation, positions in read.cigartuples:
        if operation in [0, 7, 8]:  # Consume both Q and R
            index += positions
        elif operation == 1:  # Consume only Q
            # move bookmarks up to account for x extra query positons
            q_start += (index <= q_start) * positions
            q_end += (index < q_end) * positions
            index += positions
        elif operation in [2, 3]:  # Consume only R
            # move bookmarks down to account for x ref skips
            q_end -= (index < q_end) * min(positions, q_end - index)
            q_start -= (index < q_start) * min(positions, q_start - index)
        elif operation == 4:  # Consume only Q, but only at start and end
            q_start += (index <= q_start or q_start <= 0) * positions
            q_start += (
                q_start <= 0 or (index < q_start and index + positions > q_start)
            ) * (index + positions - q_start)
            q_end += (index <= q_end or q_end <= 0) * positions
            index += positions
            q_end += (index - positions < q_end and index > q_start) * (
                index - q_end - positions
            )
    return read.seq[max(q_start, 0) : q_end]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", nargs="+", required=True, help="BAM files to parse")
    parser.add_argument("-m", required=True, help="Path to metadata")
    parser.add_argument("-r", required=True, help="Path to reference database")
    parser.add_argument(
        "-w",
        "--whitelist",
        required=True,
        help="CSV whitelist containing sequences to retain when unpacking the .bam file. Must contain a 'sequence' column.",
    )
    args = parser.parse_args()

    alignment_files = args.i
    ref_dir = Path(args.r)
    whitelist_df = pd.read_csv(args.whitelist)
    if "sequence" not in whitelist_df.columns:
        parser.error("Whitelist CSV must contain a 'sequence' column.")
    whitelist = set(
        whitelist_df["sequence"].dropna().astype(str).str.upper().str.strip()
    )
    whitelist.discard("")
    if not whitelist:
        parser.error("Whitelist contains no sequences.")

    metadata_df = pd.read_csv(args.m, sep="\t", index_col=None)
    if not {"sample_id", "reference"}.issubset(metadata_df.columns):
        parser.error("Metadata must contain 'sample_id' and 'reference' columns.")
    for alignment in alignment_files:
        print("----------------------------")
        print(f"parsing: {alignment}")
        print("----------------------------")

        alignment_path = Path(alignment)
        sample_name = alignment_path.stem
        metadata_samp = metadata_df.loc[
            metadata_df["sample_id"].astype(str).str.contains(
                sample_name, regex=False, na=False
            )
        ]
        if len(metadata_samp) != 1:
            raise ValueError(
                f"Expected one metadata row matching {sample_name}, found "
                f"{len(metadata_samp)}."
            )
        reference_name = str(metadata_samp.iloc[0]["reference"])

        if not reference_name.endswith((".fasta", ".fa", ".fna")):
            reference_name = reference_name + ".fasta"
        ref_path = ref_dir / reference_name

        with open(ref_path, "rt") as reference_reads:
            reference = SeqIO.parse(reference_reads, "fasta")
            reference_record = next(reference, None)
        if reference_record is None:
            raise ValueError(f"No reference sequence found in {ref_path}.")
        reference_sequence_dna = reference_record.seq.upper()
        reference_size = len(reference_sequence_dna)
        reference_sequence_aa = str(reference_sequence_dna.translate())

        print("extracting sequences")
        sequences_counter = Counter()
        total_valid_sequences = 0
        with pysam.AlignmentFile(alignment, "rb") as samfile:
            for read in samfile:
                try:
                    sequence = extract_sequence(read, [0, reference_size])
                except (TypeError, ValueError, IndexError):
                    continue
                if (
                    sequence
                    and len(sequence) == reference_size
                    and "N" not in sequence.upper()
                ):
                    total_valid_sequences += 1
                    sequence = sequence.upper()
                    if sequence in whitelist:
                        sequences_counter[sequence] += 1

        print(f"{total_valid_sequences} valid sequences for {sample_name}")
        print(
            f"{sum(sequences_counter.values())} reads retained from whitelist "
            f"({len(sequences_counter)} sequences)"
        )
        seq_count_total = sum(sequences_counter.values())
        mutation_analysis_dict_list = []
        for dna_variant, sequence_count in sequences_counter.items():
            sequence_dna_tmp = dna_variant
            sequence_aa_tmp = translate_sequence(dna_variant)
            normalized_read_count = (
                sequence_count / seq_count_total if seq_count_total else 0
            )

            dna_muts = find_mutations(reference_sequence_dna, sequence_dna_tmp)
            number_dna_mutations = len(dna_muts)
            aa_muts = find_mutations(reference_sequence_aa, sequence_aa_tmp)
            number_aa_mutations = len(aa_muts)
            mutation_analysis_dict_list.append(
                {
                    "dna_sequence": sequence_dna_tmp,
                    "aa_sequence": sequence_aa_tmp,
                    "read_count": sequence_count,
                    "normalized_read_count": normalized_read_count,
                    "dna_mutations": dna_muts,
                    "aa_mutations": aa_muts,
                    "number_dna_mutations": number_dna_mutations,
                    "number_aa_mutations": number_aa_mutations,
                }
            )

        mutations_df_outdir = alignment_path.parent.parent / "mutation_dfs_whitelist"
        mutations_df_outdir.mkdir(parents=True, exist_ok=True)
        mutations_df_outpath = (
            mutations_df_outdir / f"{sample_name}_whitelist_mutation_analysis.csv"
        )
        mutations_df = pd.DataFrame(
            mutation_analysis_dict_list,
            columns=[
                "dna_sequence",
                "aa_sequence",
                "read_count",
                "normalized_read_count",
                "dna_mutations",
                "aa_mutations",
                "number_dna_mutations",
                "number_aa_mutations",
            ],
        )
        mutations_df = mutations_df.sort_values(
            by="read_count", ascending=False
        ).reset_index(drop=True)
        mutations_df.to_csv(mutations_df_outpath, index=True)
        print(f"Mutation table written to: {mutations_df_outpath}")


if __name__ == "__main__":
    main()
