#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on May 16 16:27:44 2024
@author: alcantar
example run 1: python 02c_mutation_analysis.py  \
-i ../../minibinders_orthorep_data/minibinders_orthorep_outputs/maa_001/alignments/mb317-*.bam \
-m ../../minibinders_orthorep_data/ngs_raw/maa_001/demultiplex/maa_001_demux_metadata.txt \
-r ../../minibinders_orthorep_data/ngs_raw/maa_001/references/ \
-n 5
example run 2: python 02c_mutation_analysis.py  -i ../../minibinders_orthorep_data/minibinders_orthorep_outputs/maa_002/alignments/mb171-*.bam \
-m ../../minibinders_orthorep_data/ngs_raw/maa_002/demultiplex/maa_002_demux_metadata.txt \
-r ../../minibinders_orthorep_data/ngs_raw/maa_002/references/ \
-n 5
example run 3: python 02c_mutation_analysis.py  -i ../../minibinders_orthorep_data/minibinders_orthorep_outputs/maa_003/alignments/mb376-*.bam \
-m ../../minibinders_orthorep_data/ngs_raw/maa_003/demultiplex/maa_003_metadata.txt \
-r ../../minibinders_orthorep_data/ngs_raw/maa_003/references/ \
-n 5

"""
# activate virtual enviroment before running script
# source activate minibinders

import argparse
import glob
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import pysam
from Bio import SeqIO
from kneed import KneeLocator
from utils_nd import *


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


def find_knee(data, start, S=0.2):
    """
    Find the knee (elbow) of a rank-ordered sequence-frequency distribution.

    Scans from the lowest rank (right/tail end of the descending curve) upward
    and returns the read-count y-value at the first knee found.  The full
    rank-ordered curve is always plotted so context is preserved.

    PARAMETERS
    -----------
    data : list of int
        Raw sequence counts (order does not matter; they will be sorted here).
    start : int
        Number of leading entries to skip before fitting the knee.  Passed in
        as the value of the -n / --num_seq_filter argument, which serves as a
        burn-in offset rather than a hard CPM threshold in this workflow.
    S : float
        Sensitivity parameter passed to KneeLocator.  Higher values make the
        knee detection less sensitive (fewer, broader knees detected); lower
        values make it more sensitive.  Default 0.2 matches manzanaa behaviour.

    RETURNS
    --------
    knee_read_count : int or None
        The read-count value (y) at the detected knee, to be used directly as
        a filtering threshold.  Returns None if KneeLocator cannot detect a knee.
    fig : matplotlib.figure.Figure
        The knee-plot figure, ready to be saved or displayed.
    """
    x = list(range(len(data)))
    y = sorted(data, reverse=True)

    # Build an ascending curve from the tail end: y values go from smallest
    # count (last rank) up to the first rank after the skip offset.
    # x_asc is re-indexed from 0 so KneeLocator sees a clean ascending sequence.
    y_tail = y[start:][::-1]  # counts from lowest rank to highest rank
    x_asc = list(range(len(y_tail)))  # fresh 0-based ascending indices

    knee = KneeLocator(x_asc, y_tail, curve="convex", direction="increasing", S=S)
    print(f"knee is {knee.knee}, elbow is {knee.elbow}")

    # Map the ascending index back to the original descending rank for plotting
    if knee.knee is not None:
        knee_rank = (len(y) - 1) - knee.knee  # original rank in the full y array
        knee_read_count = y[knee_rank]  # read count at that rank
    else:
        knee_rank = None
        knee_read_count = None

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title("Knee Point")
    ax.plot(x, y, "b", label="data")
    if knee_rank is not None:
        ax.vlines(
            knee_rank,
            ax.get_ylim()[0],
            ax.get_ylim()[1],
            linestyles="--",
            label="knee/elbow",
        )
        x_max = knee_rank * 2
        ax.set_xlim(left=0, right=x_max)
        y_max = knee_read_count * 2
        ax.set_ylim(bottom=0, top=y_max)
    ax.legend(loc="best")
    ax.set_xlabel("Read Count Rank")
    ax.set_ylabel("Total Read Count")
    return knee_read_count, fig


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", nargs="+", help="List of fastq files to parse")
    parser.add_argument("-m", help="Path to metadata")
    parser.add_argument("-r", help="Path to reference database")
    parser.add_argument(
        "-n", default=10, help="knee start offset (leading entries to skip)"
    )
    parser.add_argument(
        "-S",
        default=0.2,
        type=float,
        help="KneeLocator sensitivity (S parameter). Higher = less sensitive. Default 0.2.",
    )

    args = parser.parse_args()

    alignment_files = args.i  # glob.glob('../../minibinders_data/minibinders_outputs/maa_001/alignments/*.bam') # input
    metadata_path = args.m  #'../../minibinders_data/ngs_raw/maa_001/demultiplex/maa_001_demux_metadata.txt' # input
    ref_dir = args.r  #'../../minibinders_data/ngs_raw/maa_001/references/' # input
    num_seq_filter = int(args.n)  # 100 #input
    knee_sensitivity = float(args.S)

    if ref_dir[-1] != "/":
        ref_dir = ref_dir + "/"

    metadata_df = pd.read_csv(metadata_path, sep="\t", index_col=None)

    for alignment in alignment_files:
        print("----------------------------")
        print(f"parsing: {alignment}")
        print("----------------------------")

        sample_name = alignment.split("/")[-1].replace(".bam", "")
        metadata_samp = metadata_df.copy()[
            metadata_df["sample_id"].str.contains(sample_name)
        ]
        reference_name = list(metadata_samp["reference"])[0]

        if ".fasta" not in reference_name:
            reference_name = reference_name + ".fasta"
        ref_path = ref_dir + reference_name

        print(metadata_samp["reference_size"])
        samfile = pysam.AlignmentFile(alignment, "rb")
        with open(ref_path, "rt") as reference_reads:
            reference = SeqIO.parse(reference_reads, "fasta")
            for reads in reference:
                reads = reads.upper()
                reference_sequence_dna = reads.seq
                reference_size = len(reference_sequence_dna)
        reference_sequence_aa = str(reference_sequence_dna.translate())

        # Open the BAM file
        sequences = []
        aa_sequences = []
        print("extracting sequences")
        for read in samfile:
            # extract sequences and only take sequences if they match the expected size
            try:
                sequence = extract_sequence(read, [0, reference_size])
                if len(sequence) == reference_size and "N" not in sequence:
                    sequences.append(sequence)
            except Exception as e:
                pass

        # filter sequences:
        # find the knee of the rank-ordered sequence frequency distribution
        print("filtering sequences")
        sequences_counter = Counter(sequences)
        print(f"{sum(sequences_counter.values())} total sequences for {sample_name}")

        knee, fig = find_knee(
            list(sequences_counter.values()), num_seq_filter, S=knee_sensitivity
        )
        print(f"read count knee: {knee}")

        # use knee as read-count threshold; fall back to 0 if no knee detected
        num_sequences_filter = int(knee) if knee else int(0)
        print(f"read count threshold: {num_sequences_filter}")

        filtered_sequences = {
            seq: count
            for seq, count in sequences_counter.items()
            if count >= num_sequences_filter
        }

        filtered_sequences_dict = filtered_sequences
        seq_count_total = sum(filtered_sequences_dict.values())
        print("finding mutations")

        mutation_analysis_dict_list = []
        for dna_variant in filtered_sequences_dict:
            sequence_dna_tmp = dna_variant
            sequence_aa_tmp = translate_sequence(dna_variant)
            sequence_cnt_tmp = filtered_sequences_dict[dna_variant]
            normalized_read_count = sequence_cnt_tmp / seq_count_total

            dna_muts = find_mutations(reference_sequence_dna, sequence_dna_tmp)
            number_dna_mutations = len(dna_muts)
            aa_muts = find_mutations(reference_sequence_aa, sequence_aa_tmp)
            number_aa_mutations = len(aa_muts)
            mutation_analysis_dict_list.append(
                {
                    "dna_sequence": sequence_dna_tmp,
                    "aa_sequence": sequence_aa_tmp,
                    "read_count": sequence_cnt_tmp,
                    "normalized_read_count": normalized_read_count,
                    "dna_mutations": dna_muts,
                    "aa_mutations": aa_muts,
                    "number_dna_mutations": number_dna_mutations,
                    "number_aa_mutations": number_aa_mutations,
                }
            )

        mutations_df_outdir = "/".join(alignment.split("/")[:-1]).replace(
            "alignments", "mutation_dfs/"
        )
        make_dir(mutations_df_outdir)
        mutations_df_outname = (
            sample_name + f"thresh{str(knee)}_S{knee_sensitivity}_mutation_analysis.csv"
        )
        mutations_df_outpath = mutations_df_outdir + mutations_df_outname
        mutations_df = pd.DataFrame(mutation_analysis_dict_list)
        mutations_df = mutations_df.sort_values(
            by="read_count", ascending=False
        ).reset_index(drop=True)
        mutations_df.to_csv(mutations_df_outpath)

        # save knee plot alongside the CSV
        fig.savefig(
            mutations_df_outdir
            + f"{sample_name}knee{str(knee)}_S{knee_sensitivity}.png"
        )
        plt.close(fig)


if __name__ == "__main__":
    main()
