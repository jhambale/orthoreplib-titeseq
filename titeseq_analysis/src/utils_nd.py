import csv
import os
import sys
from itertools import combinations

import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator


def reverse_complement(seq):
    """
    creates the reverese complement on a sequence

    PARAMETERS
    -----------
    seq: str
        string consisting of the four nucleotides: a,t,c,g

    RETURNS
    ----------
    seq_rcomp: str
        reverse complement of the given string

    """

    base_pair_dict = {"a": "t", "c": "g", "g": "c", "t": "a", "n": "n"}

    seq_rcomp = [base_pair_dict[nt] for nt in seq[::-1]]
    seq_rcomp = "".join(seq_rcomp)
    return seq_rcomp


def make_dir(dir_path):
    """
    check if a directory exists. if it does not, create the directory.

    PARAMETERS
    --------------------
    dir_path: string
        path to directory

    RETURNS
    --------------------
    none

    """

    CHECK_FOLDER = os.path.isdir(dir_path)
    if not CHECK_FOLDER:
        os.makedirs(dir_path)
        print(f"created folder : {dir_path}")


def find_R1(R2_path):
    """
    obtains the corresponding R1 file given a path to R2
    this functions performs the following two checks:
    i) checks if "_R1_" shows up in the file name
    ii) checks if the R2 file actually exists
    if either of these checks fail, the user in notified and the script ends

    PARAMETERS
    -----------
    R2_path: str
        path to an R2 file. "_R2_" should be present and only show up once

    RETURNS
    ----------
    R1_name: str
        path to corresponding R1 file

    this is a partner function to "find_R1"

    """

    fname = R2_path.split("/")[-1]
    dir_name = "/".join(R2_path.split("/")[:-1]) + "/"

    if "_R1_" in fname:
        print(
            "Error: It seems like you provided R1 files. \n"
            "Please provide _R2_ files and try again."
        )
        sys.exit()

    fname_R1 = fname.replace("_R2_", "_R1_")
    R1_name = dir_name + fname_R1

    if not os.path.isfile(R1_name):
        print(
            f"Error: The following R1 file was not found: \n"
            "{R1_name}\n"
            "Please check your inputs and R2 files, and try again"
        )
        sys.exit()
    return R1_name


def find_R2(R1_path):
    """
    obtains the corresponding R2 file given a path to R1
    this functions performs the following two checks:
    i) checks if "_R2_" shows up in the file name
    ii) checks if the R1 file actually exists
    if either of these checks fail, the user in notified and the script ends

    this is a partner function to "find_R1"

    PARAMETERS
    -----------
    R1_path: str
        path to an R1 file. "_R1_" should be present and only show up once

    RETURNS
    ----------
    R2_name: str
        path to corresponding R2 file

    """

    fname = R1_path.split("/")[-1]
    dir_name = "/".join(R1_path.split("/")[:-1]) + "/"

    if "_R2_" in fname:
        print(
            "Error: It seems like you provided R2 files. \n"
            "Please provide _R1_ files and try again."
        )
        sys.exit()

    fname_R2 = fname.replace("_R1_", "_R2_")
    R2_name = dir_name + fname_R2

    if not os.path.isfile(R2_name):
        print(
            f"Error: The following R2 file was not found: \n"
            "{R2_name}\n"
            "Please check your inputs and R2 files, and try again"
        )
        sys.exit()
    return R2_name


def find_mutations(seq1, seq2):
    """

    identify mutations between two sequences. this function to agnostic to the
    kinds of sequences that are input. this function will only work if seq1 and
    seq2 are the same length. this should be validated outside of this function

    PARAMETERS
    -----------
    seq1: str
        first dna sequence
    seq2: str
        second dna sequence

    RETURNS
    -----------
    mutations: list of str
        list of mutations. each mutation is of the form "YnZ". Y is the original
        character, n is the location of the mutation, and Z is the new character

    """

    mutations = []
    for i in range(len(seq1)):
        if seq1[i] != seq2[i]:
            mutations.append((f"{seq1[i]}{i + 1}{seq2[i]}"))
    return mutations


def translate_sequence(dna_seq):
    """
    translates a given dna sequence. only works for dna sequences that do
    not contain non-canonical nucleotides.
    adapted from: https://www.geeksforgeeks.org/dna-protein-python-3/
    """

    dna_to_aa = {
        "ATA": "I",
        "ATC": "I",
        "ATT": "I",
        "ATG": "M",
        "ACA": "T",
        "ACC": "T",
        "ACG": "T",
        "ACT": "T",
        "AAC": "N",
        "AAT": "N",
        "AAA": "K",
        "AAG": "K",
        "AGC": "S",
        "AGT": "S",
        "AGA": "R",
        "AGG": "R",
        "CTA": "L",
        "CTC": "L",
        "CTG": "L",
        "CTT": "L",
        "CCA": "P",
        "CCC": "P",
        "CCG": "P",
        "CCT": "P",
        "CAC": "H",
        "CAT": "H",
        "CAA": "Q",
        "CAG": "Q",
        "CGA": "R",
        "CGC": "R",
        "CGG": "R",
        "CGT": "R",
        "GTA": "V",
        "GTC": "V",
        "GTG": "V",
        "GTT": "V",
        "GCA": "A",
        "GCC": "A",
        "GCG": "A",
        "GCT": "A",
        "GAC": "D",
        "GAT": "D",
        "GAA": "E",
        "GAG": "E",
        "GGA": "G",
        "GGC": "G",
        "GGG": "G",
        "GGT": "G",
        "TCA": "S",
        "TCC": "S",
        "TCG": "S",
        "TCT": "S",
        "TTC": "F",
        "TTT": "F",
        "TTA": "L",
        "TTG": "L",
        "TAC": "Y",
        "TAT": "Y",
        "TAA": "_",
        "TAG": "_",
        "TGC": "C",
        "TGT": "C",
        "TGA": "_",
        "TGG": "W",
    }

    protein = ""
    if len(dna_seq) % 3 == 0:
        for i in range(0, len(dna_seq), 3):
            codon = dna_seq[i : i + 3].upper()
            protein += dna_to_aa[codon]

    return protein


def hamming_distance(sequence_1, sequence_2):
    """
    calculate hamming distance between two strings: that is, what is the minimal
    number of substitutions to get from string 1 to string 2. this script assumes
    that len(sequence_1) == len(sequence_2). this function is adapted from:
    http://claresloggett.github.io/python_workshops/improved_hammingdist.html

    PARAMETERS
    --------------------
    sequence_1: str
        first string or nucleic acid sequence to compare
    sequence_2: str
        second string or nucleic acid sequence to compare

    RETURNS
    --------------------
    distance: int
        the hamming distance between sequence_1 and sequence_2
    """

    # initialize distance to 0
    distance = 0
    len_sequence_1 = len(sequence_1)

    # loop over entire string / sequence and compare each position of
    # sequence 1 vs. 2
    for base in range(len_sequence_1):
        # Add 1 to the distance if these two characters are not equal
        if sequence_1[base] != sequence_2[base]:
            distance += 1
    # return hamming distance
    return distance


def csv_to_fasta(input_file, output_file):
    """
    takes an input 2-column table of ids and sequences and reformats to match that of fasta format. must strip sample name from csv file in main function and feed it appended to a .fasta file extension

    PARAMETERS
    --------------------
    input_file: 2 column list
        set of ids (column 1) and sequences (column 2)


    RETURNS
    --------------------
    output_file: 1 column list
        transformed id's and sequences packaged in fasta form
    """
    with open(input_file, "r") as csv_file, open(output_file, "w") as fasta_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)  # skips header

        for row in csv_reader:
            if len(row) >= 2:
                identifier = row[0].strip()
                sequence = row[1].strip()
                if identifier and sequence:
                    fasta_file.write(f">{identifier}\n{sequence}\n")
        print(f"fasta saved as {output_file}")


def retrieve_filter_counts(
    master_df, bin1_dict, bin2_dict, bin3_dict, bin4_dict, count_threshold=1
):
    """
    Populate a dataframe with read counts for each amino acid sequence in each bin.
    Practically, this takes in several dictionaries which map an amino acid sequence
    with it's corresponding read count in a given bin (i.e., WT-like binding) which are
    used to populate a dataframe.

    bin_1 - no binding
    bin_2 - low binding
    bin_3 - WT-like binding
    bin_4 - high binding

    PARAMETERS
    -----------
    master_df: pandas dataframe
        dataframe with all the amino acid sequences that will be analyzed / for which
        we want read counts
    bin1_dict: dictionary
        dictionary mapping aa sequence to read count in no binding bin
    bin2_dict: dictionary
        dictionary mapping aa sequence to read count in low binding bin
    bin3_dict: dictionary
        dictionary mapping aa sequence to read count in wt-like binding bin
    bin4_dict: dictionary
        dictionary mapping aa sequence to read count in high binding bin
    count_threshold: int
        minimun number of reads across conditions to consider sequence to analysis

    RETURNS
    -----------
    df: pandas dataframe
        dataframe with read counts per bin populated
    """

    # Use the 'get' method of dictionaries on the 'dna_sequence' column with vectorized operations
    df = master_df.copy()

    # Add new columns directly using pandas' Series mapping
    df["bin_1"] = df["aa_sequence"].map(bin1_dict).fillna(0).astype(int)
    df["bin_2"] = df["aa_sequence"].map(bin2_dict).fillna(0).astype(int)
    df["bin_3"] = df["aa_sequence"].map(bin3_dict).fillna(0).astype(int)
    df["bin_4"] = df["aa_sequence"].map(bin4_dict).fillna(0).astype(int)

    # Filter rows where the sum of the last four columns meets the threshold
    df = df[df[["bin_1", "bin_2", "bin_3", "bin_4"]].sum(axis=1) >= count_threshold]

    # Reset the index
    df.reset_index(inplace=True, drop=True)
    return df


def aggregate_by_aa_sequence(initial_df):
    """
    aggregate variants by their amino acid sequence. this is meant to consolidate
    sequences that have different dna sequences but the same amino acid sequence.
    this ignores subtle effects that different dna encodings can have on expression

    NOTE: for each amino acid sequence, this returns the most abundant DNA sequence mapping
    to that amino acid sequence

    PARAMETERS
    -----------
    initial_df: pandas dataframe
        dataframe with dna sequences, amino acid sequences, and read counts

    RETURNS
    -----------
    consolidated_df: pandas dataframe
        dataframe with consolidated dna sequences
    """

    df = initial_df.copy()

    consolidated_df = (
        df.sort_values(
            by="read_count", ascending=False
        )  # Sort by read_count in descending order
        .groupby("aa_sequence", as_index=False)
        .agg(
            {
                "dna_sequence": list,  # create a list of all dna sequences which map to an amino acid sequence
                "read_count": "sum",  # sum the read counts
                "dna_mutations": list,  # create a list of all dna mutations which map to an amino acid sequence
                "aa_mutations": "first",  # should be the same across all variants, so only take one representative entry
                "number_dna_mutations": list,  # create a list of all the number of dna mutations in each dna variant
                "number_aa_mutations": "first",  # should be the same across all variants, so only take one representative entry
            }
        )
    )

    return consolidated_df


def plot_hamming_distances(
    annotated_counts_df, out_dir, out_name_1, out_name_2, expt_id, color
):
    """
    Plot hamming distance to parent sequence and pairwise hamming distance
    as a histogram.


    annotated_counts_df: pandas dataframe
        dataframe with different variants and number of mutations
    out_dir: str
        directory
    out_name_1: str
        output name for hamming distance plot versus parent
    out_name_2: str
        output name for pairwise hamming distance plot

    RETURNS
    -----------
    NONE



    """

    # plot versus parent
    num_sequences = annotated_counts_df.shape[0]
    # Set the Seaborn style and context
    sns.set(style="white")

    # Set global font size and family
    plt.rcParams.update({"font.size": 12, "font.family": "Arial"})
    # Set the Seaborn style
    sns.set_style("ticks")

    g = sns.histplot(
        data=annotated_counts_df,
        x="number_aa_mutations",
        binwidth=1,
        stat="probability",
        edgecolor="black",
        color=color,
        linewidth=0.5,
    )
    ax = plt.gca()
    # Adjust the ticks
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ticks = g.get_xticks()
    new_ticks = ticks[:-1] + 0.5
    g.set_xticks(new_ticks)
    g.set_xticklabels([str(int(t)) for t in new_ticks])

    g.set_xlabel("amino acid mutations")
    g.set_ylabel(f"frequency (n={num_sequences} sequences)")
    g.spines["right"].set_visible(False)
    g.spines["top"].set_visible(False)
    for spine in ["left", "bottom"]:
        g.spines[spine].set_linewidth(0.5)
    g.tick_params(width=0.5)
    g.set_xlim([new_ticks[0] + 0.5, new_ticks[-1] - 0.5])
    outname_hamming_dir = out_dir
    outname_hamming_to_parent_plot = outname_hamming_dir + out_name_1
    plt.tight_layout()
    # Show the plot
    plt.savefig(f"{outname_hamming_to_parent_plot}.png", dpi=400)
    # plt.savefig(f'{outname_hamming_to_parent_plot}.pdf', dpi=400)
    # plt.savefig(f'{outname_hamming_to_parent_plot}.svg', dpi=400)
    plt.close()

    # plot pairwise hamming
    aa_sequences = annotated_counts_df["aa_sequence"].to_list()
    hamming_pairwise_aa = []
    for aa_seq_combo in combinations(aa_sequences, 2):
        hamming_pairwise_aa.append(hamming_distance(aa_seq_combo[0], aa_seq_combo[1]))

    g = sns.histplot(
        data=hamming_pairwise_aa,
        binwidth=1,
        stat="probability",
        edgecolor="black",
        color=color,
        linewidth=0.5,
    )
    # sns.histplot(data=norm_counts_annotated_df, x="number_aa_mutations",
    #                  binwidth=1,stat='probability',
    #                  edgecolor='black', linewidth=0.5, color='gray')
    ax = plt.gca()
    # Adjust the ticks
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ticks = g.get_xticks()
    new_ticks = ticks[:-1] + 0.5
    g.set_xticks(new_ticks)
    g.set_xticklabels([str(int(t)) for t in new_ticks])
    g.set_ylabel(f"frequency (n={num_sequences} sequences)")

    g.set_xlabel("amino acid mutations")
    g.spines["right"].set_visible(False)
    g.spines["top"].set_visible(False)
    for spine in ["left", "bottom"]:
        g.spines[spine].set_linewidth(0.5)
    g.tick_params(width=0.5)
    g.set_xlim([new_ticks[0] + 0.5, new_ticks[-1] - 0.5])

    outname_hamming_dir = out_dir
    outname_hamming_pairwise_plot = outname_hamming_dir + out_name_2
    plt.tight_layout()
    # Show the plot
    plt.savefig(f"{outname_hamming_pairwise_plot}.png", dpi=400)
    # plt.savefig(f'{outname_hamming_pairwise_plot}.pdf', dpi=400)
    # plt.savefig(f'{outname_hamming_pairwise_plot}.svg', dpi=400)
    plt.close()


import matplotlib.pyplot as plt
import numpy as np


def simple_venn(circle1_size, circle2_size, overlap_size):
    """
    Create a simple Venn diagram using basic matplotlib circles.

    Parameters:
    -----------
    circle1_size : float
        Total size of circle 1
    circle2_size : float
        Total size of circle 2
    overlap_size : float
        Size of the overlapping region
    """

    # Validate inputs
    if overlap_size > min(circle1_size, circle2_size):
        raise ValueError("Overlap size cannot be larger than the smaller circle")

    # Calculate radii (assuming area proportional to size)
    r1 = np.sqrt(circle1_size / np.pi)
    r2 = np.sqrt(circle2_size / np.pi)

    # Calculate distance between centers for desired overlap
    # This is an approximation - for exact overlap area calculation,
    # more complex math would be needed
    if overlap_size == 0:
        d = r1 + r2 + 0.1
    elif overlap_size == min(circle1_size, circle2_size):
        d = abs(r1 - r2)
    else:
        # Approximate distance for partial overlap
        overlap_ratio = overlap_size / min(circle1_size, circle2_size)
        d = r1 + r2 - 2 * min(r1, r2) * overlap_ratio

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw circles
    circle1 = plt.Circle(
        (-d / 2, 0),
        r1,
        color="blue",
        alpha=0.5,
        label=f"Circle 1 (size={circle1_size})",
    )
    circle2 = plt.Circle(
        (d / 2, 0), r2, color="red", alpha=0.5, label=f"Circle 2 (size={circle2_size})"
    )

    ax.add_patch(circle1)
    ax.add_patch(circle2)

    # Set axis properties
    ax.set_xlim(-r1 - d / 2 - 1, r2 + d / 2 + 1)
    ax.set_ylim(-max(r1, r2) - 1, max(r1, r2) + 1)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # Add labels
    ax.text(
        -d / 2,
        0,
        f"{circle1_size}",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
    )
    ax.text(
        d / 2,
        0,
        f"{circle2_size}",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
    )
    ax.text(0, 0, f"Overlap\n{overlap_size}", ha="center", va="center", fontsize=10)

    # Title and legend
    plt.title(
        f"Venn Diagram\nCircle 1: {circle1_size}, Circle 2: {circle2_size}, Overlap: {overlap_size}",
        fontsize=14,
        fontweight="bold",
    )
    plt.legend(loc="upper right")

    plt.tight_layout()
    # plt.show()

    return fig


# Quick usage function
def quick_venn():
    """Quick function to create Venn diagram with 3 inputs"""
    c1 = float(input("Circle 1 size: "))
    c2 = float(input("Circle 2 size: "))
    overlap = float(input("Overlap size: "))
    simple_venn(c1, c2, overlap)
