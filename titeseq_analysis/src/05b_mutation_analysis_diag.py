#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Diagnostic version with multiple filtering strategies and detailed output.
This helps identify why knee detection is too sensitive and provides alternatives.

Created on June 24, 2026
@author: alcantar modified by jhambale
Modified with diagnostic features and alternative filtering methods
"""

import argparse
from collections import Counter

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pysam
from Bio import SeqIO
from kneed import KneeLocator
from utils_nd import find_mutations, make_dir, translate_sequence


def extract_sequence(read, limits):
    """
    Given a pysam read and the index limits of a desired region, extract out the
    sequence of the desired region of the read, adjusting for any insertions or
    deletions in the alignment. written by V. Hu. Adapted by M. Alcantar
    to work with versions below python 3.10 (i.e., removed the match operator)
    """
    index = 0  # Pointer to position on the query
    q_start, q_end = limits  # start and end positions on the reference
    q_start -= read.reference_start
    q_end -= read.reference_start
    for operation, positions in read.cigartuples:
        if operation in [0, 7, 8]:  # Consume both Q and R
            index += positions
        elif operation == 1:  # Consume only Q
            q_start += (index <= q_start) * positions
            q_end += (index < q_end) * positions
            index += positions
        elif operation in [2, 3]:  # Consume only R
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


def analyze_distribution(data, sample_name):
    """
    Analyze the read count distribution and print diagnostic information.
    """
    y = sorted(data, reverse=True)

    print("\n" + "=" * 60)
    print(f"DISTRIBUTION ANALYSIS for {sample_name}")
    print("=" * 60)

    # Basic stats
    print(f"\nBasic Statistics:")
    print(f"  Total unique sequences: {len(y)}")
    print(f"  Total read count: {sum(y)}")
    print(f"  Max read count: {y[0] if y else 0}")
    print(f"  Min read count: {y[-1] if y else 0}")
    print(f"  Mean read count: {np.mean(y):.2f}")
    print(f"  Median read count: {np.median(y):.2f}")

    # Top sequences
    print(f"\nTop 20 sequence read counts:")
    for i in range(min(20, len(y))):
        print(f"  Rank {i + 1:3d}: {y[i]:8d} reads")

    # Analyze drops in read counts
    print(f"\nLargest drops in read counts (top 20):")
    if len(y) > 1:
        drops = []
        for i in range(len(y) - 1):
            drop_pct = (y[i] - y[i + 1]) / y[i] * 100 if y[i] > 0 else 0
            drops.append((i + 1, y[i], y[i + 1], drop_pct))

        # Sort by drop percentage
        drops.sort(key=lambda x: x[3], reverse=True)
        for i in range(min(20, len(drops))):
            rank, count1, count2, drop_pct = drops[i]
            print(
                f"  Rank {rank:3d} -> {rank + 1:3d}: {count1:8d} -> {count2:8d} ({drop_pct:6.2f}% drop)"
            )

    # Cumulative coverage
    print(f"\nCumulative read coverage:")
    total = sum(y)
    cumsum = 0
    thresholds = [50, 80, 90, 95, 99]
    for i, count in enumerate(y):
        cumsum += count
        pct = cumsum / total * 100
        if i < 10 or i + 1 in [20, 50, 100, 200, 500, 1000]:
            print(f"  Top {i + 1:4d} sequences: {pct:6.2f}% of reads")
        for t in thresholds:
            if pct >= t:
                thresholds.remove(t)
                print(f"  >>> {t}% coverage at rank {i + 1}")

    return y


def find_threshold_multiple_methods(
    data,
    sample_name,
    method="auto",
    min_reads=10,
    min_sequences=10,
    percentile=None,
    top_n=None,
    coverage_pct=None,
    knee_S=100.0,
):
    """
    Find threshold using multiple methods and compare results.

    Methods:
    - 'knee': Traditional knee detection
    - 'percentile': Keep sequences above Nth percentile of read counts
    - 'top_n': Keep top N sequences
    - 'coverage': Keep sequences until X% of total reads covered
    - 'min_reads': Simple minimum read count threshold
    - 'gradient': Find maximum gradient change
    - 'auto': Try multiple methods and show comparison
    """
    y = sorted(data, reverse=True)
    x = list(range(len(y)))
    total_reads = sum(y)

    results = {}

    # Method 1: Simple minimum read count
    if method in ["min_reads", "auto"]:
        threshold = min_reads
        n_seqs = sum(1 for c in y if c >= threshold)
        results["min_reads"] = {
            "threshold": threshold,
            "n_sequences": n_seqs,
            "method_name": f"Min {min_reads} reads",
        }

    # Method 2: Top N sequences
    if method in ["top_n", "auto"]:
        n = top_n if top_n else min(100, len(y))
        threshold = y[min(n - 1, len(y) - 1)] if y else 0
        results["top_n"] = {
            "threshold": threshold,
            "n_sequences": n,
            "method_name": f"Top {n} sequences",
        }

    # Method 3: Percentile-based
    if method in ["percentile", "auto"]:
        pct = percentile if percentile else 95
        percentile_value = np.percentile(y, pct) if y else 0
        # Find the minimum read count at or above the percentile value
        # This ensures we include all sequences at the minimum threshold found
        min_threshold = min((c for c in y if c >= percentile_value), default=0)
        n_seqs = sum(1 for c in y if c >= min_threshold)
        results["percentile"] = {
            "threshold": min_threshold,
            "n_sequences": n_seqs,
            "method_name": f"{pct}th percentile",
        }

    # Method 4: Coverage-based
    if method in ["coverage", "auto"]:
        target_pct = coverage_pct if coverage_pct else 90
        cumsum = 0
        for i, count in enumerate(y):
            cumsum += count
            if cumsum >= total_reads * target_pct / 100:
                threshold = count
                n_seqs = i + 1
                break
        else:
            threshold = y[-1] if y else 0
            n_seqs = len(y)
        results["coverage"] = {
            "threshold": threshold,
            "n_sequences": n_seqs,
            "method_name": f"{target_pct}% read coverage",
        }

    # Method 5: Gradient-based (find steepest drop)
    if method in ["gradient", "auto"] and len(y) > 2:
        # Calculate discrete gradient (rate of change)
        gradients = []
        for i in range(1, min(len(y) - 1, 200)):  # Look at first 200 sequences
            # Use log scale to normalize across different magnitudes
            if y[i] > 0 and y[i - 1] > 0:
                log_gradient = np.log10(y[i - 1]) - np.log10(y[i])
                gradients.append((i, log_gradient))

        if gradients:
            # Find position with maximum gradient
            max_grad_idx = max(gradients, key=lambda x: x[1])[0]
            threshold = y[max_grad_idx]
            n_seqs = max_grad_idx + 1
        else:
            threshold = y[min(9, len(y) - 1)] if y else 0
            n_seqs = min(10, len(y))

        results["gradient"] = {
            "threshold": threshold,
            "n_sequences": n_seqs,
            "method_name": "Max gradient",
        }

    # Method 6: Modified knee detection (on original data, not reversed)
    if method in ["knee", "auto"] and len(y) > 10:
        try:
            # Try knee on the descending curve directly
            # Use first 500 points to avoid noise in tail
            y_subset = y[: min(500, len(y))]
            x_subset = list(range(len(y_subset)))

            # Try different curve/direction combinations
            configs = [
                ("concave", "decreasing"),
                ("convex", "decreasing"),
            ]

            knee_results = []
            for curve, direction in configs:
                try:
                    knee = KneeLocator(
                        x_subset,
                        y_subset,
                        curve=curve,
                        direction=direction,
                        S=knee_S,
                        interp_method="polynomial",
                    )
                    if knee.knee is not None:
                        knee_results.append((knee.knee, f"{curve}_{direction}"))
                except:
                    pass

            if knee_results:
                # Use the median knee position if multiple found
                knee_positions = [k[0] for k in knee_results]
                knee_idx = int(np.median(knee_positions))
                threshold = y[knee_idx] if knee_idx < len(y) else y[-1]
                n_seqs = knee_idx + 1

                results["knee"] = {
                    "threshold": threshold,
                    "n_sequences": n_seqs,
                    "method_name": f"Knee (S={knee_S})",
                }
        except Exception as e:
            print(f"  Knee detection failed: {e}")

    # Apply minimum sequences constraint to all methods
    for method_name in results:
        if results[method_name]["n_sequences"] < min_sequences:
            # Adjust to minimum
            new_threshold = y[min(min_sequences - 1, len(y) - 1)] if y else 0
            results[method_name]["threshold"] = new_threshold
            results[method_name]["n_sequences"] = min_sequences
            results[method_name]["method_name"] += f" (adjusted to min {min_sequences})"

    # Print comparison
    print("\n" + "=" * 60)
    print("THRESHOLD COMPARISON")
    print("=" * 60)
    print(f"{'Method':<30} {'Threshold':>10} {'N Sequences':>12}")
    print("-" * 60)
    for method_name, res in results.items():
        print(
            f"{res['method_name']:<30} {res['threshold']:>10} {res['n_sequences']:>12}"
        )

    # Create diagnostic plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Define distinct colors for each method
    method_colors = {
        "min_reads": "#FF6B6B",  # Red
        "top_n": "#4ECDC4",  # Teal
        "percentile": "#45B7D1",  # Sky blue
        "coverage": "#96CEB4",  # Green
        "gradient": "#FFEAA7",  # Yellow
        "knee": "#DDA0DD",  # Plum
    }

    # Define line styles for variety
    method_styles = {
        "min_reads": "-",
        "top_n": "--",
        "percentile": "-.",
        "coverage": ":",
        "gradient": "-",
        "knee": "--",
    }

    # Plot 1: Linear scale with thresholds
    ax = axes[0, 0]
    ax.plot(x, y, "b-", linewidth=2, alpha=0.5, label="Data")
    for method_name, res in results.items():
        color = method_colors.get(method_name, "gray")
        style = method_styles.get(method_name, "--")
        ax.axhline(
            res["threshold"],
            linestyle=style,
            color=color,
            linewidth=2,
            alpha=0.8,
            label=res["method_name"],
        )
    ax.set_xlabel("Rank")
    ax.set_ylabel("Read Count")
    ax.set_title("Linear Scale - All Methods")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(0, min(200, len(x)))
    ax.grid(True, alpha=0.3)

    # Plot 2: Log scale
    ax = axes[0, 1]
    x_log = [i + 1 for i in range(len(y))]
    y_pos = [v for v in y if v > 0]
    x_log = x_log[: len(y_pos)]
    if y_pos:
        ax.loglog(x_log, y_pos, "b-", linewidth=2, alpha=0.5, label="Data")
        for method_name, res in results.items():
            if res["threshold"] > 0:
                color = method_colors.get(method_name, "gray")
                style = method_styles.get(method_name, "--")
                ax.axhline(
                    res["threshold"],
                    linestyle=style,
                    color=color,
                    linewidth=2,
                    alpha=0.8,
                    label=res["method_name"],
                )
    ax.set_xlabel("Rank (log)")
    ax.set_ylabel("Read Count (log)")
    ax.set_title("Log-Log Scale")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3, which="both")

    # Plot 3: Gradient
    ax = axes[0, 2]
    if len(y) > 2:
        gradients = []
        for i in range(1, min(len(y), 200)):
            if y[i] > 0 and y[i - 1] > 0:
                grad = (y[i - 1] - y[i]) / y[i - 1]
                gradients.append(grad)
        if gradients:
            ax.plot(range(1, len(gradients) + 1), gradients, "g-", linewidth=1)
            ax.set_xlabel("Rank")
            ax.set_ylabel("Relative Drop")
            ax.set_title("Relative Drop Between Consecutive Ranks")
            ax.grid(True, alpha=0.3)

    # Plot 4: Cumulative coverage
    ax = axes[1, 0]
    cumsum = np.cumsum(y)
    cum_pct = cumsum / total_reads * 100
    ax.plot(x, cum_pct, "darkgreen", linewidth=2.5, label="Cumulative Coverage")

    # Add coverage threshold lines
    coverage_thresholds = [50, 80, 90, 95, 99]
    coverage_colors = ["#FFD93D", "#FF9A00", "#FF6B6B", "#C73E1D", "#8B0000"]
    for thresh, color in zip(coverage_thresholds, coverage_colors):
        ax.axhline(
            thresh,
            linestyle="--",
            color=color,
            alpha=0.5,
            linewidth=1,
            label=f"{thresh}% coverage",
        )

    # Mark where each method's threshold lands on the coverage curve
    for method_name, res in results.items():
        n = res["n_sequences"]
        if n < len(cum_pct):
            color = method_colors.get(method_name, "gray")
            ax.plot(
                n,
                cum_pct[n],
                "o",
                color=color,
                markersize=8,
                label=f"{method_name[:4]}: {cum_pct[n]:.1f}%",
            )

    ax.set_xlabel("Rank")
    ax.set_ylabel("Cumulative % of Reads")
    ax.set_title("Cumulative Read Coverage")
    ax.legend(loc="lower right", fontsize=7, ncol=2)
    ax.set_xlim(0, min(500, len(x)))
    ax.grid(True, alpha=0.3)

    # Plot 5: Zoomed linear view
    ax = axes[1, 1]
    zoom_n = min(50, len(y))
    ax.bar(
        range(zoom_n),
        y[:zoom_n],
        color="steelblue",
        alpha=0.6,
        edgecolor="navy",
        linewidth=0.5,
    )

    # Add vertical lines for each method's cutoff
    legend_elements = []
    for method_name, res in results.items():
        n = res["n_sequences"]
        if n <= zoom_n:
            color = method_colors.get(method_name, "gray")
            ax.axvline(n - 0.5, color=color, linestyle="--", alpha=0.8, linewidth=2)
            # Add text label at top
            ax.text(
                n - 0.5,
                max(y[:zoom_n]) * 0.95,
                method_name[:4],
                rotation=90,
                ha="right",
                va="top",
                fontsize=7,
                color=color,
            )

    ax.set_xlabel("Rank")
    ax.set_ylabel("Read Count")
    ax.set_title(f"Top {zoom_n} Sequences (Bar Chart)")
    ax.grid(True, axis="y", alpha=0.3)

    # Plot 6: Method comparison
    ax = axes[1, 2]
    methods = list(results.keys())
    thresholds = [results[m]["threshold"] for m in methods]
    n_seqs = [results[m]["n_sequences"] for m in methods]

    # Use the same colors as the other plots
    bar_colors = [method_colors.get(m, "gray") for m in methods]

    x_pos = np.arange(len(methods))
    bars = ax.bar(
        x_pos, n_seqs, color=bar_colors, alpha=0.7, edgecolor="black", linewidth=1
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.set_ylabel("Number of Sequences")
    ax.set_title("Sequences Included by Method")
    ax.grid(True, axis="y", alpha=0.3)

    # Add threshold values as text
    for i, (n, t) in enumerate(zip(n_seqs, thresholds)):
        ax.text(
            i,
            n + max(n_seqs) * 0.02,
            f"thresh={t}",
            ha="center",
            fontsize=7,
            fontweight="bold",
        )
        ax.text(
            i,
            n / 2,
            f"{n}\nseqs",
            ha="center",
            va="center",
            fontsize=8,
            color="white",
            fontweight="bold",
        )

    plt.suptitle(f"Threshold Analysis for {sample_name}")
    plt.tight_layout()

    return results, fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", nargs="+", help="List of bam files to parse")
    parser.add_argument("-m", "--metadata", help="Path to metadata")
    parser.add_argument("-r", help="Path to reference database")
    parser.add_argument(
        "--method",
        default="auto",
        choices=[
            "auto",
            "knee",
            "percentile",
            "top_n",
            "coverage",
            "min_reads",
            "gradient",
        ],
        help="Filtering method. 'auto' tries all methods. Default: auto",
    )
    parser.add_argument(
        "--min-reads",
        default=10,
        type=int,
        help="Minimum read count threshold. Default: 10",
    )
    parser.add_argument(
        "--min-sequences",
        default=10,
        type=int,
        help="Minimum sequences to include. Default: 10",
    )
    parser.add_argument(
        "--top-n",
        default=100,
        type=int,
        help="Number of top sequences for 'top_n' method. Default: 100",
    )
    parser.add_argument(
        "--percentile",
        default=95,
        type=float,
        help="Percentile threshold for 'percentile' method. Default: 95",
    )
    parser.add_argument(
        "--coverage",
        default=90,
        type=float,
        help="Read coverage percentage for 'coverage' method. Default: 90",
    )
    parser.add_argument(
        "-S",
        "--knee-S",
        default=100.0,
        type=float,
        help="Sensitivity for knee detection. Default: 100",
    )
    parser.add_argument(
        "--use-method",
        default=None,
        help="Which method's threshold to actually use for filtering. Default: top_n",
    )

    args = parser.parse_args()

    alignment_files = args.i
    metadata_path = args.metadata
    ref_dir = args.r

    if ref_dir[-1] != "/":
        ref_dir = ref_dir + "/"

    metadata_df = pd.read_csv(metadata_path, sep="\t", index_col=None)

    for alignment in alignment_files:
        print("\n" + "=" * 80)
        print(f"PROCESSING: {alignment}")
        print("=" * 80)

        sample_name = alignment.split("/")[-1].replace(".bam", "")
        metadata_samp = metadata_df.copy()[
            metadata_df["sample_id"].str.contains(sample_name)
        ]
        reference_name = list(metadata_samp["reference"])[0]

        if ".fasta" not in reference_name:
            reference_name = reference_name + ".fasta"
        ref_path = ref_dir + reference_name

        print(f"Reference: {reference_name}")
        samfile = pysam.AlignmentFile(alignment, "rb")
        with open(ref_path, "rt") as reference_reads:
            reference = SeqIO.parse(reference_reads, "fasta")
            for reads in reference:
                reads = reads.upper()
                reference_sequence_dna = reads.seq
                reference_size = len(reference_sequence_dna)
        reference_sequence_aa = str(reference_sequence_dna.translate())

        # Extract sequences
        sequences = []
        print("Extracting sequences...")
        for read in samfile:
            try:
                sequence = extract_sequence(read, [0, reference_size])
                if len(sequence) == reference_size and "N" not in sequence:
                    sequences.append(sequence)
            except Exception:
                pass

        # Count sequences
        sequences_counter = Counter(sequences)

        # Analyze distribution
        y = analyze_distribution(list(sequences_counter.values()), sample_name)

        # Find thresholds using multiple methods
        results, fig = find_threshold_multiple_methods(
            list(sequences_counter.values()),
            sample_name,
            method=args.method,
            min_reads=args.min_reads,
            min_sequences=args.min_sequences,
            top_n=args.top_n,
            percentile=args.percentile,
            coverage_pct=args.coverage,
            knee_S=args.knee_S,
        )

        # Decide which method to use
        if args.use_method and args.use_method in results:
            chosen_method = args.use_method
        elif "top_n" in results:
            chosen_method = "top_n"
        elif results:
            chosen_method = list(results.keys())[0]
        else:
            print("No filtering method succeeded!")
            continue

        threshold = results[chosen_method]["threshold"]
        print(f"\n>>> USING METHOD: {results[chosen_method]['method_name']}")
        print(f">>> THRESHOLD: {threshold}")

        # Apply filtering
        filtered_sequences = {
            seq: count for seq, count in sequences_counter.items() if count >= threshold
        }

        filtered_sequences_dict = filtered_sequences
        seq_count_total = sum(filtered_sequences_dict.values())
        print(f">>> Sequences passing filter: {len(filtered_sequences_dict)}")
        print(f">>> Total reads passing filter: {seq_count_total}")

        # Find mutations
        print("\nFinding mutations...")
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

        # Save results
        mutations_df_outdir = "/".join(alignment.split("/")[:-1]).replace(
            "alignments", f"mutation_dfs_{chosen_method}/"
        )
        make_dir(mutations_df_outdir)

        method_str = (
            results[chosen_method]["method_name"]
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
        )
        mutations_df_outname = (
            sample_name
            + f"_diagnostic_{method_str}_thresh{threshold}_mutation_analysis.csv"
        )
        mutations_df_outpath = mutations_df_outdir + mutations_df_outname
        mutations_df = pd.DataFrame(mutation_analysis_dict_list)
        mutations_df = mutations_df.sort_values(
            by="read_count", ascending=False
        ).reset_index(drop=True)
        mutations_df.to_csv(mutations_df_outpath)
        print(f"\nSaved CSV: {mutations_df_outpath}")

        # Save plot
        plot_name = mutations_df_outdir + f"{sample_name}_diagnostic_{method_str}.png"
        fig.savefig(plot_name, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot: {plot_name}")


if __name__ == "__main__":
    main()
