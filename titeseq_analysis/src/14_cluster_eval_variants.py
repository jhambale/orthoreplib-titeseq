#!/usr/bin/env python3

"""
Created on 2026 May 11
@author: jhambale

14_cluster_eval_variants.py
===========================
Performs GMM-based clustering on midpoint EC50 values from Tite-Seq passed
variants, plots mutation-distance histograms per cluster, and runs the same
average-score plotting and per-point-mutation context analysis as done in
Neutral_Drift_Miguel/.../19_neutvariants.py.

Steps
-----
1.  Load passed-variants CSV (output of 10_compute_Kds.py).
2.  Compute log10(midpoint EC50) per replicate; GMM fit (BIC-optimal, max 3 components)
    → labels: "Tight Binder", "Moderate Binder", "Weak Binder".
3.  Plot replicate scatter coloured by GMM label (mirroring 14_ scatter).
4.  Plot per-cluster mutation-distance histograms relative to the 'parent'
    variant_id (using aa_mutations column; 'parent' = [] or the reference
    fasta when -r is provided).
5a. (NEW) Compare point mutation frequencies across the three binding groups.
    Extracts mutations from variant_id and aa_mutations columns, calculates
    frequency in each group, and ranks by maximum pairwise difference.
    Output: terminal summary + CSV 'mutation_frequency_comparison.csv'
5b. Sort variants by number_aa_mutations and plot average EC50 per sequence
    with WT-distance zone separators (plot_seqavgs from 19_).
6.  Build single-mutant families from the library aa_sequences; for each
    point mutation with >= min_contexts contexts:
        - plot binding-score (log10 midpoint EC50 avg) for each context
        - compute pairwise context distances and histogram them
        - accumulate delta-EC50 values for ridge plot
7.  Ridge plot of delta-log10(midpoint EC50) per point mutation.
8.  DMS-like heatmap of single-mutation effect on log10(midpoint EC50).

Usage
-----
python 14_cluster_eval_variants.py \\
    -i ../../titeseq_data/outputs/jh_002/tite_seq_Kds2/tite_seq_Kds_passed_variants.csv \\
    -r ../../titeseq_data/raw/jh_002/references/pmaa23_nbonly.fasta \\
    -m 2 -x 3 -d 0.2
"""

import argparse
import ast
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from Bio import SeqIO
from matplotlib.ticker import MaxNLocator
from sklearn.mixture import GaussianMixture
from utils_nd import (
    find_mutations,
    hamming_distance,
    make_dir,
    translate_sequence,
)

# ---------------------------------------------------------------------------
# Plot helpers  (ported / adapted from 19_neutvariants.py)
# ---------------------------------------------------------------------------

CATEGORY_COLORS = {
    "Tight Binder": "#1a7a4a",  # dark green — smallest midpoint EC50
    "Moderate Binder": "#3B71B2",  # blue
    "Weak Binder": "grey",  # largest midpoint EC50
}


# ---------------------------------------------------------------------------
# Mutation frequency comparison across groups
# ---------------------------------------------------------------------------


def compile_and_compare_mutations(
    tight_df: pd.DataFrame,
    moderate_df: pd.DataFrame,
    weak_df: pd.DataFrame,
    out_dir: Path,
) -> pd.DataFrame:
    # Extract point mutations from variant_id/aa_mutations columns for each group.
    # Count occurrences, calculate frequency (count/total), compute pairwise differences.
    # Rank mutations by max_pairwise_diff (highest = most different across groups).
    # High difference (>0.2) suggests functional importance.
    # Beneficial: tight_frequency high + weak_frequency low.
    # Deleterious: tight_frequency low + weak_frequency high.
    # Output: (1) terminal summary top 20, (2) CSV 'mutation_frequency_comparison.csv'
    # with columns: mutation, tight_count, tight_frequency, moderate_count, moderate_frequency,
    # weak_count, weak_frequency, tight_vs_moderate_diff, tight_vs_weak_diff,
    # moderate_vs_weak_diff, max_pairwise_diff, coefficient_of_variation.

    def extract_mutations_from_group(group_df: pd.DataFrame, group_name: str) -> dict:
        # Extract mutations from variant_id and aa_mutations columns.
        # Returns dict with mutation as key, count as value.
        mutations = {}

        for idx, row in group_df.iterrows():
            row_muts = set()

            # Extract from variant_id (regex [A-Z]\d+[A-Z])
            try:
                if "variant_id" in row:
                    val = row["variant_id"]
                    # Skip if None or NaN (check float for NaN compatibility)
                    if val is not None and (
                        not isinstance(val, float) or not pd.isna(val)
                    ):
                        variant_id = str(val)
                        single_mut_pattern = r"([A-Z])(\d+)([A-Z])"
                        matches = re.findall(single_mut_pattern, variant_id)
                        for wt, pos, mut in matches:
                            mutation = f"{wt}{pos}{mut}"
                            row_muts.add(mutation)
            except (TypeError, ValueError, AttributeError):
                pass

            # Extract from aa_mutations (parsed as list)
            try:
                if "aa_mutations" in row:
                    val = row["aa_mutations"]
                    # Skip if None or NaN
                    if val is not None and (
                        not isinstance(val, float) or not pd.isna(val)
                    ):
                        aa_muts = ast.literal_eval(str(val))
                        if isinstance(aa_muts, list):
                            for mut in aa_muts:
                                if isinstance(mut, str) and re.match(
                                    r"[A-Z]\d+[A-Z]", mut
                                ):
                                    row_muts.add(mut)
            except (ValueError, SyntaxError, TypeError, AttributeError):
                pass

            # Count mutations
            for mut in row_muts:
                if mut not in mutations:
                    mutations[mut] = 0
                mutations[mut] += 1

        return mutations

    # Extract mutations from each group
    print("\nCompiling mutations from each binding group...")
    tight_muts = extract_mutations_from_group(tight_df, "Tight Binder")
    moderate_muts = extract_mutations_from_group(moderate_df, "Moderate Binder")
    weak_muts = extract_mutations_from_group(weak_df, "Weak Binder")

    print(f"  Tight Binder: {len(tight_muts)} unique mutations")
    print(f"  Moderate Binder: {len(moderate_muts)} unique mutations")
    print(f"  Weak Binder: {len(weak_muts)} unique mutations")

    # Get all unique mutations across groups
    all_mutations = (
        set(tight_muts.keys()) | set(moderate_muts.keys()) | set(weak_muts.keys())
    )
    print(f"  Total unique mutations across groups: {len(all_mutations)}")

    # Calculate frequencies and comparisons
    comparison_records = []

    tight_total = len(tight_df)
    moderate_total = len(moderate_df)
    weak_total = len(weak_df)

    for mutation in sorted(all_mutations):
        tight_count = tight_muts.get(mutation, 0)
        moderate_count = moderate_muts.get(mutation, 0)
        weak_count = weak_muts.get(mutation, 0)

        # Calculate frequencies (percentage of group with this mutation)
        tight_freq = tight_count / tight_total if tight_total > 0 else 0
        moderate_freq = moderate_count / moderate_total if moderate_total > 0 else 0
        weak_freq = weak_count / weak_total if weak_total > 0 else 0

        # Calculate frequency differences (pairwise)
        tight_vs_moderate = abs(tight_freq - moderate_freq)
        tight_vs_weak = abs(tight_freq - weak_freq)
        moderate_vs_weak = abs(moderate_freq - weak_freq)

        # Maximum difference across any pair
        max_diff = max(tight_vs_moderate, tight_vs_weak, moderate_vs_weak)

        # Overall coefficient of variation (std / mean, treating each group as equal importance)
        freqs = [tight_freq, moderate_freq, weak_freq]
        mean_freq = np.mean(freqs)
        std_freq = np.std(freqs)
        cv = std_freq / mean_freq if mean_freq > 0 else 0

        comparison_records.append(
            {
                "mutation": mutation,
                "tight_count": tight_count,
                "tight_frequency": tight_freq,
                "moderate_count": moderate_count,
                "moderate_frequency": moderate_freq,
                "weak_count": weak_count,
                "weak_frequency": weak_freq,
                "tight_vs_moderate_diff": tight_vs_moderate,
                "tight_vs_weak_diff": tight_vs_weak,
                "moderate_vs_weak_diff": moderate_vs_weak,
                "max_pairwise_diff": max_diff,
                "coefficient_of_variation": cv,
            }
        )

    # Create DataFrame and sort by max difference
    comparison_df = pd.DataFrame(comparison_records)
    comparison_df = comparison_df.sort_values(
        "max_pairwise_diff", ascending=False
    ).reset_index(drop=True)

    # Print top mutations with largest frequency differences
    print("\n" + "=" * 80)
    print("MUTATIONS WITH LARGEST FREQUENCY DIFFERENCES BETWEEN GROUPS")
    print("=" * 80)

    top_n = min(20, len(comparison_df))
    for idx, row in comparison_df.head(top_n).iterrows():
        print(f"\n{idx + 1}. Mutation: {row['mutation']}")
        print(
            f"   Tight Binder:    {row['tight_count']:3d}/{tight_total} ({row['tight_frequency'] * 100:6.2f}%)"
        )
        print(
            f"   Moderate Binder: {row['moderate_count']:3d}/{moderate_total} ({row['moderate_frequency'] * 100:6.2f}%)"
        )
        print(
            f"   Weak Binder:     {row['weak_count']:3d}/{weak_total} ({row['weak_frequency'] * 100:6.2f}%)"
        )
        print(f"   Tight vs Moderate: {row['tight_vs_moderate_diff']:.4f}")
        print(f"   Tight vs Weak:     {row['tight_vs_weak_diff']:.4f}")
        print(f"   Moderate vs Weak:  {row['moderate_vs_weak_diff']:.4f}")
        print(
            f"   Max Difference:    {row['max_pairwise_diff']:.4f}  |  CV: {row['coefficient_of_variation']:.4f}"
        )

    print("\n" + "=" * 80)

    # Save to CSV
    csv_path = out_dir / "mutation_frequency_comparison.csv"
    comparison_df.to_csv(csv_path, index=False)
    print(f"\nMutation frequency comparison saved to: {csv_path}")

    return comparison_df


def plot_mutation_frequency_comparison(
    comparison_df: pd.DataFrame, fasta_name: str, outpath: str, top_n: int = 25
):
    """
    Grouped bar plot of mutation frequencies across binding groups, sorted by average frequency.
    Shows frequency (0-1) for Tight, Moderate, and Weak Binder groups for each mutation.
    Top mutations by average frequency are displayed.
    X-axis: mutation strings (e.g., A25T); Y-axis: frequency; bars colored by group.
    """
    # Calculate average frequency across all binding groups
    comparison_df["avg_frequency"] = (
        comparison_df["tight_frequency"]
        + comparison_df["moderate_frequency"]
        + comparison_df["weak_frequency"]
    ) / 3.0

    # Sort by average frequency and take top_n
    top_df = comparison_df.nlargest(top_n, "avg_frequency").copy()
    top_df = top_df.sort_values(
        "avg_frequency", ascending=False
    )  # descending for left-to-right (highest avg freq on left)

    mutations = top_df["mutation"].values
    tight_freqs = top_df["tight_frequency"].values
    moderate_freqs = top_df["moderate_frequency"].values
    weak_freqs = top_df["weak_frequency"].values

    x = np.arange(len(mutations))
    width = 0.25  # width of each bar

    fig, ax = plt.subplots(figsize=(16, 7))

    # Plot bars for each group with matching colors
    bars1 = ax.bar(
        x - width,
        tight_freqs,
        width,
        label="Tight Binder",
        color=CATEGORY_COLORS["Tight Binder"],
        alpha=0.85,
        edgecolor="black",
        linewidth=0.5,
    )
    bars2 = ax.bar(
        x,
        moderate_freqs,
        width,
        label="Moderate Binder",
        color=CATEGORY_COLORS["Moderate Binder"],
        alpha=0.85,
        edgecolor="black",
        linewidth=0.5,
    )
    bars3 = ax.bar(
        x + width,
        weak_freqs,
        width,
        label="Weak Binder",
        color=CATEGORY_COLORS["Weak Binder"],
        alpha=0.85,
        edgecolor="black",
        linewidth=0.5,
    )

    # Formatting
    ax.set_ylabel("Frequency", fontsize=13, fontweight="bold")
    ax.set_xlabel("Mutation", fontsize=13, fontweight="bold")
    ax.set_title(
        f"Mutation Frequencies by Binding Group (Top {top_n} by avg frequency, highest on left) - {fasta_name}",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(mutations, rotation=45, ha="right")
    ax.set_ylim([0, 1.0])
    ax.legend(fontsize=11, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(outpath, dpi=400, bbox_inches="tight")
    print(f"Mutation frequency comparison plot saved: {outpath}")
    plt.close()


def plot_seqavgs(library_org_df: pd.DataFrame, fasta_name: str, org_outpath: str):
    """
    Plot average log10(midpoint EC50) per sequence ordered by mutation count
    parent, with vertical dashed separators between mutation-distance groups.

    Mirrors plot_seqavgs() in 19_neutvariants.py but uses log10(midpoint EC50)
    the y-axis value.
    """
    colors_org = sns.color_palette("husl", len(library_org_df))
    wtdistances_org = []
    high_var = []

    plt.figure(figsize=(20, 6))
    for seq in range(len(library_org_df)):
        score_avg = library_org_df["log10_EC50_midpoint_avg"].iloc[seq]
        score_err = library_org_df["log10_EC50_midpoint_err"].iloc[seq]
        plt.plot(seq, score_avg, marker="o", ms=4, color=colors_org[seq])
        plt.errorbar(
            seq,
            score_avg,
            yerr=score_err,
            capsize=4,
            elinewidth=0.5,
            alpha=0.5,
            color=colors_org[seq],
        )
        wtdistances_org.append(library_org_df["number_aa_mutations"].iloc[seq])
        if score_err > (math.ceil(1 / 3 * 100) / 100):
            high_var.append(seq)

    plt.xlabel("Sequence")
    plt.ylabel("Average log10(midpoint EC50 [nM])")
    plt.title(f"Average log10(midpoint EC50) for sequences in {fasta_name}")
    plt.xlim(-0.5, None)

    xmino, xmaxo = plt.xlim()
    linedist_org = [0]
    for disto in range(len(wtdistances_org[:-1])):
        if wtdistances_org[disto] != wtdistances_org[disto + 1]:
            nextdisto = wtdistances_org[disto + 1] - wtdistances_org[disto]
            for lineo in range(nextdisto):
                linexo = (disto + 1.5) + (lineo / nextdisto * 0.4)
                plt.axvline(linexo, color="k", linestyle="--", linewidth=0.5)
                linedist_org.append(linexo)
            plt.text(
                (linedist_org[-1] + linedist_org[-(int(nextdisto) + 1)]) / 2,
                plt.ylim()[0] - 0.05 * abs(plt.ylim()[1] - plt.ylim()[0]),
                f"{wtdistances_org[disto]} aa from WT",
                ha="center",
                va="top",
                rotation=90,
                alpha=0.3,
                size=8,
            )
    plt.text(
        (xmaxo + linedist_org[-1]) / 2,
        plt.ylim()[0] - 0.05 * abs(plt.ylim()[1] - plt.ylim()[0]),
        f"{wtdistances_org[-1]} aa from WT",
        ha="center",
        va="top",
        rotation=90,
        alpha=0.3,
        size=8,
    )
    plt.tight_layout()
    plt.savefig(org_outpath, bbox_inches="tight", dpi=400)
    print(f"averages figure saved at {org_outpath}")
    plt.close()
    return high_var


def plot_mutation_contexts(
    pmutslice: pd.DataFrame,
    colors,
    muti: str,
    mutidist_outpath: str,
):
    """
    Plot log10(midpoint EC50) before (WT residue, circle) and after (mutated residue,
    cross) the point mutation for every context, with WT-distance separators.

    Mirrors plot_mutation_contexts() in 19_neutvariants.py, substituting
    log10(midpoint EC50) for binding score.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    wtdistances = []
    for row in range(len(pmutslice)):
        if (
            pmutslice["delta_log10EC50_midpoint"].iloc[row]
            == pmutslice["log10EC50_midpoint2"].iloc[row]
            - pmutslice["log10EC50_midpoint1"].iloc[row]
        ):
            ax.plot(
                row + 1,
                pmutslice["log10EC50_midpoint1"].iloc[row],
                marker="o",
                ms=5,
                color=colors[row],
            )
            ax.plot(
                row + 1,
                pmutslice["log10EC50_midpoint2"].iloc[row],
                marker="x",
                ms=5,
                color=colors[row],
            )
        else:
            ax.plot(
                row + 1,
                pmutslice["log10EC50_midpoint2"].iloc[row],
                marker="o",
                ms=5,
                color=colors[row],
            )
            ax.plot(
                row + 1,
                pmutslice["log10EC50_midpoint1"].iloc[row],
                marker="x",
                ms=5,
                color=colors[row],
            )
        wtdistances.append(pmutslice["wild_type_distance"].iloc[row])

    ax.plot([], [], marker="o", color="k", label="WT residue context")
    ax.plot([], [], marker="x", color="k", label="mutated residue context")
    ax.set_xlabel("Context")
    ax.set_ylabel("log10(midpoint EC50 [nM])")
    ax.set_title(f"log10(midpoint EC50) for contexts with {muti}")
    ax.legend(
        loc="upper right", bbox_to_anchor=(1.4, 1), borderaxespad=0, frameon=False
    )
    ax.set_xlim(0.5, None)

    xmin, xmax = ax.get_xlim()
    linedist = [0]
    for dist in range(len(wtdistances[:-1])):
        if wtdistances[dist] != wtdistances[dist + 1]:
            nextdist = wtdistances[dist + 1] - wtdistances[dist]
            for line in range(nextdist):
                linex = (dist + 1.5) + (line / nextdist * 0.4)
                ax.axvline(linex, color="k", linestyle="--", linewidth=0.5)
                linedist.append(linex)
            ax.text(
                (linedist[-1] + linedist[-(int(nextdist) + 1)]) / 2,
                ax.get_ylim()[0],
                f"{wtdistances[dist]} aa from WT",
                ha="center",
                va="top",
                rotation=90,
                alpha=0.3,
                size=8,
            )
    ax.text(
        (xmax + linedist[-1]) / 2,
        ax.get_ylim()[0],
        f"{wtdistances[-1]} aa from WT",
        ha="center",
        va="top",
        rotation=90,
        alpha=0.3,
        size=8,
    )
    plt.savefig(mutidist_outpath, bbox_inches="tight", dpi=400)
    print(f"context figure saved at {mutidist_outpath}")
    plt.close()


def plot_context_distances(
    hamming_dist_contexts: dict,
    mutk: str,
    contexts_dict: dict,
    mutkdist_outpath: str,
):
    """
    Histogram of pairwise aa distances between the context sequences that
    share a given point mutation.  Mirrors plot_context_distances() in 19_.
    """
    dists = hamming_dist_contexts[mutk]
    n_seqs = len(contexts_dict[mutk])

    fig, ax = plt.subplots(figsize=(8, 6))

    # When all pairwise distances are identical the data range is 0, which
    # makes seaborn/numpy refuse to create bins with binwidth=1.  Fall back
    # to a single bar (bins=[val-0.5, val+0.5]) in that edge case.
    unique_vals = np.unique(dists)
    if len(unique_vals) <= 1:
        single_val = int(unique_vals[0]) if len(unique_vals) == 1 else 0
        ax.bar(
            [single_val],
            [1.0],
            width=1.0,
            align="center",
            edgecolor="black",
            linewidth=0.5,
        )
        ax.set_xticks([single_val])
        ax.set_xticklabels([str(single_val)])
        g = ax
    else:
        g = sns.histplot(
            data=hamming_dist_contexts,
            x=dists,
            binwidth=1,
            stat="probability",
            edgecolor="black",
            linewidth=0.5,
            ax=ax,
        )
        g.xaxis.set_major_locator(MaxNLocator(integer=True))
        ticks = ax.get_xticks()
        new_ticks = ticks[:-1] + 0.5
        g.set_xticks(new_ticks)
        g.set_xticklabels([str(int(t)) for t in new_ticks])
        if len(new_ticks):
            g.set_xlim([new_ticks[0] + 0.5, new_ticks[-1] - 0.5])

    ax.set_xlabel("aa mutations")
    ax.set_ylabel(f"frequency (n={n_seqs} sequences)")
    ax.set_title(f"context distances for {mutk} point mutation")
    for spine in ["right", "top"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_linewidth(0.5)
    ax.tick_params(width=0.5)
    plt.savefig(mutkdist_outpath, dpi=400)
    print(f"distance figure saved at {mutkdist_outpath}")
    plt.close()


def plot_context_ridges(
    sorted_mut_dict: dict,
    ridgedata_df: pd.DataFrame,
    fasta_name: str,
    ridge_outpath: str,
):
    """
    Ridge (joy) plot of delta-log10(midpoint EC50) per point mutation.
    Mirrors plot_context_ridges() in 19_neutvariants.py.
    """
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

    order = (
        ridgedata_df.groupby("mutation")["dlog10EC50_midpoint"]
        .median()
        .sort_values(ascending=False)
        .index
    )

    pal = sns.cubehelix_palette(
        n_colors=len(order), start=0.5, rot=-0.75, light=0.75, dark=0.25
    )
    palette_dict = dict(zip(order, pal))

    g = sns.FacetGrid(
        ridgedata_df,
        row="mutation",
        hue="mutation",
        aspect=10,
        height=1,
        palette=palette_dict,
        row_order=order,
    )

    g.map(
        sns.kdeplot,
        "dlog10EC50_midpoint",
        bw_adjust=0.5,
        clip_on=False,
        fill=True,
        alpha=1,
        linewidth=1,
        warn_singular=False,
    )
    g.map(plt.axhline, y=0, lw=2, clip_on=False)
    g.map(sns.kdeplot, "dlog10EC50_midpoint", clip_on=False, color="white", lw=2, bw_adjust=0.5)
    g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

    def label(x, color, label):
        ax = plt.gca()
        ax.text(
            0,
            0.2,
            label,
            fontweight="bold",
            color=color,
            ha="left",
            va="center",
            transform=ax.transAxes,
        )

    g.map(label, "dlog10EC50_midpoint")
    g.figure.subplots_adjust(hspace=-0.3)
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.set_xlabels("Δlog10(midpoint EC50 [nM])", fontsize=14)
    g.despine(bottom=True, left=True)
    plt.suptitle(f"Point-mutation Δlog10(midpoint EC50)  ({fasta_name})", fontsize=16, y=0.95)
    plt.savefig(ridge_outpath, format="png", dpi=500, bbox_inches="tight")
    print(f"ridge figure saved at {ridge_outpath}")
    plt.close()


def plot_dms(heatmap_matrix: pd.DataFrame, fasta_name: str, heatmap_outpath: str):
    """
    Deep-mutational-scanning-style heatmap of single-mutation Δlog10(midpoint EC50).
    Mirrors plot_dms() in 19_neutvariants.py.
    """
    plt.rcParams.update({"font.size": 25})
    plt.figure(figsize=(len(heatmap_matrix.columns) / 1.5, 40))
    ax = sns.heatmap(
        heatmap_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm_r",  # reversed: blue = lower midpoint EC50 = better
        center=0,
        linewidths=0.5,
        linecolor="gray",
        cbar_kws={"label": "Δlog10(midpoint EC50 [nM])"},
        annot_kws={"fontsize": 25, "rotation": 90},
    )
    ax.set_xticklabels(heatmap_matrix.columns, rotation=90, fontsize=20)
    ax.set_yticklabels(heatmap_matrix.index, rotation=0, fontsize=20)
    plt.xlabel("Sequence Position (WT residue)")
    plt.ylabel("Mutant residue")
    plt.title(f"Single-mutation Δlog10(midpoint EC50) heatmap for {fasta_name}", fontsize=40)
    plt.savefig(heatmap_outpath, format="png", dpi=500, bbox_inches="tight")
    print(f"DMS figure saved at {heatmap_outpath}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="GMM clustering of replicate midpoint EC50 values and context analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "-i",
        required=True,
        help="Path to tite_seq_Kds_passed_variants.csv from script 10_compute_Kds.py",
    )
    parser.add_argument(
        "-r",
        default=None,
        help="Path to reference fasta (used for parent aa sequence and DMS analysis).",
    )
    parser.add_argument(
        "-m",
        type=int,
        default=2,
        help="Max mutant-distance (up to m aa apart) for context analysis (default: 2).",
    )
    parser.add_argument(
        "-x",
        type=int,
        default=3,
        help="Minimum number of contexts required for a point-mutation plot (default: 3).",
    )
    parser.add_argument(
        "-d",
        type=float,
        default=0.2,
        help="Minimum |Δlog10(midpoint EC50)| for multi-aa context plots (default: 0.2).",
    )
    parser.add_argument(
        "-p",
        type=int,
        default=25,
        help="Number of point mutations to include in the grouped bars plot (default: 25).",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load data and set up output directories
    # ------------------------------------------------------------------
    csv_path = Path(args.i)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    # Extract experiment id from filename (e.g. jh_002)
    match = re.search(r"(jh_\d+)", csv_path.name)
    expt_id = match.group(1) if match else csv_path.stem

    out_dir = csv_path.parent / f"cluster_eval_{expt_id}"
    out_dir_figs = out_dir / "plots"
    out_dir_pointmuts = out_dir / "point_mutations"
    for d in [out_dir, out_dir_figs, out_dir_pointmuts]:
        make_dir(str(d))

    print(f"Experiment ID : {expt_id}")
    print(f"Output directory: {out_dir}")

    df = pd.read_csv(csv_path)
    if "passed" not in df.columns:
        raise ValueError(
            "Input CSV must include the 'passed' column from script 10 so "
            "script 14 can verify and retain only passed variants."
        )
    passed_mask = df["passed"].astype(str).str.strip().str.lower().isin(
        {"true", "1", "yes"}
    )
    excluded_not_passed = int((~passed_mask).sum())
    df = df.loc[passed_mask].copy()
    print(
        f"Loaded {len(df)} passed variants; excluded {excluded_not_passed} "
        "rows not marked passed."
    )
    if df.empty:
        raise ValueError("Input CSV contains no rows marked as passed.")

    # ------------------------------------------------------------------
    # 2. Identify replicate midpoint EC50 columns
    # ------------------------------------------------------------------
    ec50_cols = [
        c for c in df.columns if re.match(r"rep\d+_EC50_midpoint_nM$", c)
    ]
    if len(ec50_cols) < 1:
        raise ValueError(
            "No rep*_EC50_midpoint_nM columns found. Make sure the input is "
            "tite_seq_Kds_passed_variants.csv from 10_compute_Kds.py."
        )
    print(f"Replicate midpoint EC50 columns: {ec50_cols}")

    # Retain only passed variants with valid midpoint EC50 values for every replicate.
    for col in ec50_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    valid_ec50_mask = (df[ec50_cols] > 0).all(axis=1) & np.isfinite(
        df[ec50_cols]
    ).all(axis=1)
    invalid_ec50_count = int((~valid_ec50_mask).sum())
    if invalid_ec50_count:
        print(
            f"Excluded {invalid_ec50_count} passed variants without positive, "
            "finite midpoint EC50 values for every replicate."
        )
    df = df.loc[valid_ec50_mask].copy()
    if df.empty:
        raise ValueError("No passed variants have valid midpoint EC50 values.")

    # ------------------------------------------------------------------
    # 2a. Compute log10(midpoint EC50) and parent aa sequence
    # ------------------------------------------------------------------
    for col in ec50_cols:
        df[f"log10_{col}"] = np.log10(df[col])

    log10_cols = [f"log10_{c}" for c in ec50_cols]
    df["log10_EC50_midpoint_avg"] = df[log10_cols].mean(axis=1)
    df["log10_EC50_midpoint_err"] = df[log10_cols].std(axis=1)

    # Determine parent aa sequence
    parent_aa_seq = None
    fasta_name = expt_id
    if args.r is not None:
        ref_fasta_path = Path(args.r)
        if ref_fasta_path.is_file():
            ref_fasta = SeqIO.read(str(ref_fasta_path), "fasta")
            parent_aa_seq = translate_sequence(ref_fasta.seq)
            fasta_name = ref_fasta_path.stem
            print(f"Reference fasta: {fasta_name}")
        else:
            print(f"[WARN] Reference fasta not found: {ref_fasta_path}")

    # Fall back: identify parent from variant_id == 'parent' or aa_mutations == '[]'
    if parent_aa_seq is None:
        parent_rows = df[
            (df.get("variant_id", pd.Series(dtype=str)) == "parent")
            | (df.get("aa_mutations", pd.Series(dtype=str)).astype(str) == "[]")
        ]
        if not parent_rows.empty:
            parent_aa_seq = parent_rows["aa_sequence"].iloc[0]
            print(f"Parent aa sequence inferred from data ({len(parent_aa_seq)} aa).")
        else:
            print(
                "[WARN] Could not determine parent aa sequence; context analysis will be skipped."
            )

    # ------------------------------------------------------------------
    # 2b. Count aa mutations relative to parent
    # ------------------------------------------------------------------
    if parent_aa_seq is not None:
        df["number_aa_mutations"] = df["aa_sequence"].apply(
            lambda s: hamming_distance(str(s), parent_aa_seq)
        )
    elif "number_aa_mutations" not in df.columns:
        # try to derive from aa_mutations list length
        def _count_muts(m):
            try:
                lst = ast.literal_eval(str(m))
                return len(lst)
            except Exception:
                return np.nan

        df["number_aa_mutations"] = df.get("aa_mutations", pd.Series(dtype=str)).apply(
            _count_muts
        )

    # ------------------------------------------------------------------
    # 3. GMM fit on log10(midpoint EC50) values  (BIC-optimal, max 3 components)
    # ------------------------------------------------------------------
    X_df = df[log10_cols].apply(pd.to_numeric, errors="coerce").dropna()
    df = df.loc[X_df.index].reset_index(drop=True)
    X_df = X_df.reset_index(drop=True)
    X = X_df.values
    mean_log10_ec50 = X_df.mean(axis=1).values

    print(f"Scoring matrix: {X.shape[0]} sequences × {X.shape[1]} replicates")
    for col in X_df.columns:
        print(
            f"  {col}: min={X_df[col].min():.3f}  max={X_df[col].max():.3f}  mean={X_df[col].mean():.3f}"
        )

    # GMM is fit on the 1-D mean log10(midpoint EC50) across replicates so that
    # classification depends purely on the per-sequence average, not on
    # replicate-specific scatter or deviation from the diagonal.
    X_1d = mean_log10_ec50.reshape(-1, 1)

    best_gmm = None
    best_bic = np.inf
    best_n = 1
    for n in range(1, 4):
        g = GaussianMixture(n_components=n, covariance_type="full", random_state=42)
        g.fit(X_1d)
        bic = g.bic(X_1d)
        print(f"  GMM n={n}: BIC={bic:.2f}  converged={g.converged_}")
        if bic < best_bic:
            best_bic = bic
            best_gmm = g
            best_n = n

    print(f"GMM selected n_components={best_n} (lowest BIC={best_bic:.2f})")

    gmm_labels = best_gmm.predict(X_1d)
    component_means = best_gmm.means_.flatten()  # 1-D: one mean per component
    component_weights = best_gmm.weights_
    sorted_idx = np.argsort(component_means)  # low → high log10 midpoint EC50
    for rank, ci in enumerate(sorted_idx.tolist()):
        print(
            f"  component {rank}: mean_log10_midpoint_EC50={component_means[ci]:.3f}"
            f"  weight={component_weights[ci]:.4f}"
            f"  ({component_weights[ci] * len(X_1d):.0f} seqs)"
        )

    # Assign labels directly from the sorted component means — no threshold
    # percentiles needed because we are in 1-D and components are already
    # ordered tight → moderate → weak by their mean log10(midpoint EC50).
    category_order = ["Tight Binder", "Moderate Binder", "Weak Binder"]
    raw_labels = np.empty(len(X_1d), dtype=object)
    for rank, ci in enumerate(sorted_idx.tolist()):
        label = category_order[min(rank, len(category_order) - 1)]
        raw_labels[gmm_labels == ci] = label

    # Fallback: ensure all three categories are populated when n_components < 3
    if "Tight Binder" not in raw_labels:
        print("  [WARN] No Tight Binder component — assigning lower tertile")
        raw_labels[mean_log10_ec50 <= np.percentile(mean_log10_ec50, 33)] = "Tight Binder"
    if "Weak Binder" not in raw_labels:
        print("  [WARN] No Weak Binder component — assigning upper tertile")
        raw_labels[mean_log10_ec50 >= np.percentile(mean_log10_ec50, 67)] = "Weak Binder"
    if "Moderate Binder" not in raw_labels:
        tight_mask = raw_labels == "Tight Binder"
        split_t = np.median(mean_log10_ec50[tight_mask])
        raw_labels[(raw_labels == "Tight Binder") & (mean_log10_ec50 >= split_t)] = (
            "Moderate Binder"
        )

    df["predicted_category"] = raw_labels
    df["color"] = df["predicted_category"].map(CATEGORY_COLORS)

    print("Category assignment:")
    for cat in ["Tight Binder", "Moderate Binder", "Weak Binder"]:
        n_cat = (df["predicted_category"] == cat).sum()
        pct = 100 * n_cat / len(df)
        print(f"  {cat}: {n_cat} sequences ({pct:.1f}%)")

    # Write results CSV
    df.drop("color", axis=1).to_csv(
        out_dir / f"tite_seq_clustered_{expt_id}.csv", index=False
    )

    # ------------------------------------------------------------------
    # 4. Replicate scatter coloured by GMM label
    # ------------------------------------------------------------------
    if len(log10_cols) >= 2:
        corr_mat = X_df.corr(method="pearson")
        mask_upper = np.triu(np.ones_like(corr_mat, dtype=bool))
        corr_masked = corr_mat.mask(mask_upper)
        best_pair = corr_masked.stack().idxmax()
        best_i, best_j = best_pair
        best_r = corr_masked.loc[best_i, best_j]
        print(f"Best replicate pair: ({best_i}, {best_j})  Pearson r={best_r:.3f}")

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(
            df[best_j],
            df[best_i],
            c=df["color"],
            edgecolors="black",
            s=80,
            alpha=0.75,
        )
        # Legend patches
        import matplotlib.patches as mpatches

        legend_patches = [
            mpatches.Patch(color=CATEGORY_COLORS[cat], label=cat)
            for cat in ["Tight Binder", "Moderate Binder", "Weak Binder"]
        ]
        ax.legend(handles=legend_patches, fontsize=12)

        parent_mask = df["predicted_category"].notna()  # highlight if possible
        if "variant_id" in df.columns:
            parent_mask = df["variant_id"].astype(str).isin(["parent"])
        ax.scatter(
            df[best_j][parent_mask],
            df[best_i][parent_mask],
            c=df["color"][parent_mask],
            edgecolors="black",
            s=500,
            linewidth=2,
            zorder=10,
            label="Parent",
        )
        ax.annotate(
            "parent",
            (df[best_j][parent_mask].mean(), df[best_i][parent_mask].mean()),
            textcoords="offset points",
            xytext=(10, -5),
            ha="left",
        )

        min_val = min(df[best_i].min(), df[best_j].min())
        max_val = max(df[best_i].max(), df[best_j].max())
        ax.plot(
            [min_val, max_val], [min_val, max_val], "b--", linewidth=1, label="y=x"
        )  # y=x unchanged: axes swapped so line stays correct

        # Strip the log prefix and midpoint concentration suffix for the legend.
        def _rep_label(col_name: str) -> str:
            return re.sub(
                r"^log10_(.+)_EC50_midpoint_nM$", r"\1", col_name
            )

        ax.set_xlabel(
            f"$\\log_{{10}}(EC_{{50}})$ {_rep_label(best_j)} (nM)", fontsize=14
        )
        ax.set_ylabel(
            f"$\\log_{{10}}(EC_{{50}})$ {_rep_label(best_i)} (nM)", fontsize=14
        )
        ax.set_title(
            "GMM clustering of Tite-Seq EC50 Values by Replicate", fontsize=14
        )
        ax.text(
            0.02,
            0.95,
            f"Pearson r = {best_r:.3f}",
            transform=ax.transAxes,
            fontsize=13,
            verticalalignment="top",
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        scatter_path = str(out_dir_figs / f"gmm_scatter_{expt_id}.png")
        plt.savefig(scatter_path, dpi=400)
        plt.savefig(str(out_dir_figs / f"gmm_scatter_{expt_id}.svg"), dpi=400)
        plt.close()
        print(f"GMM scatter saved at {scatter_path}")

    # ------------------------------------------------------------------
    # 5. Mutation-distance histograms per cluster  (mirroring 14_)
    # ------------------------------------------------------------------
    tight_df = df[df["predicted_category"] == "Tight Binder"].copy()
    moderate_df = df[df["predicted_category"] == "Moderate Binder"].copy()
    weak_df = df[df["predicted_category"] == "Weak Binder"].copy()

    # Compare mutation frequencies across groups
    try:
        mutation_comparison_df = compile_and_compare_mutations(
            tight_df, moderate_df, weak_df, out_dir
        )
    except Exception as e:
        print(f"[WARN] Mutation frequency comparison failed: {e}")
        import traceback

        traceback.print_exc()
        mutation_comparison_df = None

    # Plot mutation frequency comparison
    if mutation_comparison_df is not None and not mutation_comparison_df.empty:
        try:
            plot_outpath = str(
                out_dir_figs / f"mutation_frequency_comparison_{expt_id}.png"
            )
            plot_mutation_frequency_comparison(
                mutation_comparison_df, fasta_name, plot_outpath, top_n=args.p
            )
        except Exception as e:
            print(f"[WARN] Mutation frequency plot failed: {e}")

    # Per-cluster individual histograms
    for cat_df, cat_name, cat_color in [
        (tight_df, "tight", CATEGORY_COLORS["Tight Binder"]),
        (moderate_df, "moderate", CATEGORY_COLORS["Moderate Binder"]),
        (weak_df, "weak", CATEGORY_COLORS["Weak Binder"]),
    ]:
        if cat_df.empty or "number_aa_mutations" not in cat_df.columns:
            continue
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.set_style("ticks")
        sns.histplot(
            data=cat_df,
            x="number_aa_mutations",
            binwidth=1,
            stat="probability",
            edgecolor="black",
            color=cat_color,
            linewidth=0.5,
            ax=ax,
        )
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ticks = ax.get_xticks()
        if len(ticks) > 1:
            new_ticks = ticks[:-1] + 0.5
            ax.set_xticks(new_ticks)
            ax.set_xticklabels([str(int(t)) for t in new_ticks])
            ax.set_xlim([new_ticks[0] + 0.5, new_ticks[-1] - 0.5])
        ax.set_xlabel("aa mutations from parent")
        ax.set_ylabel(f"frequency (n={len(cat_df)} sequences)")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        for sp in ["left", "bottom"]:
            ax.spines[sp].set_linewidth(0.5)
        ax.tick_params(width=0.5)
        plt.tight_layout()
        plt.savefig(
            str(out_dir_figs / f"hamming_vs_parent_{cat_name}_{expt_id}.png"), dpi=400
        )
        plt.close()

    # Combined overlaid histogram with KDE (mirroring the combined plot in 14_)
    num_sequences = len(df)
    tight_vals = tight_df["number_aa_mutations"].dropna().to_numpy()
    moderate_vals = moderate_df["number_aa_mutations"].dropna().to_numpy()
    weak_vals = weak_df["number_aa_mutations"].dropna().to_numpy()

    tight_mean, tight_med = (
        (np.mean(tight_vals), np.median(tight_vals))
        if len(tight_vals)
        else (np.nan, np.nan)
    )
    moderate_mean, moderate_med = (
        (np.mean(moderate_vals), np.median(moderate_vals))
        if len(moderate_vals)
        else (np.nan, np.nan)
    )
    weak_mean, weak_med = (
        (np.mean(weak_vals), np.median(weak_vals))
        if len(weak_vals)
        else (np.nan, np.nan)
    )

    combined_vals = [v for v in [tight_vals, moderate_vals, weak_vals] if v.size]
    xmin_hist = 0
    xmax_hist = int(np.max([v.max() for v in combined_vals])) if combined_vals else 10

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.set_style("ticks")

    for cat_df, cat_name, cat_color, cat_mean, cat_med in [
        (
            tight_df,
            "Tight Binder",
            CATEGORY_COLORS["Tight Binder"],
            tight_mean,
            tight_med,
        ),
        (
            moderate_df,
            "Moderate Binder",
            CATEGORY_COLORS["Moderate Binder"],
            moderate_mean,
            moderate_med,
        ),
        (weak_df, "Weak Binder", CATEGORY_COLORS["Weak Binder"], weak_mean, weak_med),
    ]:
        if cat_df.empty:
            continue
        sns.histplot(
            data=cat_df,
            x="number_aa_mutations",
            binwidth=1,
            stat="density",
            edgecolor="black",
            color=cat_color,
            linewidth=0.5,
            alpha=0.3,
            label=f"{cat_name} (n={len(cat_df)} [{cat_mean:.1f}|{cat_med:.1f}])",
            ax=ax,
        )
        sns.kdeplot(
            data=cat_df,
            x="number_aa_mutations",
            color=cat_color,
            lw=1.5,
            fill=False,
            cut=0,
            clip=(xmin_hist, xmax_hist),
            bw_adjust=1.5,
            ax=ax,
            zorder=5,
        )

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ticks = ax.get_xticks()
    if len(ticks) > 1:
        new_ticks = ticks[:-1] + 0.5
        ax.set_xticks(new_ticks)
        ax.set_xticklabels([str(int(t)) for t in new_ticks])
        ax.set_xlim([new_ticks[0] + 0.5, new_ticks[-1] - 0.5])
    ax.set_xlabel("amino acid mutations from parent")
    ax.set_ylabel(f"density (n={num_sequences} sequences)")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_linewidth(0.5)
    ax.tick_params(width=0.5)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(str(out_dir_figs / f"hamming_all_clusters_{expt_id}.png"), dpi=400)
    plt.savefig(str(out_dir_figs / f"hamming_all_clusters_{expt_id}.svg"), dpi=400)
    plt.close()
    print("Combined mutation histogram saved.")

    # Write results summary
    total = len(df)
    with open(str(out_dir / "results_summary.txt"), "w") as fh:
        for cat in ["Tight Binder", "Moderate Binder", "Weak Binder"]:
            n_cat = (df["predicted_category"] == cat).sum()
            fh.write(f"{cat}: {n_cat}/{total} ({100 * n_cat / total:.2f}%)\n")

    # ------------------------------------------------------------------
    # 6. Average log10(midpoint EC50) per sequence sorted by mutation distance.
    # ------------------------------------------------------------------
    try:
        library_org_df = df.sort_values(
            [
                "number_aa_mutations",
                "log10_EC50_midpoint_avg",
                "log10_EC50_midpoint_err",
            ]
        ).reset_index(drop=True)
        org_outpath = str(
            out_dir_figs / f"{expt_id}_seqs_avglog10_midpoint_EC50.png"
        )
        high_var = plot_seqavgs(library_org_df, fasta_name, org_outpath)
        if len(high_var) == 0:
            print("No sequences with high variance.")
        else:
            print(f"{len(high_var)} sequences with high midpoint EC50 variance.")
    except Exception as e:
        print(f"[WARN] plot_seqavgs failed: {e}")
        library_org_df = df.sort_values("number_aa_mutations").reset_index(drop=True)

    # ------------------------------------------------------------------
    # 7. Context / point-mutation analysis  (ported from 19_neutvariants.py)
    # ------------------------------------------------------------------
    if parent_aa_seq is None:
        print("[WARN] Skipping context analysis: parent aa sequence not determined.")
        return

    mutant_lists = args.m
    min_contexts = args.x
    min_deltascore = args.d

    lib_aaseq = library_org_df["aa_sequence"]  # already sorted by number_aa_mutations

    short_path_highscore_dict = {}

    for mutnum in range(1, mutant_lists + 1):
        try:
            print(f"Calculating groups of mutants {mutnum} aa apart …")
            muts_dict: dict = {}
            for i in range(len(lib_aaseq)):
                aaseq = lib_aaseq.iloc[i]
                mutlist = []
                for j in range(i + 1, len(lib_aaseq)):
                    seqcomp = lib_aaseq.iloc[j]
                    if seqcomp != aaseq and hamming_distance(aaseq, seqcomp) == mutnum:
                        mutlist.append(seqcomp)
                if mutlist:
                    muts_dict[aaseq] = mutlist
            print(f"  {len(muts_dict)} groups of mutants {mutnum} aa apart")

            # Build comprehensive pair table
            mutrec = []
            for sequence in muts_dict:
                seqdist = hamming_distance(sequence, parent_aa_seq)
                for variant in muts_dict[sequence]:
                    vardist = hamming_distance(variant, parent_aa_seq)
                    seqslice = library_org_df[library_org_df["aa_sequence"] == sequence]
                    varslice = library_org_df[library_org_df["aa_sequence"] == variant]
                    if seqslice.empty or varslice.empty:
                        continue
                    ec50_1 = seqslice["log10_EC50_midpoint_avg"].item()
                    ec50_2 = varslice["log10_EC50_midpoint_avg"].item()
                    if seqdist <= vardist:
                        dist = seqdist
                        delta_ec50 = ec50_2 - ec50_1
                        pointmut = find_mutations(sequence, variant)
                        wt_muts = find_mutations(parent_aa_seq, sequence)
                    else:
                        dist = vardist
                        delta_ec50 = ec50_1 - ec50_2
                        pointmut = find_mutations(variant, sequence)
                        wt_muts = find_mutations(parent_aa_seq, variant)
                    mutrec.append(
                        {
                            "sequence1": sequence,
                            "sequence2": variant,
                            "point_mutation": pointmut,
                            "wild_type_distance": dist,
                            "wild_type_mutations": wt_muts,
                            "position": int(
                                "".join(c for c in pointmut[0] if c.isdigit())
                            )
                            if pointmut
                            else -1,
                            "log10EC50_midpoint1": ec50_1,
                            "log10EC50_midpoint2": ec50_2,
                            "delta_log10EC50_midpoint": delta_ec50,
                        }
                    )

            mutrec_df = pd.DataFrame(mutrec)
            mutrec_sort_df = mutrec_df.sort_values(["wild_type_distance", "position"])

            contexts_dict: dict = {}

            # Flatten to single point mutations
            mutrec_sort_df["point_mutation"] = (
                mutrec_sort_df["point_mutation"].astype(str).apply(ast.literal_eval)
            )
            allpointmuts = mutrec_sort_df["point_mutation"]
            allpmuts_flat = [pmut for sublist in allpointmuts for pmut in sublist]

            for muti in set(allpmuts_flat):
                pmutslice = mutrec_sort_df[
                    mutrec_sort_df["point_mutation"].apply(lambda x: muti in x)
                ]
                pmutslice = pmutslice.sort_values(
                    [
                        "wild_type_distance",
                        "log10EC50_midpoint1",
                        "log10EC50_midpoint2",
                    ]
                )
                pmutslice = pmutslice.dropna(
                    subset=["delta_log10EC50_midpoint"]
                ).copy()

                if mutnum == 1:
                    if len(pmutslice) >= min_contexts:
                        position = int(muti[1:-1]) - 1
                        if muti[0] == parent_aa_seq[position]:
                            muti_outdir = out_dir_pointmuts / muti
                        else:
                            muti_outdir = out_dir_pointmuts / f"{muti}_nonparent"
                        make_dir(str(muti_outdir))

                        mutidist_outpath = str(muti_outdir / f"scoreplots_{muti}.png")
                        colors = sns.color_palette("husl", len(pmutslice))
                        plot_mutation_contexts(
                            pmutslice, colors, muti, mutidist_outpath
                        )
                        pmutslice.to_csv(
                            str(muti_outdir / f"{muti}_{mutnum}aa_table.csv")
                        )

                        # Collect WT-side context sequences
                        contexts = []
                        for seq1, seq2 in zip(
                            pmutslice["sequence1"], pmutslice["sequence2"]
                        ):
                            d1 = hamming_distance(seq1, parent_aa_seq)
                            d2 = hamming_distance(seq2, parent_aa_seq)
                            contexts.append(seq1 if d1 < d2 else seq2)
                        contexts_dict[muti] = contexts
                else:
                    sig_pmut = pmutslice[
                        pmutslice["delta_log10EC50_midpoint"].abs()
                        > min_deltascore
                    ]
                    if len(sig_pmut) > 0:
                        position = int(muti[1:-1]) - 1
                        if muti[0] == parent_aa_seq[position]:
                            muti_outdir = out_dir_pointmuts / muti
                        else:
                            muti_outdir = out_dir_pointmuts / f"{muti}_nonparent"
                        make_dir(str(muti_outdir))
                        sig_pmut.to_csv(
                            str(muti_outdir / f"{muti}_{mutnum}aa_table.csv")
                        )
                        short_path_highscore_dict[muti] = sig_pmut

            # Save multi-aa significant-delta table
            if mutnum > 1 and short_path_highscore_dict:
                all_sig_df = pd.concat(
                    [d.assign(key=k) for k, d in short_path_highscore_dict.items()],
                    ignore_index=True,
                )
                all_sig_df.drop_duplicates(
                    subset=["sequence1", "sequence2"], keep="first"
                ).rename(columns={"key": "mutation_of_interest"}).sort_values(
                    "delta_log10EC50_midpoint"
                ).to_csv(
                    str(
                        out_dir_pointmuts
                        / f"{expt_id}_all{mutnum}aa_deldlog10_midpoint_EC50{min_deltascore}.csv"
                    )
                )

        except Exception as e:
            print(f"[ERROR] mutnum={mutnum}: {e}")
            import traceback

            traceback.print_exc()

        # ------------------------------------------------------------------
        # Context distances + ridge plot  (only for single-mutant families)
        # ------------------------------------------------------------------
        if mutnum == 1:
            # Pairwise distances between context sequences
            hamming_dist_contexts: dict = {}
            for mutj in contexts_dict:
                hamming_dists = []
                seqs = contexts_dict[mutj]
                for p in range(len(seqs)):
                    for q in range(p + 1, len(seqs)):
                        hamming_dists.append(hamming_distance(seqs[p], seqs[q]))
                hamming_dist_contexts[mutj] = hamming_dists

            for mutk in hamming_dist_contexts:
                position = int(mutk[1:-1]) - 1
                if mutk[0] == parent_aa_seq[position]:
                    mutk_outdir = out_dir_pointmuts / mutk
                else:
                    mutk_outdir = out_dir_pointmuts / f"{mutk}_nonparent"
                make_dir(str(mutk_outdir))
                mutkdist_outpath = str(mutk_outdir / f"contextdist_{mutk}.png")
                plot_context_distances(
                    hamming_dist_contexts, mutk, contexts_dict, mutkdist_outpath
                )

            # Accumulate delta-log10(midpoint EC50) per point mutation.
            mut_dict: dict = {}
            noref_dict: dict = {}
            for mutation in mutrec_sort_df["point_mutation"]:
                mutation_string = mutation[0]
                position = int(mutation_string[1:-1]) - 1
                mask = (
                    mutrec_sort_df["point_mutation"].apply(
                        lambda x: x[0] if isinstance(x, list) else x
                    )
                    == mutation_string
                )
                if mutation_string[0] == parent_aa_seq[position]:
                    mut_dict[mutation_string] = list(
                        mutrec_sort_df[mask]["delta_log10EC50_midpoint"].values
                    )
                else:
                    noref_dict[mutation_string] = list(
                        mutrec_sort_df[mask]["delta_log10EC50_midpoint"].values
                    )

            sorted_mut_dict = {k: sorted(v, reverse=True) for k, v in mut_dict.items()}

            # Ridge plot
            ridgedata = []
            for key, values in sorted_mut_dict.items():
                if len(values) >= min_contexts:
                    for v in values:
                        ridgedata.append(
                            {"mutation": key, "dlog10EC50_midpoint": v}
                        )
            ridgedata_df = pd.DataFrame(ridgedata)
            if not ridgedata_df.empty:
                ridge_outpath = str(
                    out_dir_figs
                    / f"pointmut_ridges_{min_contexts}contexts_{expt_id}.png"
                )
                plot_context_ridges(
                    sorted_mut_dict, ridgedata_df, fasta_name, ridge_outpath
                )

            # ------------------------------------------------------------------
            # DMS-like heatmap of single-mutation Δlog10(midpoint EC50).
            # ------------------------------------------------------------------
            amino_acids_dict = {
                "A": "Alanine",
                "C": "Cysteine",
                "D": "Aspartic acid",
                "E": "Glutamic acid",
                "F": "Phenylalanine",
                "G": "Glycine",
                "H": "Histidine",
                "I": "Isoleucine",
                "K": "Lysine",
                "L": "Leucine",
                "M": "Methionine",
                "N": "Asparagine",
                "P": "Proline",
                "Q": "Glutamine",
                "R": "Arginine",
                "S": "Serine",
                "T": "Threonine",
                "V": "Valine",
                "W": "Tryptophan",
                "Y": "Tyrosine",
            }

            # Parent log10(midpoint EC50) — use average over all parent rows.
            parent_rows = library_org_df[
                library_org_df["aa_sequence"].astype(str) == parent_aa_seq
            ]
            if parent_rows.empty:
                print(
                    "[WARN] Parent sequence not in library; DMS heatmap will use 0 as reference."
                )
                parent_log10_ec50 = 0.0
            else:
                parent_log10_ec50 = parent_rows["log10_EC50_midpoint_avg"].mean()
            print(f"Parent log10(midpoint EC50): {parent_log10_ec50:.3f}")

            mutscore_dict: dict = {}
            for pos, res in enumerate(parent_aa_seq):
                for mut in amino_acids_dict:
                    singlemut = (
                        parent_aa_seq[:pos] + mut + parent_aa_seq[(pos + 1) :]
                        if len(parent_aa_seq) > pos
                        else parent_aa_seq
                    )
                    if singlemut == parent_aa_seq:
                        continue
                    mut_seq_df = library_org_df[
                        library_org_df["aa_sequence"] == singlemut
                    ]
                    if mut_seq_df.empty:
                        continue
                    smutscore = mut_seq_df["log10_EC50_midpoint_avg"].item()
                    mutation_key = f"{res}{pos + 1}{mut}"
                    mutscore_dict[mutation_key] = smutscore - parent_log10_ec50

            positions = list(range(1, len(parent_aa_seq) + 1))
            wtresidues = list(parent_aa_seq)
            wt_df = pd.DataFrame({"position": positions, "wtres": wtresidues})

            rec = []
            for mut, score in mutscore_dict.items():
                wt = mut[0]
                pos = int(mut[1:-1])
                mut_aa = mut[-1]
                rec.append(
                    {"position": pos, "wtres": wt, "mutres": mut_aa, "score": score}
                )

            if not rec:
                print(
                    "[WARN] No single mutants found in library — skipping DMS heatmap."
                )
                print("------next amino acid distance------")
                continue

            mut_df = pd.DataFrame(rec)

            rec_df = pd.merge(
                wt_df, mut_df, on=["position", "wtres"], how="left"
            ).fillna("")
            positions_sorted = sorted(rec_df["position"].unique())
            wt_seq_map = (
                rec_df.drop_duplicates("position")
                .set_index("position")["wtres"]
                .to_dict()
            )
            wt_pos_labels = [f"{p}{wt_seq_map.get(p, '')}" for p in positions_sorted]
            all_aas = list(amino_acids_dict.keys())
            pos2col = dict(zip(positions_sorted, wt_pos_labels))

            heatmap_matrix = pd.DataFrame(
                index=all_aas, columns=wt_pos_labels, dtype=float
            )
            mut_df_valid = rec_df[
                rec_df["mutres"].notna()
                & (rec_df["mutres"] != "")
                & rec_df["score"].notna()
                & (rec_df["score"] != "")
            ]
            for _, row in mut_df_valid.iterrows():
                pos = int(row["position"])
                if pos not in pos2col:
                    continue
                col = pos2col[pos]
                score = (
                    row["score"].item()
                    if hasattr(row["score"], "item")
                    else row["score"]
                )
                heatmap_matrix.at[row["mutres"], col] = score

            heatmap_outpath = str(out_dir_figs / f"1mut_heatmap_{expt_id}.png")
            plot_dms(heatmap_matrix, fasta_name, heatmap_outpath)

        print("------next amino acid distance------")

    print(f"\nDone. Results saved to {out_dir}")


if __name__ == "__main__":
    main()
