#!/usr/bin/env python
# coding: utf-8

"""
Force-directed graph layout visualization for protein variant fitness landscape.

This script generates force-directed layout plots for protein sequence variants,
with binding affinity (Kd) values as coloring. Variants are connected in the graph
if they differ by a single amino acid mutation.

Expected input: CSV file with columns including:
  - variant_id: List of amino acid mutations (e.g., ['A23D', 'A50V', ...])
  - aa_sequence: Full amino acid sequence
  - log10_Kd_avg or (log10_rep1_Kd and log10_rep2_Kd): Log-transformed Kd values
  - passed: Boolean indicating if variant passed QC

Author: Yutong Lexie Yu
"""

import argparse
import logging
import os
import random
import re
import warnings
from pathlib import Path

import igraph as ig
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Set matplotlib parameters
mpl.rcParams["figure.dpi"] = 200


# Note: reachable() function kept for potential future use with binary sequences
# Currently not used with protein sequence data


def load_and_prepare_data(data_path):
    """
    Load variant data and prepare for analysis.

    Args:
        data_path: Path to CSV file with variant data

    Returns:
        pandas.DataFrame with processed variant data
    """
    logger.info(f"Loading data from {data_path}")

    # Read CSV, handling potential quoted columns
    df = pd.read_csv(data_path, dtype={"variant_id": "str"})

    # Create a unique integer ID for each variant (sequence)
    df["variant_int"] = range(len(df))

    logger.info(f"Loaded {len(df)} total sequences")
    return df


def calculate_kd_values(df):
    """
    Calculate/prepare log Kd values from data.

    Args:
        df: pandas.DataFrame with Kd measurements

    Returns:
        pandas.DataFrame with calculated Kd values
    """
    logger.info("Processing Kd values")

    # Check which Kd columns are available
    kd_cols = [
        col
        for col in df.columns
        if col in ["log10_Kd_avg", "log10_rep1_Kd", "log10_rep2_Kd"]
    ]

    if "log10_Kd_avg" in df.columns:
        # Use pre-calculated average
        df["log10Kd"] = df["log10_Kd_avg"]
        logger.info("Using 'log10_Kd_avg' column")
    elif "log10_rep1_Kd" in df.columns and "log10_rep2_Kd" in df.columns:
        # Calculate mean from replicates
        df["log10Kd"] = df[["log10_rep1_Kd", "log10_rep2_Kd"]].mean(
            axis=1, skipna=False
        )
        logger.info("Calculated mean from 'log10_rep1_Kd' and 'log10_rep2_Kd'")
    else:
        raise ValueError(
            f"Could not find expected Kd columns. Available columns: {df.columns.tolist()}"
        )

    # Only keep passed variants for FDL graph
    if "passed" in df.columns:
        df = df[df["passed"] == True].copy()
        logger.info(f"Filtered to {len(df)} passed variants")

    return df


def check_data_completeness(df_original, df_processed):
    """
    Print data quality and completeness statistics.

    Args:
        df_original: pandas.DataFrame with all loaded data
        df_processed: pandas.DataFrame after filtering
    """
    logger.info("Data completeness check:")

    total_rows = len(df_original)
    passed_rows = len(df_processed)

    print(f"\nTotal variants in file:  {total_rows:>6}")
    print(f"Passed variants (used):  {passed_rows:>6}")
    print(f"Filtered out:            {total_rows - passed_rows:>6}")

    # Check for NaN values in key column
    n_nan_kd = df_processed["log10Kd"].isna().sum()
    if n_nan_kd > 0:
        print(f"\n  Variants with NaN Kd:    {n_nan_kd:>6}  ← excluded from FDL graph")
        print(f"  Variants with valid Kd: {len(df_processed) - n_nan_kd:>6}")
    else:
        print(f"\n  All variants have valid Kd values")


def build_fitness_graph(df):
    """
    Build igraph Graph object for force-directed layout.

    For protein sequences, edges connect variants that differ by a single amino acid mutation.

    Args:
        df: pandas.DataFrame with variant data and fitness values

    Returns:
        tuple: (igraph.Graph, list of vertex integers, dict of fitnesses)
    """
    logger.info("Building fitness graph")

    # Extract fitness values
    fitnesses = df.set_index("variant_int")["log10Kd"].to_dict()

    # All variants are vertices (already filtered for passed status)
    vertices = list(df["variant_int"].values)

    logger.info(f"Graph vertices: {len(vertices)}")

    # Create graph
    g = ig.Graph()
    g.add_vertices([str(s) for s in vertices])

    # For protein sequences, connect variants that differ by a single mutation
    # We'll build edges based on Hamming distance (number of amino acid differences)
    edges = []
    sequences = {
        v: df[df["variant_int"] == v]["aa_sequence"].values[0] for v in vertices
    }

    for i, v1 in enumerate(vertices):
        for v2 in vertices[i + 1 :]:
            seq1 = sequences[v1]
            seq2 = sequences[v2]

            # Count amino acid differences
            if len(seq1) == len(seq2):
                diff_count = sum(1 for a, b in zip(seq1, seq2) if a != b)
                if diff_count == 1:  # Single amino acid mutation
                    edges.append((str(v1), str(v2)))

    if edges:
        g.add_edges(edges)
        logger.info(f"Graph edges: {len(edges)} (connecting single-mutation neighbors)")
    else:
        logger.warning("No single-mutation neighbor edges found")

    # Weight edges by inverse Kd difference (only if edges exist)
    if edges:
        g.es["weight"] = [
            1 / (0.01 + np.abs(fitnesses[int(t)] - fitnesses[int(s)])) for s, t in edges
        ]

    return g, vertices, fitnesses


def compute_force_directed_layout(g, vertices, seed_value=21191):
    """
    Compute force-directed layout coordinates using DrL algorithm.

    Args:
        g: igraph.Graph object
        vertices: List of vertex integers
        seed_value: Random seed for reproducibility

    Returns:
        tuple: (x_coords dict, y_coords dict)
    """
    logger.info(f"Computing force-directed layout (seed={seed_value})")

    # Check if graph has edges
    if g.ecount() == 0:
        logger.warning("Graph has no edges - using random layout")
        # Fall back to random layout
        random.seed(seed_value)
        xcoord = {v: random.uniform(-100, 100) for v in vertices}
        ycoord = {v: random.uniform(-100, 100) for v in vertices}
        return xcoord, ycoord

    # Set random seed
    random.seed(seed_value)

    # Generate random initial seed positions
    seed = np.empty((len(vertices), 2))
    for i in range(len(vertices)):
        xcoord_init = random.uniform(-100, 100)
        ycoord_init = random.uniform(-100, 100)
        seed[i, :] = [xcoord_init, ycoord_init]

    # Compute layout using DrL algorithm with edge weights
    try:
        layout = g.layout_drl(weights="weight", seed=seed)
    except Exception as e:
        logger.warning(f"DrL layout failed ({e}), using Fruchterman-Reingold instead")
        layout = g.layout_fruchterman_reingold(weights="weight", seed=seed)

    # Extract coordinates
    coord = np.array(layout.coords)

    xcoord = {vertices[i]: -1.0 * coord[i, 1] for i in range(len(vertices))}
    ycoord = {vertices[i]: coord[i, 0] for i in range(len(vertices))}

    return xcoord, ycoord


def add_layout_to_dataframe(df, xcoord, ycoord):
    """
    Add FDL coordinates to dataframe.

    Args:
        df: pandas.DataFrame with variant data
        xcoord: Dictionary mapping variant_int to x coordinate
        ycoord: Dictionary mapping variant_int to y coordinate

    Returns:
        pandas.DataFrame with fdl_x and fdl_y columns added
    """
    df["fdl_x"] = df.variant_int.map(xcoord)
    df["fdl_y"] = df.variant_int.map(ycoord)
    return df


def get_connected_vertex_ints(g, vertices):
    """
    Return the set of vertex integers that have at least one edge (degree > 0).

    Args:
        g: igraph.Graph object
        vertices: List of vertex integers (same order as igraph vertex indices)

    Returns:
        Set of vertex integers belonging to the connected subgraph
    """
    return {vertices[i] for i, deg in enumerate(g.degree()) if deg > 0}


def compute_edge_segments(g, vertices, xcoord, ycoord):
    """
    Compute line segment coordinates for all graph edges.

    Args:
        g: igraph.Graph object
        vertices: List of vertex integers (same order as igraph vertex indices)
        xcoord: Dict mapping vertex integer to x coordinate
        ycoord: Dict mapping vertex integer to y coordinate

    Returns:
        List of [(x1, y1), (x2, y2)] pairs for use with LineCollection
    """
    segments = []
    for i, j in g.get_edgelist():
        v1, v2 = vertices[i], vertices[j]
        segments.append([(xcoord[v1], ycoord[v1]), (xcoord[v2], ycoord[v2])])
    return segments


def plot_all_variants(df, out_dir, expt_id="", edge_segments=None, n_total=None):
    """
    Plot connected variants in force-directed layout space coloured by log10(Kd),
    with edges drawn between single-mutation neighbours. Only sequences with at
    least one edge are plotted.

    Args:
        df: pandas.DataFrame of connected variants with FDL coordinates
        out_dir: Output directory for saving plots
        expt_id: Experiment identifier used in the plot title
        edge_segments: List of [(x1,y1),(x2,y2)] pairs from compute_edge_segments()
        n_total: Total variant count before network filtering (for title display)
    """
    logger.info("Plotting all variants")

    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw edges first so nodes render on top
    if edge_segments:
        lc = LineCollection(
            edge_segments, linewidths=0.5, colors="gray", alpha=0.3, zorder=1
        )
        ax.add_collection(lc)

    # Plot nodes coloured by Kd
    sc = ax.scatter(
        x=df.fdl_x,
        y=df.fdl_y,
        s=5,
        c=df.log10Kd,
        alpha=0.7,
        cmap="viridis",
        edgecolors="face",
        zorder=2,
    )
    plt.colorbar(sc, fraction=0.02, label="log10(Kd) [nM]")

    n_in_network = len(df)
    n_total_label = n_total if n_total is not None else n_in_network
    ax.set_title(
        f"Force-directed layout\n"
        f"(edges: 1 aa mutation apart, coloured by log10(Kd) [nM])\n"
        f"{n_in_network}/{n_total_label} sequences in network  ({expt_id})",
        fontsize=14,
    )
    ax.set_aspect("equal")
    ax.set_axis_off()
    plt.tight_layout()

    out_path = os.path.join(out_dir, "Force_directed_layout_all_variants.png")
    plt.savefig(out_path, dpi=400, bbox_inches="tight")
    logger.info(f"Saved: {out_path}")
    plt.close()


def plot_variants_by_mutation_count(
    df, out_dir, expt_id="", edge_segments=None, n_total=None
):
    """
    Plot connected variants in FDL space coloured by number of amino acid mutations
    from parent, with edges drawn between single-mutation neighbours. Only sequences
    with at least one edge are plotted.

    Args:
        df: pandas.DataFrame of connected variants with FDL coordinates
        out_dir: Output directory for saving plots
        expt_id: Experiment identifier used in the plot title
        edge_segments: List of [(x1,y1),(x2,y2)] pairs from compute_edge_segments()
        n_total: Total variant count before network filtering (for title display)
    """
    logger.info("Plotting variants by mutation count")

    if "number_aa_mutations" in df.columns:
        df["n_mutations"] = df["number_aa_mutations"].astype(int)
    elif "variant_id" in df.columns:
        df["n_mutations"] = df["variant_id"].apply(
            lambda x: len(eval(x)) if isinstance(x, str) and x.startswith("[") else 0
        )
    else:
        logger.warning("Could not determine mutation counts, skipping this plot")
        return

    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw edges first so nodes render on top
    if edge_segments:
        lc = LineCollection(
            edge_segments, linewidths=0.5, colors="gray", alpha=0.3, zorder=1
        )
        ax.add_collection(lc)

    scatter = ax.scatter(
        x=df.fdl_x,
        y=df.fdl_y,
        s=20,
        c=df.n_mutations,
        cmap="cool",
        alpha=0.6,
        edgecolors="black",
        linewidth=0.5,
        zorder=2,
    )
    plt.colorbar(scatter, fraction=0.02, label="aa mutations from parent")

    n_in_network = len(df)
    n_total_label = n_total if n_total is not None else n_in_network
    ax.set_title(
        f"Force-directed layout — aa mutations from parent\n"
        f"{n_in_network}/{n_total_label} sequences in network  ({expt_id})",
        fontsize=14,
    )
    ax.set_aspect("equal")
    ax.set_axis_off()
    plt.tight_layout()

    out_path = os.path.join(out_dir, "force_directed_layout_by_mutation_count.png")
    plt.savefig(out_path, dpi=400, bbox_inches="tight")
    logger.info(f"Saved: {out_path}")
    plt.close()


def plot_fitness_by_mutation_count(df, out_dir, expt_id=""):
    """
    Boxplot of log10(Kd) as a function of number of amino acid mutations from parent.

    Args:
        df: pandas.DataFrame with variant data
        out_dir: Output directory for saving plots
        expt_id: Experiment identifier used in the plot title
    """
    logger.info("Plotting fitness by mutation count")

    if "number_aa_mutations" in df.columns:
        df_plot = df.copy()
        df_plot["number_aa_mutations"] = df_plot["number_aa_mutations"].astype(int)
        mut_col = "number_aa_mutations"
    elif "variant_id" in df.columns:
        df_plot = df.copy()
        try:
            df_plot["n_muts"] = df_plot["variant_id"].apply(
                lambda x: (
                    len(eval(x)) if isinstance(x, str) and x.startswith("[") else 0
                )
            )
            mut_col = "n_muts"
        except Exception:
            logger.warning("Could not parse mutation counts from variant_id")
            return
    else:
        logger.warning("No mutation count information available")
        return

    sns.set_style("ticks")
    fig, ax = plt.subplots(figsize=(10, 5))
    mut_order = sorted(df_plot[mut_col].unique())
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Using categorical units")
        sns.boxplot(
            data=df_plot,
            x=df_plot[mut_col].astype(str),
            y="log10Kd",
            order=[str(v) for v in mut_order],
            color="steelblue",
            fliersize=2,
            linewidth=0.8,
            ax=ax,
        )
    ax.set_xlabel("amino acid mutations from parent")
    ax.set_ylabel("log10(Kd) [nM]")
    ax.set_title(f"log10(Kd) vs. amino acid mutations from parent  ({expt_id})")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.tick_params(width=0.5)
    plt.tight_layout()

    out_path = os.path.join(out_dir, "fitness_by_mutation_count.png")
    plt.savefig(out_path, dpi=400, bbox_inches="tight")
    logger.info(f"Saved: {out_path}")
    plt.close()

    # Print summary statistics
    print("\nFitness by mutation count:")
    summary = (
        df_plot.groupby(mut_col)["log10Kd"]
        .agg(["count", "mean", "median", "min", "max", "std"])
        .round(2)
    )
    print(summary)


def save_output_data(df, out_dir):
    """
    Save processed dataframe to CSV.

    Args:
        df: pandas.DataFrame with variant data
        out_dir: Output directory for saving files
    """
    out_path = os.path.join(
        out_dir, "Force_directed_layout_all_variants_vhh_mature_flua.csv"
    )
    df.to_csv(out_path, index=False)
    logger.info(f"Saved: {out_path}")


def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace with parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Generate force-directed layout visualizations for protein variant fitness landscapes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 19_force_directed_layout.py \\
    --data-path tite_seq_clustered_tite_seq_Kds_passed_variants.csv \\
    --output-dir plots/

  python 19_force_directed_layout.py \\
    --data-path data.csv \\
    --output-dir results/ \\
    --seed 42 \\
    --verbose
        """,
    )

    parser.add_argument(
        "--data-path",
        required=True,
        help="Path to CSV file containing variant data with Kd measurements",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for saving plots and processed data",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=21191,
        help="Random seed for reproducible layout. Default: 21191",
    )

    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging output"
    )

    return parser.parse_args()


def main():
    """
    Main execution function.
    """
    args = parse_arguments()

    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Suppress noisy matplotlib/seaborn categorical axis warnings
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", message="Using categorical units")

    # Validate input file exists
    if not os.path.exists(args.data_path):
        logger.error(f"Data file not found: {args.data_path}")
        return 1

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Output directory: {args.output_dir}")

    try:
        # Load and prepare data
        df_original = load_and_prepare_data(args.data_path)

        # Calculate Kd values
        df_processed = calculate_kd_values(df_original)

        # Check data completeness
        check_data_completeness(df_original, df_processed)

        # Derive experiment ID from input filename (e.g. jh_002)
        match = re.search(r"(jh_\d+)", args.data_path)
        expt_id = match.group(1) if match else Path(args.data_path).stem

        # Build graph and compute layout
        g, vertices, fitnesses = build_fitness_graph(df_processed)
        xcoord, ycoord = compute_force_directed_layout(
            g, vertices, seed_value=args.seed
        )

        # Add coordinates to dataframe
        df_processed = add_layout_to_dataframe(df_processed, xcoord, ycoord)

        # Pre-compute edge segments for both FDL scatter plots
        edge_segments = compute_edge_segments(g, vertices, xcoord, ycoord)
        logger.info(f"Edge segments computed: {len(edge_segments)}")

        # Filter to connected vertices only for FDL scatter plots
        connected_ints = get_connected_vertex_ints(g, vertices)
        df_network = df_processed[
            df_processed["variant_int"].isin(connected_ints)
        ].copy()
        n_total = len(df_processed)
        logger.info(
            f"Network filter: {len(df_network)}/{n_total} sequences have at least one edge"
        )

        # Generate visualizations
        plot_all_variants(
            df_network,
            args.output_dir,
            expt_id=expt_id,
            edge_segments=edge_segments,
            n_total=n_total,
        )
        plot_variants_by_mutation_count(
            df_network,
            args.output_dir,
            expt_id=expt_id,
            edge_segments=edge_segments,
            n_total=n_total,
        )
        plot_fitness_by_mutation_count(df_processed, args.output_dir, expt_id=expt_id)

        # Save output data
        save_output_data(df_processed, args.output_dir)

        logger.info("Pipeline completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"Error during execution: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
