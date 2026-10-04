#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on November 10 16:04 2025
@author: @jhambale
example run:
    python 15_network_nxrep.py \
    -i ../../titeseq_data/outputs/jh_002/tite_seq_Kds2/cluster_eval_tite_seq_Kds_passed_variants/tite_seq_clustered_tite_seq_Kds_passed_variants.csv

Expects titeseq clustered variant CSV with columns:
    aa_sequence, log10_rep1_Kd, log10_rep2_Kd, number_aa_mutations, predicted_category
"""

import argparse
import json
import os
import re
from typing import Dict, List, Tuple

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from utils_nd import hamming_distance


def build_sequence_network(
    sequences: List[str], scores: List[float]
) -> Tuple[nx.Graph, Dict]:
    """
    Build a network where nodes are sequences and edges connect sequences
    that differ by exactly 1 amino acid.

    Args:
        sequences: List of amino acid sequences
        scores: List of function scores corresponding to each sequence

    Returns:
        G: NetworkX graph
        node_data: Dictionary mapping sequences to their scores
    """
    # Create graph
    G = nx.Graph()

    # Create node data dictionary
    node_data = {seq: score for seq, score in zip(sequences, scores)}

    # Add nodes with their scores as attributes
    for seq, score in node_data.items():
        G.add_node(seq, score=score)

    # Add edges between sequences that differ by exactly 1 amino acid
    for i, seq1 in enumerate(sequences):
        for j, seq2 in enumerate(sequences[i + 1 :], start=i + 1):
            if hamming_distance(seq1, seq2) == 1:
                G.add_edge(seq1, seq2)

    return G, node_data


def build_sequence_network_with_frequency(
    sequences: List[str], scores: List[float], frequencies: List[int]
) -> Tuple[nx.Graph, Dict, Dict]:
    """
    Build a network with sequence, score, and frequency data.

    Args:
        sequences: List of amino acid sequences (unique)
        scores: List of function scores corresponding to each sequence
        frequencies: List of observation counts for each sequence

    Returns:
        G: NetworkX graph
        node_data: Dictionary mapping sequences to their scores
        frequency_data: Dictionary mapping sequences to their frequencies
    """
    # Create dictionaries
    node_data = {seq: score for seq, score in zip(sequences, scores)}
    frequency_data = {seq: freq for seq, freq in zip(sequences, frequencies)}

    # Create graph
    G = nx.Graph()

    # Add nodes with their scores and frequencies as attributes
    for seq, score, freq in zip(sequences, scores, frequencies):
        G.add_node(seq, score=score, frequency=freq)

    # Add edges between sequences that differ by exactly 1 amino acid
    print(f"Building edges for {len(sequences)} sequences...")
    edge_count = 0
    for i, seq1 in enumerate(sequences):
        if (i + 1) % 100 == 0:  # Progress indicator
            print(f"  Processed {i + 1}/{len(sequences)} sequences...")
        for j, seq2 in enumerate(sequences[i + 1 :], start=i + 1):
            if hamming_distance(seq1, seq2) == 1:
                G.add_edge(seq1, seq2)
                edge_count += 1

    print(f"Created {edge_count} edges")

    return G, node_data, frequency_data


def visualize_network(
    G: nx.Graph,
    node_data: Dict,
    expt_id="",
    layout="spring",
    figsize=(12, 10),
    cmap="viridis",
    node_size=200,
    # size_by="frequency",  # variable node size (commented out - uniform size used instead)
    # min_node_size=100,
    # max_node_size=1000,
    save_path=None,
):
    """
    Visualize the sequence network with nodes colored by function score.

    Args:
        G: NetworkX graph
        node_data: Dictionary mapping sequences to scores
        layout: Layout algorithm ('spring', 'circular', 'kamada_kawai', 'spectral')
        figsize: Figure size tuple
        cmap: Colormap name for heatmap coloring
        node_size: Size of nodes
        save_path: Optional path to save the figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Filter out isolated nodes (degree 0) before plotting
    n_total = G.number_of_nodes()
    isolated = [node for node, degree in G.degree() if degree == 0]
    n_isolated = len(isolated)
    G_plot = G.copy()
    G_plot.remove_nodes_from(isolated)

    # Choose layout
    if layout == "spring":
        pos = nx.spring_layout(G_plot, k=2, iterations=50, seed=42)
    elif layout == "circular":
        pos = nx.circular_layout(G_plot)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G_plot)
    elif layout == "spectral":
        pos = nx.spectral_layout(G_plot)
    else:
        pos = nx.spring_layout(G_plot, seed=42)

    # Get scores for coloring
    scores = [node_data[node] for node in G_plot.nodes()]

    # # Get sizes based on frequency or other attribute
    # if size_by == "frequency":
    #     sizes = [G_plot.nodes[node].get("frequency", 1) for node in G_plot.nodes()]
    # elif size_by == "degree":
    #     sizes = [G_plot.degree(node) for node in G_plot.nodes()]
    # else:
    #     sizes = [1] * len(G_plot.nodes())

    # # Normalize sizes to the specified range
    # if max(sizes) > min(sizes):
    #     sizes_normalized = [
    #         min_node_size
    #         + (s - min(sizes))
    #         / (max(sizes) - min(sizes))
    #         * (max_node_size - min_node_size)
    #         for s in sizes
    #     ]
    # else:
    #     sizes_normalized = [min_node_size] * len(sizes)

    # Normalize scores for colormap
    norm = mcolors.Normalize(vmin=min(scores), vmax=max(scores))
    cmap_obj = cm.get_cmap(cmap)

    # Draw network
    nx.draw_networkx_edges(G_plot, pos, alpha=0.3, width=1.5, edge_color="gray", ax=ax)

    # Draw nodes with uniform size, color based on score
    nodes = nx.draw_networkx_nodes(
        G_plot,
        pos,
        node_color=scores,
        node_size=node_size,
        cmap=cmap_obj,
        vmin=min(scores),
        vmax=max(scores),
        alpha=0.8,
        ax=ax,
    )

    # Add colorbar for scores
    cbar = plt.colorbar(nodes, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("log10 Kd (avg)", rotation=270, labelpad=20, fontsize=12)

    # # Add size legend (commented out - uniform node size used)
    # if size_by:
    #     legend_sizes = [min(sizes), np.median(sizes), max(sizes)]
    #     legend_labels = [f"{int(s)}" for s in legend_sizes]
    #     legend_handles = []
    #     for size, label in zip(legend_sizes, legend_labels):
    #         size_normalized = (
    #             min_node_size
    #             + (size - min(sizes))
    #             / (max(sizes) - min(sizes))
    #             * (max_node_size - min_node_size)
    #             if max(sizes) > min(sizes)
    #             else min_node_size
    #         )
    #         handle = plt.scatter([], [], s=size_normalized, c="gray", alpha=0.6,
    #                              edgecolors="black", linewidths=1)
    #         legend_handles.append(handle)
    #     legend = ax.legend(legend_handles, legend_labels, scatterpoints=1,
    #                        title=f"{size_by.capitalize()}", loc="upper left",
    #                        frameon=True, fontsize=10)
    #     legend.get_title().set_fontsize(12)

    # Set title and remove axes
    title = "Amino Acid Sequence Network\n(Edges connect sequences differing by 1 amino acid)"
    title += f"\n{n_total - n_isolated}/{n_total} sequences in network"
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")

    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    # plt.show()


def analyze_network(G: nx.Graph, node_data: Dict, output_file=None):
    """
    Print basic network statistics and optionally save to file.

    Args:
        G: NetworkX graph
        node_data: Dictionary mapping sequences to scores
        output_file: Optional path to save the analysis output
    """
    # Collect all output in a list
    output_lines = []

    output_lines.append("=" * 50)
    output_lines.append("NETWORK STATISTICS")
    output_lines.append("=" * 50)
    output_lines.append(f"Number of sequences (nodes): {G.number_of_nodes()}")
    output_lines.append(f"Number of connections (edges): {G.number_of_edges()}")
    output_lines.append(
        f"Average degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}"
    )
    output_lines.append(f"Network density: {nx.density(G):.4f}")
    output_lines.append(
        f"Number of connected components: {nx.number_connected_components(G)}"
    )

    # Score statistics
    scores = list(node_data.values())
    output_lines.append("\nFunction Score Statistics:")
    output_lines.append(f"  Min: {min(scores):.4f}")
    output_lines.append(f"  Max: {max(scores):.4f}")
    output_lines.append(f"  Mean: {np.mean(scores):.4f}")
    output_lines.append(f"  Std: {np.std(scores):.4f}")

    # Find most connected sequences
    degrees = dict(G.degree())
    top_connected = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]
    output_lines.append("\nTop 5 most connected sequences:")
    for seq, degree in top_connected:
        output_lines.append(
            f"  {seq}: {degree} connections (score: {node_data[seq]:.4f})"
        )

    output_lines.append("=" * 50)

    # Print to terminal
    for line in output_lines:
        print(line)

    # Save to file if path provided
    if output_file:
        with open(output_file, "w") as f:
            f.write("\n".join(output_lines))
        print(f"\nAnalysis saved to: {output_file}")


def export_for_cytoscape(
    G: nx.Graph, node_data: Dict, expt_id: str, output_prefix: str
):
    """
    Export network data in multiple formats compatible with Cytoscape.

    Args:
        G: NetworkX graph
        node_data: Dictionary mapping sequences to scores
        output_prefix: Prefix for output files (e.g., 'jh_019_network')
    """
    base_dir = os.path.dirname(output_prefix)
    base_name = os.path.basename(output_prefix)

    # 1. Export as GraphML (RECOMMENDED - preserves all attributes)
    graphml_file = os.path.join(base_dir, f"{base_name}.graphml")
    nx.write_graphml(G, graphml_file)
    print(f"GraphML file saved: {graphml_file}")

    # 2. Export as GML
    gml_file = os.path.join(base_dir, f"{base_name}.gml")
    nx.write_gml(G, gml_file)
    print(f"GML file saved: {gml_file}")

    # 3. Export node table (CSV)
    node_table = []
    degrees = dict(G.degree())
    for node in G.nodes():
        node_table.append(
            {
                "sequence": node,
                "function_score": node_data[node],
                "degree": degrees[node],
                "sequence_length": len(node),
            }
        )

    node_df = pd.DataFrame(node_table)
    node_csv = os.path.join(base_dir, f"{base_name}_nodes.csv")
    node_df.to_csv(node_csv, index=False)
    print(f"Node table saved: {node_csv}")

    # 4. Export edge table (CSV)
    edge_table = []
    for edge in G.edges():
        seq1, seq2 = edge
        # Find the position where they differ
        diff_positions = [i for i, (c1, c2) in enumerate(zip(seq1, seq2)) if c1 != c2]

        edge_table.append(
            {
                "source": seq1,
                "target": seq2,
                "interaction": "one_mutation",
                "mutation_position": diff_positions[0] if diff_positions else -1,
                "source_score": node_data[seq1],
                "target_score": node_data[seq2],
                "score_difference": abs(node_data[seq1] - node_data[seq2]),
            }
        )

    edge_df = pd.DataFrame(edge_table)
    edge_csv = os.path.join(base_dir, f"{base_name}_edges.csv")
    edge_df.to_csv(edge_csv, index=False)
    print(f"Edge table saved: {edge_csv}")

    # 5. Export as Cytoscape.js JSON
    cyjs_data = nx.cytoscape_data(G)
    cyjs_file = os.path.join(base_dir, f"{base_name}.cyjs")
    with open(cyjs_file, "w") as f:
        json.dump(cyjs_data, f, indent=2)
    print(f"Cytoscape.js JSON saved: {cyjs_file}")

    # 6. Export as adjacency list
    adjlist_file = os.path.join(base_dir, f"{base_name}.adjlist")
    nx.write_adjlist(G, adjlist_file)
    print(f"Adjacency list saved: {adjlist_file}")

    print("\n" + "=" * 50)
    print("CYTOSCAPE IMPORT INSTRUCTIONS")
    print("=" * 50)
    print("\nOption 1 (RECOMMENDED): Import GraphML")
    print("  1. Open Cytoscape")
    print("  2. File → Import → Network from File")
    print("  3. Select: {graphml_file}")
    print("  4. All node attributes (including scores) will be imported automatically")

    print("\nOption 2: Import from CSV tables")
    print("  1. Open Cytoscape")
    print("  2. File → Import → Network from File")
    print("  3. Select: {edge_csv}")
    print("  4. Set 'source' as Source Node and 'target' as Target Node")
    print("  5. File → Import → Table from File")
    print("  6. Select: {node_csv}")
    print("  7. Import as Node Table, key column: 'sequence'")

    print("\nOption 3: Import Cytoscape.js JSON")
    print("  1. Open Cytoscape")
    print("  2. File → Import → Network from File")
    print("  3. Select: {cyjs_file}")
    print("=" * 50)


def export_node_attributes_detailed(G: nx.Graph, node_data: Dict, output_file: str):
    """
    Export detailed node attributes table for Cytoscape import.
    Includes additional network metrics.

    Args:
        G: NetworkX graph
        node_data: Dictionary mapping sequences to scores
        output_file: Path to save the CSV file
    """

    # Calculate various centrality measures
    degree_cent = nx.degree_centrality(G)

    # Only calculate these if graph is connected, otherwise do per component
    if nx.is_connected(G):
        betweenness_cent = nx.betweenness_centrality(G)
        closeness_cent = nx.closeness_centrality(G)
    else:
        betweenness_cent = {node: 0 for node in G.nodes()}
        closeness_cent = {node: 0 for node in G.nodes()}

    # Clustering coefficient
    clustering = nx.clustering(G)

    # Build detailed table
    node_table = []
    for node in G.nodes():
        # Get neighbors
        neighbors = list(G.neighbors(node))
        neighbor_scores = [node_data[n] for n in neighbors]

        node_table.append(
            {
                "sequence": node,
                "function_score": node_data[node],
                "degree": G.degree(node),
                "degree_centrality": degree_cent[node],
                "betweenness_centrality": betweenness_cent[node],
                "closeness_centrality": closeness_cent[node],
                "clustering_coefficient": clustering[node],  # type:ignore
                "num_neighbors": len(neighbors),
                "avg_neighbor_score": np.mean(neighbor_scores)
                if neighbor_scores
                else 0,
                "max_neighbor_score": max(neighbor_scores) if neighbor_scores else 0,
                "min_neighbor_score": min(neighbor_scores) if neighbor_scores else 0,
                "sequence_length": len(node),
            }
        )

    df = pd.DataFrame(node_table)
    df.to_csv(output_file, index=False)
    print(f"Detailed node attributes saved: {output_file}")

    return df


# ============================================
# EXAMPLE USAGE
# ============================================


# Main execution
def main():

    # add arguments via command line

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", nargs="+", help="input mutation analysis csv files")
    parser.add_argument("-r", help="Path to reference fasta")
    parser.add_argument(
        "-c",
        default=None,
        required=False,
        help="counts per million cutoff used for generation of input csvs",
    )

    args = parser.parse_args()

    # initialize variables

    all_score_paths = args.i

    score_seq_path = [path for path in all_score_paths if isinstance(path, str)]

    print(score_seq_path)

    # extract experiment id (e.g. jh_017) from the filename
    fname = score_seq_path[0].split("/")[-3]
    match = re.search(r"(jh_\d+)", fname)
    expt_id = match.group(1) if match else fname
    print(f"experiment id: {expt_id}")

    # output goes into cleaned_libs/plots/jh_XXX/network_plots/
    cleaned_libs_dir = "/".join(score_seq_path[0].split("/")[:-1])
    dir_prefix = cleaned_libs_dir + f"/plots/network_plots/"
    os.makedirs(dir_prefix, exist_ok=True)
    print(f"output directory: {dir_prefix}")

    score_seq_df = pd.read_csv(score_seq_path[0])

    sequences = score_seq_df["aa_sequence"]

    score_seq_df["average_score"] = score_seq_df[
        ["log10_rep1_Kd", "log10_rep2_Kd"]
    ].mean(axis=1)

    function_scores = score_seq_df["average_score"]

    # Use number_aa_mutations as a proxy for node sizing (no bin read counts in titeseq CSVs)
    frequencies = score_seq_df["number_aa_mutations"].tolist()

    # print(frequencies[:10])

    # Build network
    print("Building sequence network...")
    # G, node_data = build_sequence_network(sequences, function_scores)

    G, node_data, frequency_data = build_sequence_network_with_frequency(
        sequences, function_scores, frequencies
    )

    network_txt = dir_prefix + f"{expt_id}_network_stats.txt"

    # Analyze network
    analyze_network(G, node_data, network_txt)

    # Export for Cytoscape
    print("\nExporting network data for Cytoscape...")
    export_for_cytoscape(
        G, node_data, expt_id, output_prefix=dir_prefix + f"{expt_id}_network"
    )

    # Export detailed node attributes
    export_node_attributes_detailed(
        G, node_data, output_file=dir_prefix + f"{expt_id}_network_nodes_detailed.csv"
    )

    layouts = ["spring", "circular", "kamada_kawai", "spectral"]

    # Visualize network in each network style

    for style in layouts:
        print(f"\nVisualizing network ({style})...")
        visualize_network(
            G,
            node_data,
            expt_id,
            layout=str(style),  # Try 'circular', 'kamada_kawai', 'spectral'
            cmap="RdYlGn_r",  # _r suffix inverts the map: green=low log10 Kd (tight binder), red=high (weak)
            node_size=200,
            # size_by="frequency",  # variable node size (commented out)
            # min_node_size=100,
            # max_node_size=1000,
            save_path=dir_prefix + f"{expt_id}_sequence_network_{style}.png",
        )  # Set to None to not save
        # save_path = dir_prefix + f'{expt_id}_sequence_network_{style}.png'  # Set to None to not save


if __name__ == "__main__":
    main()

# if __name__ == "__main__":
#     # Example data - replace with your actual data
#     sequences = [
#         "ACDEFG",
#         "ACDEFH",  # 1 diff from ACDEFG
#         "ACDEFI",  # 1 diff from ACDEFG
#         "BCDEFG",  # 1 diff from ACDEFG
#         "ACDEFF",  # 1 diff from ACDEFG
#         "ACDEFJ",  # 1 diff from ACDEFG
#         "ACDEFK",  # 1 diff from ACDEFG
#         "BCDEFF",  # 1 diff from BCDEFG and ACDEFF
#         "BCDEFH",  # 1 diff from BCDEFG and ACDEFH
#     ]

#     # Example function scores - replace with your actual scores
#     scores = [0.5, 0.7, 0.3, 0.8, 0.6, 0.9, 0.4, 0.75, 0.85]

#     # Build network
#     print("Building sequence network...")
#     G, node_data = build_sequence_network(sequences, scores)

#     # Analyze network
#     analyze_network(G, node_data)
#     # Visualize network
#     print("\nVisualizing network...")
#     visualize_network(G, node_data,
#                      layout='spring',  # Try 'circular', 'kamada_kawai', 'spectral'
#                      cmap='RdYlGn',    # Try 'viridis', 'plasma', 'coolwarm', 'RdYlGn'
#                      node_size=800,
#                      save_path='sequence_network.png')  # Set to None to not save
