#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 2026 May 13
@author: jhambale

17_drift_dispersity.py
==================
intakes input library of sequences and displays the amino acid sequences as a 2D dimensionally reduced scatter plot
dimension reduction done on one-hot encoded sequence data converted with PaCMAP

then clusters the reduced data with kmeans++ and outputs a sequence logo of each cluster

lastly calculates pairwise distances between the following groups:
    - all sequences
    - each cluster
    - a user input sample of sequences (randomly sampled from the input library) repeated a user input number of times

Usage
-----
python 17_drift_dispersity.py \\
    -i ../../titeseq_data/outputs/jh_002/tite_seq_Kds2/cluster_eval_tite_seq_Kds_passed_variants/tite_seq_clustered_tite_seq_Kds_passed_variants.csv \\
    -k 3 -s 50 -n 20 --seed 42
"""

import argparse
import re
import sys
from pathlib import Path

import logomaker
import matplotlib.axes
import matplotlib.figure
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pacmap
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Ensure utils_nd (in the same src/ directory) is importable
sys.path.insert(0, str(Path(__file__).parent))
from utils_nd import hamming_distance, make_dir  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Canonical 20 standard amino acids (alphabetically ordered for one-hot index)
AA_ALPHABET = list("ACDEFGHIKLMNPQRSTVWY")  # length 20
AA_INDEX = {aa: i for i, aa in enumerate(AA_ALPHABET)}

# Category colors — shared across the codebase
CATEGORY_COLORS = {
    "Tight Binder": "#1a7a4a",
    "Moderate Binder": "#3B71B2",
    "Weak Binder": "grey",
}

# Per-amino-acid colors for sequence logos
#   Groups: hydrophobic → orange, positively charged → blue,
#           negatively charged → red, polar uncharged → green,
#           special (C, G, P) → purple
AA_COLORS = {
    # Hydrophobic
    "A": "#E8853D",
    "V": "#E8853D",
    "I": "#E8853D",
    "L": "#E8853D",
    "M": "#E8853D",
    "F": "#E8853D",
    "W": "#E8853D",
    "Y": "#E87A3D",  # slightly darker orange (aromatic + hydroxyl)
    # Positively charged
    "R": "#3B71B2",
    "K": "#3B71B2",
    "H": "#6B9BD2",
    # Negatively charged
    "D": "#C0392B",
    "E": "#C0392B",
    # Polar uncharged
    "S": "#27AE60",
    "T": "#27AE60",
    "N": "#27AE60",
    "Q": "#27AE60",
    # Special
    "C": "#8E44AD",
    "G": "#8E44AD",
    "P": "#8E44AD",
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def one_hot_encode(sequences: list[str]) -> np.ndarray:
    """
    One-hot encode a list of amino acid sequences of equal length.

    Each position is encoded as a 20-dimensional binary vector, one entry
    per amino acid in AA_ALPHABET.  Unknown residues map to an all-zero
    vector for that position.

    Parameters
    ----------
    sequences : list[str]
        List of amino acid sequences (all the same length).

    Returns
    -------
    X : np.ndarray, shape (n_seqs, seq_len * 20)
        Flattened one-hot matrix.
    """
    n = len(sequences)
    seq_len = len(sequences[0])
    X = np.zeros((n, seq_len * len(AA_ALPHABET)), dtype=np.float32)
    for i, seq in enumerate(sequences):
        for pos, aa in enumerate(seq):
            if aa in AA_INDEX:
                X[i, pos * len(AA_ALPHABET) + AA_INDEX[aa]] = 1.0
    return X


def compute_pairwise_hamming(sequences: list[str]) -> list[int]:
    """
    Compute all pairwise Hamming distances for a list of sequences.

    Parameters
    ----------
    sequences : list[str]
        Amino acid sequences of equal length.

    Returns
    -------
    distances : list[int]
        Flat list of n*(n-1)/2 pairwise distances.
    """
    distances = []
    n = len(sequences)
    for i in range(n):
        for j in range(i + 1, n):
            distances.append(hamming_distance(sequences[i], sequences[j]))
    return distances


def save_fig(fig: matplotlib.figure.Figure, base_path: str) -> None:
    """Save a matplotlib figure as both .png (dpi=400) and .svg."""
    fig.savefig(f"{base_path}.png", dpi=400, bbox_inches="tight")
    fig.savefig(f"{base_path}.svg", bbox_inches="tight")
    print(f"  saved → {base_path}.png / .svg")


def apply_style(ax: matplotlib.axes.Axes) -> None:
    """Remove top and right spines (house style)."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------------------------------------------------------------------
# Step 4 helper — PaCMAP scatter coloured by predicted_category
# ---------------------------------------------------------------------------


def plot_umap_category(
    embedding: np.ndarray,
    df: pd.DataFrame,
    expt_id: str,
    plots_dir: str,
) -> None:
    """Scatter PaCMAP embedding coloured by predicted_category."""
    print("\n[Step 4] Plotting PaCMAP scatter (coloured by predicted_category)…")
    sns.set_style("ticks")
    fig, ax = plt.subplots(figsize=(8, 6))

    for category, color in CATEGORY_COLORS.items():
        mask = df["predicted_category"] == category
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=color,
            s=10,
            alpha=0.6,
            linewidths=0,
            label=category,
        )

    # Legend via Patch handles so we control colour explicitly
    legend_handles = [
        mpatches.Patch(color=color, label=cat) for cat, color in CATEGORY_COLORS.items()
    ]
    ax.legend(handles=legend_handles, frameon=False, fontsize=9)

    ax.set_xlabel("PaCMAP 1")
    ax.set_ylabel("PaCMAP 2")
    ax.set_title(f"PaCMAP of aa sequences ({expt_id})")
    apply_style(ax)
    fig.tight_layout()

    out_base = str(Path(plots_dir) / f"pacmap_scatter_{expt_id}")
    save_fig(fig, out_base)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Step 6 helper — PaCMAP scatter coloured by KMeans cluster
# ---------------------------------------------------------------------------


def plot_umap_kmeans(
    embedding: np.ndarray,
    df: pd.DataFrame,
    k: int,
    expt_id: str,
    plots_dir: str,
) -> None:
    """Scatter PaCMAP embedding coloured by KMeans cluster index."""
    print("\n[Step 6] Plotting PaCMAP scatter (coloured by KMeans cluster)…")
    sns.set_style("ticks")
    tab10 = plt.cm.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(8, 6))

    cluster_labels = df["kmeans_cluster"].values
    for c in range(k):
        mask = cluster_labels == c
        color = tab10(c / 10)
        ax.scatter(
            embedding[mask, 0],  # type: ignore[index]
            embedding[mask, 1],  # type: ignore[index]
            c=[color],
            s=10,
            alpha=0.6,
            linewidths=0,
            label=f"Cluster {c + 1}",
        )

    # Annotate cluster centres
    for c in range(k):
        mask = cluster_labels == c
        cx = embedding[mask, 0].mean()  # type: ignore[index]
        cy = embedding[mask, 1].mean()  # type: ignore[index]
        ax.text(
            cx,
            cy,
            str(c + 1),
            fontsize=11,
            fontweight="bold",
            ha="center",
            va="center",
            color="white",
            bbox=dict(
                boxstyle="round,pad=0.25", fc=tab10(c / 10), ec="none", alpha=0.8
            ),
        )

    ax.legend(frameon=False, fontsize=9)
    ax.set_xlabel("PaCMAP 1")
    ax.set_ylabel("PaCMAP 2")
    ax.set_title(f"PaCMAP KMeans clusters (k={k}, {expt_id})")
    apply_style(ax)
    fig.tight_layout()

    out_base = str(Path(plots_dir) / f"pacmap_kmeans_{expt_id}")
    save_fig(fig, out_base)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Step 7 helper — per-cluster sequence logos (pure matplotlib)
# ---------------------------------------------------------------------------


def plot_sequence_logo(
    sequences: list[str],
    cluster_idx: int,
    expt_id: str,
    plots_dir: str,
) -> None:
    """
    Build a stacked-letter sequence logo for a single cluster using logomaker.

    logomaker.alignment_to_matrix converts the aligned sequences directly
    into a probability matrix (rows = positions, columns = amino acids).
    logomaker.Logo renders each column as frequency-scaled letter glyphs
    stacked with the most frequent character on top.  Letters are coloured
    by amino-acid chemical grouping via AA_COLORS.
    """
    n = len(sequences)
    seq_len = len(sequences[0])

    # Build a raw counts matrix with no pseudocount so that positions with
    # a single dominant AA (e.g. conserved position 1 = 'Q') produce zero
    # counts — and therefore zero probability — for every other character.
    # Using pseudocount=0 means only characters actually observed are non-zero.
    counts_df = logomaker.alignment_to_matrix(
        sequences,
        to_type="counts",
        characters_to_ignore=".-",  # ignore gap / stop chars only
        pseudocount=0,
    )
    # Restrict to the 20 standard AAs; any unexpected characters are dropped
    counts_df = counts_df.reindex(columns=list(AA_ALPHABET), fill_value=0.0)

    # Normalise each row to get true observed frequencies (sum = 1 per position)
    row_sums = counts_df.sum(axis=1)
    prob_df = counts_df.div(row_sums, axis=0).fillna(0.0)

    sns.set_style("ticks")
    fig, ax = plt.subplots(figsize=(max(14, seq_len * 0.15), 4))

    logo = logomaker.Logo(
        prob_df,
        color_scheme=AA_COLORS,  # per-AA hex colors from AA_COLORS dict
        stack_order="big_on_top",  # most frequent letter on top
        # font_name="Arial",
        ax=ax,
    )

    # Style spines to match the rest of the script
    logo.style_spines(spines=("top", "right"), visible=False)
    logo.style_spines(spines=("left", "bottom"), visible=True, linewidth=0.5)

    # x-ticks: sparse, every 10 positions, 1-based labels
    logo.style_xticks(anchor=0, spacing=10, fmt="%d", rotation=0, fontsize=7)
    # logomaker uses 0-based row index; shift tick labels to 1-based
    ax.set_xticklabels(
        [str(int(t.get_text()) + 1) for t in ax.get_xticklabels()],
        fontsize=7,
    )

    ax.set_xlabel("Position")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Cluster {cluster_idx + 1} sequence logo (n={n})")
    ax.tick_params(width=0.5)
    fig.tight_layout()

    out_base = str(Path(plots_dir) / f"seqlogo_cluster{cluster_idx + 1}_{expt_id}")
    save_fig(fig, out_base)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Step 8 helpers — pairwise Hamming distance histograms
# ---------------------------------------------------------------------------


def plot_hamming_histogram(
    distances: list[int],
    title: str,
    base_path: str,
    color: str = "#3B71B2",
    n_seqs: int = 0,
) -> None:
    """
    Plot and save a histogram of pairwise Hamming distances.

    Formatted to match the aa-distance-from-parent plots in script 14:
    sns.histplot with stat='probability', integer x-ticks with half-step
    offset, thin spines/ticks, and a vertical median line.
    """
    arr = np.array(distances)
    median_dist = float(np.median(arr)) if len(arr) else 0.0

    sns.set_style("ticks")
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.histplot(
        arr,
        binwidth=1,
        stat="probability",
        color=color,
        edgecolor="black",
        linewidth=0.5,
        ax=ax,
    )
    ax.axvline(
        median_dist,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label=f"median = {median_dist:.1f}",
    )
    ax.legend(frameon=False, fontsize=9)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ticks = ax.get_xticks()
    if len(ticks) > 1:
        new_ticks = ticks[:-1] + 0.5
        ax.set_xticks(new_ticks)
        ax.set_xticklabels([str(int(t)) for t in new_ticks])
        ax.set_xlim([new_ticks[0] + 0.5, new_ticks[-1] - 0.5])

    ax.set_xlabel("pairwise Hamming distance")
    n_label = n_seqs if n_seqs else len(distances)
    ax.set_ylabel(f"frequency (n={n_label:,} sequences)")
    ax.set_title(title)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_linewidth(0.5)
    ax.tick_params(width=0.5)
    fig.tight_layout()
    save_fig(fig, base_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# K-means optimisation
# ---------------------------------------------------------------------------


def optimize_k_means(
    embedding: np.ndarray,
    max_k: int,
    expt_id: str,
    plots_dir: str,
    q_threshold: float = 0.5,
    seed: int = 42,
) -> int:
    """
    Determine the optimal number of KMeans clusters for ``embedding``.

    For each k in [2, max_k) the function records inertia and silhouette
    score, then combines them into a weighted composite score:

        weighted_score = 0.4 * norm_silhouette
                       + 0.3 * (1 - norm_inertia)
                       + 0.3 * |Δnorm_inertia|   (elbow contribution)

    The smallest k whose weighted score >= q_threshold is selected.  If
    no k meets the threshold the k with the highest weighted score is used.

    Diagnostic plots (inertia elbow and silhouette score vs. k) are saved
    to plots_dir.

    Parameters
    ----------
    embedding : np.ndarray
        2-D PaCMAP embedding of shape (n_seqs, 2).
    max_k : int
        Upper bound (exclusive) on the number of clusters to evaluate.
    expt_id : str
        Experiment identifier used in plot titles and filenames.
    plots_dir : str
        Directory in which diagnostic plots are saved.
    q_threshold : float
        Minimum weighted score required to select a k (default 0.5).
    seed : int
        Random seed for KMeans reproducibility.

    Returns
    -------
    opt_k : int
        Optimal number of clusters.
    """
    if max_k < 3:
        print("  [WARN] max_k < 3 — defaulting to k=2.")
        return 2

    means: list[int] = []
    inertias: list[float] = []
    sil_scores: list[float] = []
    results: list[dict] = []

    for k in range(2, max_k):  # evaluate k = 2 … max_k-1 (inclusive)
        km = KMeans(n_clusters=k, init="k-means++", random_state=seed, n_init="auto")
        labels = km.fit_predict(embedding)
        inertia = float(km.inertia_)
        sil = float(silhouette_score(embedding, labels))

        means.append(k)
        inertias.append(inertia)
        sil_scores.append(sil)
        results.append({"k": k, "inertia": inertia, "silhouette_score": sil})
        print(f"    k={k:2d}  inertia={inertia:.2f}  silhouette={sil:.4f}")

    # --- diagnostic plot 1: inertia elbow ---
    sns.set_style("ticks")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(means, inertias, "o-", color="#3B71B2", linewidth=1.5, markersize=5)
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Inertia")
    ax.set_title(f"KMeans inertia elbow ({expt_id})")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_linewidth(0.5)
    ax.tick_params(width=0.5)
    fig.tight_layout()
    save_fig(fig, str(Path(plots_dir) / f"kmeans_inertia_{expt_id}"))
    plt.close(fig)

    # --- diagnostic plot 2: silhouette score ---
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(means, sil_scores, "o-", color="#1a7a4a", linewidth=1.5, markersize=5)
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette score")
    ax.set_title(f"KMeans silhouette score ({expt_id})")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_linewidth(0.5)
    ax.tick_params(width=0.5)
    fig.tight_layout()
    save_fig(fig, str(Path(plots_dir) / f"kmeans_silhouette_{expt_id}"))
    plt.close(fig)

    # --- composite weighted score (ported from kmeanscluster.py) ---
    sel = pd.DataFrame(results[:-1])  # drop last row (same as source)

    sel["norm_inertia"] = (sel["inertia"] - sel["inertia"].min()) / (
        sel["inertia"].max() - sel["inertia"].min()
    )
    sel["inertia_score"] = 1 - sel["norm_inertia"]

    sel["norm_sil"] = (sel["silhouette_score"] - sel["silhouette_score"].min()) / (
        sel["silhouette_score"].max() - sel["silhouette_score"].min()
    )

    sel["inertia_change"] = sel["norm_inertia"].diff().abs().fillna(0)

    sel["weighted_score"] = (
        sel["norm_sil"] * 0.4 + sel["inertia_score"] * 0.3 + sel["inertia_change"] * 0.3
    )

    print("\n  Weighted scores per k:")
    for _, row in sel.iterrows():
        print(f"    k={int(row['k']):2d}  weighted_score={row['weighted_score']:.4f}")

    filtered = sel[sel["weighted_score"] >= q_threshold]
    if not filtered.empty:
        opt_k = int(filtered["k"].min())
    else:
        opt_k = int(sel.loc[sel["weighted_score"].idxmax(), "k"])

    print(f"  Optimal k = {opt_k} (threshold={q_threshold})")
    return opt_k


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "PaCMAP dimensionality reduction, KMeans++ clustering, sequence logos, "
            "and pairwise Hamming distance analysis of an amino-acid sequence library."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "-i",
        required=True,
        metavar="CSV",
        help=(
            "Path to input CSV containing at minimum 'aa_sequence' and "
            "'predicted_category' columns."
        ),
    )
    p.add_argument(
        "-k",
        dest="max_k",
        type=int,
        default=10,
        metavar="INT",
        help="Maximum number of KMeans clusters to evaluate when optimising k (default: 10).",
    )
    p.add_argument(
        "-q",
        dest="q_threshold",
        type=float,
        default=0.5,
        metavar="FLOAT",
        help="Minimum weighted score threshold for k selection (default: 0.5).",
    )
    p.add_argument(
        "-s",
        type=int,
        default=50,
        metavar="INT",
        help="Sample size for random sub-sampling pairwise distance repeats (default: 50).",
    )
    p.add_argument(
        "-n",
        type=int,
        default=20,
        metavar="INT",
        help="Number of random sub-sampling repeats (default: 20).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="INT",
        help="Global random seed (default: 42).",
    )
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)

    csv_path = Path(args.i)
    max_k: int = args.max_k
    q_threshold: float = args.q_threshold
    sample_size: int = args.s
    n_repeats: int = args.n
    seed: int = args.seed

    # -----------------------------------------------------------------------
    # 1. LOAD DATA
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("[Step 1] Loading data…")
    print("=" * 60)

    if not csv_path.is_file():
        print(f"[ERROR] Input CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    # Extract experiment id from the parent-directory path (e.g. "jh_002")
    match = re.search(r"(jh_\d+)", str(csv_path))
    expt_id: str = match.group(1) if match else csv_path.stem
    print(f"  Experiment ID : {expt_id}")

    # Output directories
    out_dir = csv_path.parent / "drift_dispersity"
    plots_dir = out_dir / "plots"
    make_dir(str(out_dir))
    make_dir(str(plots_dir))

    df = pd.read_csv(csv_path)
    print(f"  Sequences loaded : {len(df):,}")

    # Validate required columns
    for col in ("aa_sequence", "predicted_category"):
        if col not in df.columns:
            print(f"[ERROR] CSV must contain column '{col}'.", file=sys.stderr)
            sys.exit(1)

    # Work with clean list of strings
    sequences: list[str] = df["aa_sequence"].astype(str).tolist()
    n_seqs: int = len(sequences)

    # -----------------------------------------------------------------------
    # 2. ONE-HOT ENCODING
    # -----------------------------------------------------------------------
    print("\n[Step 2] One-hot encoding sequences…")
    X = one_hot_encode(sequences)
    print(f"  One-hot matrix shape : {X.shape}")  # (n_seqs, seq_len * 20)

    # -----------------------------------------------------------------------
    # 3. PaCMAP DIMENSIONALITY REDUCTION
    # -----------------------------------------------------------------------
    print("\n[Step 3] Running PaCMAP (n_components=2)…")
    reducer = pacmap.PaCMAP(n_components=2, random_state=seed)
    pacmap_embedding: np.ndarray = reducer.fit_transform(X)
    print(f"  PaCMAP embedding shape : {pacmap_embedding.shape}")

    # -----------------------------------------------------------------------
    # 4. SCATTER PLOT — coloured by predicted_category
    # -----------------------------------------------------------------------
    plot_umap_category(pacmap_embedding, df, expt_id, str(plots_dir))

    # -----------------------------------------------------------------------
    # 5. OPTIMIZE k, THEN FIT KMEANS++
    # -----------------------------------------------------------------------
    print(f"\n[Step 5] Optimising k (max_k={max_k})…")
    k: int = optimize_k_means(
        pacmap_embedding,
        max_k=max_k,
        expt_id=expt_id,
        plots_dir=str(plots_dir),
        q_threshold=q_threshold,
        seed=seed,
    )
    print(f"  Fitting KMeans++ with optimal k={k}…")
    kmeans = KMeans(n_clusters=k, init="k-means++", random_state=seed, n_init="auto")
    cluster_labels: np.ndarray = kmeans.fit_predict(pacmap_embedding)
    df["kmeans_cluster"] = cluster_labels

    for c in range(k):
        count = int((cluster_labels == c).sum())
        print(f"  Cluster {c + 1} : {count:,} sequences")

    # -----------------------------------------------------------------------
    # 6. SCATTER PLOT — coloured by KMeans cluster
    # -----------------------------------------------------------------------
    plot_umap_kmeans(pacmap_embedding, df, k, expt_id, str(plots_dir))

    # -----------------------------------------------------------------------
    # 7. SEQUENCE LOGOS PER CLUSTER
    # -----------------------------------------------------------------------
    print("\n[Step 7] Generating per-cluster sequence logos…")
    for c in range(k):
        mask = df["kmeans_cluster"] == c
        cluster_seqs = df.loc[mask, "aa_sequence"].astype(str).tolist()
        print(f"  Cluster {c + 1} — {len(cluster_seqs):,} sequences")
        plot_sequence_logo(cluster_seqs, c, expt_id, str(plots_dir))

    # -----------------------------------------------------------------------
    # 8. PAIRWISE HAMMING DISTANCES — three groups
    # -----------------------------------------------------------------------
    print("\n[Step 8] Computing pairwise Hamming distances…")

    # --- Group A: all sequences ---
    print(f"\n  Group A — all sequences (n={n_seqs:,})…")
    print("  (this may take a moment for large libraries)")
    all_distances = compute_pairwise_hamming(sequences)
    print(f"    Computed {len(all_distances):,} pairwise distances.")
    plot_hamming_histogram(
        distances=all_distances,
        title=f"Pairwise distances — all sequences ({expt_id})",
        base_path=str(plots_dir / f"pairwise_dist_all_{expt_id}"),
        color="#3B71B2",
        n_seqs=n_seqs,
    )

    # --- Group B: each cluster individually, then combined overlay ---
    print("\n  Group B — per-cluster distances…")
    tab10 = plt.cm.get_cmap("tab10")
    cluster_distances: dict[int, list[int]] = {}

    for c in range(k):
        mask = df["kmeans_cluster"] == c
        c_seqs = df.loc[mask, "aa_sequence"].astype(str).tolist()
        print(f"    Cluster {c + 1} (n={len(c_seqs):,})…")
        dists = compute_pairwise_hamming(c_seqs)
        cluster_distances[c] = dists
        print(f"      {len(dists):,} pairwise distances computed.")

        cluster_color = tab10(c / 10)
        plot_hamming_histogram(
            distances=dists,
            title=f"Pairwise distances — cluster {c + 1} ({expt_id})",
            base_path=str(plots_dir / f"pairwise_dist_cluster{c + 1}_{expt_id}"),
            color=cluster_color,
            n_seqs=len(c_seqs),
        )

    # Combined overlay histogram (bins only, no KDE)
    print("  Plotting combined cluster overlay histogram…")

    sns.set_style("ticks")
    fig, ax = plt.subplots(figsize=(7, 5))
    for c in range(k):
        dists_c = cluster_distances[c]
        arr_c = np.array(dists_c)
        c_mean = float(np.mean(arr_c)) if len(arr_c) else np.nan
        c_med = float(np.median(arr_c)) if len(arr_c) else np.nan
        n_c = (df["kmeans_cluster"] == c).sum()
        color_c = tab10(c / 10)
        sns.histplot(
            arr_c,
            binwidth=1,
            stat="probability",
            color=color_c,
            edgecolor="black",
            linewidth=0.5,
            alpha=0.4,
            label=f"Cluster {c + 1} (n={n_c:,} [{c_mean:.1f}|{c_med:.1f}])",
            ax=ax,
        )

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ticks = ax.get_xticks()
    if len(ticks) > 1:
        new_ticks = ticks[:-1] + 0.5
        ax.set_xticks(new_ticks)
        ax.set_xticklabels([str(int(t)) for t in new_ticks])
        ax.set_xlim([new_ticks[0] + 0.5, new_ticks[-1] - 0.5])
    ax.set_xlabel("pairwise Hamming distance")
    ax.set_ylabel(f"probability (n={n_seqs:,} sequences)")
    ax.set_title(f"Pairwise distances — all clusters ({expt_id})")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_linewidth(0.5)
    ax.tick_params(width=0.5)
    fig.tight_layout()
    save_fig(fig, str(plots_dir / f"pairwise_dist_all_clusters_{expt_id}"))
    plt.close(fig)

    # --- Group C: random sub-sampling ---
    print(f"\n  Group C — {n_repeats} random sub-samples of {sample_size} sequences…")
    if sample_size > n_seqs:
        print(
            f"  [WARN] sample_size ({sample_size}) > n_seqs ({n_seqs}); "
            "clamping to n_seqs."
        )
        sample_size = n_seqs

    median_pairwise_dists: list[float] = []
    for repeat in range(n_repeats):
        rng = np.random.default_rng(seed + repeat)
        indices = rng.choice(n_seqs, size=sample_size, replace=False)
        sampled_seqs = [sequences[idx] for idx in indices]
        dists = compute_pairwise_hamming(sampled_seqs)
        median_dist = float(np.median(dists)) if dists else 0.0
        median_pairwise_dists.append(median_dist)
        if (repeat + 1) % 5 == 0 or repeat == 0:
            print(
                f"    repeat {repeat + 1}/{n_repeats} — median dist = {median_dist:.3f}"
            )

    # Summary stats
    arr = np.array(median_pairwise_dists)
    print(
        f"\n  Random sub-sample summary ({n_repeats} repeats of {sample_size} seqs):\n"
        f"    mean   = {arr.mean():.4f}\n"
        f"    std    = {arr.std():.4f}\n"
        f"    min    = {arr.min():.4f}\n"
        f"    max    = {arr.max():.4f}\n"
        f"    median = {float(np.median(arr)):.4f}"
    )

    sns.set_style("ticks")
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.histplot(
        arr,
        binwidth=1,
        stat="probability",
        color="#1a7a4a",
        edgecolor="black",
        linewidth=0.5,
        ax=ax,
    )
    overall_median = float(np.median(arr))
    ax.axvline(
        overall_median,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label=f"median = {overall_median:.2f}",
    )
    ax.legend(frameon=False, fontsize=9)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ticks = ax.get_xticks()
    if len(ticks) > 1:
        new_ticks = ticks[:-1] + 0.5
        ax.set_xticks(new_ticks)
        ax.set_xticklabels([str(int(t)) for t in new_ticks])
        ax.set_xlim([new_ticks[0] + 0.5, new_ticks[-1] - 0.5])
    ax.set_xlabel("median pairwise Hamming distance")
    ax.set_ylabel(f"frequency (n={n_repeats} repeats)")
    ax.set_title(
        f"Median pairwise dist — {n_repeats} random samples of {sample_size} seqs ({expt_id})"
    )
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_linewidth(0.5)
    ax.tick_params(width=0.5)
    fig.tight_layout()
    save_fig(
        fig,
        str(
            plots_dir / f"pairwise_dist_{n_repeats}samples_{sample_size}seqs_{expt_id}"
        ),
    )
    plt.close(fig)

    # -----------------------------------------------------------------------
    # 9. SAVE SUMMARY CSV
    # -----------------------------------------------------------------------
    print("\n[Step 9] Saving summary CSV…")
    out_csv = out_dir / f"{expt_id}_drift_dispersity.csv"
    df.to_csv(str(out_csv), index=False)
    print(f"  Summary CSV saved → {out_csv}")

    print(f"\nDone. Results saved to {out_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
